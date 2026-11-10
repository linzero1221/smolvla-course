#!/usr/bin/env python3
"""Prepare the LIBERO-Object training subset.

Steps
1. Map the LIBERO-Object BDDL tasks in the simulator to the task indices of
   ``lerobot/libero`` by matching the original English instructions.
2. Audit the BDDL ``:objects`` lists and confirm that neither held-out object
   appears in any of the three training scenes.
3. Write ``data/task_map.csv`` and ``data/object_audit.csv``.
4. Write ``data/train_episodes.csv`` with one row per kept episode.
5. Optionally export ``data/libero_object_train/`` with only the training
   episodes and re-aggregated statistics.

Usage
    python scripts/prepare_data.py --config configs/experiment.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def simulator_tasks() -> list[dict]:
    """Return the LIBERO-Object task list as ordered by the simulator."""
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    tasks = []
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        tasks.append(
            {
                "sim_task_id": task_id,
                "bddl_stem": task.name,
                "language": task.language,
                "n_init_states": len(suite.get_task_init_states(task_id)),
            }
        )
    return tasks


def bddl_objects(bddl_stem: str) -> tuple[list[str], list[str]]:
    """Parse ``:objects`` and ``:obj_of_interest`` from a BDDL file."""
    from libero.libero import get_libero_path

    path = Path(get_libero_path("bddl_files")) / "libero_object" / f"{bddl_stem}.bddl"
    text = path.read_text(encoding="utf-8")

    def block(name: str) -> list[str]:
        match = re.search(rf"\(:{name}(.*?)\)", text, re.S)
        if not match:
            raise ValueError(f"BDDL block :{name} not found in {path}")
        return [line.strip() for line in match.group(1).splitlines() if line.strip()]

    objects = []
    for line in block("objects"):
        instance, _, obj_type = line.partition("-")
        objects.append(obj_type.strip() or instance.strip())
    interests = [item.split()[0] for item in block("obj_of_interest") if item.split()]
    return objects, interests


def dataset_task_table(dataset_root: Path) -> dict[str, int]:
    """Map task text to task index from the dataset metadata."""
    import pandas as pd

    table = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    texts = table.index.astype(str).tolist()
    indices = table["task_index"].astype(int).tolist()
    return dict(zip(texts, indices, strict=True))


def episode_table(dataset_root: Path) -> list[dict]:
    """Read episode metadata and attach the task index of every episode.

    ``meta/episodes`` does not store ``task_index``, so the mapping is read from
    the data parquet shards (only two small columns).
    """
    import pandas as pd

    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError("No episode metadata found; download meta/episodes first")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    shards = sorted((dataset_root / "data").rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError("No data shards found; download the full dataset first")
    tasks = pd.concat(
        [
            pd.read_parquet(path, columns=["episode_index", "task_index"]).drop_duplicates("episode_index")
            for path in shards
        ],
        ignore_index=True,
    ).drop_duplicates("episode_index")
    frame = frame.merge(tasks, on="episode_index", how="left", validate="one_to_one")
    if frame["task_index"].isna().any():
        missing = frame.loc[frame["task_index"].isna(), "episode_index"].tolist()
        raise ValueError(f"episodes without a task index: {missing[:10]}")
    frame["task_index"] = frame["task_index"].astype(int)
    return frame.sort_values("episode_index").to_dict(orient="records")


def resolve_dataset_root(config: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=config["dataset"]["repo_id"],
            revision=config["dataset"]["revision"],
            repo_type="dataset",
        )
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_train_subset(config: dict, dataset_root: Path, output_dir: Path, keep: list[int]) -> None:
    """Export a dataset with only ``keep`` episodes and re-aggregated stats."""
    from lerobot.datasets.dataset_tools import delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=config["dataset"]["repo_id"],
        root=dataset_root,
        video_backend=config["dataset"]["video_backend"],
    )

    # lerobot/libero stores episode metadata without the `meta/episodes/*` index
    # columns or the per-episode `tasks` column that LeRobot 0.6.1's
    # delete_episodes expects. Add them in memory (single meta file, chunk 0 /
    # file 0) instead of patching the library.
    import pandas as pd

    episode_tasks = {}
    for shard in sorted((dataset_root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(shard, columns=["episode_index", "task_index"]).drop_duplicates("episode_index")
        for row in frame.itertuples(index=False):
            episode_tasks[int(row.episode_index)] = [dataset.meta.tasks.iloc[int(row.task_index)].name]

    episodes = dataset.meta.episodes
    if "tasks" not in episodes.features:
        episodes = episodes.add_column("tasks", [episode_tasks[i] for i in range(len(episodes))])
    for column in ("meta/episodes/chunk_index", "meta/episodes/file_index"):
        if column not in episodes.features:
            episodes = episodes.add_column(column, [0] * len(episodes))
    dataset.meta.episodes = episodes

    drop = [index for index in range(dataset.meta.total_episodes) if index not in set(keep)]
    if output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists; remove it or pick another output")
    result = delete_episodes(
        dataset,
        episode_indices=drop,
        output_dir=output_dir,
        repo_id="course/libero_object_train",
    )

    # The source episode metadata has no per-episode stats, so delete_episodes
    # writes an empty stats.json. Recompute numeric statistics (state, action)
    # from the retained frames only. Image/video stats are left untouched; the
    # policy normalizes them separately and the training config uses the
    # dataset's numeric stats for state/action normalization.
    from lerobot.datasets.dataset_tools import recompute_stats

    result = recompute_stats(result, skip_image_video=True)

    total_frames = int(result.meta.total_frames)
    stats_path = output_dir / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for feature in ("action", "observation.state"):
        count = stats[feature]["count"]
        count = count[0] if isinstance(count, list) else count
        assert int(count) == total_frames, f"{feature} stats were not re-aggregated"
    assert int(result.meta.total_episodes) == len(keep)

    tasks = {str(task) for task in result.meta.tasks.index}
    expected = {task["bddl_stem"].replace("_", " ") for task in config["tasks"]["seen"]}
    unexpected = tasks - expected
    assert not unexpected, f"unexpected tasks in training subset: {unexpected}"
    print(f"training subset: {result.meta.total_episodes} episodes, {total_frames} frames, {len(tasks)} tasks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiment.json")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--export-train", action="store_true", help="export data/libero_object_train/")
    args = parser.parse_args()

    from libero.libero import get_libero_path  # noqa: F401  (fail early if the env is missing)

    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config, args.dataset_root)
    tasks = simulator_tasks()
    task_index = dataset_task_table(dataset_root)
    episodes = episode_table(dataset_root)

    seen = {item["bddl_stem"]: item for item in config["tasks"]["seen"]}
    unseen = {item["bddl_stem"]: item for item in config["tasks"]["unseen"]}

    task_rows = []
    audit_rows = []
    for task in tasks:
        stem = task["bddl_stem"]
        language = task["language"]
        if language not in task_index:
            raise KeyError(f"task text not found in dataset: {language!r}")
        split = "seen" if stem in seen else "unseen" if stem in unseen else "other"
        objects, interests = bddl_objects(stem)
        task_episodes = [row for row in episodes if int(row["task_index"]) == task_index[language]]

        task_rows.append(
            {
                "bddl_stem": stem,
                "object": stem.removeprefix("pick_up_the_").removesuffix(
                    "_and_place_it_in_the_basket"
                ),
                "split": split,
                "sim_task_id": task["sim_task_id"],
                "sim_language": language,
                "dataset_task_index": task_index[language],
                "dataset_task_text": language,
                "n_init_states": task["n_init_states"],
                "n_episodes": len(task_episodes),
                "n_frames": sum(int(row["length"]) for row in task_episodes),
            }
        )
        audit_rows.append(
            {
                "bddl_stem": stem,
                "split": split,
                "target_object": stem.removeprefix("pick_up_the_").removesuffix(
                    "_and_place_it_in_the_basket"
                ),
                "obj_of_interest": ";".join(interests),
                "n_objects": len(objects),
                "objects": ";".join(objects),
                "contains_bbq_sauce": "bbq_sauce" in objects,
                "contains_chocolate_pudding": "chocolate_pudding" in objects,
            }
        )

    write_csv(
        args.output_dir / "task_map.csv",
        task_rows,
        [
            "bddl_stem",
            "object",
            "split",
            "sim_task_id",
            "sim_language",
            "dataset_task_index",
            "dataset_task_text",
            "n_init_states",
            "n_episodes",
            "n_frames",
        ],
    )
    write_csv(
        args.output_dir / "object_audit.csv",
        audit_rows,
        [
            "bddl_stem",
            "split",
            "target_object",
            "obj_of_interest",
            "n_objects",
            "objects",
            "contains_bbq_sauce",
            "contains_chocolate_pudding",
        ],
    )

    # Held-out objects must not appear in any training scene.
    for row in audit_rows:
        if row["split"] != "seen":
            continue
        assert not row["contains_bbq_sauce"], row["bddl_stem"]
        assert not row["contains_chocolate_pudding"], row["bddl_stem"]

    seen_indices = {row["dataset_task_index"] for row in task_rows if row["split"] == "seen"}
    keep_rows = [row for row in episodes if int(row["task_index"]) in seen_indices]
    keep = [int(row["episode_index"]) for row in keep_rows]
    episode_rows = []
    for row in keep_rows:
        task_row = next(item for item in task_rows if item["dataset_task_index"] == int(row["task_index"]))
        episode_rows.append(
            {
                "source_episode_index": int(row["episode_index"]),
                "source_task_index": int(row["task_index"]),
                "bddl_stem": task_row["bddl_stem"],
                "object": task_row["object"],
                "frames": int(row["length"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
            }
        )
    write_csv(
        args.output_dir / "train_episodes.csv",
        episode_rows,
        [
            "source_episode_index",
            "source_task_index",
            "bddl_stem",
            "object",
            "frames",
            "dataset_from_index",
            "dataset_to_index",
        ],
    )

    summary = {
        "dataset": config["dataset"]["repo_id"],
        "revision": config["dataset"]["revision"],
        "total_episodes": len(episodes),
        "train_episodes": len(keep),
        "train_frames": sum(int(row["length"]) for row in keep_rows),
        "tasks": {row["bddl_stem"]: row["split"] for row in task_rows},
        "min_init_states": min(row["n_init_states"] for row in task_rows),
    }
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"task map: {len(task_rows)} tasks -> {args.output_dir / 'task_map.csv'}")
    print(f"object audit: {len(audit_rows)} tasks -> {args.output_dir / 'object_audit.csv'}")
    print(f"training episodes: {len(keep)} / {len(episodes)} -> {args.output_dir / 'train_episodes.csv'}")
    print(f"training frames: {summary['train_frames']}")
    print(f"minimum init states per task: {summary['min_init_states']}")

    if args.export_train:
        export_train_subset(config, dataset_root, args.output_dir / "libero_object_train", keep)
    return 0


if __name__ == "__main__":
    sys.exit(main())

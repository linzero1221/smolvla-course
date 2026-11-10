#!/usr/bin/env python3
"""Run the three formal evaluation conditions on a fixed checkpoint.

Conditions (see configs/experiment.json):
  seen_original    3 seen tasks x original prompt x 10 test init states = 30
  seen_paraphrase  3 seen tasks x 2 paraphrases x the same 10 init states = 60
  unseen_original  2 held-out tasks x original prompt x 10 test init states = 20

Every episode is a separate simulator reset pinned to a fixed initial-state id.
The paraphrase conditions reuse the exact same (task, init state, seed) pairs;
only the English text passed to the policy changes. Success is decided by the
LIBERO environment predicate, never by the text. A timeout counts as failure.

Usage:
    python scripts/evaluate.py --checkpoint outputs/train/pilot_s42/checkpoints/000200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_plan(config: dict, validation: bool = False) -> list[dict]:
    """Build the evaluation plan.

    ``validation`` uses the validation init states (0-4) on the three seen tasks
    with the original prompts only: 3 tasks x 5 states = 15 episodes. It is used
    to pick a checkpoint, never for the formal test report.
    """
    if validation:
        ids = config["evaluation"]["validation_init_state_ids"]
        return [
            {
                "condition": "validation_original",
                "object": task["object"],
                "bddl_stem": task["bddl_stem"],
                "init_state_id": init_state_id,
                "prompt_id": "original",
                "prompt": task["bddl_stem"].replace("_", " "),
            }
            for task in config["tasks"]["seen"]
            for init_state_id in ids
        ]

    test_ids = config["evaluation"]["test_init_state_ids"]
    episodes = []
    for task in config["tasks"]["seen"]:
        for init_state_id in test_ids:
            episodes.append(
                {
                    "condition": "seen_original",
                    "object": task["object"],
                    "bddl_stem": task["bddl_stem"],
                    "init_state_id": init_state_id,
                    "prompt_id": "original",
                    "prompt": task["bddl_stem"].replace("_", " "),
                }
            )
    for task in config["tasks"]["seen"]:
        for prompt_index, text in enumerate(task["paraphrases"], start=1):
            for init_state_id in test_ids:
                episodes.append(
                    {
                        "condition": "seen_paraphrase",
                        "object": task["object"],
                        "bddl_stem": task["bddl_stem"],
                        "init_state_id": init_state_id,
                        "prompt_id": f"paraphrase_{prompt_index}",
                        "prompt": text,
                    }
                )
    for task in config["tasks"]["unseen"]:
        for init_state_id in test_ids:
            episodes.append(
                {
                    "condition": "unseen_original",
                    "object": task["object"],
                    "bddl_stem": task["bddl_stem"],
                    "init_state_id": init_state_id,
                    "prompt_id": "original",
                    "prompt": task["bddl_stem"].replace("_", " "),
                }
            )
    return episodes


def resolve_policy_source(checkpoint: Path) -> Path:
    """Accept a run dir, a checkpoint dir or its pretrained_model subdir."""
    for candidate in (checkpoint, checkpoint / "pretrained_model"):
        if (candidate / "config.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No config.json found under {checkpoint}")


def episode_seed(base_seed: int, task_id: int, init_state_id: int) -> int:
    """Deterministic per-episode seed shared by original and paraphrase conditions."""
    return base_seed + 1000 * task_id + init_state_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/experiment.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/eval")
    parser.add_argument("--conditions", nargs="*", default=None, help="subset of conditions to run")
    parser.add_argument("--validation", action="store_true", help="run the 15 validation episodes instead")
    parser.add_argument("--max-episodes", type=int, default=None, help="debug: cap the plan size")
    parser.add_argument("--no-video", action="store_true", help="skip MP4 recording")
    args = parser.parse_args()

    import numpy as np
    import torch
    from libero.libero import benchmark
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.io_utils import write_video

    config = load_config(args.config)
    policy_dir = resolve_policy_source(args.checkpoint)

    policy_cfg = PreTrainedConfig.from_pretrained(str(policy_dir))
    policy_cfg.pretrained_path = policy_dir
    policy_cfg.device = "cuda"
    policy_cfg.n_action_steps = config["evaluation"]["n_action_steps"]
    policy_cfg.push_to_hub = False

    env_cfg = LiberoEnvConfig(
        task=config["dataset"]["suite"],
        task_ids=[0],
        observation_width=config["evaluation"]["observation_size"][1],
        observation_height=config["evaluation"]["observation_size"][0],
        control_mode=config["evaluation"]["control_mode"],
        hard_reset=config["evaluation"]["hard_reset"],
    )

    # Released checkpoints use camera1/camera2; checkpoints trained here use
    # image/image2. Rename only when the policy was trained with camera names.
    rename_map = {}
    input_keys = set(policy_cfg.input_features or {})
    if "observation.images.camera1" in input_keys:
        rename_map = {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        }

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename_map or None)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    env_preprocessor, env_postprocessor = env_cfg.get_env_processors()

    plan = episode_plan(config, validation=args.validation)
    if args.conditions:
        unknown = set(args.conditions) - {item["condition"] for item in plan}
        if unknown:
            raise ValueError(f"unknown conditions: {sorted(unknown)}")
        plan = [item for item in plan if item["condition"] in args.conditions]
    if args.max_episodes:
        plan = plan[: args.max_episodes]

    suite = benchmark.get_benchmark_dict()[config["dataset"]["suite"]]()
    task_ids = {task.name: index for index, task in enumerate(suite.tasks)}

    # Accept <run>/checkpoints/<step>, <run>/checkpoints/<step>/pretrained_model,
    # or <run>/checkpoints/<step>/pretrained_model/<file>. Name the run dir
    # <run>_<step> so repeated evaluations never overwrite each other.
    resolved = args.checkpoint.resolve()
    if resolved.is_file():
        resolved = resolved.parent
    parts = list(resolved.parts)
    step_name = next((part for part in reversed(parts) if part.isdigit()), "checkpoint")
    if "checkpoints" in parts:
        run_name = parts[parts.index("checkpoints") - 1]
    else:
        run_name = resolved.parent.name
    output_dir = args.output_dir / f"{run_name}_{step_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    fields = [
        "run_id", "checkpoint", "condition", "task_id", "object", "init_state_id",
        "prompt_id", "prompt", "seed", "status", "success", "steps", "video_path",
    ]
    base_seed = config["evaluation"]["seed"]
    width, height = config["evaluation"]["observation_size"]

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, item in enumerate(plan):
            task_id = task_ids[item["bddl_stem"]]
            seed = episode_seed(base_seed, task_id, item["init_state_id"])
            env = LiberoEnv(
                task_suite=suite,
                task_id=task_id,
                task_suite_name=config["dataset"]["suite"],
                camera_name="agentview_image,robot0_eye_in_hand_image",
                observation_width=width,
                observation_height=height,
                init_states=True,
                episode_index=item["init_state_id"],
                n_envs=1,
                control_mode=config["evaluation"]["control_mode"],
                hard_reset=config["evaluation"]["hard_reset"],
            )
            frames: list[np.ndarray] = []
            status, success, steps = "ok", False, 0
            try:
                observation, _ = env.reset(seed=seed)
                policy.reset()
                if not args.no_video:
                    frames.append(env.render())
                for step in range(env._max_episode_steps):
                    batch = preprocess_observation(
                        {"pixels": observation["pixels"], "robot_state": observation["robot_state"]}
                    )
                    batch["task"] = [item["prompt"]]
                    batch = env_preprocessor(batch)
                    batch = preprocessor(batch)
                    with torch.inference_mode():
                        action = policy.select_action(batch)
                    action = postprocessor(action)
                    action = env_postprocessor({"action": action})["action"]
                    observation, _, terminated, truncated, info = env.step(action.to("cpu").numpy()[0])
                    steps = step + 1
                    if not args.no_video:
                        frames.append(env.render())
                    if terminated or truncated:
                        success = bool(info.get("is_success", False))
                        break
                else:
                    status, success = "timeout", False
            except Exception as error:  # noqa: BLE001 - one bad episode must not abort the run
                status = f"error: {type(error).__name__}: {error}"
                success = False
            finally:
                env.close()

            video_path = ""
            if frames and not args.no_video:
                name = f"{item['condition']}_{item['object']}_s{item['init_state_id']}_{item['prompt_id']}.mp4"
                write_video(videos_dir / name, frames, fps=20)
                video_path = str((videos_dir / name).relative_to(output_dir))

            row = {
                "run_id": output_dir.name,
                "checkpoint": str(policy_dir),
                "condition": item["condition"],
                "task_id": task_id,
                "object": item["object"],
                "init_state_id": item["init_state_id"],
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "seed": seed,
                "status": status,
                "success": success,
                "steps": steps,
                "video_path": video_path,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"[{index + 1}/{len(plan)}] {item['condition']} {item['object']} "
                f"state={item['init_state_id']} {item['prompt_id']} -> success={success} steps={steps} ({status})"
            )

    print(f"\nwrote {output_dir / 'episodes.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run one LIBERO episode with a live pygame window (two cameras side by side)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "x11")

import numpy as np
import torch

REPO = Path("/mnt/d/AiCoding/VLA/.claude/worktrees/desktop-codex-lin-wsl-setup-ba03f133")
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import pygame
from libero.libero import benchmark
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors

CHECKPOINT = REPO / "outputs/train/formal_s42_resume/checkpoints/020000"
TASK = "pick_up_the_salad_dressing_and_place_it_in_the_basket"
PROMPT = "pick up the salad dressing and place it in the basket"
INIT_STATE = 9
PANEL = 420


def to_surface(frame_rgb):
    rgb = np.ascontiguousarray(frame_rgb)
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    return pygame.transform.scale(surf, (PANEL, PANEL))


def main():
    policy_cfg = PreTrainedConfig.from_pretrained(str(CHECKPOINT / "pretrained_model"))
    policy_cfg.pretrained_path = CHECKPOINT / "pretrained_model"
    policy_cfg.device = "cuda"
    policy_cfg.n_action_steps = 10

    env_cfg = LiberoEnvConfig(
        task="libero_object", task_ids=[0],
        observation_width=256, observation_height=256,
        control_mode="relative", hard_reset=True,
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(policy_cfg.pretrained_path),
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    env_pre, env_post = env_cfg.get_env_processors()

    suite = benchmark.get_benchmark_dict()["libero_object"]()
    task_id = next(i for i, t in enumerate(suite.tasks) if t.name == TASK)
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name="libero_object",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        obs_type="pixels_agent_pos", observation_width=256, observation_height=256,
        init_states=True, episode_index=INIT_STATE, n_envs=1,
        control_mode="relative", hard_reset=True,
    )

    pygame.init()
    screen = pygame.display.set_mode((PANEL * 2 + 30, PANEL + 90))
    pygame.display.set_caption("LIBERO-Object closed-loop rollout - SmolVLA LoRA 020000")
    font = pygame.font.SysFont("dejavusans", 20)
    small = pygame.font.SysFont("dejavusans", 15)

    obs, _ = env.reset(seed=42)
    policy.reset()
    step, success = 0, False

    def draw(obs, status):
        screen.fill((20, 22, 28))
        screen.blit(to_surface(obs["pixels"]["image"]), (10, 60))
        screen.blit(to_surface(obs["pixels"]["image2"]), (PANEL + 20, 60))
        screen.blit(font.render("agentview (third person)", True, (230, 230, 230)), (10, 34))
        screen.blit(font.render("wrist camera", True, (230, 230, 230)), (PANEL + 20, 34))
        screen.blit(small.render(status, True, (120, 230, 120)), (10, PANEL + 66))
        pygame.display.flip()

    running = True
    while running and step < env._max_episode_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        batch = preprocess_observation({
            "pixels": {k: v[None] for k, v in obs["pixels"].items()},
            "robot_state": {k: {n: v[None] for n, v in g.items()} for k, g in obs["robot_state"].items()},
        })
        batch["task"] = [PROMPT]
        batch = env_pre(batch)
        batch = preprocessor(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        action = env_post({"action": action})["action"]
        obs, _, terminated, truncated, info = env.step(action.to("cpu").numpy()[0])
        step += 1
        success = bool(info.get("is_success", False)) or success
        draw(obs, f"task: salad_dressing | init state {INIT_STATE} | step {step}/280 | success={success}")
        if terminated or truncated:
            break

    label = "SUCCESS" if success else "TIMEOUT / FAILURE"
    print(f"episode finished: steps={step} success={success} ({label})")

    # hold the final frame for the screenshot
    clock = pygame.time.Clock()
    for _ in range(1200):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        draw(obs, f"FINAL: {label} after {step} steps | task: salad_dressing | init state {INIT_STATE}")
        clock.tick(20)
        if not running:
            break

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()

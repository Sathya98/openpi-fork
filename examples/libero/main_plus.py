"""LIBERO-plus evaluation client.

Same client-server loop as main.py, but for the LIBERO-plus robustness benchmark
(https://github.com/sylvestf/LIBERO-plus). Each suite multiplexes ~2500 perturbed
task variants (light, textures, camera, ...); which perturbation dimension to
evaluate is selected via --args.perturbation, which filters tasks through the
benchmark's task_classification.json. Per the LIBERO-plus protocol each task
variant is evaluated for a single trial.

Must run inside the libero_plus_env (LIBERO-plus importable as `libero`).
"""

import dataclasses
import json
import logging
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from episode_runner import LIBERO_ENV_RESOLUTION
from episode_runner import make_libero_env
from episode_runner import run_episode
from plus_system.perturbations import PERTURBATION_CATEGORIES


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 8

    #################################################################################################################
    # LIBERO-plus environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_object"  # libero_spatial, libero_object, libero_goal, libero_10
    perturbation: str = "light"  # one of PERTURBATION_CATEGORIES keys, or "all"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # LIBERO-plus protocol: 1 trial per (already perturbed) task variant
    max_tasks: Optional[int] = None  # Evaluate only the first N selected tasks (smoke tests)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero_plus/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def _select_task_ids(task_suite, suite_name: str, perturbation: str) -> list:
    """Return suite task indices belonging to one perturbation category (or all)."""
    classification_path = (
        pathlib.Path(get_libero_path("benchmark_root")) / "benchmark" / "task_classification.json"
    )
    classification = json.loads(classification_path.read_text())[suite_name]

    if perturbation == "all":
        selected_names = {entry["name"] for entry in classification}
    else:
        category = PERTURBATION_CATEGORIES[perturbation]
        selected_names = {entry["name"] for entry in classification if entry["category"] == category}

    name_to_id = {task_suite.get_task(i).name: i for i in range(task_suite.n_tasks)}
    missing = selected_names - set(name_to_id)
    if missing:
        raise ValueError(f"{len(missing)} classified tasks not found in suite (e.g. {sorted(missing)[:3]})")
    return sorted(name_to_id[name] for name in selected_names)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    logging.info(f"Task suite: {args.task_suite_name}")

    task_ids = _select_task_ids(task_suite, args.task_suite_name, args.perturbation)
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]
    logging.info(f"Perturbation '{args.perturbation}': {len(task_ids)} tasks selected")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Success statistics, written to JSON after every task (partial results survive job kills).
    results_path = pathlib.Path(args.video_out_path).parent / "results.json"
    results = {
        "task_suite": args.task_suite_name,
        "perturbation": args.perturbation,
        "num_tasks": len(task_ids),
        "num_trials_per_task": args.num_trials_per_task,
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "tasks": {},
    }

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = make_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_successes = []
        for episode_idx in range(args.num_trials_per_task):
            logging.info(f"\nTask: {task_description}")
            logging.info(f"Starting episode {task_episodes+1}...")

            result = run_episode(
                env,
                client,
                task_description,
                initial_states[episode_idx],
                max_steps=max_steps,
                replan_steps=args.replan_steps,
                resize_size=args.resize_size,
                num_steps_wait=args.num_steps_wait,
            )
            done = result.success

            task_episodes += 1
            total_episodes += 1
            if done:
                task_successes += 1
                total_successes += 1
            episode_successes.append(done)

            # Save a replay video of the episode (task.name is unique per perturbed variant)
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task.name}_{suffix}.mp4",
                [np.asarray(x) for x in result.replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        env.close()

        results["tasks"][task.name] = {
            "task_id": task_id,
            "description": task_description,
            "successes": task_successes,
            "failures": task_episodes - task_successes,
            "num_trials": task_episodes,
            "success_rate": task_successes / task_episodes,
            "episode_successes": episode_successes,
        }
        results["total"] = {
            "num_trials": total_episodes,
            "successes": total_successes,
            "failures": total_episodes - total_successes,
            "success_rate": total_successes / total_episodes,
        }
        results_path.write_text(json.dumps(results, indent=2))

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Results written to {results_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)

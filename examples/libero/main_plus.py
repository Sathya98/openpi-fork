"""LIBERO-plus evaluation client.

Same client-server loop as main.py, but for the LIBERO-plus robustness benchmark
(https://github.com/sylvestf/LIBERO-plus). Each suite multiplexes ~2500 perturbed
task variants (light, textures, camera, ...); which perturbation dimension to
evaluate is selected via --args.perturbation, which filters tasks through the
benchmark's task_classification.json. Per the LIBERO-plus protocol each task
variant is evaluated for a single trial.

Must run inside the libero_plus_env (LIBERO-plus importable as `libero`).
"""

import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from libero_eef_control_wrapper import LiberoEEFControlWrapper
from plus_system.perturbations import PERTURBATION_CATEGORIES

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


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
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_successes = []
        for episode_idx in range(args.num_trials_per_task):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            episode_successes.append(bool(done))

            # Save a replay video of the episode (task.name is unique per perturbed variant)
            suffix = "success" if done else "failure"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task.name}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
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


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    # Must be a str: LIBERO-plus parses camera/robot/noise suffixes out of the filename.
    task_bddl_file = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    # horizon=5000: the EEF wrapper takes up to 16 sim steps per policy action, so the
    # default robosuite horizon (1000) truncates episodes before the 280-policy-step cap.
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "horizon": 5000,
    }
    env = LiberoEEFControlWrapper(OffScreenRenderEnv(**env_args))
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)

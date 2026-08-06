"""Shared LIBERO rollout code: environment construction and one episode against a policy server.

Used by the single-job client (main_plus.py) and by the sharded runner
(plus_system/run_shard.py) so both execute exactly the same episode.
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import List

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools

from libero_eef_control_wrapper import LiberoEEFControlWrapper

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class EpisodeResult:
    success: bool
    num_policy_steps: int  # actions actually executed (excludes the settling steps)
    replay_images: List  # empty when record_images=False


def make_libero_env(task, resolution, seed):
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


def run_episode(
    env,
    client,
    task_description,
    initial_state,
    *,
    max_steps,
    replan_steps,
    resize_size=224,
    num_steps_wait=10,
    record_images=True,
) -> EpisodeResult:
    """Reset the env to `initial_state` and roll out one episode against the policy server."""
    # Reset environment
    env.reset()
    action_plan = collections.deque()

    # Set initial states
    obs = env.set_init_state(initial_state)

    # Setup
    t = 0
    num_policy_steps = 0
    done = False
    replay_images = []

    while t < max_steps + num_steps_wait:
        try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
            wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

            # Save preprocessed image for replay video
            if record_images:
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
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                # Query model to get action
                action_chunk = client.infer(element)["actions"]
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            num_policy_steps += 1
            if done:
                break
            t += 1

        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    return EpisodeResult(success=bool(done), num_policy_steps=num_policy_steps, replay_images=replay_images)


def quat2axisangle(quat):
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

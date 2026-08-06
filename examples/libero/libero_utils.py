import os
import re
import pickle
import cv2

import numpy as np
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# LIBERO environment creation helpers
# ---------------------------------------------------------------------------

def derive_libero_task_info(source_path):
    """Derive LIBERO suite name, task name, and episode ID from a Drema source_path.

    Expected path format:
      .../libero_<suite>_full_regenerated/<task_name>/<task_name>_episode<N>_start
    or:
      .../libero_<suite>_full/<task_name>_episode<N>_start
    """
    parts = source_path.rstrip("/").split("/")
    leaf = parts[-1]

    episode_match = re.search(r"_episode(\d+)_start$", leaf)
    if not episode_match:
        raise ValueError(
            f"Cannot parse episode ID from source_path leaf: {leaf}. "
            f"Expected format: <task_name>_episode<N>_start"
        )
    episode_id = int(episode_match.group(1))
    task_name = leaf[:episode_match.start()]

    suite_name = None
    for part in parts:
        suite_match = re.match(r"libero_(spatial|object|goal|10|90)", part)
        if suite_match:
            suite_name = f"libero_{suite_match.group(1)}"
            break

    if suite_name is None:
        raise ValueError(
            f"Cannot derive LIBERO suite name from source_path: {source_path}. "
            f"Expected a parent directory matching libero_<suite>."
        )

    return suite_name, task_name, episode_id


def create_libero_env(suite_name, task_name, episode_id, resolution=256, render=True):
    """Create a LIBERO environment wrapped with EEF control for a specific task.

    When render=False, offscreen rendering is disabled (useful on nodes without
    EGL support). Physics and joint states still work; observations just won't
    contain camera images.

    Returns (wrapped_env, task_description, initial_state).
    """
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero as _libero_pkg
    from libero.libero import benchmark
    from libero_eef_control_wrapper import LiberoEEFControlWrapper

    bddl_root = os.path.join(os.path.dirname(_libero_pkg.__file__), "bddl_files")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    target_task = None
    target_task_id = None
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        if task.name == task_name:
            target_task = task
            target_task_id = task_id
            break

    if target_task is None:
        raise ValueError(
            f"Task '{task_name}' not found in suite '{suite_name}'. "
            f"Available tasks: {[task_suite.get_task(i).name for i in range(task_suite.n_tasks)]}"
        )

    bddl_file = os.path.join(
        bddl_root,
        target_task.problem_folder,
        target_task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        has_renderer=False,
        has_offscreen_renderer=render,
        camera_heights=resolution,
        camera_widths=resolution,
        horizon=5000,
    )
    env.seed(0)

    import torch
    init_states_path = os.path.join(
        os.path.dirname(_libero_pkg.__file__), "init_files",
        target_task.problem_folder, target_task.init_states_file,
    )
    try:
        init_states = torch.load(init_states_path, weights_only=False)
    except TypeError:
        # torch < 1.13 has no weights_only kwarg (and loads everything by default anyway)
        init_states = torch.load(init_states_path)
    initial_state = init_states[episode_id]

    wrapped_env = LiberoEEFControlWrapper(env)

    return wrapped_env, target_task.language, initial_state


def compute_libero_action(p_cur, q_cur, p_target, q_target, gripper_action, pos_limit=0.05, rot_limit=0.1):
    """Compute 7D normalized LIBERO action from current and target EEF poses."""
    delta_pos = p_target - p_cur
    norm_pos = np.clip(delta_pos / pos_limit, -1.0, 1.0)

    r_delta = R.from_quat(q_target) * R.from_quat(q_cur).inv()
    norm_rot = np.clip(r_delta.as_rotvec() / rot_limit, -1.0, 1.0)

    return np.array([*norm_pos, *norm_rot, gripper_action])


# ---------------------------------------------------------------------------
# Control loop helpers
# ---------------------------------------------------------------------------

############### ADDED FOR NEW CONTROL LOOP ###############
def compute_normalized_action(p_cur, q_cur, p_next, q_next, pos_limit=0.05, rot_limit=0.1):
    """
    Computes Robosuite-normalized OSC-Pose action from current and target eef poses.

    Args:
        p_cur, p_next : np.array(3)  - position
        q_cur, q_next : np.array(4)  - quaternions (w, x, y, z)
        pos_limit : float            - controller output_max for position (usually ~0.05 m)
        rot_limit : float            - controller output_max for rotation (rad)

    Returns:
        action : np.array(6) - normalized action for OSC-Pose (no gripper)
    """

    # --- Position delta ---
    delta_pos = p_next - p_cur

    # Normalize position delta to [-1, 1]
    norm_pos = np.clip(delta_pos / pos_limit, -1.0, 1.0)

    # --- Orientation delta (quaternion) ---
    r_cur = R.from_quat(q_cur)
    r_next = R.from_quat(q_next)

    # rotation needed to go from current to next
    r_delta = r_next * r_cur.inv()

    delta_rot = r_delta.as_rotvec()

    # Normalize rotation delta to [-1,1]
    norm_rot = np.clip(delta_rot / rot_limit, -1.0, 1.0)

    # Combine action (no gripper)
    action = np.concatenate([norm_pos, norm_rot])

    return action

def compute_goal_from_action(action_norm, p_cur, q_cur, pos_limit=0.05, rot_limit=0.1):

    # denormalize action
    delta_p = action_norm[:3] * pos_limit
    delta_q = action_norm[3:6] * rot_limit

    p_goal = p_cur + delta_p
    q_goal = R.from_rotvec(delta_q) * R.from_quat(q_cur)

    return p_goal, q_goal.as_quat()


def reached_pose(p_cur, q_cur, p_goal, q_goal, pos_thresh=0.002, rot_thresh=0.02):
    """Return True if current pose is close enough to goal pose."""
    pos_err = np.linalg.norm(p_goal - p_cur)

    # orientation error via quaternion distance
    dot = abs(np.dot(q_cur, q_goal))
    rot_err = 2 * np.arccos(np.clip(dot, -1.0, 1.0))

    return pos_err < pos_thresh and rot_err < rot_thresh

def move_to_pose(env, action, p_cur, q_cur, max_steps=15):
    """
    Executes actions until the EE reaches the target pose.

    Args:
        env  : robosuite environment
        compute_action_fn : fn(current_pos, current_quat, goal_pos, goal_quat)->normalized action

    """
    p_goal, q_goal = compute_goal_from_action(action, p_cur, q_cur)
    step_action = action.copy()

    obs, reward, done, info = env.step(step_action)

    for step in range(max_steps):

        p_cur = obs["robot0_eef_pos"]
        q_cur = obs["robot0_eef_quat"]

        # check if reached
        if reached_pose(p_cur, q_cur, p_goal, q_goal) or done:
            return obs, reward, done, info, True, step+1

        # compute normalized action for this step
        step_action[:6] = compute_normalized_action(p_cur, q_cur, p_goal, q_goal)

        # step environment
        obs, reward, done, info = env.step(step_action)

    return obs, reward, done, info, False, step

##########################################################

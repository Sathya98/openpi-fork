"""Wrapper around LIBERO environments that replaces single-step control
with closed-loop end-effector pose tracking.

Instead of forwarding each action as one ``env.step`` call, the wrapper
computes a goal EEF pose from the action and then repeatedly steps the
underlying environment until the end-effector reaches that pose (or a
maximum number of inner steps is exhausted).
"""

import numpy as np

from libero_utils import move_to_pose


class LiberoEEFControlWrapper:
    """Wraps a LIBERO ``OffScreenRenderEnv`` with closed-loop EEF pose control.

    The action space semantics stay the same (normalized OSC-Pose deltas +
    gripper), but each ``step`` call now internally loops until the target
    pose is reached, giving more accurate execution.

    Parameters
    ----------
    env : OffScreenRenderEnv
        The underlying LIBERO environment.
    """

    def __init__(self, env):
        self.env = env

        self._p_cur = None
        self._q_cur = None

    # ------------------------------------------------------------------
    # Forwarded environment methods
    # ------------------------------------------------------------------

    def reset(self):
        obs = self.env.reset()
        self._update_eef_state(obs)
        return obs

    def set_init_state(self, state):
        obs = self.env.set_init_state(state)
        self._update_eef_state(obs)
        return obs

    # def get_observation(self):
    #     obs = self.env.get_observation()
    #     self._update_eef_state(obs)
    #     return obs

    def seed(self, s):
        return self.env.seed(s)

    def close(self):
        return self.env.close()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def __getattr__(self, name):
        """Forward any other attribute access to the wrapped env."""
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Core: closed-loop step
    # ------------------------------------------------------------------

    def step(self, action):
        """Execute *action* with closed-loop EEF pose tracking.

        Parameters
        ----------
        action : array-like
            Normalized OSC-Pose action of length >= 7.  Elements 0-2 are
            position deltas, 3-5 are rotation deltas, 6 is the gripper
            command.  Any additional elements are passed through unchanged.

        Returns
        -------
        obs : dict
        reward : float
        done : bool
        info : dict
        """
        action = np.asarray(action, dtype=np.float64)
        obs, reward, done, info, _, _ = move_to_pose(
            self.env, action, self._p_cur, self._q_cur,
        )
        self._update_eef_state(obs)
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_eef_state(self, obs):
        if obs is not None:
            self._p_cur = obs["robot0_eef_pos"]
            self._q_cur = obs["robot0_eef_quat"]

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_drema_example() -> dict:
    """Creates a random input example for the Drema policy (single front camera)."""
    return {
        # State is carried through the pipeline (PadStatesAndActions / norm stats need it),
        # but pi0.5 with discrete_state_input=False never feeds it to the model.
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DremaInputs(transforms.DataTransformFn):
    """Input transform for the DREMA dataset (single front camera).

    Same as ``LiberoInputs`` with one change: this robot has only a front camera. The dataset's
    ``wrist_image`` is a byte-identical copy of the front image, so we do NOT feed it. Both wrist
    slots are zero-filled and masked off (``image_mask=False`` for pi0/pi0.5), telling the model
    those cameras are absent. Only ``base_0_rgb`` is active.

    State is still passed through (the data pipeline requires the key), but pi0.5 with
    ``discrete_state_input=False`` does not consume it -- the policy runs on image + prompt only.
    """

    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        # pi0/pi0.5 mask out absent cameras; pi0-FAST does not use image masking.
        wrist_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                # No wrist camera on this robot -> zero-fill and mask off.
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": wrist_mask,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DremaOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the dataset action format (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Return only the first 7 action dims (the rest is padding up to the model action_dim).
        return {"actions": np.asarray(data["actions"][:, :7])}


def _r6_to_matrix(r6: np.ndarray) -> np.ndarray:
    """(..., 6) = [R[:,0], R[:,1]] -> (..., 3, 3), Gram-Schmidt, so a predicted pair need not be orthonormal."""
    a, b = r6[..., :3], r6[..., 3:6]
    x = a / np.linalg.norm(a, axis=-1, keepdims=True)
    y = b - x * np.sum(x * b, axis=-1, keepdims=True)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def _matrix_to_r6(m: np.ndarray) -> np.ndarray:
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


@dataclasses.dataclass(frozen=True)
class RelativePoseActions(transforms.DataTransformFn):
    """Pose actions (xyz + r6, then gripper) as offsets from the state at the chunk start.

    `DeltaActions` subtracts all nine pose numbers; for the rotation that is not a rotation.
    Here xyz is p_k - p_0 and the rotation is R_k R_0^T (base frame, like xyz), as r6. The
    gripper stays absolute.
    """

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state, actions = np.asarray(data["state"]), np.array(data["actions"], copy=True)
        r0 = _r6_to_matrix(state[3:9])
        actions[..., :3] -= state[:3]
        actions[..., 3:9] = _matrix_to_r6(_r6_to_matrix(actions[..., 3:9]) @ r0.T)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsolutePoseActions(transforms.DataTransformFn):
    """The inverse of `RelativePoseActions`: p_k = p_0 + d, R_k = dR R_0."""

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state, actions = np.asarray(data["state"]), np.array(data["actions"], copy=True)
        r0 = _r6_to_matrix(state[3:9])
        actions[..., :3] += state[:3]
        actions[..., 3:9] = _matrix_to_r6(_r6_to_matrix(actions[..., 3:9]) @ r0)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class DremaPoseOutputs(transforms.DataTransformFn):
    """FrankaJointPos actions: hand pose (xyz + r6) and gripper, 10 dims."""

    action_dim: int = 10

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}

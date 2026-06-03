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

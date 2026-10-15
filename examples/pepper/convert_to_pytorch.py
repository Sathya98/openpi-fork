"""A pepper checkpoint (JAX, full finetune or LoRA) -> PyTorch `model.safetensors`, e.g. for RL.

openpi's converter drops LoRA weights without a word, so for a LoRA checkpoint the
adapters are first folded into the base weights: `w += A @ B * alpha/rank` for the
attention einsums, `w += A @ B` for the MLP (openpi's FeedForward adds LoRA unscaled).
The result is a plain pi0.5 with the training config's horizon and transforms;
`assets/` (norm stats) is copied next to it.

Needs openpi's patched transformers (`src/openpi/models_pytorch/transformers_replace`).

    python examples/pepper/convert_to_pytorch.py \\
        --checkpoint-dir <ckpts>/pi05_pepper3_rel_lorafv_b32_7k5/4500 \\
        --config-name pi05_pepper3_rel_lorafv_b32_7k5 --output-path <out>
"""

import dataclasses
import json
import pathlib
import shutil
import sys

import numpy as np
import tyro

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import convert_jax_model_to_pytorch as conv  # noqa: E402

import openpi.training.config as _config  # noqa: E402


def merge_lora(flat: dict[str, np.ndarray], scaling: dict[str, float]) -> list[str]:
    """Fold every LoRA pair into its base weight, in place. `flat` uses openpi's PaliGemma keys."""
    merged = []
    for key in [k for k in flat if k.endswith("lora_a")]:
        a, b = flat.pop(key), flat.pop(key[: -len("lora_a")] + "lora_b")
        if key.endswith("/lora_a"):  # lora.Einsum: <name>/{w, lora_a, lora_b}, scaled
            base = key[: -len("lora_a")] + "w"
            scale = scaling["_1" if "_1/" in key else ""]
        else:  # lora.FeedForward: <name>_lora_{a,b} next to <name>, unscaled
            base, scale = key[: -len("_lora_a")], 1.0
        if "attn_vec_einsum" in base:
            # "BTNH,NHD->BTD": the output has no head axis, so LoRA's second einsum
            # ("BTL,NLD->BTD") sums B over heads: dW[n,h,d] = sum_l A[n,h,l] sum_n' B[n',l,d].
            delta = np.einsum("...nhl,...ld->...nhd", a, b.sum(axis=-3))
        else:  # q ("BTD,NDH->BTNH"), kv ("BSD,2KDH->2BSKH"), MLP: per leading axis
            delta = np.matmul(a, b)
        flat[base] = flat[base] + scale * delta
        merged.append(base)
    return merged


def main(checkpoint_dir: str, config_name: str, output_path: str, precision: str = "bfloat16") -> None:
    model_config = _config.get_config(config_name).model
    variants = (model_config.paligemma_variant, model_config.action_expert_variant)
    # alpha / rank of each LoRA'd Gemma: "" = the VLM, "_1" = the action expert.
    scaling = {}
    for suffix, variant in zip(("", "_1"), variants):
        lora = conv.openpi.models.gemma.get_config(variant).lora_configs
        scaling[suffix] = lora["attn"].scaling_value if lora else 1.0
    plain = dataclasses.replace(
        model_config,
        paligemma_variant=variants[0].removesuffix("_lora"),
        action_expert_variant=variants[1].removesuffix("_lora"),
    )

    restore = conv.slice_initial_orbax_checkpoint

    def restore_merged(checkpoint_dir, restore_precision=None):
        params = restore(checkpoint_dir, restore_precision)
        merged = merge_lora(params["paligemma_params"], scaling)
        print(f"merged {len(merged)} LoRA pairs" if merged else "no LoRA weights: full finetune")
        return params

    conv.slice_initial_orbax_checkpoint = restore_merged
    conv.convert_pi0_checkpoint(checkpoint_dir, precision, output_path, plain)

    # The converter looks for assets next to the step dir; openpi saves them inside it.
    shutil.copytree(pathlib.Path(checkpoint_dir) / "assets", pathlib.Path(output_path) / "assets", dirs_exist_ok=True)
    # RL4VLAs reads the norm stats from <checkpoint>/<asset_id>/, not assets/<asset_id>/.
    for asset in (pathlib.Path(checkpoint_dir) / "assets").iterdir():
        shutil.copytree(asset, pathlib.Path(output_path) / asset.name, dirs_exist_ok=True)
    cfg = pathlib.Path(output_path) / "config.json"
    cfg.write_text(json.dumps({**json.loads(cfg.read_text()), "config_name": config_name,
                               "source": str(checkpoint_dir), "lora_merged": "lora" in "".join(variants)}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)

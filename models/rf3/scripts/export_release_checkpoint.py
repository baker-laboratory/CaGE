#!/usr/bin/env python
"""Turn a training checkpoint into a redistributable inference-only checkpoint.

Three transforms, each undone only by re-exporting from the original:

1. Keep only the keys inference reads (``model``, ``train_cfg``, and the two
   scalar counters). Drops the optimizer and scheduler state.
2. Replace every absolute path under a cluster-internal root in ``train_cfg``
   with a placeholder. ``train_cfg`` itself must stay -- the inference engine
   rebuilds the network from ``train_cfg.model.net`` and the transform pipeline
   from the first ``train_cfg.datasets.val`` entry -- so paths are scrubbed by
   value, never by deleting keys.
3. Unless ``--keep-ema``, drop the online half of the EMA wrapper and promote
   the shadow weights to the top level, setting ``train_cfg.model.ema = None``
   so ``construct_model`` builds a bare net. ``EMA.forward`` dispatches to the
   shadow in eval, so this is exactly the weight set inference already used.

Before writing, rebuilds the network from the scrubbed ``train_cfg`` and refuses
to export unless every model parameter can be copied from the checkpoint. This is
not a formality: ``load_weights_with_policies`` answers a missing or mis-shaped
parameter by falling back to ``REINIT`` -- a randomly initialized tensor -- behind a
single WARNING that this repository's logging config suppresses. A renaming mistake
would otherwise ship as a quietly half-random model.

Writes the checkpoint, prints its sha256, and leaves a sidecar JSON recording
what was done.
"""

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

# Keys the inference path reads; everything else is training state.
KEYS_TO_KEEP = ("model", "train_cfg", "current_epoch", "global_step")

INTERNAL_ROOTS = (
    "net",
    "projects",
    "squash",
    "mnt",
    "home",
    "scratch",
    "gscratch",
    "work",
)
INTERNAL_PATH_RE = re.compile(rf"^/({'|'.join(INTERNAL_ROOTS)})(/|$)")
REDACTED = "<redacted-training-path>"

# Absolute paths that are not cluster-internal and may legitimately survive.
ALLOWED_ABSOLUTE_RE = re.compile(r"^/(tmp|dev|usr|etc|opt|var)(/|$)")


def scrub_paths(node: DictConfig | ListConfig) -> int:
    """Replace internal absolute paths with a placeholder, in place."""
    keys = list(node.keys()) if isinstance(node, DictConfig) else range(len(node))
    n = 0
    for key in keys:
        if OmegaConf.is_interpolation(node, key):
            continue  # resolves through another key, which gets scrubbed on its own
        try:
            value = node[key]
        except Exception:
            continue  # MISSING ('???') and friends
        if isinstance(value, (DictConfig, ListConfig)):
            n += scrub_paths(value)
        elif isinstance(value, str) and INTERNAL_PATH_RE.match(value):
            node[key] = REDACTED
            n += 1
    return n


def audit_absolute_paths(cfg: DictConfig) -> list[str]:
    """Return absolute-path-looking strings left in cfg, minus the allowed roots."""
    text = OmegaConf.to_yaml(cfg)
    found = set(re.findall(r"(/[A-Za-z0-9._][A-Za-z0-9._/-]{3,})", text))
    return sorted(p for p in found if not ALLOWED_ABSOLUTE_RE.match(p))


def promote_ema_shadow(model_sd: dict) -> dict:
    """Return the shadow half of an EMA state dict, re-keyed to the bare net."""
    online = {k for k in model_sd if k.startswith("model.")}
    shadow = {k for k in model_sd if k.startswith("shadow.")}
    other = set(model_sd) - online - shadow
    if not shadow:
        raise SystemExit(
            "no shadow.* tensors: checkpoint is not EMA-wrapped, use --keep-ema"
        )
    if other:
        raise SystemExit(
            f"unexpected non-EMA keys, refusing to guess: {sorted(other)[:5]}"
        )
    stripped = {k[len("model.") :] for k in online}
    promoted = {k[len("shadow.") :] for k in shadow}
    if stripped != promoted:
        only_online = sorted(stripped - promoted)[:5]
        only_shadow = sorted(promoted - stripped)[:5]
        raise SystemExit(
            f"EMA halves disagree; online-only {only_online}, shadow-only {only_shadow}"
        )
    return {
        k[len("shadow.") :]: v for k, v in model_sd.items() if k.startswith("shadow.")
    }


def verify_against_model(cfg: DictConfig, model_sd: dict) -> int:
    """Refuse the export unless the rebuilt model's keys and shapes match model_sd."""
    import hydra

    net = hydra.utils.instantiate(cfg.model.net, _recursive_=False)
    if cfg.model.get("ema") is not None:
        from foundry.training.EMA import EMA

        net = EMA(net, **cfg.model.ema)
    reference = net.state_dict()

    missing = sorted(set(reference) - set(model_sd))
    unused = sorted(set(model_sd) - set(reference))
    mis_shaped = [
        k
        for k in sorted(set(reference) & set(model_sd))
        if tuple(reference[k].shape) != tuple(model_sd[k].shape)
    ]
    if missing or unused or mis_shaped:
        raise SystemExit(
            f"model/checkpoint mismatch, refusing to write: "
            f"{len(missing)} would be REINIT'd {missing[:3]}, "
            f"{len(unused)} unused {unused[:3]}, "
            f"{len(mis_shaped)} mis-shaped {mis_shaped[:3]}"
        )
    return len(reference)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, help="training checkpoint to export from")
    parser.add_argument("dest", type=Path, help="release checkpoint to write")
    parser.add_argument(
        "--keep-ema",
        action="store_true",
        help="keep both EMA halves and the ema config",
    )
    parser.add_argument(
        "--registry-name", default=None, help="registry name, recorded in the sidecar"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip rebuilding the model to check the key set (needs rf3 importable)",
    )
    args = parser.parse_args()

    if args.dest.exists():
        raise SystemExit(f"refusing to overwrite {args.dest}")

    print(f"reading {args.source}")
    ckpt = torch.load(args.source, map_location="cpu", weights_only=False, mmap=True)
    dropped = sorted(set(ckpt) - set(KEYS_TO_KEEP))
    print(f"  dropping top-level keys: {dropped}")

    cfg = OmegaConf.create(ckpt["train_cfg"])
    OmegaConf.set_struct(cfg, False)
    n_scrubbed = scrub_paths(cfg)
    print(f"  scrubbed {n_scrubbed} internal paths in train_cfg")

    model_sd = dict(ckpt["model"])
    n_tensors_in = len(model_sd)
    if args.keep_ema:
        print(f"  keeping EMA layout ({n_tensors_in} tensors)")
    else:
        model_sd = promote_ema_shadow(model_sd)
        cfg.model.ema = None
        print(
            f"  promoted EMA shadow: {n_tensors_in} -> {len(model_sd)} tensors, "
            "train_cfg.model.ema = None"
        )

    leftover = audit_absolute_paths(cfg)
    if leftover:
        raise SystemExit(
            f"absolute paths still in train_cfg, refusing to write: {leftover}"
        )

    if args.no_verify:
        verified = None
        print("  skipping model key-set verification (--no-verify)")
    else:
        verified = verify_against_model(cfg, model_sd)
        print(f"  verified: all {verified} model parameters copy from the checkpoint")

    # Clone so the saved file carries one contiguous storage per tensor rather
    # than views into the source checkpoint's storages.
    out = {
        "model": {
            k: (v.detach().clone().contiguous() if torch.is_tensor(v) else v)
            for k, v in model_sd.items()
        },
        "train_cfg": cfg,
        "current_epoch": ckpt.get("current_epoch"),
        "global_step": ckpt.get("global_step"),
    }

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.dest)
    digest = sha256_of(args.dest)
    size = args.dest.stat().st_size
    print(f"wrote {args.dest}  {size / 2**30:.3f} GiB  sha256 {digest}")

    sidecar = args.dest.with_suffix(args.dest.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "filename": args.dest.name,
                "registry_name": args.registry_name,
                "sha256": digest,
                "size_bytes": size,
                "exported_on": date.today().isoformat(),
                "exported_by": "models/rf3/scripts/export_release_checkpoint.py",
                "source_basename": args.source.name,
                "current_epoch": out["current_epoch"],
                "global_step": out["global_step"],
                "dropped_top_level_keys": dropped,
                "internal_paths_scrubbed": n_scrubbed,
                "path_placeholder": REDACTED,
                "ema": "kept"
                if args.keep_ema
                else "shadow promoted, online half dropped",
                "n_tensors": len(out["model"]),
                "model_keys_verified": verified,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()

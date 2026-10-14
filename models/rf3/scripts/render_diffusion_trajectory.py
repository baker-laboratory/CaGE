"""Render mechanism-figure frames from a trajectory written by `dump_trajectories=true`.

Each frame shows the step's x0 estimates as cartoons (cond in purple over uncond in grey
under CFG, fading in as the noise drops) and, as dots, the atoms of x_t still more than
`--dot_cutoff` Å from the guided x0 -- so the cloud thins out as the structure resolves.
Dots are a fixed purple/grey mix per atom; `--dot_spread` < 1 draws them closer to the
structure (a display choice). The last frame is x_0 alone, motif in purple, or with
`--x0 mix` purple where CFG pushed hardest. `--noise_first` makes the first frame pure noise.
All frames share one camera and have transparent backgrounds; the .pse keeps every frame as
its own objects and scene (PgUp/PgDn), plus the full trajectories (disabled) for scrubbing.

  pymol -cq render_diffusion_trajectory.py -- TRAJ_DIR OUT_DIR \
      [--t_hat 4608,13,5,3,1,0] [--motif "chain A and resi 31+48 or chain X"]

To re-render from another angle: open the .pse, rotate, run `print(cmd.get_view())`, and
pass the 18 numbers as `--view`.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

from pymol import cmd

COLOURS = {
    "traj_purple": [0.55, 0.30, 0.88],
    "traj_grey": [0.82, 0.82, 0.85],
    "traj_dot_grey": [0.55, 0.55, 0.60],
}
DOT_ATOMS = {
    "ca": "((polymer and name CA) or not polymer)",
    "backbone": "((polymer and name N+CA+C+O) or not polymer)",
    "heavy": "(not elem H)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("traj_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--t_hat",
        default="4608,13,5,3,1,0",
        help="noise levels to render; nearest recorded step is used, 0 = x_0",
    )
    p.add_argument("--motif", default="", help="PyMOL selection coloured purple in x_0")
    p.add_argument(
        "--cond", default=None, help="guidance stream to draw (default: first)"
    )
    p.add_argument(
        "--cartoon",
        choices=["both", "cfg", "uncond"],
        default="both",
        help="both = cond over uncond; cfg = guided x0 only; uncond = main x0 only",
    )
    p.add_argument(
        "--x0",
        choices=["guided", "both", "mix"],
        default="guided",
        help="last frame: guided x0 in grey (motif purple); both = final cond "
        "(purple) over final uncond (grey), which coincide; mix = guided x0 "
        "with the residues CFG pushed hardest in purple (see --x0_purple)",
    )
    p.add_argument(
        "--x0_purple",
        type=float,
        default=0.5,
        help="fraction of residues coloured purple under --x0 mix",
    )
    p.add_argument(
        "--noise_first",
        action="store_true",
        help="first frame: every x_t atom as an isotropic ball, no structure",
    )
    p.add_argument(
        "--noise_size",
        type=float,
        default=0.8,
        help="pure-noise ball radius (95th pct) relative to x_0's",
    )
    p.add_argument(
        "--noise_scale",
        type=float,
        default=0.4,
        help="size (sphere_scale) of every noise dot in every frame",
    )
    p.add_argument(
        "--noise_matte",
        type=float,
        default=0.0,
        help="specular of every frame (0 = matte); lighting is global in PyMOL, "
        "so cartoons follow the dots",
    )
    p.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="PATH:COLOUR",
        help="also render a structure (e.g. a Cond./Uncond. reference) "
        "superposed on x_0 with the same camera; COLOUR is purple or grey",
    )
    p.add_argument(
        "--dot_atoms",
        choices=list(DOT_ATOMS),
        default="heavy",
        help="which x_t atoms become dots (heavy = densest)",
    )
    p.add_argument(
        "--dot_purple", type=float, default=0.5, help="fraction of purple dots"
    )
    p.add_argument(
        "--dot_cutoff",
        type=float,
        default=1.5,
        help="draw x_t atoms at least this far (Å) from the step's x0",
    )
    p.add_argument(
        "--dot_spread",
        type=float,
        default=1.0,
        help="display dots at x0 + spread * (x_t - x0); <1 hugs the structure "
        "(schematic, not the true x_t), 1 = true x_t",
    )
    p.add_argument(
        "--fade",
        type=float,
        default=0.6,
        help="cartoon transparency at the noisiest step (0 at the last)",
    )
    p.add_argument(
        "--dot_envelope",
        type=float,
        default=6.0,
        help="keep dots within this many Å of x_0 (0 = keep all); the camera "
        "fits the envelope, so no dot is cut at the frame edge",
    )
    p.add_argument(
        "--zoom_buffer",
        type=float,
        default=None,
        help="Å around x_0 in frame (default: envelope + 2, or 5 without one)",
    )
    p.add_argument(
        "--view", default=None, help="18 comma-separated floats from get_view()"
    )
    p.add_argument("--size", type=int, default=1200, help="frame width/height in px")
    p.add_argument(
        "--outline", action="store_true", help="ray_trace_mode 1 (black outlines)"
    )
    return p.parse_args(sys.argv[1:])


def nearest_state(t_hats: list[float], t: float) -> int:
    """1-based state whose t_hat is closest to t in log space."""

    def log(x: float) -> float:
        return math.log(max(x, 1e-6))

    return min(range(len(t_hats)), key=lambda i: abs(log(t_hats[i]) - log(t))) + 1


def noise_level(t: float, t_min: float, t_max: float) -> float:
    """0 at the final step, 1 at the noisiest one (log scale)."""
    return min(max(math.log(t / t_min) / math.log(t_max / t_min), 0.0), 1.0)


def store_distance_to(ref: str, state: int, atoms: str) -> dict:
    """Put each x_t atom's distance to `ref` (same step) in `noisy`'s q column.

    Returns `ref`'s coordinates by atom ID.
    """
    # iterate_state, not get_coords: PyMOL builds here can ship a mismatched numpy
    x0, dist = {}, {}
    cmd.iterate_state(
        state, f"{ref} and {atoms}", "x0[ID] = (x, y, z)", space={"x0": x0}
    )
    cmd.iterate_state(
        state,
        f"noisy and {atoms}",
        "dist[ID] = math.dist((x, y, z), x0[ID])",
        space={"x0": x0, "dist": dist, "math": math},
    )
    cmd.alter(f"noisy and {atoms}", "q = dist[ID]", space={"dist": dist})
    return x0


def pull_towards(obj: str, x0: dict, spread: float) -> None:
    """Move each atom of `obj` to x0 + spread * (x - x0)."""

    def pull(i, p):
        return tuple(a + spread * (b - a) for a, b in zip(x0[i], p))

    cmd.alter_state(1, obj, "x, y, z = pull(ID, (x, y, z))", space={"pull": pull})


def cfg_push(cond: str, t_hat: dict) -> dict:
    """Per residue, the cond-uncond x0 CA distance averaged over the guided steps.

    Each step is divided by its mean over residues, so every step counts equally.
    """
    push = {}
    for m, t in enumerate(t_hat["denoised"]):
        if t not in t_hat[cond]:
            continue
        ca = {"u": {}, "c": {}}
        cmd.iterate_state(
            m + 1,
            "denoised and polymer and name CA",
            "ca['u'][(chain, resi)] = (x, y, z)",
            space={"ca": ca},
        )
        cmd.iterate_state(
            t_hat[cond].index(t) + 1,
            f"{cond.replace('-', '_')} and polymer and name CA",
            "ca['c'][(chain, resi)] = (x, y, z)",
            space={"ca": ca},
        )
        dist = {key: math.dist(xyz, ca["c"][key]) for key, xyz in ca["u"].items()}
        mean = sum(dist.values()) / len(dist) or 1.0
        for key, d in dist.items():
            push[key] = push.get(key, 0.0) + d / mean
    return push


def noise_ball(obj: str, centre: list, r95: float, r_clip: float) -> None:
    """Rescale `obj` about its centroid onto `centre` so 95% of atoms fall within r95.

    Atoms then beyond r_clip are dropped, so the ball fits the camera.
    """
    xyz = []
    cmd.iterate_state(1, obj, "xyz.append((x, y, z))", space={"xyz": xyz})
    mid = [sum(p[i] for p in xyz) / len(xyz) for i in range(3)]
    radii = sorted(math.dist(p, mid) for p in xyz)
    scale = r95 / radii[int(0.95 * (len(radii) - 1))]

    def move(p):
        return [c + scale * (v - m) for c, v, m in zip(centre, p, mid)]

    cmd.alter_state(1, obj, "x, y, z = move((x, y, z))", space={"move": move})
    far = []
    cmd.iterate_state(
        1,
        obj,
        "far.append(ID) if math.dist((x, y, z), centre) > r else None",
        space={"far": far, "math": math, "centre": centre, "r": r_clip},
    )
    if far:
        cmd.remove(f"{obj} and ID {'+'.join(map(str, far))}")


def style_structure(
    obj: str, colour: str, transparency: float, motif: str = ""
) -> None:
    """Cartoon protein and ball-and-stick ligand in one colour; motif in purple."""
    cmd.dss(obj)
    cmd.show("cartoon", f"{obj} and polymer")
    cmd.show("sticks", f"{obj} and not polymer")
    cmd.show("spheres", f"{obj} and not polymer")
    cmd.set("sphere_scale", 0.25, obj)
    cmd.color(colour, obj)
    if motif:
        cmd.color("traj_purple", f"{obj} and ({motif})")
    for setting in (
        "cartoon_transparency",
        "stick_transparency",
        "sphere_transparency",
    ):
        cmd.set(setting, transparency, obj)


def style_dots(obj: str, scale: float) -> None:
    cmd.show("spheres", obj)
    cmd.set("sphere_scale", scale, obj)
    cmd.color("traj_dot_grey", obj)
    cmd.color("traj_purple", f"{obj} and b > 0.5")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_hat = json.loads((args.traj_dir / "frames.json").read_text())["t_hat"]
    cond_names = [n for n in t_hat if n.startswith("denoised_cond-")]
    cond = f"denoised_cond-{args.cond}" if args.cond else (cond_names or [None])[0]
    guided = "denoised_cfg" if "denoised_cfg" in t_hat else "denoised"
    mode = args.cartoon if cond else "uncond"
    dot_atoms = DOT_ATOMS[args.dot_atoms]

    cmd.reinitialize()
    for name, rgb in COLOURS.items():
        cmd.set_color(name, rgb)
    to_load = {"noisy", "denoised", guided} | ({cond} if cond else set())
    for name in to_load:
        cmd.load(str(args.traj_dir / f"{name}.cif"), name.replace("-", "_"))
    cmd.remove("elem H")
    cmd.create("x0_envelope", guided, 1, 1)  # final structure: dot envelope and camera
    cmd.hide("everything")
    # Fixed purple/grey assignment per atom (b = 1 -> purple), kept through every frame
    cmd.alter(
        "noisy",
        "b = float(random.Random(ID).random() < frac)",
        space={"random": random, "frac": args.dot_purple},
    )

    t_all = t_hat["noisy"]
    t_min, t_max = min(t_all), max(t_all)
    # x_0's centre and radii: size of the pure-noise ball
    x0_xyz = []
    cmd.iterate_state(
        1, "x0_envelope", "x0_xyz.append((x, y, z))", space={"x0_xyz": x0_xyz}
    )
    centre = [sum(p[i] for p in x0_xyz) / len(x0_xyz) for i in range(3)]
    x0_radii = sorted(math.dist(p, centre) for p in x0_xyz)
    layers = {
        "cfg": [(guided, "traj_grey")],
        "uncond": [("denoised", "traj_grey")],
        "both": [("denoised", "traj_grey"), (cond, "traj_purple")],
    }[mode]
    frames = []
    for k, t in enumerate(float(t) for t in args.t_hat.split(",")):
        state = nearest_state(t_all, t)
        t_real = t_all[state - 1]
        objs = []
        if k == 0 and args.noise_first:  # pure noise: every x_t atom, no structure
            obj = f"f{k}_xt"
            cmd.create(obj, f"noisy and {dot_atoms}", state, 1)
            noise_ball(
                obj,
                centre,
                args.noise_size * x0_radii[int(0.95 * (len(x0_radii) - 1))],
                x0_radii[-1] + args.dot_envelope,
            )
            style_dots(obj, args.noise_scale)
            objs.append(obj)
        elif state == 1 and args.x0 == "mix" and cond:  # x_0 coloured by CFG push
            obj = f"f{k}_x0"
            cmd.create(obj, guided, 1, 1)
            style_structure(obj, "traj_grey", 0.0)
            push = cfg_push(cond, t_hat)
            top = set(
                sorted(push, key=push.get, reverse=True)[
                    : round(args.x0_purple * len(push))
                ]
            )
            cmd.alter(obj, "b = float((chain, resi) in top)", space={"top": top})
            cmd.color("traj_purple", f"{obj} and polymer and b > 0.5")
            objs.append(obj)
        elif state == 1 and args.x0 == "both" and cond:  # final cond over final uncond
            for name, colour in (("denoised", "traj_grey"), (cond, "traj_purple")):
                obj = f"f{k}_x0_{'uncond' if name == 'denoised' else 'cond'}"
                cmd.create(obj, name.replace("-", "_"), 1, 1)
                style_structure(obj, colour, 0.0)
                objs.append(obj)
        elif state == 1:  # x_0: the final structure alone
            obj = f"f{k}_x0"
            cmd.create(obj, guided, 1, 1)
            style_structure(obj, "traj_grey", 0.0, args.motif)
            objs.append(obj)
        else:
            x0 = store_distance_to(guided, state, dot_atoms)
            dots = f"noisy and {dot_atoms} and q > {args.dot_cutoff}"
            if cmd.count_atoms(dots):
                cmd.create(f"f{k}_xt", dots, state, 1)
                if args.dot_spread != 1.0:
                    pull_towards(f"f{k}_xt", x0, args.dot_spread)
                if args.dot_envelope > 0:
                    cmd.remove(
                        f"f{k}_xt and not (f{k}_xt within {args.dot_envelope} "
                        "of x0_envelope)"
                    )
                if cmd.count_atoms(f"f{k}_xt"):
                    style_dots(f"f{k}_xt", args.noise_scale)
                    objs.append(f"f{k}_xt")
                else:
                    cmd.delete(f"f{k}_xt")
            transparency = args.fade * noise_level(t_real, t_min, t_max)
            for name, colour in layers:
                if (
                    t_real not in t_hat[name]
                ):  # stream not evaluated outside the CFG window
                    continue
                obj = f"f{k}_{name.replace('-', '_')}"
                cmd.create(
                    obj, name.replace("-", "_"), t_hat[name].index(t_real) + 1, 1
                )
                # Under "both", purple already means cond; no motif highlight mid-trajectory
                style_structure(
                    obj, colour, transparency, "" if mode == "both" else args.motif
                )
                objs.append(obj)
        frames.append(
            {
                "frame": k,
                "t_hat": t_real,
                "state": state,
                "objects": objs,
                "scene": f"frame{k}_t{t_real:.3g}",
            }
        )

    refs = []
    for spec in args.ref:
        path, colour = spec.rsplit(":", 1)
        obj = f"ref_{Path(path).stem}"
        cmd.load(path, obj)
        cmd.remove(f"{obj} and elem H")
        cmd.super(f"{obj} and name CA", "x0_envelope and name CA")
        # A different fold may superpose only in part; keep it centred on x_0
        shift = [
            a - b
            for a, b in zip(cmd.centerofmass("x0_envelope"), cmd.centerofmass(obj))
        ]
        cmd.translate(shift, obj, camera=0)
        style_structure(obj, {"purple": "traj_purple"}.get(colour, "traj_grey"), 0.0)
        refs.append({"scene": obj, "objects": [obj]})

    if args.view:
        view = [
            float(x) for x in args.view.replace("(", "").replace(")", "").split(",")
        ]
    else:
        buffer = args.zoom_buffer
        if buffer is None:
            buffer = args.dot_envelope + 2 if args.dot_envelope > 0 else 5.0
        cmd.orient("x0_envelope and polymer")
        cmd.zoom("x0_envelope", buffer=buffer, complete=1)
        view = cmd.get_view()

    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)
    cmd.set("orthoscopic", 1)
    cmd.set("ray_shadow", 0)
    cmd.set("ray_trace_mode", 1 if args.outline else 0)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.45)
    cmd.set("specular", args.noise_matte)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.space("cmyk")

    # One scene per frame (PgUp/PgDn in PyMOL); the session opens on the first
    for f in frames + refs:
        cmd.disable("all")
        for obj in f["objects"]:
            cmd.enable(obj)
        cmd.set_view(view)
        cmd.scene(f["scene"], "store")
    cmd.scene(frames[0]["scene"], "recall")
    cmd.save(str(args.out_dir / "trajectory_frames.pse"))

    for f in frames + refs:
        cmd.scene(f["scene"], "recall")
        f["png"] = f"{f['scene']}.png"  # relative to out_dir
        cmd.png(
            str(args.out_dir / f["png"]),
            width=args.size,
            height=args.size,
            dpi=300,
            ray=1,
        )
    (args.out_dir / "frames_used.json").write_text(json.dumps(frames, indent=1))


main()
cmd.quit()

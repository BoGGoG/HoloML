import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
import scienceplots
from src.Vaidya_AdS import (
    geodesic_length_from_traj,
    geodesic_length_reg,
    get_mass_and_dmdv,
    integrate_geodesic,
    length_profile_vs_x,
    lengths_vs_rstar,
    speed_stats,
    ds_dlambda,
)

plt.style.use(["science", "grid"])
mpl.rcParams.update(
    {
        # Fonts
        "font.family": "sans-serif",  # "sans-serif" or "serif"
        "font.serif": [
            "Times New Roman"
        ],  # Or ["Computer Modern Roman"] for LaTeX style
        "mathtext.fontset": "cm",  # Computer Modern for math
        "font.size": 14,  # Base font size (good for papers)
        # Axes
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "axes.linewidth": 1.8,
        # Ticks
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        # Legend
        "legend.fontsize": 14,
        "legend.frameon": False,
        # Lines
        "lines.linewidth": 3.2,
        "lines.markersize": 6,
        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def geodesics_and_lengths(v0_list, r_star_list, n_steps=40000, dt=0.002, r_cut=200.0):
    traj_list = []
    length_list = []
    v0_rstar_list = []

    for v0 in v0_list:
        for r_star in r_star_list:
            traj = integrate_geodesic(r_star, v0, n_steps=n_steps, dt=dt)
            length = geodesic_length_from_traj(traj, dt=dt, r_cut=r_cut)
            traj_list.append(traj)
            length_list.append(length)
            v0_rstar_list.append((v0, r_star))
    traj_list = np.array(traj_list)
    length_list = np.array(length_list)
    v0_rstar_list = np.array(v0_rstar_list)
    return traj_list, length_list, v0_rstar_list


def convergence_test_with_plots(
    v0_list,
    r_star_list,
    n_steps0=20000,
    dt0=0.004,
    n_refinements=4,
    r_cut=200.0,
    out_dir="convergence_plots",
    make_speed_plots=True,
):
    """
    Run convergence test and generate illustrative plots:
      - L vs dt (per (v0,r_star))
      - |ΔL| vs dt (log-log)
      - optional: ds/dλ profiles for coarse vs fine
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dts = np.array([dt0 / (2**k) for k in range(n_refinements)], dtype=float)
    n_steps = np.array(
        [int(n_steps0 * (2**k)) for k in range(n_refinements)], dtype=int
    )
    lam_max = float(n_steps0 * dt0)

    print(f"Target λ_max = n_steps * dt = {lam_max:g}")
    print(f"r_cut = {r_cut:g}")
    print(f"Saving plots to: {out_dir.resolve()}\n")

    # Store results in dict keyed by (v0, r_star)
    results = {}

    for v0 in v0_list:
        for r_star in r_star_list:
            key = (float(v0), float(r_star))
            Ls = []
            Lregs = []
            trajs = []

            print(f"=== v0={v0:g}, r_star={r_star:g} ===")
            for k, (ns, dt) in enumerate(zip(n_steps, dts)):
                traj = integrate_geodesic(r_star, v0, n_steps=int(ns), dt=float(dt))
                L = float(geodesic_length_from_traj(traj, dt=float(dt), r_cut=r_cut))
                Lreg = float(geodesic_length_reg(L, r_cut=r_cut))

                Ls.append(L)
                Lregs.append(Lreg)

                # Keep only coarse+fine trajectories for speed plots (to save memory)
                if make_speed_plots and (k == 0 or k == len(dts) - 1):
                    trajs.append(np.array(traj))

                if k == 0:
                    print(
                        f"  k={k}  dt={dt:.6g}  n_steps={ns:<7d}  L={L:.12g}  Lreg={Lreg:.12g}"
                    )
                else:
                    dL = abs(Ls[k] - Ls[k - 1])
                    dLreg = abs(Lregs[k] - Lregs[k - 1])
                    print(
                        f"  k={k}  dt={dt:.6g}  n_steps={ns:<7d}  "
                        f"L={L:.12g}  |ΔL|={dL:.3g}  "
                        f"Lreg={Lreg:.12g}  |ΔLreg|={dLreg:.3g}"
                    )

            results[key] = {
                "dts": dts.copy(),
                "n_steps": n_steps.copy(),
                "L": np.array(Ls, dtype=float),
                "Lreg": np.array(Lregs, dtype=float),
                "trajs": trajs,  # [coarse, fine] if make_speed_plots else []
                "lam_max": lam_max,
                "r_cut": float(r_cut),
            }
            print()

    # ---------- Plot 1: L vs dt (one combined figure) ----------
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    for (v0, r_star), d in results.items():
        ax.plot(
            d["dts"],
            d["L"],
            marker="o",
            linewidth=1.5,
            label=rf"$v_0={v0:g},\ r_\ast={r_star:g}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\Delta\lambda$ (dt)")
    ax.set_ylabel(r"Length $L$")
    ax.set_title(
        rf"Convergence: $L$ vs $\Delta\lambda$  (fixed $\lambda_{{\max}}={lam_max:g}$)"
    )
    ax.grid(True, which="both", linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "L_vs_dt.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 2: |ΔL| vs dt (log-log), with reference slope dt^4 ----------
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    for (v0, r_star), d in results.items():
        L = d["L"]
        dt_arr = d["dts"]
        # successive differences aligned with dt[1:]
        dL = np.abs(L[1:] - L[:-1])
        ax.plot(
            dt_arr[1:],
            dL,
            marker="o",
            linewidth=1.5,
            label=rf"$v_0={v0:g},\ r_\ast={r_star:g}$",
        )

    # reference line ~ dt^4 anchored to the first curve’s first point (if any)
    any_key = next(iter(results.keys()), None)
    if any_key is not None:
        d0 = results[any_key]
        ref_x = d0["dts"][1:]
        ref_y0 = np.abs(d0["L"][1] - d0["L"][0])
        if ref_y0 > 0:
            ref = ref_y0 * (ref_x / ref_x[0]) ** 4
            ax.plot(
                ref_x,
                ref,
                linestyle="--",
                linewidth=2.0,
                label=r"reference $\propto (\Delta\lambda)^4$",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\Delta\lambda$ (dt)")
    ax.set_ylabel(r"$|L_k - L_{k-1}|$")
    ax.set_title(r"Convergence: successive differences (log-log)")
    ax.grid(True, which="both", linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "dL_vs_dt_loglog.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 3 (optional): speed profile ds/dλ for coarse vs fine ----------
    if make_speed_plots:
        for (v0, r_star), d in results.items():
            trajs = d["trajs"]
            if len(trajs) != 2:
                continue

            traj_coarse, traj_fine = trajs
            sdot_coarse = np.array(jax.vmap(ds_dlambda)(traj_coarse))
            sdot_fine = np.array(jax.vmap(ds_dlambda)(traj_fine))

            lam_coarse = np.arange(len(sdot_coarse)) * d["dts"][0]
            lam_fine = np.arange(len(sdot_fine)) * d["dts"][-1]

            fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0))
            ax.plot(
                lam_coarse,
                sdot_coarse,
                linewidth=1.5,
                label=rf"coarse dt={d['dts'][0]:g}",
            )
            ax.plot(
                lam_fine, sdot_fine, linewidth=1.5, label=rf"fine dt={d['dts'][-1]:g}"
            )
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$ds/d\lambda$")
            ax.set_title(rf"Speed profile $ds/d\lambda$  (v0={v0:g}, r*={r_star:g})")
            ax.grid(True, linestyle=":")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(out_dir / f"sdot_profile_v0{v0:g}_rstar{r_star:g}.png", dpi=200)
            plt.close(fig)

    # Save raw results for later analysis
    np.savez_compressed(
        out_dir / "convergence_results.npz",
        dts=dts,
        n_steps=n_steps,
        v0_list=np.array(v0_list, dtype=float),
        r_star_list=np.array(r_star_list, dtype=float),
        # Flatten per-key results into arrays (shape: Npairs x n_refinements)
        L=np.array(
            [results[(float(v0), float(r))]["L"] for v0 in v0_list for r in r_star_list]
        ),
        Lreg=np.array(
            [
                results[(float(v0), float(r))]["Lreg"]
                for v0 in v0_list
                for r in r_star_list
            ]
        ),
        lam_max=np.array(lam_max),
        r_cut=np.array(r_cut),
    )

    print("Saved:")
    print(" - L_vs_dt.png")
    print(" - dL_vs_dt_loglog.png")
    if make_speed_plots:
        print(" - sdot_profile_*.png (per (v0,r*))")
    print(" - convergence_results.npz")


if __name__ == "__main__":
    plots_dir = Path("checks_plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    #
    # v0_list = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    # r_star_list = list(np.linspace(0.001, 20, 10))
    #
    # ### Geodesic length vs r_star for multiple v0 ###
    # traj_list, length_list, v0_rstar_list = geodesics_and_lengths(
    #     v0_list, r_star_list, n_steps=40000, dt=0.002, r_cut=200.0
    # )
    #
    # # speed stats for reference
    # speed_stats_list = [speed_stats(t) for t in traj_list]
    # print("Speed stats")
    # for (v0, r_star), stats in zip(v0_rstar_list, speed_stats_list):
    #     print(f"v0={v0}, r_star={r_star}: {stats}")
    #
    # # save speed stats to a file in plots_dir
    # path_stats = plots_dir / "speed_stats.txt"
    # np.savetxt(
    #     path_stats,
    #     np.array(
    #         [
    #             [v0, r_star, *stats.values()]
    #             for (v0, r_star), stats in zip(v0_rstar_list, speed_stats_list)
    #         ]
    #     ),
    #     fmt="%10.5f",
    #     header="v0 r_star min_speed max_speed mean_speed std_speed",
    # )
    # print(f"Saved speed stats to {path_stats}")

    v0_list = [-1.0, 0.1, 1.0]
    r_star_list = [0.5, 1.0, 2.0, 5.0]

    convergence_test_with_plots(
        v0_list=v0_list,
        r_star_list=r_star_list,
        n_steps0=20000,
        dt0=0.004,
        n_refinements=4,
        r_cut=200.0,
        out_dir=plots_dir / "convergence_test",
        make_speed_plots=True,
    )

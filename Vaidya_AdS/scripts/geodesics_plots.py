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
    get_mass_and_dmdv,
    integrate_geodesic,
    length_profile_vs_x,
    lengths_vs_rstar,
    speed_stats,
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


def plot_figure_5(plots_dir):
    plots_dir.mkdir(exist_ok=True)
    v0_values = [-2, -1, 0, 0.1, 0.5, 1]
    # r_stars = list(np.linspace(0.0001, 0.0009, 10))
    # r_stars = list(np.linspace(0.001, 0.009, 10))
    # r_stars += list(np.linspace(0.01, 0.09, 50))
    # r_stars += list(np.linspace(0.1, 0.9, 50))
    # r_stars += list(np.linspace(1.1, 20.0, 50))
    r_stars = list(np.linspace(0.0001, 20, 200))

    # Setup Colormap
    norm = Normalize(vmin=min(r_stars), vmax=max(r_stars))
    cmap = plt.cm.cool
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Create figure with specific layout adjustment
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), subplot_kw={"projection": "polar"})

    # Adjust subplots to leave room on the right for the colorbar
    # left, bottom, right, top parameters define the plotting area
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)

    axes_flat = axes.flatten()

    for i, v0 in enumerate(v0_values):
        ax = axes_flat[i]

        # Boundary
        theta_boundary = np.linspace(-np.pi / 2, np.pi / 2, 100)
        ax.plot(theta_boundary, [np.pi / 2] * 100, "k-", linewidth=2)

        # Horizon
        m_val, _ = get_mass_and_dmdv(v0, m_0=1.0, v_s=1.0)
        if m_val > 0:
            r_horizon = np.sqrt(m_val)  # apparent horizon
            R_horizon = np.arctan(r_horizon)
            theta_h = np.linspace(0, 2 * np.pi, 200)
            ax.plot(theta_h, [R_horizon] * 200, color="brown", linewidth=2)

        # Geodesics
        for r_star in r_stars:
            if m_val > 0 and r_star <= np.sqrt(m_val):
                continue

            traj = integrate_geodesic(r_star, v0)
            v_t, r_t, x_t = traj[:, 0], traj[:, 1], traj[:, 2]
            R_plot = np.arctan(r_t)
            color = cmap(norm(r_star))

            ax.plot(x_t, R_plot, color=color, linewidth=1)
            ax.plot(-x_t, R_plot, color=color, linewidth=1)

        ax.set_title(r"$v_0$" + f"= {v0}", fontsize=12)
        ax.set_ylim(0, np.pi / 2 + 0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)

    # Manually add axes for the colorbar to the right of the subplots
    # [left, bottom, width, height] in figure coordinate fractions
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Turning Point ($r_\ast$)", fontsize=12)
    plot_path = plots_dir / "figure_5_geodesics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {plot_path}")
    plt.close()


def plot_lengths_sweep(
    v0=0.1,
    rstar_min=1e-3,
    rstar_max=20.0,
    n_rstars=200,
    n_steps=40000,
    dt=0.002,
    r_cut=200.0,
    out_dir=None,
    show=True,
):
    """
    Produce plots for L, L_reg, h, v_inf as functions of r_star, and parametric plots
    versus boundary half-width h.
    """
    # r_star often spans decades; geomspace is usually more informative than linspace
    r_stars = np.geomspace(rstar_min, rstar_max, n_rstars)

    Ls, Lregs, hs, vins = lengths_vs_rstar(
        r_stars, v0, n_steps=n_steps, dt=dt, r_cut=r_cut
    )

    # --- 1) L and L_reg vs r_star ---
    fig1, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(r_stars, Ls, label="L")
    ax.plot(r_stars, Lregs, label="L_reg")
    ax.set_xscale("log")
    ax.set_xlabel(r"$r_\ast$")
    ax.set_ylabel("Length")
    ax.set_title(rf"Geodesic lengths vs $r_\ast$  (v0={v0}, r_cut={r_cut})")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig1.set_constrained_layout(True)

    # --- 2) Diagnostics: h(r_star) and v_inf(r_star) ---
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.plot(r_stars, hs)
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$h = x(r_{\rm cut})$")
    ax1.grid(True, which="both", linestyle=":")

    ax2.plot(r_stars, vins)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$r_\ast$")
    ax2.set_ylabel(r"$v_\infty = v(r_{\rm cut})$")
    ax2.grid(True, which="both", linestyle=":")

    fig2.suptitle(rf"Cutoff diagnostics (v0={v0}, r_cut={r_cut})")
    fig1.set_constrained_layout(True)

    # --- 3) Parametric plots vs boundary scale h ---
    # Sort by h for nice monotone curves (h may not be perfectly monotone in r_star numerically)
    order = np.argsort(hs)
    h_sorted = hs[order]
    Lreg_sorted = Lregs[order]
    vin_sorted = vins[order]

    fig3, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(h_sorted, Lreg_sorted)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$L_{\rm reg}$")
    ax.set_title(rf"$L_{{\rm reg}}(h)$  (v0={v0}, r_cut={r_cut})")
    ax.grid(True, linestyle=":")
    fig1.set_constrained_layout(True)

    fig4, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(h_sorted, vin_sorted)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$v_\infty$")
    ax.set_title(rf"$v_\infty(h)$  (v0={v0}, r_cut={r_cut})")
    ax.grid(True, linestyle=":")
    fig1.set_constrained_layout(True)

    # Optionally save
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig1.savefig(os.path.join(out_dir, f"lengths_vs_rstar_v0{v0:g}.png"), dpi=200)
        fig2.savefig(
            os.path.join(out_dir, f"diagnostics_vs_rstar_v0{v0:g}.png"), dpi=200
        )
        fig3.savefig(os.path.join(out_dir, f"Lreg_vs_h_v0{v0:g}.png"), dpi=200)
        fig4.savefig(os.path.join(out_dir, f"vinf_vs_h_v0{v0:g}.png"), dpi=200)

        # Save raw arrays too (super useful)
        np.savez_compressed(
            os.path.join(out_dir, f"sweep_v0{v0:g}.npz"),
            r_stars=r_stars,
            L=Ls,
            Lreg=Lregs,
            h=hs,
            v_inf=vins,
            n_steps=np.int64(n_steps),
            dt=np.float64(dt),
            r_cut=np.float64(r_cut),
        )

    if show:
        plt.show()
    else:
        plt.close("all")


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


if __name__ == "__main__":
    plots_dir = Path("geodesics_plots")
    # plot_figure_5(plots_dir)

    v0_list = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    r_star_list = list(np.linspace(0.001, 20, 100))

    ### Geodesic length vs r_star for multiple v0 ###
    traj_list, length_list, v0_rstar_list = geodesics_and_lengths(
        v0_list, r_star_list, n_steps=40000, dt=0.002, r_cut=200.0
    )

    # plot length vs r_star for each v0
    for i_v0, v0 in enumerate(v0_list):
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        lengths = length_list[i_v0 * len(r_star_list) : (i_v0 + 1) * len(r_star_list)]
        r_stars = v0_rstar_list[
            i_v0 * len(r_star_list) : (i_v0 + 1) * len(r_star_list), 1
        ]
        plt.plot(r_stars, lengths, marker="o")
        plt.title(f"Geodesic Lengths for v0={v0}")
        plt.xlabel("r_star")
        plt.ylabel("Geodesic Length L")
        plt.grid(True, which="both", linestyle=":")
        plot_path = plots_dir / f"lengths_v0_{v0}.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Saved figure to {plot_path}")

    # --- single plot: length vs r_star for all v0 (different colors) ---
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    ax1, ax2 = axs

    cmap = plt.cm.viridis  # any matplotlib colormap
    colors = cmap(np.linspace(0.0, 1.0, len(v0_list)))

    n_r = len(r_star_list)

    for i_v0, (v0, color) in enumerate(zip(v0_list, colors)):
        start = i_v0 * n_r
        stop = (i_v0 + 1) * n_r

        lengths = length_list[start:stop]
        r_stars = v0_rstar_list[start:stop, 1].astype(float)

        # If you want consistent ordering (just in case):
        order = np.argsort(r_stars)
        r_stars = r_stars[order]
        lengths = lengths[order]

        ax1.plot(
            r_stars,
            lengths,
            marker="o",
            markersize=2,
            linewidth=1.5,
            color=color,
            label=rf"$v_0={v0:g}$",
        )
        ax2.plot(
            r_stars,
            lengths,
            marker="o",
            markersize=2,
            linewidth=1.5,
            color=color,
            label=rf"$v_0={v0:g}$",
        )

    ax1.set_ylim(0, 16)
    ax1.set_title("Geodesic length $L$ vs turning point $r_\\ast$ (all $v_0$)")
    ax1.set_xlabel(r"$r_\ast$")
    ax1.set_ylabel(r"Geodesic length $L$")
    ax1.grid(True, which="both", linestyle=":")

    ax2.set_title("Geodesic length $L$ vs turning point $r_\\ast$ (all $v_0$)")
    ax2.set_xlabel(r"$r_\ast$")
    ax2.set_ylabel(r"Geodesic length $L$")
    ax2.grid(True, which="both", linestyle=":")
    ax2.set_yscale("log")

    # Legend: put it outside to avoid covering curves
    ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    plot_path = plots_dir / "lengths_all_v_-2_2.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to {plot_path}")
    plt.close(fig)

    ### Geodesic length profile L_half(x) for various r_star ###
    r_star_list = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0]
    for r_star in r_star_list:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(v0_list)))

        for v0, c in zip(v0_list, colors):
            traj = integrate_geodesic(r_star, v0, n_steps=40000, dt=0.002)
            x, Lx = length_profile_vs_x(traj, dt=0.002, r_cut=200.0)
            ax.plot(x, Lx, color=c, label=rf"$v_0={v0:g}$")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$L_{\mathrm{half}}(x)$")
        ax.set_title(
            rf"Geodesic length profile $L_{{\mathrm{{half}}}}(x)$ for $r_\ast={r_star}$"
        )
        ax.grid(True, linestyle=":")
        ax.legend(frameon=False)
        ax.set_ylim(0, 8)
        fig.tight_layout()
        plot_path = plots_dir / f"length_profile_vs_x_rstar{r_star}.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Saved figure to {plot_path}")

    ### Geodesic length profile L_half(x) for various v0 ###
    for v0 in v0_list:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(r_star_list)))

        for r_star, c in zip(r_star_list, colors):
            traj = integrate_geodesic(r_star, v0, n_steps=40000, dt=0.002)
            x, Lx = length_profile_vs_x(traj, dt=0.002, r_cut=200.0)
            ax.plot(x, Lx, color=c, label=rf"$r_\ast={r_star}$")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$L_{\mathrm{half}}(x)$")
        ax.set_title(
            rf"Geodesic length profile $L_{{\mathrm{{half}}}}(x)$ for $v_0={v0:g}$"
        )
        ax.grid(True, linestyle=":")
        ax.legend(frameon=False)
        ax.set_ylim(0, 8)
        fig.tight_layout()
        plot_path = plots_dir / f"length_profile_vs_x_v0{v0}.png"
        plt.savefig(plot_path, dpi=200)
        print(f"Saved figure to {plot_path}")

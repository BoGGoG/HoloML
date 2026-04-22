import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys, os
    from pathlib import Path
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    return Normalize, Path, ScalarMappable, mo, mpl, np, plt, sys


@app.cell
def _(Path, sys):
    _nb_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_nb_dir.parent / "src"))

    import jax
    jax.config.update("jax_enable_x64", True)

    import scienceplots

    from Vaidya_AdS import (
        geodesic_length_from_traj,
        get_mass_and_dmdv,
        integrate_geodesic,
        length_profile_vs_x,
        lengths_vs_rstar,
    )

    return (
        geodesic_length_from_traj,
        get_mass_and_dmdv,
        integrate_geodesic,
        length_profile_vs_x,
        lengths_vs_rstar,
    )


@app.cell
def _(mpl, plt):
    plt.style.use(["science", "grid"])
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "axes.linewidth": 1.5,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 4,
        }
    )
    return


@app.cell
def _(mo):
    mo.md("""
    # Vaidya-AdS Geodesics Explorer

    Interactive version of `scripts/geodesics_plots.py`. Navigate each section below.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Figure 5 — Geodesics in compactified AdS
    """)
    return


@app.cell
def _(mo):
    n_rstars_5 = mo.ui.slider(30, 200, value=80, label=r"number of $r_\star$ points")
    n_rstars_5
    return (n_rstars_5,)


@app.cell
def _(
    Normalize,
    ScalarMappable,
    get_mass_and_dmdv,
    integrate_geodesic,
    n_rstars_5,
    np,
    plt,
):
    _v0_values = [-2, -1, 0, 0.1, 0.5, 1]
    _r_stars = np.linspace(0.0001, 20, n_rstars_5.value)
    _norm = Normalize(vmin=float(_r_stars[0]), vmax=float(_r_stars[-1]))
    _cmap = plt.cm.cool
    _sm = ScalarMappable(norm=_norm, cmap=_cmap)
    _sm.set_array([])

    _fig5, _axes5 = plt.subplots(
        2, 3, figsize=(14, 8), subplot_kw={"projection": "polar"}
    )
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)

    for _i, _v0 in enumerate(_v0_values):
        _ax = _axes5.flatten()[_i]

        _ax.plot(
            np.linspace(-np.pi / 2, np.pi / 2, 100),
            [np.pi / 2] * 100,
            "k-",
            linewidth=2,
        )

        _m_val, _ = get_mass_and_dmdv(float(_v0), m_0=1.0, v_s=1.0)
        _m = float(_m_val)
        if _m > 0:
            _ax.plot(
                np.linspace(0, 2 * np.pi, 200),
                [np.arctan(np.sqrt(_m))] * 200,
                color="brown",
                linewidth=2,
            )

        for _r_star in _r_stars:
            if _m > 0 and _r_star <= np.sqrt(_m):
                continue
            _traj = integrate_geodesic(_r_star, float(_v0))
            _R = np.arctan(np.array(_traj[:, 1]))
            _x = np.array(_traj[:, 2])
            _c = _cmap(_norm(_r_star))
            _ax.plot(_x, _R, color=_c, linewidth=0.8)
            _ax.plot(-_x, _R, color=_c, linewidth=0.8)

        _ax.set_title(r"$v_0 = $" + f"{_v0}", fontsize=11)
        _ax.set_ylim(0, np.pi / 2 + 0.1)
        _ax.set_yticks([])
        _ax.set_xticks([])
        _ax.spines["polar"].set_visible(False)

    _cbar_ax5 = _fig5.add_axes([0.88, 0.15, 0.02, 0.7])
    _cbar5 = _fig5.colorbar(_sm, cax=_cbar_ax5)
    _cbar5.set_label(r"Turning point $r_\ast$", fontsize=11)
    _fig5
    return


@app.cell
def _(mo):
    mo.md("""
    ## Geodesic lengths vs turning point $r_\star$
    """)
    return


@app.cell
def _(mo):
    v0_sweep = mo.ui.slider(-2.0, 2.0, value=0.1, step=0.1, label="$v_0$")
    n_rstars_sweep = mo.ui.slider(50, 300, value=100, label=r"$r_\star$ points")
    r_cut_sweep = mo.ui.number(value=200.0, label="$r_{\\rm cut}$")
    mo.hstack([v0_sweep, n_rstars_sweep, r_cut_sweep])
    return n_rstars_sweep, r_cut_sweep, v0_sweep


@app.cell
def _(lengths_vs_rstar, n_rstars_sweep, np, r_cut_sweep, v0_sweep):
    _r_stars_sw = np.geomspace(1e-3, 20.0, n_rstars_sweep.value)
    Ls_sw, Lregs_sw, hs_sw, vins_sw = lengths_vs_rstar(
        _r_stars_sw,
        float(v0_sweep.value),
        n_steps=50_000,
        dt=0.002,
        r_cut=float(r_cut_sweep.value),
    )
    r_stars_sw = _r_stars_sw
    return Lregs_sw, Ls_sw, hs_sw, r_stars_sw, vins_sw


@app.cell
def _(Lregs_sw, Ls_sw, plt, r_cut_sweep, r_stars_sw, v0_sweep):
    _fig1, _ax1 = plt.subplots(figsize=(7, 4.5))
    _ax1.plot(r_stars_sw, Ls_sw, label=r"$L$")
    _ax1.plot(r_stars_sw, Lregs_sw, label=r"$L_{\rm reg}$")
    _ax1.set_xscale("log")
    _ax1.set_xlabel(r"$r_\ast$")
    _ax1.set_ylabel("Length")
    _ax1.set_title(
        rf"Geodesic lengths  ($v_0={v0_sweep.value:g}$,  $r_{{\rm cut}}={r_cut_sweep.value:g}$)"
    )
    _ax1.legend()
    _fig1.set_constrained_layout(True)
    _fig1
    return


@app.cell
def _(hs_sw, plt, r_stars_sw, v0_sweep, vins_sw):
    _fig2, (_ax2a, _ax2b) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    _ax2a.plot(r_stars_sw, hs_sw)
    _ax2a.set_xscale("log")
    _ax2a.set_ylabel(r"$h = x(r_{\rm cut})$")
    _ax2b.plot(r_stars_sw, vins_sw)
    _ax2b.set_xscale("log")
    _ax2b.set_xlabel(r"$r_\ast$")
    _ax2b.set_ylabel(r"$v_\infty = v(r_{\rm cut})$")
    _fig2.suptitle(rf"Cutoff diagnostics  ($v_0={v0_sweep.value:g}$)")
    _fig2.set_constrained_layout(True)
    _fig2
    return


@app.cell
def _(Lregs_sw, hs_sw, np, plt, r_cut_sweep, v0_sweep, vins_sw):
    _order = np.argsort(hs_sw)
    _h_s = hs_sw[_order]
    _Lr_s = Lregs_sw[_order]
    _vi_s = vins_sw[_order]

    _fig3, (_ax3a, _ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
    _ax3a.plot(_h_s, _Lr_s)
    _ax3a.set_xlabel(r"$h$")
    _ax3a.set_ylabel(r"$L_{\rm reg}$")
    _ax3a.set_title(
        rf"$L_{{\rm reg}}(h)$  ($v_0={v0_sweep.value:g}$,  $r_{{\rm cut}}={r_cut_sweep.value:g}$)"
    )

    _ax3b.plot(_h_s, _vi_s)
    _ax3b.set_xlabel(r"$h$")
    _ax3b.set_ylabel(r"$v_\infty$")
    _ax3b.set_title(rf"$v_\infty(h)$  ($v_0={v0_sweep.value:g}$)")
    _fig3.set_constrained_layout(True)
    _fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ## Geodesic lengths for all $v_0$
    """)
    return


@app.cell
def _(mo):
    n_rstars_all = mo.ui.slider(20, 100, value=50, label=r"number of $r_\star$ per v0")
    run_all = mo.ui.run_button(label="Compute all-v0 sweep")
    mo.hstack([n_rstars_all, run_all])
    return n_rstars_all, run_all


@app.cell
def _(
    geodesic_length_from_traj,
    integrate_geodesic,
    mo,
    n_rstars_all,
    np,
    run_all,
):
    mo.stop(not run_all.value, mo.md("Click **Compute** to run the full v0 sweep."))

    _v0_list_all = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    _rstar_list_all = list(np.linspace(0.001, 20, n_rstars_all.value))

    _lengths, _v0rs = [], []
    for _v0a in _v0_list_all:
        for _rs in _rstar_list_all:
            _traj = integrate_geodesic(_rs, _v0a, n_steps=50_000, dt=0.0005)
            _lengths.append(float(geodesic_length_from_traj(_traj, dt=0.0005, r_cut=200.0)))
            _v0rs.append((_v0a, _rs))

    all_lengths = np.array(_lengths)
    all_v0_rstar = np.array(_v0rs)
    all_v0_list = _v0_list_all
    all_rstar_list = _rstar_list_all
    return all_lengths, all_rstar_list, all_v0_list, all_v0_rstar


@app.cell
def _(all_lengths, all_rstar_list, all_v0_list, all_v0_rstar, np, plt):
    _n_r = len(all_rstar_list)
    _colors_all = plt.cm.viridis(np.linspace(0, 1, len(all_v0_list)))

    _fig_all, (_axa1, _axa2) = plt.subplots(1, 2, figsize=(14, 5))

    for _i, (_v0a, _col) in enumerate(zip(all_v0_list, _colors_all)):
        _ls = all_lengths[_i * _n_r : (_i + 1) * _n_r]
        _rs = all_v0_rstar[_i * _n_r : (_i + 1) * _n_r, 1].astype(float)
        _ord = np.argsort(_rs)
        _rs, _ls = _rs[_ord], _ls[_ord]
        for _axx in [_axa1, _axa2]:
            _axx.plot(
                _rs, _ls,
                marker="o", markersize=2, linewidth=1.5,
                color=_col, label=rf"$v_0={_v0a:g}$",
            )

    _axa1.set_ylim(0, 18)
    _axa2.set_ylim(1.0, 50)
    _axa2.set_yscale("log")
    for _axx in [_axa1, _axa2]:
        _axx.set_xlabel(r"$r_\ast$")
        _axx.set_ylabel(r"$L$")
        _axx.set_title(r"Geodesic length $L$ vs $r_\ast$  (all $v_0$)")
    _axa1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    _fig_all.set_constrained_layout(True)
    _fig_all
    return


@app.cell
def _(mo):
    mo.md("""
    ## Length profile $L_{\mathrm{half}}(x)$
    """)
    return


@app.cell
def _(mo):
    _v0_opts = {str(v): float(v) for v in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]}
    _rs_opts = {str(r): r for r in [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]}

    profile_mode = mo.ui.radio(
        options=[r"Fix $v_0$ (vary $r_\\ast$)", "Fix $r_\star$ (vary $v_0$)"],
        value=r"Fix $v_0$ (vary $r_\\ast$)",
        label="Plot mode",
    )
    v0_profile = mo.ui.dropdown(_v0_opts, value="0.0", label="$v_0$")
    rstar_profile = mo.ui.dropdown(_rs_opts, value="1.0", label="$r_\\ast$")
    mo.hstack([profile_mode, v0_profile, rstar_profile])
    return profile_mode, rstar_profile, v0_profile


@app.cell
def _(
    integrate_geodesic,
    length_profile_vs_x,
    np,
    plt,
    profile_mode,
    rstar_profile,
    v0_profile,
):
    _v0_list_p = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    _rs_list_p = [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]
    _fix_v0 = "v_0" in profile_mode.value and "r_\\ast$)" in profile_mode.value

    _fig_p, (_axp1, _axp2) = plt.subplots(1, 2, figsize=(14, 5))

    if _fix_v0:
        _v0p = v0_profile.value
        _cols = plt.cm.viridis(np.linspace(0, 1, len(_rs_list_p)))
        for _rsp, _c in zip(_rs_list_p, _cols):
            _traj = integrate_geodesic(_rsp, _v0p, n_steps=50_000, dt=0.0005)
            _xp, _Lxp = length_profile_vs_x(_traj, dt=0.0005, r_cut=200.0)
            _axp1.plot(_xp, _Lxp, color=_c, label=rf"$r_\ast={_rsp}$")
            _axp2.plot(np.arctanh(_xp), _Lxp, color=_c, label=rf"$r_\ast={_rsp}$")
        _title_sfx = rf"$v_0={_v0p:g}$"
    else:
        _rsp = rstar_profile.value
        _cols = plt.cm.viridis(np.linspace(0, 1, len(_v0_list_p)))
        for _v0p, _c in zip(_v0_list_p, _cols):
            _traj = integrate_geodesic(_rsp, _v0p, n_steps=50_000, dt=0.0005)
            _xp, _Lxp = length_profile_vs_x(_traj, dt=0.0005, r_cut=200.0)
            _axp1.plot(_xp, _Lxp, color=_c, label=rf"$v_0={_v0p:g}$")
            _axp2.plot(np.arctanh(_xp), _Lxp, color=_c, label=rf"$v_0={_v0p:g}$")
        _title_sfx = rf"$r_\ast={_rsp}$"

    _axp1.set_xlabel(r"$x$")
    _axp2.set_xlabel(r"$\mathrm{arctanh}(x)$")
    for _ap in [_axp1, _axp2]:
        _ap.set_ylabel(r"$L_{\mathrm{half}}(x)$")
        _ap.set_ylim(0, 8)
        _ap.set_xlim(0, 5)
        _ap.legend(frameon=False, fontsize=9)
    _axp1.set_title(rf"Length profile: {_title_sfx}")
    _axp2.set_title(rf"Length profile (arctanh): {_title_sfx}")
    _fig_p.set_constrained_layout(True)
    _fig_p
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regularized length $L_{\mathrm{reg}}(h)$

    $L_{\rm reg} = 2L_{\rm half} - 2\log(2\,r_{\rm cut})$
    """)
    return


@app.cell
def _(mo):
    _v0_opts_r = {str(v): float(v) for v in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]}
    v0_reg = mo.ui.dropdown(_v0_opts_r, value="0.0", label="$v_0$")
    v0_reg
    return (v0_reg,)


@app.cell
def _(integrate_geodesic, length_profile_vs_x, np, plt, v0_reg):
    _rs_list_r = [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]
    _r_cut_r = 200.0
    _v0r = v0_reg.value
    _cols_r = plt.cm.viridis(np.linspace(0, 1, len(_rs_list_r)))

    _fig_r, (_axr1, _axr2) = plt.subplots(1, 2, figsize=(14, 5))

    for _rsr, _cr in zip(_rs_list_r, _cols_r):
        _traj_r = integrate_geodesic(_rsr, _v0r, n_steps=50_000, dt=0.0005)
        _xh, _Lxh = length_profile_vs_x(_traj_r, dt=0.0005, r_cut=_r_cut_r)
        _Lreg = 2 * _Lxh - 2 * np.log(2 * _r_cut_r)
        _h = 2 * _xh
        _axr1.plot(_h, _Lreg, color=_cr, label=rf"$r_\ast={_rsr}$")
        _axr2.plot(np.arctanh(_h), _Lreg, color=_cr, label=rf"$r_\ast={_rsr}$")

    _axr1.set_xlabel(r"$h$")
    _axr2.set_xlabel(r"$\mathrm{arctanh}(h)$")
    for _ar in [_axr1, _axr2]:
        _ar.set_ylabel(r"$L_{\mathrm{reg}}(h)$")
        _ar.set_ylim(-15, 0)
        _ar.set_xlim(0, 8)
        _ar.legend(frameon=False, fontsize=9)
    _axr1.set_title(rf"Regularized length: $v_0={_v0r:g}$")
    _axr2.set_title(rf"Regularized length (arctanh): $v_0={_v0r:g}$")
    _fig_r.set_constrained_layout(True)
    _fig_r
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

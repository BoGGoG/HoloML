import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import sys
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
    )

    return (
        geodesic_length_from_traj,
        get_mass_and_dmdv,
        integrate_geodesic,
        length_profile_vs_x,
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

    Geodesics are computed **once** in the Figure 5 section and reused by all plots below.
    Adjust the grid parameters and recompute only when you need a finer grid.
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
    n_rstars_5 = mo.ui.slider(30, 200, value=80, label=r"# $r_\ast$ in grid")
    n_steps_5 = mo.ui.slider(5_000, 80_000, value=20_000, step=5_000, label="n_steps")
    mo.hstack([n_rstars_5, n_steps_5])
    return n_rstars_5, n_steps_5


@app.cell
def _(integrate_geodesic, n_rstars_5, n_steps_5, np):
    # All v0 values used across the notebook (superset of Figure 5's subset)
    V0_LIST = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

    # Profile r_stars are pinned so downstream dropdowns always have exact matches
    _profile_rstars = [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]

    R_STARS = np.unique(
        np.concatenate([np.linspace(0.0001, 20, n_rstars_5.value), _profile_rstars])
    )
    DT = 0.002

    trajs = {}
    for _v0 in V0_LIST:
        for _r in R_STARS:
            trajs[(_v0, _r)] = integrate_geodesic(_r, _v0, n_steps=n_steps_5.value, dt=DT)
    return DT, R_STARS, V0_LIST, trajs


@app.cell
def _(Normalize, R_STARS, ScalarMappable, get_mass_and_dmdv, np, plt, trajs):
    _fig5_v0 = [-2, -1, 0, 0.5, 1]
    _norm = Normalize(vmin=float(R_STARS[0]), vmax=float(R_STARS[-1]))
    _cmap = plt.cm.cool
    _sm = ScalarMappable(norm=_norm, cmap=_cmap)
    _sm.set_array([])

    _fig5, _axes5 = plt.subplots(
        2, 3, figsize=(14, 8), subplot_kw={"projection": "polar"}
    )
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)

    for _i, _v0 in enumerate(_fig5_v0):
        _v0k = float(_v0)
        _ax = _axes5.flatten()[_i]

        _ax.plot(
            np.linspace(-np.pi / 2, np.pi / 2, 100),
            [np.pi / 2] * 100, "k-", linewidth=2,
        )

        _m, _ = get_mass_and_dmdv(_v0k)
        _m = float(_m)
        if _m > 0:
            _ax.plot(
                np.linspace(0, 2 * np.pi, 200),
                [np.arctan(np.sqrt(_m))] * 200,
                color="brown", linewidth=2,
            )

        for _r in R_STARS:
            if _m > 0 and _r <= np.sqrt(_m):
                continue
            _traj = trajs[(_v0k, _r)]
            _R = np.arctan(np.array(_traj[:, 1]))
            _x = np.array(_traj[:, 2])
            _c = _cmap(_norm(_r))
            _ax.plot(_x, _R, color=_c, linewidth=0.8)
            _ax.plot(-_x, _R, color=_c, linewidth=0.8)

        _ax.set_title(r"$v_0 = $" + f"{_v0}", fontsize=11)
        _ax.set_ylim(0, np.pi / 2 + 0.1)
        _ax.set_yticks([])
        _ax.set_xticks([])
        _ax.spines["polar"].set_visible(False)

    _cbar_ax = _fig5.add_axes([0.88, 0.15, 0.02, 0.7])
    _cbar = _fig5.colorbar(_sm, cax=_cbar_ax)
    _cbar.set_label(r"Turning point $r_\ast$", fontsize=11)
    _fig5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plots of geodesic lengths
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Geodesic Lenghts Plots
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Geodesic lengths vs $r_\ast$
    """)
    return


@app.cell
def _(V0_LIST, mo):
    v0_sweep = mo.ui.dropdown(
        {str(v): v for v in V0_LIST}, value="0.0", label="$v_0$"
    )
    v0_sweep
    return (v0_sweep,)


@app.cell
def _(DT, R_STARS, geodesic_length_from_traj, np, trajs, v0_sweep):
    _r_cut = 200.0
    _v0 = v0_sweep.value
    _Ls, _Lregs, _hs, _vins = [], [], [], []

    for _r in R_STARS:
        _traj = trajs[(_v0, _r)]
        _L = float(geodesic_length_from_traj(_traj, dt=DT, r_cut=_r_cut))
        _Lregs.append(_L - 2 * np.log(2 * _r_cut))
        _Ls.append(_L)
        _rarr = np.array(_traj[:, 1])
        _hit = int(np.argmax(_rarr >= _r_cut))
        if not np.any(_rarr >= _r_cut):
            _hit = _traj.shape[0] - 1
        _hs.append(float(_traj[_hit, 2]))
        _vins.append(float(_traj[_hit, 0]))

    Ls_sw = np.array(_Ls)
    Lregs_sw = np.array(_Lregs)
    hs_sw = np.array(_hs)
    vins_sw = np.array(_vins)
    return Lregs_sw, Ls_sw, hs_sw, vins_sw


@app.cell
def _(Lregs_sw, Ls_sw, R_STARS, plt, v0_sweep):
    _fig1, _ax1 = plt.subplots(figsize=(7, 4.5))
    _ax1.plot(R_STARS, Ls_sw, label=r"$L$")
    _ax1.plot(R_STARS, Lregs_sw, label=r"$L_{\rm reg}$")
    _ax1.set_xscale("log")
    _ax1.set_xlabel(r"$r_\ast$")
    _ax1.set_ylabel("Length")
    _ax1.set_title(rf"Geodesic lengths  ($v_0 = {v0_sweep.value:g}$)")
    _ax1.legend()
    _fig1.set_constrained_layout(True)
    _fig1
    return


@app.cell
def _(R_STARS, hs_sw, plt, v0_sweep, vins_sw):
    _fig2, (_ax2a, _ax2b) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    _ax2a.plot(R_STARS, hs_sw)
    _ax2a.set_xscale("log")
    _ax2a.set_ylabel(r"$h = x(r_{\rm cut})$")
    _ax2b.plot(R_STARS, vins_sw)
    _ax2b.set_xscale("log")
    _ax2b.set_xlabel(r"$r_\ast$")
    _ax2b.set_ylabel(r"$v_\infty = v(r_{\rm cut})$")
    _fig2.suptitle(rf"Cutoff diagnostics  ($v_0 = {v0_sweep.value:g}$)")
    _fig2.set_constrained_layout(True)
    _fig2
    return


@app.cell
def _(Lregs_sw, hs_sw, np, plt, v0_sweep, vins_sw):
    _order = np.argsort(hs_sw)
    _h_s = hs_sw[_order]
    _Lr_s = Lregs_sw[_order]
    _vi_s = vins_sw[_order]

    _fig3, (_ax3a, _ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
    _ax3a.plot(_h_s, _Lr_s)
    _ax3a.set_xlabel(r"$h$")
    _ax3a.set_ylabel(r"$L_{\rm reg}$")
    _ax3a.set_title(rf"$L_{{\rm reg}}(h)$  ($v_0 = {v0_sweep.value:g}$)")
    _ax3b.plot(_h_s, _vi_s)
    _ax3b.set_xlabel(r"$h$")
    _ax3b.set_ylabel(r"$v_\infty$")
    _ax3b.set_title(rf"$v_\infty(h)$  ($v_0 = {v0_sweep.value:g}$)")
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
def _(DT, R_STARS, V0_LIST, geodesic_length_from_traj, np, plt, trajs):
    _r_cut = 200.0
    _colors_all = plt.cm.viridis(np.linspace(0, 1, len(V0_LIST)))
    _fig_all, (_axa1, _axa2) = plt.subplots(1, 2, figsize=(14, 5))

    for _v0a, _col in zip(V0_LIST, _colors_all):
        _ls = np.array([
            float(geodesic_length_from_traj(trajs[(_v0a, _r)], dt=DT, r_cut=_r_cut))
            for _r in R_STARS
        ])
        for _axx in [_axa1, _axa2]:
            _axx.plot(
                R_STARS, _ls,
                marker="o", markersize=2, linewidth=1.5,
                color=_col, label=rf"$v_0={_v0a:g}$",
            )

    _axa1.set_ylim(0, 18)
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
    mo.md(r"""
    ## Length profile $L_{\mathrm{half}}(x)$
    """)
    return


@app.cell
def _(V0_LIST, mo):
    _v0_opts = {str(v): v for v in V0_LIST}
    _rs_opts = {str(r): r for r in [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]}

    profile_mode = mo.ui.radio(
        options=["Fix $v_0$ (vary $r_\\ast$)", r"Fix $r_\ast$ (vary $v_0$)"],
        value="Fix $v_0$ (vary $r_\\ast$)",
        label="Plot mode",
    )
    v0_profile = mo.ui.dropdown(_v0_opts, value="0.0", label="$v_0$")
    rstar_profile = mo.ui.dropdown(_rs_opts, value="1.0", label=r"$r_\ast$")
    mo.hstack([profile_mode, v0_profile, rstar_profile])
    return profile_mode, rstar_profile, v0_profile


@app.cell
def _(
    DT,
    V0_LIST,
    length_profile_vs_x,
    np,
    plt,
    profile_mode,
    rstar_profile,
    trajs,
    v0_profile,
):
    _rs_list_p = [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]
    _fix_v0 = "v_0" in profile_mode.value

    _fig_p, (_axp1, _axp2) = plt.subplots(1, 2, figsize=(14, 5))

    if _fix_v0:
        _v0p = v0_profile.value
        _cols = plt.cm.viridis(np.linspace(0, 1, len(_rs_list_p)))
        for _rsp, _c in zip(_rs_list_p, _cols):
            _xp, _Lxp = length_profile_vs_x(trajs[(_v0p, _rsp)], dt=DT, r_cut=200.0)
            _axp1.plot(_xp, _Lxp, color=_c, label=rf"$r_\ast={_rsp}$")
            _axp2.plot(np.arctanh(_xp), _Lxp, color=_c, label=rf"$r_\ast={_rsp}$")
        _title_sfx = rf"$v_0={_v0p:g}$"
    else:
        _rsp = rstar_profile.value
        _cols = plt.cm.viridis(np.linspace(0, 1, len(V0_LIST)))
        for _v0p, _c in zip(V0_LIST, _cols):
            _xp, _Lxp = length_profile_vs_x(trajs[(_v0p, _rsp)], dt=DT, r_cut=200.0)
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
def _(V0_LIST, mo):
    v0_reg = mo.ui.dropdown({str(v): v for v in V0_LIST}, value="0.0", label="$v_0$")
    v0_reg
    return (v0_reg,)


@app.cell
def _(DT, length_profile_vs_x, np, plt, trajs, v0_reg):
    _rs_list_r = [0.01, 0.1, 1.0, 1.5, 2.0, 5.0, 10.0]
    _r_cut_r = 200.0
    _v0r = v0_reg.value
    _cols_r = plt.cm.viridis(np.linspace(0, 1, len(_rs_list_r)))

    _fig_r, (_axr1, _axr2) = plt.subplots(1, 2, figsize=(14, 5))

    for _rsr, _cr in zip(_rs_list_r, _cols_r):
        _xh, _Lxh = length_profile_vs_x(trajs[(_v0r, _rsr)], dt=DT, r_cut=_r_cut_r)
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
def _(mo):
    mo.md(r"""
    ## 3D Geodesics

    Compactified coordinates: $X = \arctan(r)\cos(x)$, $Y = \arctan(r)\sin(x)$, $Z = v$.
    Drag to rotate, scroll to zoom.
    """)
    return


@app.cell
def _(V0_LIST, mo):
    v0_3d = mo.ui.dropdown(
        {str(v): v for v in V0_LIST}, value="0.0", label="$v_0$"
    )
    r_cut_3d = mo.ui.slider(1.0, 50.0, value=15.0, step=1.0, label="display $r_{\\rm cut}$")
    mo.hstack([v0_3d, r_cut_3d])
    return r_cut_3d, v0_3d


@app.cell
def _(R_STARS, get_mass_and_dmdv, np, r_cut_3d, trajs, v0_3d):
    import plotly.graph_objects as go
    import plotly.colors as pc

    _v0 = v0_3d.value
    _r_cut_disp = float(r_cut_3d.value)
    _m, _ = get_mass_and_dmdv(float(_v0))
    _m = float(_m)

    _candidate_rs = [_r for _r in R_STARS if not (_m > 0 and _r <= np.sqrt(_m))]

    # ── Pass 1: truncate + downsample, then drop numerically exploded trajectories
    def _get_seg(r):
        traj = np.array(trajs[(_v0, r)])
        rarr = traj[:, 1]
        hit = int(np.argmax(rarr >= _r_cut_disp))
        if not np.any(rarr >= _r_cut_disp):
            hit = len(rarr) - 1
        seg = traj[: hit + 1]
        step = max(1, len(seg) // 300)
        return seg[::step]

    _valid_rs, _segs = [], []
    for _r in _candidate_rs:
        _seg = _get_seg(_r)
        _v_seg = _seg[:, 0]
        if np.all(np.isfinite(_v_seg)) and np.max(np.abs(_v_seg)) < 1e6:
            _valid_rs.append(_r)
            _segs.append(_seg)

    _n = len(_valid_rs)
    _palette = pc.sample_colorscale("Turbo", [i / max(_n - 1, 1) for i in range(_n)])

    _v_all = np.concatenate([s[:, 0] for s in _segs]) if _segs else np.array([0.0])
    _v_lo, _v_hi = np.percentile(_v_all, [5, 99])
    _v_pad = 0.05 * max(_v_hi - _v_lo, 1e-3)

    # ── Pass 2: build traces, masking mild remaining outliers with NaN ────────
    _theta_b = np.linspace(0, 2 * np.pi, 120)
    _data = []

    for _i, (_r, _seg) in enumerate(zip(_valid_rs, _segs)):
        _v_t = _seg[:, 0]
        _r_t = _seg[:, 1]
        _x_t = _seg[:, 2]
        _R_comp = np.arctan(_r_t)

        _mask = (_v_t < _v_lo) | (_v_t > _v_hi)
        _v_plot = np.where(_mask, np.nan, _v_t)
        _X_plot = np.where(_mask, np.nan, _R_comp * np.cos(_x_t))
        _Y_plot = np.where(_mask, np.nan, _R_comp * np.sin(_x_t))
        _Xn_plot = np.where(_mask, np.nan, _R_comp * np.cos(-_x_t))
        _Yn_plot = np.where(_mask, np.nan, _R_comp * np.sin(-_x_t))

        _kw = dict(
            mode="lines",
            showlegend=False,
            line=dict(color=_palette[_i], width=2),
            hovertemplate=f"r★={_r:.3f}<br>v=%{{z:.3f}}<extra></extra>",
        )
        _data.append(go.Scatter3d(x=_X_plot.tolist(), y=_Y_plot.tolist(), z=_v_plot.tolist(), **_kw))
        _data.append(go.Scatter3d(x=_Xn_plot.tolist(), y=_Yn_plot.tolist(), z=_v_plot.tolist(), **_kw))

    # Boundary circles at evenly-spaced v levels within the clipped range
    _X_b = (np.pi / 2) * np.cos(_theta_b)
    _Y_b = (np.pi / 2) * np.sin(_theta_b)
    for _vb in np.linspace(_v_lo, _v_hi, 5):
        _data.append(go.Scatter3d(
            x=_X_b.tolist(), y=_Y_b.tolist(), z=[float(_vb)] * 120,
            mode="lines", showlegend=False,
            line=dict(color="black", width=1, dash="dot"),
            hoverinfo="skip",
        ))

    # Apparent horizon surface: R_hor(v) = arctan(sqrt(m(v))) for m(v) > 0
    _v_hor = np.linspace(_v_lo, _v_hi, 120)
    _m_hor = np.array([float(get_mass_and_dmdv(float(_vv))[0]) for _vv in _v_hor])
    _hor_mask = _m_hor > 0
    if np.any(_hor_mask):
        _v_valid = _v_hor[_hor_mask]
        _R_valid = np.arctan(np.sqrt(_m_hor[_hor_mask]))
        _theta_hor = np.linspace(0, 2 * np.pi, 60)
        _V_surf  = np.outer(np.ones(60), _v_valid)
        _R_surf  = np.outer(np.ones(60), _R_valid)
        _Th_surf = np.outer(_theta_hor, np.ones(len(_v_valid)))
        _data.append(go.Surface(
            x=_R_surf * np.cos(_Th_surf),
            y=_R_surf * np.sin(_Th_surf),
            z=_V_surf,
            colorscale=[[0, "saddlebrown"], [1, "saddlebrown"]],
            showscale=False,
            opacity=0.35,
            hoverinfo="skip",
            name="Apparent horizon",
        ))

    # Colorbar dummy trace
    _data.append(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers",
        marker=dict(
            colorscale="Turbo",
            cmin=float(R_STARS[0]), cmax=float(R_STARS[-1]),
            color=[float(R_STARS[0])],
            colorbar=dict(title=dict(text="r★", side="right"), thickness=15, len=0.6, x=1.02),
            size=0,
        ),
        showlegend=False,
    ))

    _fig3d = go.Figure(data=_data)
    _fig3d.update_layout(
        template="simple_white",
        scene=dict(
            xaxis_title="arctan(r)·cos(x)",
            yaxis_title="arctan(r)·sin(x)",
            zaxis=dict(title="v", range=[_v_lo - _v_pad, _v_hi + _v_pad]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.8),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        margin=dict(l=0, r=80, t=40, b=0),
        height=720,
        title=dict(
            text=f"Compactified geodesics  (v₀ = {_v0:g},  display r_cut = {_r_cut_disp:g})",
            x=0.5,
        ),
    )
    _fig3d
    return


if __name__ == "__main__":
    app.run()

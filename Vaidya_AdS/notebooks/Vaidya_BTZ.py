import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import sys, os
    from pathlib import Path
    import numpy as np
    import marimo as mo

    return Path, mo, np, os, sys


@app.cell
def _(Path, os, sys):
    _nb_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_nb_dir.parent / "src"))

    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import plotly.graph_objects as go

    from Vaidya_AdS import (
        get_mass_and_dmdv,
        integrate_geodesic,
        lengths_vs_rstar,
    )
    from BTZ import f_true, h_true, l_func, l_integral_NN, S_integral_NN
    from Empty_AdS import empty_ads_geodesic_exact

    return (
        S_integral_NN,
        empty_ads_geodesic_exact,
        f_true,
        get_mass_and_dmdv,
        go,
        h_true,
        integrate_geodesic,
        jnp,
        l_func,
        l_integral_NN,
        lengths_vs_rstar,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Vaidya-AdS vs BTZ vs Empty AdS: Geodesic Comparison

    All three solvers compute the same **half-geodesic** — from the turning point
    $r_\star$ outward to the AdS boundary — in different ways:

    | | **Vaidya-AdS** | **Empty AdS** | **BTZ analytic** |
    |---|---|---|---|
    | Coordinates | $(v, r, x)$ | $(v, r, x)$ | $z = 1/r$ |
    | Method | RK4 ODE integration | Closed form | Gauss-Legendre quadrature |
    | Mass profile | $m(v)$ dynamic | $m=0$ (exact) | $f(z)=1-z^2$ (static, $m=1$) |
    | Limit | full Vaidya collapse | early time $v_0\to-\infty$ | late time $v_0\to+\infty$ |

    At late times $v_0\to+\infty$, the Vaidya result must converge to BTZ.
    At early times $v_0\to-\infty$, it must converge to the empty-AdS exact curve.
    """)
    return


@app.cell
def _(mo):
    n_rstars_slider = mo.ui.slider(5, 40, value=15, step=1, label="# r★ grid points")
    n_steps_slider = mo.ui.slider(10_000, 80_000, value=40_000, step=5_000, label="Vaidya steps")
    r_cut_inp = mo.ui.number(value=200.0, label="r_cut (UV cutoff)")
    dt_inp = mo.ui.number(value=0.002, label="Δλ (affine step)")
    mo.hstack([n_rstars_slider, n_steps_slider, r_cut_inp, dt_inp])
    return dt_inp, n_rstars_slider, n_steps_slider, r_cut_inp


@app.cell
def _(mo):
    r_min_inp = mo.ui.number(value=1.05, label="r★ min  (> 1 = outside BTZ horizon)")
    r_max_inp = mo.ui.number(value=8.0, label="r★ max")
    mo.hstack([r_min_inp, r_max_inp])
    return r_max_inp, r_min_inp


@app.cell
def _(mo):
    V0_ALL = [-5.0, -3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    v0_checks = mo.ui.array(
        [
            mo.ui.checkbox(label=f"v₀ = {v:+.1f}", value=(v in [-3.0, 0.0, 1.0, 3.0, 5.0]))
            for v in V0_ALL
        ]
    )
    mo.vstack([mo.md("**Select initial times v₀ for Vaidya integration:**"), v0_checks])
    return V0_ALL, v0_checks


@app.cell
def _(V0_ALL, v0_checks):
    V0_SELECTED = [v for v, checked in zip(V0_ALL, v0_checks.value) if checked]
    return (V0_SELECTED,)


@app.cell
def _(
    S_integral_NN,
    V0_SELECTED,
    dt_inp,
    f_true,
    h_true,
    jnp,
    l_func,
    l_integral_NN,
    lengths_vs_rstar,
    n_rstars_slider,
    n_steps_slider,
    np,
    r_cut_inp,
    r_max_inp,
    r_min_inp,
):
    # ── r★ grid ──────────────────────────────────────────────────────────────
    R_STARS = np.linspace(r_min_inp.value, r_max_inp.value, n_rstars_slider.value)
    Z_STARS = 1.0 / R_STARS  # z_star = 1/r_star (BTZ z-coordinate)

    # ── BTZ analytic l(r★) ───────────────────────────────────────────────────
    l_BTZ_analytic = np.array(l_func(jnp.array(Z_STARS)))

    # Numerical check: same integral via GL quadrature (should match analytic)
    class _BTZTrue:
        """Thin wrapper so S/l_integral_NN can call f_true, h_true."""
        def __call__(self, z):
            return f_true(z), h_true(z)

    _analytic_model = _BTZTrue()
    l_BTZ_integral = np.array(l_integral_NN(_analytic_model, jnp.array(Z_STARS), N_GL=40))

    # ── BTZ analytic regulated half-length S_finite ──────────────────────────
    # S_integral_NN returns S_finite = L_reg_half.
    # Full regulated length: L_reg_BTZ = 2 * S_finite.
    S_finite_BTZ = np.array(S_integral_NN(_analytic_model, jnp.array(Z_STARS), N_GL=40))
    L_reg_BTZ = 2.0 * S_finite_BTZ

    # ── Vaidya numeric geodesics for each v₀ ─────────────────────────────────
    vaidya = {}
    for _v0 in V0_SELECTED:
        _Ls, _Lregs, _hs, _vins = lengths_vs_rstar(
            R_STARS, _v0,
            n_steps=int(n_steps_slider.value),
            dt=float(dt_inp.value),
            r_cut=float(r_cut_inp.value),
        )
        vaidya[_v0] = {
            "l":     2.0 * np.array(_hs),
            "L_reg": np.array(_Lregs),
        }
    return L_reg_BTZ, R_STARS, l_BTZ_analytic, l_BTZ_integral, vaidya


@app.cell
def _(R_STARS, empty_ads_geodesic_exact, np, r_cut_inp):
    _r_cut = float(r_cut_inp.value)
    l_empty = np.array([
        2.0 * empty_ads_geodesic_exact(_rs, r_cut=_r_cut)[2][-1]
        for _rs in R_STARS
    ])
    L_reg_empty = 2.0 * np.arccosh(_r_cut / R_STARS) - 2.0 * np.log(2.0 * _r_cut)
    return L_reg_empty, l_empty


@app.cell
def _(mo):
    mo.md(r"""
    ## Direct solver validation: Vaidya RK4 at $m=0$ vs exact empty AdS

    Set $m_f = 0$ so the Vaidya mass profile is identically zero everywhere.
    The RK4 integrator then solves the geodesic equations in pure Poincaré AdS$_3$,
    and the result must match the closed-form solution from `Empty_AdS.py` exactly
    up to numerical integration error.  Any residual here is pure RK4 truncation error
    — it is independent of the mass profile and sets a floor on forward-model accuracy.
    """)
    return


@app.cell
def _(
    L_reg_empty,
    R_STARS,
    dt_inp,
    l_empty,
    lengths_vs_rstar,
    n_steps_slider,
    np,
    r_cut_inp,
):
    _r_cut = float(r_cut_inp.value)
    _Ls_m0, _Lregs_m0, _hs_m0, _vins_m0 = lengths_vs_rstar(
        R_STARS, 0.0,
        n_steps=int(n_steps_slider.value),
        dt=float(dt_inp.value),
        r_cut=_r_cut,
        m_i=0.0, m_f=0.0,
    )
    l_vaidya_m0 = 2.0 * np.array(_hs_m0)
    L_reg_vaidya_m0 = np.array(_Lregs_m0)
    dl_m0 = np.abs(l_vaidya_m0 - l_empty)
    dL_m0 = np.abs(L_reg_vaidya_m0 - L_reg_empty)
    return L_reg_vaidya_m0, dL_m0, dl_m0, l_vaidya_m0


@app.cell
def _(
    L_reg_empty,
    L_reg_vaidya_m0,
    R_STARS,
    dL_m0,
    dl_m0,
    go,
    l_empty,
    l_vaidya_m0,
    mo,
):
    _fig_m0 = go.Figure()
    _fig_m0.add_trace(go.Scatter(
        x=R_STARS, y=l_empty,
        mode="lines", line=dict(color="teal", dash="dot", width=2.5),
        name="Exact empty AdS",
    ))
    _fig_m0.add_trace(go.Scatter(
        x=R_STARS, y=l_vaidya_m0,
        mode="markers", marker=dict(color="darkorange", size=7, symbol="circle-open"),
        name="Vaidya RK4  (m=0)",
    ))
    _fig_m0.add_trace(go.Scatter(
        x=R_STARS, y=dl_m0,
        mode="lines+markers", marker=dict(size=4),
        line=dict(color="gray", dash="dot"),
        name="|Δℓ|",
        yaxis="y2",
    ))
    _fig_m0.update_layout(
        title="ℓ(r★): Vaidya RK4 (m=0) vs exact empty AdS",
        xaxis_title="r★",
        yaxis_title="ℓ",
        yaxis2=dict(title="|Δℓ|", overlaying="y", side="right", showgrid=False, type="log"),
        legend_title="",
        width=750, height=420,
    )

    _fig_Lm0 = go.Figure()
    _fig_Lm0.add_trace(go.Scatter(
        x=R_STARS, y=L_reg_empty,
        mode="lines", line=dict(color="teal", dash="dot", width=2.5),
        name="Exact empty AdS",
    ))
    _fig_Lm0.add_trace(go.Scatter(
        x=R_STARS, y=L_reg_vaidya_m0,
        mode="markers", marker=dict(color="darkorange", size=7, symbol="circle-open"),
        name="Vaidya RK4  (m=0)",
    ))
    _fig_Lm0.add_trace(go.Scatter(
        x=R_STARS, y=dL_m0,
        mode="lines+markers", marker=dict(size=4),
        line=dict(color="gray", dash="dot"),
        name="|ΔL_reg|",
        yaxis="y2",
    ))
    _fig_Lm0.update_layout(
        title="L_reg(r★): Vaidya RK4 (m=0) vs exact empty AdS",
        xaxis_title="r★",
        yaxis_title="L_reg",
        yaxis2=dict(title="|ΔL_reg|", overlaying="y", side="right", showgrid=False, type="log"),
        legend_title="",
        width=750, height=420,
    )

    mo.vstack([mo.ui.plotly(_fig_m0), mo.ui.plotly(_fig_Lm0)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Internal check: BTZ analytic formula vs GL quadrature

    Both compute the same quantity $\ell = \int_0^{z_\star} \frac{d\alpha}{\sqrt{f(\alpha)(z_\star^2/\alpha^2 - 1)}}$.
    The GL quadrature has a mild $\sqrt{z_\star - \alpha}$ singularity at the upper endpoint,
    causing a few-percent error at moderate $N_\text{GL}$.  Increase `N_GL` in the code to
    reduce it; the analytic formula is used for all main comparisons.
    """)
    return


@app.cell
def _(R_STARS, go, l_BTZ_analytic, l_BTZ_integral, mo, np):
    _fig_check = go.Figure()
    _fig_check.add_trace(go.Scatter(
        x=R_STARS, y=l_BTZ_analytic,
        mode="lines", line=dict(color="black", width=2),
        name="Analytic  2·arctanh(1/r★)",
    ))
    _fig_check.add_trace(go.Scatter(
        x=R_STARS, y=l_BTZ_integral,
        mode="markers", marker=dict(color="crimson", size=7, symbol="circle-open"),
        name="GL quadrature (l_integral_NN)",
    ))
    _fig_check.add_trace(go.Scatter(
        x=R_STARS, y=np.abs(l_BTZ_analytic - l_BTZ_integral),
        mode="lines", line=dict(color="gray", dash="dot"),
        name="|residual|",
        yaxis="y2",
    ))
    _fig_check.update_layout(
        title="BTZ: analytic ℓ(r★) = 2 arctanh(1/r★) vs numerical integral",
        xaxis_title="r★",
        yaxis_title="ℓ",
        yaxis2=dict(title="|residual|", overlaying="y", side="right", showgrid=False),
        legend_title="",
        width=750, height=400,
    )
    mo.ui.plotly(_fig_check)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ℓ(r★): boundary separation

    All methods compute $\ell = 2x(r_\text{cut})$ (Vaidya / empty AdS) or $\ell = 2\operatorname{arctanh}(z_\star)$
    (BTZ analytic).
    Vaidya curves at large $v_0$ must converge to BTZ; at early $v_0$ they must converge to empty AdS.
    """)
    return


@app.cell
def _(R_STARS, go, l_BTZ_analytic, l_empty, mo):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=R_STARS, y=l_BTZ_analytic,
        mode="lines", line=dict(color="black", dash="dash", width=2.5),
        name="BTZ analytic",
    ))
    _fig.add_trace(go.Scatter(
        x=R_STARS, y=l_empty,
        mode="lines", line=dict(color="teal", dash="dot", width=2),
        name="Empty AdS₃  (m=0 exact)",
    ))
    mo.ui.plotly(_fig)
    return


@app.cell
def _(R_STARS, V0_SELECTED, go, l_BTZ_analytic, l_empty, mo, np, vaidya):
    _colors = [f"hsl({int(h * 280)},80%,45%)" for h in np.linspace(0, 1, max(len(V0_SELECTED), 1))]
    _fig_l = go.Figure()

    for _i, _v0 in enumerate(V0_SELECTED):
        _fig_l.add_trace(go.Scatter(
            x=R_STARS, y=vaidya[_v0]["l"],
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=_colors[_i]),
            name=f"Vaidya  v₀={_v0:+.1f}",
        ))

    _fig_l.add_trace(go.Scatter(
        x=R_STARS, y=l_empty,
        mode="lines", line=dict(color="teal", dash="dot", width=5.5),
        name="Empty AdS₃  (m=0 exact)",
    ))
    _fig_l.add_trace(go.Scatter(
        x=R_STARS, y=l_BTZ_analytic,
        mode="lines", line=dict(color="black", dash="dash", width=5.5),
        name="BTZ analytic",
    ))

    _fig_l.update_layout(
        title="Boundary separation ℓ(r★)  —  Vaidya vs BTZ and empty AdS",
        xaxis_title="r★ (turning-point radius)",
        yaxis_title="ℓ",
        legend_title="",
        width=800, height=450,
    )
    mo.ui.plotly(_fig_l)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## L_reg(r★): regularised geodesic length

    **Convention note.** The codes use slightly different UV subtraction schemes:

    - **Vaidya / Empty AdS**: $L_\text{reg} = L - 2\log(2 r_\text{cut})$
    - **BTZ analytic**: $L_\text{reg} = 2 \times S_\text{finite}$, where $S_\text{finite}$
      is computed by subtracting 1 inside the integrand before integrating to $z=0$.

    These differ by the constant $2\log 2$; all curves should give parallel results that
    converge at large and small $v_0$ respectively.
    """)
    return


@app.cell
def _(L_reg_BTZ, L_reg_empty, R_STARS, V0_SELECTED, go, mo, np, vaidya):
    _colors = [f"hsl({int(h * 280)},80%,45%)" for h in np.linspace(0, 1, max(len(V0_SELECTED), 1))]
    _fig_L = go.Figure()

    for _i, _v0 in enumerate(V0_SELECTED):
        _fig_L.add_trace(go.Scatter(
            x=R_STARS, y=vaidya[_v0]["L_reg"],
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=_colors[_i]),
            name=f"Vaidya  v₀={_v0:+.1f}",
        ))

    _fig_L.add_trace(go.Scatter(
        x=R_STARS, y=L_reg_empty,
        mode="lines", line=dict(color="teal", dash="dot", width=2),
        name="Empty AdS₃  (m=0 exact)",
    ))
    _fig_L.add_trace(go.Scatter(
        x=R_STARS, y=L_reg_BTZ,
        mode="lines", line=dict(color="black", dash="dash", width=2.5),
        name="BTZ analytic  2·S_finite",
    ))

    _fig_L.update_layout(
        title="Regularised geodesic length L_reg(r★)  —  Vaidya vs BTZ and empty AdS",
        xaxis_title="r★",
        yaxis_title="L_reg",
        legend_title="",
        width=800, height=450,
    )
    mo.ui.plotly(_fig_L)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## L_reg(ℓ): length as a function of boundary separation

    Parametric plot: $r_\star$ is eliminated and we plot $L_\text{reg}$ against $\ell$.
    The BTZ result is $L_\text{reg} \propto \log\sinh(\ell/2)$ (up to convention constants).
    For empty AdS, $\ell \approx 2/r_\star$ and $L_\text{reg} \approx -2\log r_\star \approx 2\log(\ell/2)$.
    """)
    return


@app.cell
def _(
    L_reg_BTZ,
    L_reg_empty,
    V0_SELECTED,
    go,
    l_BTZ_analytic,
    l_empty,
    mo,
    np,
    vaidya,
):
    _colors = [f"hsl({int(h * 280)},80%,45%)" for h in np.linspace(0, 1, max(len(V0_SELECTED), 1))]
    _fig_Ll = go.Figure()

    for _i, _v0 in enumerate(V0_SELECTED):
        _l = vaidya[_v0]["l"]
        _L = vaidya[_v0]["L_reg"]
        _order = np.argsort(_l)
        _fig_Ll.add_trace(go.Scatter(
            x=_l[_order], y=_L[_order],
            mode="lines+markers", marker=dict(size=5),
            line=dict(color=_colors[_i]),
            name=f"Vaidya  v₀={_v0:+.1f}",
        ))

    _order_empty = np.argsort(l_empty)
    _fig_Ll.add_trace(go.Scatter(
        x=l_empty[_order_empty], y=L_reg_empty[_order_empty],
        mode="lines", line=dict(color="teal", dash="dot", width=2),
        name="Empty AdS₃  (m=0 exact)",
    ))
    _order_btz = np.argsort(l_BTZ_analytic)
    _fig_Ll.add_trace(go.Scatter(
        x=l_BTZ_analytic[_order_btz], y=L_reg_BTZ[_order_btz],
        mode="lines", line=dict(color="black", dash="dash", width=2.5),
        name="BTZ analytic",
    ))

    _fig_Ll.update_layout(
        title="L_reg(ℓ)  —  Vaidya vs BTZ and empty AdS",
        xaxis_title="ℓ (boundary separation)",
        yaxis_title="L_reg",
        legend_title="",
        width=800, height=450,
    )
    mo.ui.plotly(_fig_Ll)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Convergence check: $|\ell_\text{Vaidya} - \ell_\text{BTZ}|$ vs $v_0$

    For each $r_\star$ point, the residual between the Vaidya $\ell$ and the BTZ analytic
    $\ell$ is shown as a function of $v_0$.  All curves should decay to zero as $v_0 \to +\infty$.
    """)
    return


@app.cell
def _(R_STARS, V0_SELECTED, go, l_BTZ_analytic, mo, np, vaidya):
    _n_show = min(5, len(R_STARS))
    _r_idx = np.round(np.linspace(0, len(R_STARS) - 1, _n_show)).astype(int)

    _fig_conv = go.Figure()
    _cmap = [f"hsl({int(h * 220)},75%,45%)" for h in np.linspace(0, 1, _n_show)]

    for _j, _ri in enumerate(_r_idx):
        _residuals = [
            abs(vaidya[_v0]["l"][_ri] - l_BTZ_analytic[_ri])
            for _v0 in V0_SELECTED
        ]
        _fig_conv.add_trace(go.Scatter(
            x=list(V0_SELECTED),
            y=_residuals,
            mode="lines+markers",
            line=dict(color=_cmap[_j]),
            name=f"r★ = {R_STARS[_ri]:.2f}",
        ))

    _fig_conv.update_layout(
        title="|ℓ_Vaidya − ℓ_BTZ| vs v₀  (convergence to BTZ limit)",
        xaxis_title="v₀",
        yaxis_title="|Δℓ|",
        yaxis_type="log",
        legend_title="",
        width=800, height=400,
    )
    mo.ui.plotly(_fig_conv)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Convergence check: $|\ell_\text{Vaidya} - \ell_\text{empty}|$ vs $v_0$

    Symmetric check for the early-time limit: as $v_0\to-\infty$ the Vaidya geodesic
    travels entirely through the $m\approx 0$ region and must converge to the
    empty-AdS exact curve.  All curves should decay to zero for the most negative $v_0$.
    """)
    return


@app.cell
def _(R_STARS, V0_SELECTED, go, l_empty, mo, np, vaidya):
    _n_show = min(5, len(R_STARS))
    _r_idx = np.round(np.linspace(0, len(R_STARS) - 1, _n_show)).astype(int)

    _fig_conv_e = go.Figure()
    _cmap_e = [f"hsl({int(h * 220)},75%,45%)" for h in np.linspace(0, 1, _n_show)]

    for _j, _ri in enumerate(_r_idx):
        _residuals_e = [
            abs(vaidya[_v0]["l"][_ri] - l_empty[_ri])
            for _v0 in V0_SELECTED
        ]
        _fig_conv_e.add_trace(go.Scatter(
            x=list(V0_SELECTED),
            y=_residuals_e,
            mode="lines+markers",
            line=dict(color=_cmap_e[_j]),
            name=f"r★ = {R_STARS[_ri]:.2f}",
        ))

    _fig_conv_e.update_layout(
        title="|ℓ_Vaidya − ℓ_empty| vs v₀  (convergence to empty AdS limit)",
        xaxis_title="v₀",
        yaxis_title="|Δℓ|",
        yaxis_type="log",
        legend_title="",
        width=800, height=400,
    )
    mo.ui.plotly(_fig_conv_e)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Geodesic shapes in compactified coordinates

    Each geodesic arch is drawn in the compactified AdS plane:
    $$R_\text{comp} = \arctan(r), \quad x \text{ (boundary direction)}$$

    - **Top** ($R_\text{comp} = \pi/2$): AdS boundary
    - **Arch dip**: turning point $r_\star$ at $R_\text{comp} = \arctan(r_\star)$
    - **Shaded region / dotted line**: apparent horizon at each $v_0$
    - **Black dashed lines**: BTZ reference at $v_0 = 10$ ($m \approx 1$)
    - **Teal dotted lines**: empty AdS exact arches ($m = 0$)
    """)
    return


@app.cell
def _(mo):
    n_r_traj_slider = mo.ui.slider(2, 8, value=4, step=1, label="# geodesics (r★ values)")
    n_r_traj_slider
    return (n_r_traj_slider,)


@app.cell
def _(
    V0_SELECTED,
    dt_inp,
    integrate_geodesic,
    n_r_traj_slider,
    n_steps_slider,
    np,
    r_cut_inp,
    r_max_inp,
    r_min_inp,
):
    R_TRAJ = np.linspace(r_min_inp.value, r_max_inp.value, n_r_traj_slider.value)
    _r_cut = float(r_cut_inp.value)
    _n_steps = int(n_steps_slider.value)
    _dt = float(dt_inp.value)

    def _trunc(traj):
        r_col = traj[:, 1]
        hit = int(np.argmax(r_col >= _r_cut))
        if not np.any(r_col >= _r_cut):
            hit = len(r_col) - 1
        return traj[: hit + 1]

    trajs_geo = {}
    for _v0 in V0_SELECTED:
        for _r in R_TRAJ:
            trajs_geo[(_v0, float(_r))] = _trunc(
                np.array(integrate_geodesic(float(_r), _v0, n_steps=_n_steps, dt=_dt))
            )

    # BTZ reference: v0 = 10 so m(10) = tanh(10) ≈ 1
    trajs_btz_geo = {
        float(_r): _trunc(
            np.array(integrate_geodesic(float(_r), 10.0, n_steps=_n_steps, dt=_dt))
        )
        for _r in R_TRAJ
    }
    return R_TRAJ, trajs_btz_geo, trajs_geo


@app.cell
def _(
    R_TRAJ,
    V0_SELECTED,
    empty_ads_geodesic_exact,
    get_mass_and_dmdv,
    go,
    mo,
    np,
    r_cut_inp,
    trajs_btz_geo,
    trajs_geo,
):
    def _ds(arr, n=120):
        if len(arr) <= n:
            return arr
        return arr[np.round(np.linspace(0, len(arr) - 1, n)).astype(int)]

    def _arch(seg):
        x = _ds(seg[:, 2])
        R_c = np.arctan(_ds(seg[:, 1]))
        return np.concatenate([-x[::-1], x[1:]]), np.concatenate([R_c[::-1], R_c[1:]])

    _n_v0 = max(len(V0_SELECTED), 1)
    _colors = [f"hsl({int(h * 280)},80%,45%)" for h in np.linspace(0, 1, _n_v0)]
    _fig2d = go.Figure()

    for _i, _v0 in enumerate(V0_SELECTED):
        _first = True
        for _r in R_TRAJ:
            _seg = trajs_geo[(_v0, float(_r))]
            if len(_seg) < 3:
                continue
            _xa, _Ra = _arch(_seg)
            _fig2d.add_trace(go.Scatter(
                x=_xa, y=_Ra, mode="lines",
                line=dict(color=_colors[_i], width=1.8),
                name=f"Vaidya  v₀={_v0:+.1f}" if _first else None,
                showlegend=_first,
                legendgroup=f"v0_{_v0}",
            ))
            _first = False

        # Apparent horizon: shaded band and dotted line at this v0
        _m_val = float(get_mass_and_dmdv(float(_v0))[0])
        if _m_val > 0:
            _R_hor = float(np.arctan(np.sqrt(_m_val)))
            _x_bdy = float(trajs_geo[(_v0, float(R_TRAJ[0]))][-1, 2]) * 1.15
            _fig2d.add_trace(go.Scatter(
                x=[-_x_bdy, _x_bdy, _x_bdy, -_x_bdy],
                y=[0.0, 0.0, _R_hor, _R_hor],
                fill="toself", fillcolor=_colors[_i], opacity=0.07,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=f"v0_{_v0}",
            ))
            _fig2d.add_trace(go.Scatter(
                x=[-_x_bdy, _x_bdy], y=[_R_hor, _R_hor],
                mode="lines", line=dict(color=_colors[_i], dash="dot", width=1),
                showlegend=False, hoverinfo="skip",
                legendgroup=f"v0_{_v0}",
            ))

    # BTZ reference arches
    _first_btz = True
    for _r in R_TRAJ:
        _seg = trajs_btz_geo[float(_r)]
        if len(_seg) < 3:
            continue
        _xa, _Ra = _arch(_seg)
        _fig2d.add_trace(go.Scatter(
            x=_xa, y=_Ra, mode="lines",
            line=dict(color="black", dash="dash", width=5),
            name="BTZ  (v₀=10)" if _first_btz else None,
            showlegend=_first_btz,
            legendgroup="btz",
        ))
        _first_btz = False

    # Empty AdS exact arches
    _r_cut_2d = float(r_cut_inp.value)
    _first_empty = True
    for _r in R_TRAJ:
        _, _r_arr, _x_arr = empty_ads_geodesic_exact(float(_r), t_boundary=0.0,
                                                      r_cut=_r_cut_2d, n_points=200)
        _R_c = np.arctan(_r_arr)
        _xa_e = np.concatenate([-_x_arr[::-1], _x_arr[1:]])
        _Ra_e = np.concatenate([_R_c[::-1], _R_c[1:]])
        _fig2d.add_trace(go.Scatter(
            x=_xa_e, y=_Ra_e, mode="lines",
            line=dict(color="teal", dash="dot", width=5),
            name="Empty AdS₃  (m=0)" if _first_empty else None,
            showlegend=_first_empty,
            legendgroup="empty",
        ))
        _first_empty = False

    _fig2d.add_hline(
        y=np.pi / 2, line=dict(color="gray", dash="dot", width=1),
        annotation_text="boundary  π/2", annotation_position="bottom right",
    )
    _fig2d.update_layout(
        title="Geodesic arches in compactified AdS  [x  vs  arctan(r)]",
        xaxis_title="x  (boundary direction)",
        yaxis_title="R_comp = arctan(r)",
        yaxis=dict(range=[0.0, np.pi / 2 * 1.08]),
        legend_title="",
        width=860, height=520,
    )
    mo.ui.plotly(_fig2d)
    return


@app.cell
def _(
    R_TRAJ,
    V0_SELECTED,
    get_mass_and_dmdv,
    go,
    mo,
    np,
    trajs_btz_geo,
    trajs_geo,
):
    def _ds3(arr, n=100):
        if len(arr) <= n:
            return arr
        return arr[np.round(np.linspace(0, len(arr) - 1, n)).astype(int)]

    def _geo3d(seg):
        idx = np.round(np.linspace(0, len(seg) - 1, min(len(seg), 100))).astype(int)
        s = seg[idx]
        x = s[:, 2]
        R_c = np.arctan(s[:, 1])
        v = s[:, 0]
        X = R_c * np.cos(x)
        Y = R_c * np.sin(x)
        X_full = np.concatenate([X[::-1], X[1:]])
        Y_full = np.concatenate([-Y[::-1], Y[1:]])
        Z_full = np.concatenate([v[::-1], v[1:]])
        return X_full, Y_full, Z_full

    _n_v0 = max(len(V0_SELECTED), 1)
    _colors = [f"hsl({int(h * 280)},80%,45%)" for h in np.linspace(0, 1, _n_v0)]
    _data3d = []

    for _i, _v0 in enumerate(V0_SELECTED):
        _first = True
        for _r in R_TRAJ:
            _seg = trajs_geo[(_v0, float(_r))]
            if len(_seg) < 3:
                continue
            _X, _Y, _Z = _geo3d(_seg)
            _ok = np.isfinite(_X) & np.isfinite(_Y) & np.isfinite(_Z)
            _data3d.append(go.Scatter3d(
                x=_X[_ok], y=_Y[_ok], z=_Z[_ok], mode="lines",
                line=dict(color=_colors[_i], width=4),
                name=f"Vaidya  v₀={_v0:+.1f}" if _first else None,
                showlegend=_first, legendgroup=f"v0_{_v0}",
            ))
            _first = False

    # BTZ reference (dark)
    _first_btz = True
    for _r in R_TRAJ:
        _seg = trajs_btz_geo[float(_r)]
        if len(_seg) < 3:
            continue
        _X, _Y, _Z = _geo3d(_seg)
        _ok = np.isfinite(_X) & np.isfinite(_Y) & np.isfinite(_Z)
        _data3d.append(go.Scatter3d(
            x=_X[_ok], y=_Y[_ok], z=_Z[_ok], mode="lines",
            line=dict(color="#111111", width=5),
            name="BTZ  (v₀=10)" if _first_btz else None,
            showlegend=_first_btz, legendgroup="btz",
        ))
        _first_btz = False

    # Apparent horizon surface
    _v_lo = (min(V0_SELECTED) - 1.0) if V0_SELECTED else -5.0
    _v_hor = np.linspace(_v_lo, 10.0, 60)
    _m_hor = np.array([float(get_mass_and_dmdv(float(_vv))[0]) for _vv in _v_hor])
    _hmask = _m_hor > 0
    if np.any(_hmask):
        _v_v = _v_hor[_hmask]
        _R_h = np.arctan(np.sqrt(_m_hor[_hmask]))
        _th = np.linspace(0, 2 * np.pi, 40)
        _Th = np.outer(_th, np.ones(len(_v_v)))
        _Rh = np.outer(np.ones(40), _R_h)
        _Vh = np.outer(np.ones(40), _v_v)
        _data3d.append(go.Surface(
            x=_Rh * np.cos(_Th), y=_Rh * np.sin(_Th), z=_Vh,
            colorscale=[[0, "saddlebrown"], [1, "saddlebrown"]],
            showscale=False, opacity=0.3, hoverinfo="skip", name="Apparent horizon",
        ))

    _fig3d = go.Figure(data=_data3d)
    _fig3d.update_layout(
        title="Geodesics in compactified Vaidya-AdS  (drag to rotate)",
        scene=dict(
            xaxis_title="X = R_comp·cos x",
            yaxis_title="Y = R_comp·sin x",
            zaxis_title="v (advanced time)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.5),
        ),
        legend_title="",
        width=920, height=720,
    )
    mo.ui.plotly(_fig3d)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

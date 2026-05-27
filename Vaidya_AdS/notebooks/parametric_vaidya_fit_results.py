import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    import os
    import time
    from pathlib import Path
    import numpy as np
    import polars as pl
    import marimo as mo
    import altair as alt
    from scipy.optimize import minimize

    return Path, alt, minimize, mo, np, os, pl, sys, time


@app.cell
def _(Path, os, sys):
    _nb_dir = Path(__file__).resolve().parent
    _root = _nb_dir.parent
    sys.path.insert(0, str(_root / "src"))
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    jax.config.update("jax_enable_x64", True)
    from Vaidya_AdS import (
        integrate_geodesic,
        geodesic_length_from_traj,
        geodesic_length_reg,
    )

    return geodesic_length_from_traj, geodesic_length_reg, integrate_geodesic


@app.cell
def _(mo):
    mo.md(r"""
    # Parametric Vaidya Inverse Problem — Turning-Grid Fit

    Recover the Vaidya shell parameters $v_c$ (center) and $v_s$ (thickness) from
    synthetic geodesic data generated with known ground-truth values.

    **Metric** (AdS₃-Vaidya in ingoing EF coordinates):
    $$ds^2 = -f(r,v)\,dv^2 + 2\,dv\,dr + r^2\,dx^2, \qquad
    f(r,v) = r^2 - m(v)$$

    **Mass profile** (vacuum-to-BTZ shell collapse, $m_i=0$, $m_f=1$):
    $$m(v;\,v_c, v_s) = \tfrac{1}{2}\!\left[1 + \tanh\!\left(\tfrac{v - v_c}{v_s}\right)\right]$$

    **Inverse problem:** minimize turning-grid MSE over $(v_c, v_s)$:
    $$\mathcal{L}(v_c, v_s) = \frac{1}{N}\sum_{i}
    \bigl[L_{\mathrm{reg}}^{\mathrm{pred}}(r_{\star,i},v_{\star,i};\,v_c,v_s)
    -L_{\mathrm{reg}}^{\mathrm{target}}(r_{\star,i},v_{\star,i})\bigr]^2$$

    > **Bulk-label note:** The grid is parameterized by turning-point coordinates
    > $(r_\star, v_\star)$, which are internal bulk quantities. The real boundary-only
    > inverse problem will use $(\ell, t_{\mathrm{bdy}})$ instead.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Parameters
    """)
    return


@app.cell
def _(mo):
    true_vc = mo.ui.slider(-1.5, 1.5, step=0.05, value=0.0, label=r"True $v_c$")
    true_vs = mo.ui.slider(0.1, 2.0, step=0.05, value=0.5, label=r"True $v_s$")
    mo.vstack([
        mo.md("### True mass profile"),
        mo.hstack([true_vc, true_vs]),
    ])
    return true_vc, true_vs


@app.cell
def _(mo):
    init_vc = mo.ui.slider(-1.5, 1.5, step=0.05, value=0.3, label=r"Initial guess $v_c$")
    init_vs = mo.ui.slider(0.1, 2.0, step=0.05, value=0.8, label=r"Initial guess $v_s$")
    mo.vstack([
        mo.md("### Optimizer initial guess"),
        mo.hstack([init_vc, init_vs]),
    ])
    return init_vc, init_vs


@app.cell
def _(mo):
    n_r_slider = mo.ui.slider(4, 12, step=1, value=8, label=r"$n_{r_\star}$")
    n_v_slider = mo.ui.slider(4, 12, step=1, value=9, label=r"$n_{v_\star}$")
    r_cut_inp = mo.ui.number(value=200.0, label=r"$r_{\mathrm{cut}}$")
    dt_inp = mo.ui.number(value=0.002, label=r"$\Delta\lambda$")
    n_steps_slider = mo.ui.slider(5000, 40000, step=5000, value=20000, label=r"$n_\mathrm{steps}$")
    mo.vstack([
        mo.md("### Grid and solver"),
        mo.hstack([n_r_slider, n_v_slider]),
        mo.hstack([r_cut_inp, dt_inp, n_steps_slider]),
    ])
    return dt_inp, n_r_slider, n_steps_slider, n_v_slider, r_cut_inp


@app.cell
def _(alt, init_vc, init_vs, mo, np, pl, true_vc, true_vs):
    _v = np.linspace(-3.5, 3.5, 600)

    def _m(v, vc, vs):
        return 0.5 * (1.0 + np.tanh((v - vc) / vs))

    _n = len(_v)
    _df = pl.DataFrame({
        "v": np.tile(_v, 2),
        "m": np.concatenate([
            _m(_v, true_vc.value, true_vs.value),
            _m(_v, init_vc.value, init_vs.value),
        ]),
        "curve": ["truth"] * _n + ["initial guess"] * _n,
    })
    _preview = (
        alt.Chart(_df)
        .mark_line()
        .encode(
            x=alt.X("v:Q", title="v"),
            y=alt.Y("m:Q", title="m(v)", scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color(
                "curve:N",
                scale=alt.Scale(
                    domain=["truth", "initial guess"],
                    range=["#1976d2", "#f57c00"],
                ),
                legend=alt.Legend(title=""),
            ),
            strokeDash=alt.StrokeDash(
                "curve:N",
                scale=alt.Scale(
                    domain=["truth", "initial guess"],
                    range=[[1, 0], [6, 3]],
                ),
            ),
            tooltip=["v:Q", "m:Q", "curve:N"],
        )
        .properties(
            title="Mass profile preview (updates live with sliders)",
            width=500,
            height=200,
        )
    )
    mo.vstack([mo.md("### Profile preview"), _preview])
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶  Generate data & run fit", kind="success")
    run_button
    return (run_button,)


@app.cell
def _(
    alt,
    dt_inp,
    geodesic_length_from_traj,
    geodesic_length_reg,
    init_vc,
    init_vs,
    integrate_geodesic,
    minimize,
    mo,
    n_r_slider,
    n_steps_slider,
    n_v_slider,
    np,
    pl,
    r_cut_inp,
    run_button,
    time,
    true_vc,
    true_vs,
):
    mo.stop(
        not run_button.value,
        mo.vstack([
            mo.md("---"),
            mo.md("Configure parameters above, then click **▶ Generate data & run fit**."),
        ]),
    )

    # ── Parameters ────────────────────────────────────────────────────────────
    M_I, M_F = 0.0, 1.0
    V_C_TRUE = float(true_vc.value)
    V_S_TRUE = float(true_vs.value)
    V_C_INIT = float(init_vc.value)
    V_S_INIT = float(init_vs.value)
    R_CUT = float(r_cut_inp.value)
    DT = float(dt_inp.value)
    N_STEPS = int(n_steps_slider.value)
    R_STARS = np.linspace(1.05, 5.0, int(n_r_slider.value))
    V_STARS = np.linspace(-1.2, 1.2, int(n_v_slider.value))
    N_GRID = len(R_STARS) * len(V_STARS)

    # ── Helper: integrate one geodesic and return all observables ──────────────
    def _run_one(r_star, v_star, v_c, v_s):
        traj = np.array(integrate_geodesic(
            r_star, v_star,
            n_steps=N_STEPS, dt=DT,
            m_i=M_I, m_f=M_F, v_c=v_c, v_s=v_s,
        ))
        rs = traj[:, 1]
        hit_k = next((k for k in range(1, len(rs)) if rs[k] >= R_CUT), None)
        if hit_k is None:
            return None
        # Linearly interpolate to exact r_cut for boundary observables
        alpha = (R_CUT - rs[hit_k - 1]) / (rs[hit_k] - rs[hit_k - 1])
        s = traj[hit_k - 1] + alpha * (traj[hit_k] - traj[hit_k - 1])
        ell = 2.0 * float(s[2])
        t_bdy = float(s[0]) + 1.0 / R_CUT
        # L_reg uses discrete cutoff (cancels in the MSE loss)
        L = float(geodesic_length_from_traj(
            traj, DT, r_cut=R_CUT, m_i=M_I, m_f=M_F, v_c=v_c, v_s=v_s,
        ))
        L_reg = float(geodesic_length_reg(L, R_CUT))
        return {"r_star": r_star, "v_star": v_star,
                "ell": ell, "t_bdy": t_bdy, "L_reg": L_reg}

    # ── Step 1: Generate target data ───────────────────────────────────────────
    mo.output.replace(mo.md(f"**Step 1 / 3 — Generating target data** (0 / {N_GRID})…"))
    _t0 = time.time()
    target_records = []
    _done = 0
    for _r in R_STARS:
        for _v in V_STARS:
            _done += 1
            _rec = _run_one(_r, _v, V_C_TRUE, V_S_TRUE)
            if _rec:
                target_records.append(_rec)
            mo.output.replace(mo.md(
                f"**Step 1 / 3 — Generating target data** ({_done} / {N_GRID})…"
            ))
    t_data = time.time() - _t0
    n_accepted = len(target_records)

    # ── Step 2: Optimize (v_c, v_s) with Nelder-Mead ──────────────────────────
    # Log-parameterize v_s to enforce positivity: q = [v_c, log(v_s)]
    def _pack(vc, vs):
        return np.array([vc, np.log(vs)])

    def _unpack(q):
        return float(q[0]), float(np.exp(q[1]))

    loss_history = []

    def _loss(q):
        vc, vs = _unpack(q)
        sq, n = 0.0, 0
        for rec in target_records:
            pred = _run_one(rec["r_star"], rec["v_star"], vc, vs)
            if pred is not None:
                sq += (pred["L_reg"] - rec["L_reg"]) ** 2
                n += 1
        val = sq / n if n > 0 else 1e10
        ev = len(loss_history) + 1
        loss_history.append({"eval": ev, "v_c": vc, "v_s": vs, "loss": val})
        if ev % 5 == 0 or ev == 1:
            mo.output.replace(mo.md(
                f"**Step 2 / 3 — Optimizer** — eval {ev} | "
                f"v_c = {vc:+.4f} | v_s = {vs:.4f} | loss = {val:.3e}"
            ))
        return val

    _t1 = time.time()
    _opt = minimize(
        _loss,
        _pack(V_C_INIT, V_S_INIT),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 600},
    )
    t_fit = time.time() - _t1
    v_c_fit, v_s_fit = _unpack(_opt.x)
    n_evals = len(loss_history)

    # ── Step 3: Evaluate at fitted parameters ──────────────────────────────────
    mo.output.replace(mo.md("**Step 3 / 3 — Computing prediction at fitted parameters…**"))
    pred_rows = []
    for rec in target_records:
        _p = _run_one(rec["r_star"], rec["v_star"], v_c_fit, v_s_fit)
        _Lp = _p["L_reg"] if _p else None
        pred_rows.append({
            "r_star": rec["r_star"],
            "v_star": rec["v_star"],
            "ell": rec["ell"],
            "t_bdy": rec["t_bdy"],
            "L_reg_target": rec["L_reg"],
            "L_reg_pred": _Lp,
            "residual": (_Lp - rec["L_reg"]) if _Lp is not None else None,
            "r_label": f"{rec['r_star']:.2f}",
            "v_label": f"{rec['v_star']:.2f}",
        })

    results_df = pl.DataFrame(pred_rows)
    loss_df = pl.DataFrame(loss_history)

    # ── Build plots ────────────────────────────────────────────────────────────

    # Summary table
    _err_vc = abs(v_c_fit - V_C_TRUE)
    _err_vs = abs(v_s_fit - V_S_TRUE)
    _summary = mo.md(rf"""
    ---
    ## Results

    | | $v_c$ | $v_s$ |
    |---|---:|---:|
    | **Truth** | {V_C_TRUE:.4f} | {V_S_TRUE:.4f} |
    | **Initial guess** | {V_C_INIT:.4f} | {V_S_INIT:.4f} |
    | **Fit** | **{v_c_fit:.6f}** | **{v_s_fit:.6f}** |
    | $|\Delta|$ | {_err_vc:.2e} | {_err_vs:.2e} |

    Final MSE loss: **{float(_opt.fun):.3e}** &nbsp;|&nbsp;
    Nelder-Mead evals: **{n_evals}** &nbsp;|&nbsp;
    {t_fit:.1f} s (fit) + {t_data:.1f} s (data) &nbsp;|&nbsp;
    {n_accepted}/{N_GRID} geodesics accepted
    """)

    # Mass profile comparison
    _v_arr = np.linspace(-3.5, 3.5, 600)
    _nv = len(_v_arr)

    def _mp(v, vc, vs):
        return 0.5 * (1.0 + np.tanh((v - vc) / vs))

    _mp_df = pl.DataFrame({
        "v": np.tile(_v_arr, 3),
        "m(v)": np.concatenate([
            _mp(_v_arr, V_C_TRUE, V_S_TRUE),
            _mp(_v_arr, V_C_INIT, V_S_INIT),
            _mp(_v_arr, v_c_fit, v_s_fit),
        ]),
        "curve": ["truth"] * _nv + ["initial guess"] * _nv + ["fit"] * _nv,
    })
    _mass_chart = (
        alt.Chart(_mp_df)
        .mark_line()
        .encode(
            x=alt.X("v:Q", title="v"),
            y=alt.Y("m(v):Q", scale=alt.Scale(domain=[-0.05, 1.05])),
            color=alt.Color(
                "curve:N",
                scale=alt.Scale(
                    domain=["truth", "initial guess", "fit"],
                    range=["#1976d2", "#f57c00", "#d32f2f"],
                ),
                legend=alt.Legend(title=""),
            ),
            strokeDash=alt.StrokeDash(
                "curve:N",
                scale=alt.Scale(
                    domain=["truth", "initial guess", "fit"],
                    range=[[1, 0], [8, 4], [3, 3]],
                ),
            ),
            tooltip=["v:Q", "m(v):Q", "curve:N"],
        )
        .properties(title="Mass profile: truth / initial guess / fit", width=500, height=260)
    )

    # Loss convergence (log scale)
    _loss_chart = (
        alt.Chart(loss_df)
        .mark_line(point=alt.OverlayMarkDef(size=30), color="#7b1fa2")
        .encode(
            x=alt.X("eval:Q", title="Optimizer evaluation"),
            y=alt.Y("loss:Q", title="MSE loss", scale=alt.Scale(type="log")),
            tooltip=["eval:Q", alt.Tooltip("v_c:Q", format=".5f"),
                     alt.Tooltip("v_s:Q", format=".5f"),
                     alt.Tooltip("loss:Q", format=".3e")],
        )
        .properties(title="Loss convergence (log scale)", width=500, height=260)
    )

    # L_reg heatmap on turning-point grid
    _sorted_r = sorted(results_df["r_label"].unique().to_list(), key=float)
    _sorted_v = sorted(results_df["v_label"].unique().to_list(), key=float)

    _lreg_heatmap = (
        alt.Chart(results_df)
        .mark_rect()
        .encode(
            x=alt.X("r_label:O", sort=_sorted_r, title="r★"),
            y=alt.Y("v_label:O", sort=_sorted_v, title="v★"),
            color=alt.Color(
                "L_reg_target:Q",
                scale=alt.Scale(scheme="viridis"),
                title="L_reg (target)",
            ),
            tooltip=["r_star:Q", "v_star:Q",
                     alt.Tooltip("L_reg_target:Q", format=".4f"),
                     alt.Tooltip("ell:Q", format=".4f"),
                     alt.Tooltip("t_bdy:Q", format=".4f")],
        )
        .properties(title="Target L_reg(r★, v★)", width=360, height=290)
    )

    # Residuals heatmap
    _res_heatmap = (
        alt.Chart(results_df)
        .mark_rect()
        .encode(
            x=alt.X("r_label:O", sort=_sorted_r, title="r★"),
            y=alt.Y("v_label:O", sort=_sorted_v, title="v★"),
            color=alt.Color(
                "residual:Q",
                scale=alt.Scale(scheme="blueorange", domainMid=0),
                title="pred − target",
            ),
            tooltip=["r_star:Q", "v_star:Q",
                     alt.Tooltip("residual:Q", format=".3e"),
                     alt.Tooltip("L_reg_target:Q", format=".4f"),
                     alt.Tooltip("L_reg_pred:Q", format=".4f")],
        )
        .properties(title="Residuals: L_reg_pred − L_reg_target", width=360, height=290)
    )

    # Scatter: predicted vs target
    _lo = float(results_df["L_reg_target"].min())
    _hi = float(results_df["L_reg_target"].max())
    _diag = (
        alt.Chart(pl.DataFrame({"x": [_lo, _hi], "y": [_lo, _hi]}))
        .mark_line(color="gray", strokeDash=[5, 3], opacity=0.6)
        .encode(x="x:Q", y="y:Q")
    )
    _fit_scatter = (
        alt.Chart(results_df)
        .mark_circle(size=65, opacity=0.85)
        .encode(
            x=alt.X("L_reg_target:Q", title="L_reg target"),
            y=alt.Y("L_reg_pred:Q", title="L_reg predicted"),
            color=alt.Color("r_star:Q", scale=alt.Scale(scheme="plasma"), title="r★"),
            tooltip=["r_star:Q", "v_star:Q",
                     alt.Tooltip("L_reg_target:Q", format=".4f"),
                     alt.Tooltip("L_reg_pred:Q", format=".4f"),
                     alt.Tooltip("residual:Q", format=".3e")],
        )
        .properties(title="Predicted vs target (diagonal = perfect fit)", width=340, height=320)
    )

    # Boundary observables
    _bdy = (
        alt.Chart(results_df)
        .mark_circle(size=70, opacity=0.85)
        .encode(
            x=alt.X("ell:Q", title="ℓ (boundary separation)"),
            y=alt.Y("t_bdy:Q", title="t_bdy (boundary time)"),
            color=alt.Color(
                "L_reg_target:Q",
                scale=alt.Scale(scheme="viridis"),
                title="L_reg",
            ),
            tooltip=["r_star:Q", "v_star:Q",
                     alt.Tooltip("ell:Q", format=".4f"),
                     alt.Tooltip("t_bdy:Q", format=".4f"),
                     alt.Tooltip("L_reg_target:Q", format=".4f")],
        )
        .properties(
            title="Boundary observables (ℓ, t_bdy) — future inverse problem space",
            width=400,
            height=320,
        )
    )

    # ── Final display ──────────────────────────────────────────────────────────
    mo.vstack([
        _summary,
        mo.md("### Mass profile & loss convergence"),
        mo.hstack([_mass_chart, _loss_chart]),
        mo.md("### $L_{\\mathrm{reg}}$ on turning-point grid"),
        mo.hstack([_lreg_heatmap, _res_heatmap]),
        mo.md("### Fit quality & boundary observables"),
        mo.hstack([(_diag + _fit_scatter), _bdy]),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

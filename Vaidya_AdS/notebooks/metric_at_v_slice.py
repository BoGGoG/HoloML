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

    return Path, alt, mo, np, os, pl, sys, time


@app.cell
def _(Path, os, sys):
    _nb_dir = Path(__file__).resolve().parent
    _root = _nb_dir.parent
    sys.path.insert(0, str(_root / "src"))
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from Vaidya_AdS import (
        integrate_geodesic,
        geodesic_length_from_traj,
        geodesic_length_reg,
    )
    print(jax.default_backend())
    return (
        geodesic_length_from_traj,
        geodesic_length_reg,
        integrate_geodesic,
        jax,
        jnp,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Metric Reconstruction at a Fixed $v$ Slice

    For a fixed probe time $v_0$, recover the effective mass $m_0 = m(v_0)$ from
    geodesic data at turning points $(r_{\star,i},\, v_0)$ via gradient descent.

    **Metric** (AdS₃-Vaidya in ingoing EF coordinates):
    $$ds^2 = -f(r,v)\,dv^2 + 2\,dv\,dr + r^2\,dx^2,\qquad f(r,v) = r^2 - m(v)$$

    **Target data:** Vaidya geodesics with true profile $m(v;\,v_c,v_s)$, all turning at $v_\star = v_0$.

    **Model:** instantaneous static geometry with constant mass $m_0$ — exact analytic formula
    derived from the conserved quantities $E=0$, $L=r_\star$ of the static geodesic:
    $$L_{\rm reg}^{\rm static}(r_\star,\,m_0)
    = 2\operatorname{arcsinh}\!\sqrt{\frac{r_{\rm cut}^2 - r_\star^2}{r_\star^2 - m_0}}
    - 2\log(2r_{\rm cut})$$

    **Loss and gradient** (exact, via `jax.grad`):
    $$\mathcal{L}(m_0)=\frac{1}{N}\sum_{i=1}^N
    \bigl[L_{\rm reg}^{\rm static}(r_{\star,i},\,m_0)
    -L_{\rm reg}^{\rm Vaidya}(r_{\star,i},\,v_0)\bigr]^2,
    \qquad
    m_0 \leftarrow \Pi_{[0,1]}\!\left[m_0 - \eta\,\nabla_{m_0}\mathcal{L}\right]$$

    > **Note:** The fitted $m_0$ is the effective mass that best reproduces the geodesic
    > data at the $v_0$ slice under the static approximation. It will in general differ
    > from the instantaneous true value $m(v_0)$ because the geodesics traverse the
    > evolving spacetime as they propagate to the boundary.
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
        mo.md("### True Vaidya mass profile"),
        mo.hstack([true_vc, true_vs]),
    ])
    return true_vc, true_vs


@app.cell
def _(mo):
    v0_slider = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.0, label=r"Probe time $v_0$")
    init_m0 = mo.ui.slider(0.05, 0.95, step=0.05, value=0.2, label=r"Initial guess $m_0$")
    mo.vstack([
        mo.md("### Time slice & initial guess"),
        mo.hstack([v0_slider, init_m0]),
    ])
    return init_m0, v0_slider


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(0.01, 1.0, step=0.01, value=0.3, label=r"Learning rate $\eta$")
    n_iter_slider = mo.ui.slider(5, 100, step=5, value=30, label="GD iterations")
    mo.vstack([
        mo.md("### Gradient descent"),
        mo.hstack([lr_slider, n_iter_slider]),
    ])
    return lr_slider, n_iter_slider


@app.cell
def _(mo):
    n_r_slider = mo.ui.slider(4, 20, step=1, value=8, label=r"$n_{r_\star}$")
    r_cut_inp = mo.ui.number(value=200.0, label=r"$r_{\mathrm{cut}}$")
    dt_inp = mo.ui.number(value=0.002, label=r"$\Delta\lambda$")
    n_steps_slider = mo.ui.slider(5000, 40000, step=5000, value=20000, label=r"$n_{\mathrm{steps}}$")
    mo.vstack([
        mo.md("### Target data solver"),
        mo.hstack([n_r_slider, r_cut_inp]),
        mo.hstack([dt_inp, n_steps_slider]),
    ])
    return dt_inp, n_r_slider, n_steps_slider, r_cut_inp


@app.cell
def _(alt, init_m0, mo, np, pl, true_vc, true_vs, v0_slider):
    _v = np.linspace(-3.5, 3.5, 600)
    _m_profile = 0.5 * (1.0 + np.tanh((_v - true_vc.value) / true_vs.value))
    _m_at_v0 = float(0.5 * (1.0 + np.tanh((v0_slider.value - true_vc.value) / true_vs.value)))

    _df_curve = pl.DataFrame({"v": _v, "m": _m_profile})
    _df_truth = pl.DataFrame({
        "v": [float(v0_slider.value)],
        "m": [_m_at_v0],
        "label": [f"truth  m(v₀) = {_m_at_v0:.4f}"],
    })
    _df_init = pl.DataFrame({
        "v": [float(v0_slider.value)],
        "m": [float(init_m0.value)],
        "label": [f"init  m₀ = {init_m0.value:.2f}"],
    })

    _curve = (
        alt.Chart(_df_curve)
        .mark_line(color="#1976d2")
        .encode(
            x=alt.X("v:Q", title="v"),
            y=alt.Y("m:Q", title="m(v)", scale=alt.Scale(domain=[-0.05, 1.05])),
        )
    )
    _vline = (
        alt.Chart(pl.DataFrame({"v0": [float(v0_slider.value)]}))
        .mark_rule(color="#e53935", strokeDash=[6, 3])
        .encode(x=alt.X("v0:Q"))
    )
    _truth_dot = (
        alt.Chart(_df_truth)
        .mark_circle(size=120, color="#e53935")
        .encode(x="v:Q", y="m:Q", tooltip=["label:N"])
    )
    _init_dot = (
        alt.Chart(_df_init)
        .mark_circle(size=80, color="#f57c00", opacity=0.8)
        .encode(x="v:Q", y="m:Q", tooltip=["label:N"])
    )
    _chart = (
        (_curve + _vline + _truth_dot + _init_dot)
        .properties(
            title=(
                f"True m(v) — probe at v₀ = {v0_slider.value:.2f},  "
                f"m(v₀) = {_m_at_v0:.4f}   |   init m₀ = {init_m0.value:.2f}"
            ),
            width=520,
            height=220,
        )
    )
    mo.vstack([mo.md("### Profile preview"), _chart])
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶  Generate data & run gradient descent", kind="success")
    run_button
    return (run_button,)


@app.cell
def _(
    dt_inp,
    geodesic_length_from_traj,
    geodesic_length_reg,
    init_m0,
    integrate_geodesic,
    jax,
    jnp,
    lr_slider,
    mo,
    n_iter_slider,
    n_r_slider,
    n_steps_slider,
    np,
    r_cut_inp,
    run_button,
    time,
    true_vc,
    true_vs,
    v0_slider,
):
    mo.stop(
        not run_button.value,
        mo.vstack([
            mo.md("---"),
            mo.md("Configure parameters above, then click **▶ Generate data & run gradient descent**."),
        ]),
    )

    # ── Parameters ────────────────────────────────────────────────────────────
    _M_I, _M_F = 0.0, 1.0
    _V_C = float(true_vc.value)
    _V_S = float(true_vs.value)
    _V_0 = float(v0_slider.value)
    _R_CUT = float(r_cut_inp.value)
    _DT = float(dt_inp.value)
    _N_STEPS = int(n_steps_slider.value)
    _LR = float(lr_slider.value)
    _N_ITER = int(n_iter_slider.value)
    _R_STARS = np.linspace(1.05, 5.0, int(n_r_slider.value))
    _N = len(_R_STARS)
    gd_m_true = float(0.5 * (1.0 + np.tanh((_V_0 - _V_C) / _V_S)))

    # ── Analytic static BTZ model ─────────────────────────────────────────────
    # Derived from conserved quantities E=0, L=r_star of the static geodesic:
    #   L_half = arcsinh(sqrt((r_cut^2 - r_star^2) / (r_star^2 - m0)))
    @jax.jit
    def _lreg_model_single(r_star, m0):
        x = jnp.sqrt((_R_CUT**2 - r_star**2) / (r_star**2 - m0))
        return 2.0 * jnp.arcsinh(x) - 2.0 * jnp.log(2.0 * _R_CUT)

    # ── Step 1: Generate target data with Vaidya integrator ───────────────────
    mo.output.replace(mo.md(f"**Step 1 / 2 — Generating target data** (0 / {_N})…"))
    _t0 = time.time()
    gd_target = []
    for _i, _r in enumerate(_R_STARS):
        _traj = np.array(integrate_geodesic(
            float(_r), _V_0,
            n_steps=_N_STEPS, dt=_DT,
            m_i=_M_I, m_f=_M_F, v_c=_V_C, v_s=_V_S,
        ))
        _rs = _traj[:, 1]
        _hit = next((k for k in range(1, len(_rs)) if _rs[k] >= _R_CUT), None)
        if _hit is not None:
            _L = float(geodesic_length_from_traj(
                _traj, _DT, r_cut=_R_CUT, m_i=_M_I, m_f=_M_F, v_c=_V_C, v_s=_V_S,
            ))
            gd_target.append({
                "r_star": float(_r),
                "L_reg": float(geodesic_length_reg(_L, _R_CUT)),
            })
        mo.output.replace(mo.md(
            f"**Step 1 / 2 — Generating target data** ({_i + 1} / {_N})…"
        ))
    gd_n_accepted = len(gd_target)
    gd_t_data = time.time() - _t0

    # Pack into JAX arrays for vectorised autodiff
    _r_arr = jnp.array([rec["r_star"] for rec in gd_target])
    _L_arr = jnp.array([rec["L_reg"] for rec in gd_target])

    def _loss(m0):
        preds = jax.vmap(lambda r: _lreg_model_single(r, m0))(_r_arr)
        return jnp.mean((preds - _L_arr) ** 2)

    _grad_loss = jax.jit(jax.grad(_loss))

    # ── Step 2: Gradient descent ──────────────────────────────────────────────
    _m0 = float(init_m0.value)
    gd_history = []

    for _it in range(_N_ITER):
        _f_cur = float(_loss(float(_m0)))
        _grad = float(_grad_loss(float(_m0)))
        gd_history.append({"iter": _it, "m0": _m0, "loss": _f_cur, "grad": _grad})
        _m0 = float(np.clip(_m0 - _LR * _grad, 1e-4, 1.0 - 1e-4))
        if _it % 5 == 0 or _it == 0:
            mo.output.replace(mo.md(
                f"**Step 2 / 2 — Gradient descent** iter {_it + 1}/{_N_ITER}: "
                f"m₀ = {_m0:.5f},  loss = {_f_cur:.3e},  ∇ = {_grad:.3e}"
            ))
        if abs(_grad) < 1e-9:
            break

    gd_m0 = _m0
    gd_loss_final = float(_loss(float(gd_m0)))
    gd_history.append({"iter": len(gd_history), "m0": gd_m0, "loss": gd_loss_final, "grad": 0.0})
    gd_t_total = time.time() - _t0
    gd_residual = gd_m0 - gd_m_true

    for rec in gd_target:
        rec["L_reg_model"] = float(_lreg_model_single(rec["r_star"], gd_m0))

    mo.output.replace(mo.md(
        f"Done — {len(gd_history)} iters, {gd_t_total:.1f}s  |  "
        f"m₀ = {gd_m0:.6f}  (true m(v₀) = {gd_m_true:.6f},  Δ = {gd_residual:+.2e}),  "
        f"final loss = {gd_loss_final:.3e}"
    ))
    return (
        gd_history,
        gd_loss_final,
        gd_m0,
        gd_m_true,
        gd_n_accepted,
        gd_residual,
        gd_t_total,
        gd_target,
    )


@app.cell
def _(
    gd_loss_final,
    gd_m0,
    gd_m_true,
    gd_n_accepted,
    gd_residual,
    gd_t_total,
    mo,
):
    mo.md(f"""
    ## Results

    | Quantity | Value |
    |---|---|
    | Fitted $m_0$ | **{gd_m0:.6f}** |
    | True $m(v_0)$ | {gd_m_true:.6f} |
    | Residual $\\Delta m_0$ | {gd_residual:+.2e} |
    | Final loss $\\mathcal{{L}}$ | {gd_loss_final:.3e} |
    | Accepted geodesics | {gd_n_accepted} |
    | Wall time | {gd_t_total:.1f} s |

    > The fitted $m_0$ reflects the *effective* mass experienced along the geodesics,
    > which traverse the dynamic spacetime as they propagate from $v_0$ outward to
    > $r_{{\\rm cut}}$. It differs from the instantaneous $m(v_0)$ when the shell
    > is still actively collapsing during that propagation.
    """)
    return


@app.cell
def _(alt, gd_history, gd_m_true, mo, pl):
    _df = pl.DataFrame(gd_history)

    _loss_chart = (
        alt.Chart(_df)
        .mark_line(point=True, color="#1976d2")
        .encode(
            x=alt.X("iter:Q", title="Iteration"),
            y=alt.Y("loss:Q", title="MSE loss", scale=alt.Scale(type="log")),
            tooltip=["iter:Q", alt.Tooltip("loss:Q", format=".3e"), alt.Tooltip("m0:Q", format=".5f")],
        )
        .properties(title="Loss convergence", width=360, height=220)
    )

    _true_line = (
        alt.Chart(pl.DataFrame({"m_true": [gd_m_true]}))
        .mark_rule(color="#e53935", strokeDash=[5, 3])
        .encode(y=alt.Y("m_true:Q"))
    )
    _m0_chart = (
        (
            alt.Chart(_df)
            .mark_line(point=True, color="#f57c00")
            .encode(
                x=alt.X("iter:Q", title="Iteration"),
                y=alt.Y("m0:Q", title="m₀"),
                tooltip=["iter:Q", alt.Tooltip("m0:Q", format=".5f"), alt.Tooltip("grad:Q", format=".3e")],
            )
        + _true_line
        )
        .properties(title=f"m₀ trajectory  (red: true m(v₀) = {gd_m_true:.4f})", width=360, height=220)
    )

    mo.hstack([_loss_chart, _m0_chart])
    return


@app.cell
def _(alt, gd_m0, gd_m_true, gd_target, mo, pl):
    _df = pl.DataFrame(gd_target)
    _lmin = min(_df["L_reg"].min(), _df["L_reg_model"].min()) - 0.05
    _lmax = max(_df["L_reg"].max(), _df["L_reg_model"].max()) + 0.05

    _scatter = (
        alt.Chart(_df)
        .mark_circle(size=80, color="#1976d2")
        .encode(
            x=alt.X("L_reg:Q", title="L_reg  target (Vaidya)", scale=alt.Scale(domain=[_lmin, _lmax])),
            y=alt.Y(
                "L_reg_model:Q",
                title=f"L_reg  model (fitted m₀ = {gd_m0:.4f})",
                scale=alt.Scale(domain=[_lmin, _lmax]),
            ),
            tooltip=[
                alt.Tooltip("r_star:Q", format=".3f"),
                alt.Tooltip("L_reg:Q", format=".5f"),
                alt.Tooltip("L_reg_model:Q", format=".5f"),
            ],
        )
    )
    _diag = (
        alt.Chart(pl.DataFrame({"x": [_lmin, _lmax]}))
        .mark_line(color="#aaa", strokeDash=[4, 4])
        .encode(x="x:Q", y="x:Q")
    )
    _plot = (
        (_scatter + _diag)
        .properties(
            title=f"Target vs model  (true m(v₀) = {gd_m_true:.4f},  fitted m₀ = {gd_m0:.4f})",
            width=400,
            height=380,
        )
    )
    _residuals = _df.with_columns(
        (pl.col("L_reg_model") - pl.col("L_reg")).alias("residual")
    )
    _res_chart = (
        alt.Chart(_residuals)
        .mark_bar(color="#e53935", opacity=0.7)
        .encode(
            x=alt.X("r_star:Q", title="r★"),
            y=alt.Y("residual:Q", title="L_reg model − target"),
            tooltip=[
                alt.Tooltip("r_star:Q", format=".3f"),
                alt.Tooltip("residual:Q", format=".4e"),
            ],
        )
        .properties(title="Residuals vs r★", width=400, height=180)
    )
    mo.vstack([_plot, _res_chart])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

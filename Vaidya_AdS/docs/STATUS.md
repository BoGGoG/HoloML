# PROJECT STATUS

> **FILE ROLE:** Active project tracking. High-turnover file for daily tasks and milestones.
> **USAGE:** Update this file whenever a task is started, blocked, or finished.
> **DO NOT:** Put long-form research notes here; use `docs/JOURNAL.md` for reasoning and `docs/SPECS.md` for formulas/specs.

_Last updated: 2026-08-21_  (First convergence test of parametric pipeline)

## 🟢 Current Focus
Build a validated forward model for AdS$_3$-Vaidya entanglement data before attempting ML reconstruction.

The immediate scientific target is to generate reliable synthetic boundary data of the form

$$
(\ell,t) \mapsto L_{\mathrm{reg}}(\ell,t)
$$

from spacelike HRT geodesics in Vaidya-AdS. The first inverse-learning target should be the static metric at a fixed $v$, starting with a constrained/parametric profile rather than a fully general metric.

## Current Repository State
- `BTZ.py` implements the static BTZ baseline: analytic $f(z)=1-z^2$, $h(z)=1$, $\ell(z_*)=2\operatorname{arctanh}(z_*)$, finite entropy/length integrals, synthetic BTZ entropy data generation, and an Equinox network skeleton for $f(z)$ and $h(z)$.
- `Vaidya_AdS.py` implements a JAX/RK4 geodesic solver for AdS$_3$-Vaidya in ingoing EF-like coordinates,
  $$ds^2=-f(r,v)dv^2+2dvdr+r^2dx^2,\qquad f(r,v)=r^2-m(v).$$
  It computes trajectories from a symmetric turning point, proper lengths, regularized lengths, cutoff half-widths, boundary time readouts, and length profiles.
- `geodesics_plots.py` is a Marimo exploration notebook for Vaidya geodesics, compactified geometry plots, length diagnostics, and apparent-horizon visualization.
- `Vaidya_BTZ.py` compares Vaidya geodesics against both limits: late-time BTZ ($m\to1$) and early-time empty AdS ($m=0$). Plots $\ell(r_*)$, $L_{\mathrm{reg}}(r_*)$, $L_{\mathrm{reg}}(\ell)$, and convergence residuals toward each reference.
- `Empty_AdS.py` provides exact closed-form HRT geodesics in empty AdS$_3$ ($m=0$ Vaidya limit): $r=r_\ast\cosh\lambda$, $x=\tanh\lambda/r_\ast$, $v=t_{\mathrm{bdy}}-1/r$. Serves as the early-time validation reference for the Vaidya solver.
- `papers/` contains Markdown summaries of the five core papers and an overall roadmap summary.

## 🛠 Active TODO List

### High Priority
- [ ] Replace the placeholder `docs/SPECS.md` with the actual project specification: goal, metric conventions, RT/HRT formulas, Vaidya mass profile, geodesic equations, regularization convention, and ML target.
- [ ] Verify the Vaidya geodesic equations and turning-point initial conditions against the HRT/Vaidya literature before using generated data for ML.
- [x] Quantitatively validate the forward solver in known limits:
  - [x] pure AdS limit: `scripts/validate_known_limits.py`, max $|\Delta\ell|=4.5\times10^{-10}$, max $|\Delta L_{\mathrm{reg}}|=4.0\times10^{-3}$, $\kappa$-dev $<5\times10^{-12}$.
  - [x] static BTZ limit ($m_i=m_f=1$): max $|\Delta\ell|=2.0\times10^{-4}$, $\kappa$-dev $<5\times10^{-12}$.
  - [ ] convergence under `dt`, `n_steps`, and `r_cut` (systematic sweep not yet done),
  - [~] consistency of $\ell=2x(r_{\mathrm{cut}})$ and $L_{\mathrm{reg}}$: $\ell$ uses interpolated cutoff (accurate); $L_{\mathrm{reg}}$ uses discrete cutoff in the length integrator (causes ~4e-3 error in empty AdS; tracked in JOURNAL.md).
- [ ] Implement a robust forward-data generator that saves rows like `(turning_radius, turning_time, boundary_separation, boundary_time, regularized_length, solver_metadata)`.
- [ ] Add a shooting or interpolation layer to convert the native solver output $(r_*,v_0)\mapsto(\ell,v_\infty,L_{\mathrm{reg}})$ into boundary-controlled data $(\ell,t)\mapsto L_{\mathrm{reg}}$.
- [x] **Level 1 turning-grid sanity check**: fit $(v_c, v_s)$ from turning-point data $(r_\star, v_\star) \mapsto L_{\mathrm{reg}}$ — see `scripts/fit_parametric_vaidya_turning_grid.py`. Recovered $v_c=-3.37\times10^{-4}$ (true=0), $v_s=0.4998$ (true=0.5), final MSE loss $=3.95\times10^{-28}$, 72/72 samples accepted, 110 Nelder-Mead evaluations, ~152s runtime. (This is a bulk-label check only; see JOURNAL.md.)
- [ ] **Level 1 boundary-only inverse problem**: replace turning-point labels with boundary observables $(\ell, t_{\mathrm{bdy}}) \mapsto L_{\mathrm{reg}}$ — requires shooting/interpolation layer to convert $(r_\star, v_\star) \to (\ell, t_{\mathrm{bdy}})$.
- [ ] Define the first ML inverse problem: recover a parametric $m(v)$, e.g. shell amplitude/thickness/center, from synthetic entanglement data.

### Medium Priority
- [ ] Fix `geodesic_length_from_traj` to use interpolated rather than discrete cutoff; current mismatch between interpolated $\ell$ and discretely-integrated $L_{\mathrm{reg}}$ causes ~4e-3 error (see JOURNAL.md).
- [ ] Consider an event-based integrator or adaptive integration strategy for geodesics that miss the UV cutoff with fixed `n_steps`.
- [ ] Reduce or isolate the Gauss-Legendre endpoint singularity error in the BTZ comparison; consider endpoint-aware quadrature or analytic benchmarks for validation.
- [ ] Add tests for shape, monotonicity, spacelike norm positivity, cutoff hit status, and reproducible metadata in generated datasets.
- [ ] Standardize naming and directory layout: likely `src/`, `notebooks/`, `data/`, `docs/`, `tests/`.
- [ ] Decide whether the ML forward loss should use geodesic length directly, entropy $S=L/(4G_N)$, or dimensionless rescaled quantities.

### Low Priority / Future Work
- [ ] Extend from single-interval entanglement entropy to mutual information or EWCS data if the inverse problem is underdetermined.
- [ ] Generalize from parametric $m(v)$ to a neural mass profile after the constrained inverse problem works.
- [ ] Explore higher-dimensional strip geometries only after the AdS$_3$-Vaidya pipeline is stable.
- [ ] Package the paper summaries as permanent documentation under `docs/literature/`.

## ✅ Recently Completed
- 2026-08-21: First convergence test of `scripts/fit_metric_from_lreg.py` with optax SGD (LR=3.0, 20 steps). Pipeline is structurally functional: loss decreases monotonically ($1.48\times10^{-3} \to 2.77\times10^{-4}$), $m_f$ recovers to 0.996 (true=1.0), $v_c$ moves from 0.3 to 0.131 (true=0.0), but $v_s$ barely moves (0.8→0.77, true=0.5). Gradient scale disparity is the likely bottleneck; Adam or per-parameter LR should help. `stop_gradient` on $h_\mathrm{pred}$ was removed; effect TBD. See JOURNAL.md for full analysis.
- 2026-06-03: Built `scripts/fit_metric_from_lreg.py` — a fully differentiable parametric metric fitting pipeline. Generates geodesic data at a fixed probe time $v_0$ with true parameters, exports to JSON, then fits $f(r,v;\theta)$ end-to-end using `jax.grad`. Forward model uses general geodesic equations (via `jax.grad` on `f_metric`), batched RK4 with `jax.vmap` + `@jax.jit`, and a soft sigmoid cutoff for differentiable $L$ and $h$ extraction. Loss is MSE between predicted $L$ and interpolated target $L(h)$, optimised with optax (SGD or Adam).
- 2026-06-03: Found and fixed three bugs in `scripts/fit_metric_from_lreg.py`: (1) wrong sign on the $\partial_v f$ term in $\ddot r$, (2) hardcoded $f_r=2r$ in $\ddot v$ instead of using `jax.grad`, (3) `h_\mathrm{data}` was stored in descending order but passed directly to `jnp.interp` which requires ascending order — this caused completely wrong loss and gradient values. See JOURNAL.md for details.
- 2026-05-13: Implemented Level 1 turning-grid parametric fit (`scripts/fit_parametric_vaidya_turning_grid.py`). Generated 72 geodesics from true $(v_c=0, v_s=0.5)$; Nelder-Mead recovered $v_c=-3.4\times10^{-4}$, $v_s=0.4998$, loss $=3.95\times10^{-28}$ (machine zero). Outputs saved to `inverse_results/`. This is a bulk-label sanity check; real boundary-only inversion requires a shooting layer.
- 2026-05-13: Ran automated validation (`scripts/validate_known_limits.py`) against both known limits (20 $r_\star$ samples each, $r_{\mathrm{cut}}=200$, $\Delta\lambda=0.002$, 40000 steps). Zero cutoff failures. Empty AdS: $|\Delta\ell|_{\max}=4.5\times10^{-10}$, $|\Delta L_{\mathrm{reg}}|_{\max}=4.0\times10^{-3}$. Static BTZ: $|\Delta\ell|_{\max}=2.0\times10^{-4}$. Spacelike norm deviation $<5\times10^{-12}$ (machine zero) in both cases. JSON report saved to `validation_reports/known_limits_validation.json`.
- 2026-05-13: Added `src/Empty_AdS.py` with exact analytic geodesic solution for empty AdS$_3$ ($m=0$); verified against all three geodesic equations and the Poincaré–EF coordinate identity.
- 2026-05-13: Migrated `BTZ.py` from PyTorch/torchquad to JAX/Equinox; fixed Vaidya mass profile to SPECS.md convention ($m_i=0$, vacuum-to-BTZ) and propagated mass parameters through `ds_dlambda` and all length functions.
- 2026-05-13: Added concise Markdown summaries for the five core holographic-entanglement papers and an overall roadmap summary.
- 2026-05-13: Established the immediate project direction: validate BTZ/pure-AdS limits, generate Vaidya HRT data, then reconstruct a constrained mass profile before attempting general spacetime reconstruction.
- (predates journal): Implemented static BTZ analytic/quadrature baseline in `BTZ.py`.
- (predates journal): Implemented numerical AdS$_3$-Vaidya geodesic integration and length extraction in `Vaidya_AdS.py`.
- (predates journal): Built Marimo notebooks for geodesic exploration and Vaidya-vs-BTZ comparison.

## ⚠️ Blockers / Issues
- The native Vaidya solver is parameterized by turning-point data $(r_*,v_0)$, but ML training data should likely be parameterized by boundary observables $(\ell,t)$. This requires inversion/shooting/interpolation.
- Current cutoff handling uses discrete trajectory samples; this can introduce systematic error in $\ell$, $v_\infty$, and $L_{\mathrm{reg}}$.
- Fixed-step RK4 may miss `r_cut` for some geodesics unless `n_steps` is large enough.
- JAX `integrate_geodesic` requires static `n_steps`; changing it triggers recompilation.
- BTZ and Vaidya length conventions use different UV subtraction schemes; constant offsets must be tracked and documented.
- Gauss-Legendre quadrature near the BTZ endpoint singularity can have a few-percent error at moderate quadrature order.
- Full metric reconstruction from single-interval entropy is likely underdetermined; begin with constrained $m(v)$ reconstruction.

## Next Session Recommendation
The parametric pipeline converges but slowly, especially for $v_s$ (shell thickness). Immediate next steps:
1. Switch from SGD to Adam to handle per-parameter gradient scale disparity — $m_f$ has much larger gradients than $v_s$.
2. Run for more steps (200–500) to check whether the optimizer reaches the true parameters or stalls.
3. Re-enable `stop_gradient` on $h_\mathrm{pred}$ and compare convergence — the confounding gradient path may be slowing $v_s$.
4. Reduce $\delta$ (e.g. to $0.1$) if the loss floor at true parameters is too high.
5. Once the parametric fit reliably converges, replace `f_metric` with an Equinox neural network for unconstrained metric reconstruction.


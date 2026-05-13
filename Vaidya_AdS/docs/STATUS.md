# PROJECT STATUS

> **FILE ROLE:** Active project tracking. High-turnover file for daily tasks and milestones.
> **USAGE:** Update this file whenever a task is started, blocked, or finished.
> **DO NOT:** Put long-form research notes here; use `docs/JOURNAL.md` for reasoning and `docs/SPECS.md` for formulas/specs.

_Last updated: 2026-05-13_

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
- [ ] Quantitatively validate the forward solver in known limits:
  - [ ] pure AdS limit where applicable,
  - [ ] late-time BTZ limit $m(v)\to1$,
  - [ ] convergence under `dt`, `n_steps`, and `r_cut`,
  - [ ] consistency of $\ell=2x(r_{\mathrm{cut}})$ and $L_{\mathrm{reg}}$.
- [ ] Implement a robust forward-data generator that saves rows like `(turning_radius, turning_time, boundary_separation, boundary_time, regularized_length, solver_metadata)`.
- [ ] Add a shooting or interpolation layer to convert the native solver output $(r_*,v_0)\mapsto(\ell,v_\infty,L_{\mathrm{reg}})$ into boundary-controlled data $(\ell,t)\mapsto L_{\mathrm{reg}}$.
- [ ] Define the first ML inverse problem: recover a parametric $m(v)$, e.g. shell amplitude/thickness/center, from synthetic entanglement data.

### Medium Priority
- [ ] Improve cutoff handling by interpolating to exactly `r_cut` instead of using the first index with `r >= r_cut`.
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
Start by replacing `SPECS.md` with the actual physics specification, then run a small reproducible validation sweep comparing late-time Vaidya data against the BTZ analytic curve. Record residuals versus `dt`, `n_steps`, and `r_cut` before adding ML.


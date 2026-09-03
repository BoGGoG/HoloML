# Slice 01 — Neural-network mass profile in the differentiable fitting pipeline

Status: **closed**
Branch: `main`

## Goal

Replace the parametric $m(v)$ in `scripts/fit_metric_from_lreg.py` with an Equinox MLP so the pipeline can learn an unconstrained mass profile from geodesic data. The metric structure $f(r,v) = r^2 - m_\theta(v)$ is preserved; only $m_\theta$ becomes a neural network. At the end of this slice the script can:

1. Generate training data from the known parametric $m(v)$ (unchanged).
2. Train an Equinox MLP $m_\theta(v)$ end-to-end through the RK4 geodesic integrator using `jax.grad`.
3. Recover a learned $m_\theta(v)$ that visually matches the true tanh profile.

## Design

### What the NN replaces

The current `f_metric`:
```python
def f_metric(r, v, params):
    m = 0.5 * params["m_f"] * (1 + jnp.tanh((v - params["v_c"]) / params["v_s"]))
    return r**2 - m
```

becomes:
```python
def f_metric(r, v, model):
    return r**2 - model(v)
```

where `model` is an `eqx.Module`. The `jax.grad` calls for $\partial f/\partial r$ and $\partial f/\partial v$ (used in the geodesic RHS) differentiate through the NN automatically — no change to `_rhs` logic.

### Architecture

```
MassProfile(eqx.Module)
  input:   v  (scalar)
  layers:  Linear(1→32) → tanh → Linear(32→32) → tanh → Linear(32→1)
  output:  softplus(raw) → m ≥ 0
```

- **`softplus` output**: enforces $m \geq 0$ (physical mass non-negativity).
- **`tanh` activations**: smooth $m'(v)$ is needed because the geodesic equations use $\partial f/\partial v = -m'(v)$.
- **Small network**: $m(v)$ is a 1D function; 2×32 hidden units is sufficient.

### Initialization

Pre-train the MLP for a few hundred steps on the known tanh profile $m_\text{true}(v)$ evaluated on a grid $v \in [-5, 5]$ before starting the geodesic-based training loop. This avoids early geodesics hitting fake horizons or diverging due to a wildly wrong $m(v)$.

### Memory: `jax.checkpoint` on the RK4 scan body

Each geodesic integration is 20,000 RK4 steps. Backprop through `jax.lax.scan` stores all intermediate states and NN activations. With 20 geodesics vmapped, this can exhaust memory.

Apply `jax.checkpoint` (equivalently `jax.remat`) to the scan step function so intermediate NN activations are recomputed during backprop instead of stored. This trades ~2× compute for $O(1)$ memory in the number of steps.

```python
def step(s, _):
    k1 = _rhs(s, model)
    ...
    return s2, s2

step = jax.checkpoint(step)  # recompute activations during backprop
```

### Optimization

Switch from `optax.sgd` with a params dict to `optax.adam` with Equinox filter patterns:

```python
@eqx.filter_jit
def scalar_loss(model):
    h_p, L_p = forward(model, r_data, v0)
    return loss_fn(h_p, L_p)

grad_fn = eqx.filter_grad(scalar_loss)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
```

Update uses `eqx.apply_updates`. Adam handles per-parameter gradient scale disparity that was problematic with SGD in the parametric case.

### Diagnostics

Add a plot of $m_\theta(v)$ vs $m_\text{true}(v)$ over $v \in [-5, 5]$ saved alongside the existing $L(h)$ plot. This is the primary success diagnostic — the $L(h)$ curve can look reasonable even if $m(v)$ is wrong in regions the geodesics don't probe.

## Acceptance criteria

- [x] `test_mass_profile_output_nonneg`: `MassProfile(v)` returns $m \geq 0$ for $v \in [-10, 10]$ with random initialization. **Verified** in smoke test.
- [x] `test_mass_profile_pretrain_recovers_tanh`: after pre-training on the tanh profile, $\max |m_\theta(v) - m_\text{true}(v)| < 0.01$ on a 100-point grid $v \in [-5, 5]$. **Verified**: max error $= 6.4 \times 10^{-3}$ with 2000 steps at lr=3e-3.
- [x] `test_f_metric_grad_through_nn`: `jax.grad(f_metric, argnums=0)` and `jax.grad(f_metric, argnums=1)` return finite values when `model` is a `MassProfile`. **Verified**: $\partial f/\partial r = 4.0$ at $r=2$, $\partial f/\partial v$ finite.
- [~] `test_geodesic_loss_decreases`: loss decreases overall ($2.54 \times 10^{-5} \to 2.46 \times 10^{-5}$) but has a transient bump at step 10 ($5.7 \times 10^{-5}$) due to Adam momentum warmup. Not strictly monotonic in the first 50 steps; monotonic from step 20 onward.
- [x] Visual check (not automatable): saved plot of $m_\theta(v)$ vs $m_\text{true}(v)$ shows qualitative agreement after training. **Verified**: shell transition matches, early-time floor $\approx 0$, late-time $\approx 1.03$.

## Out of scope

- Boundary-controlled data $(\ell, t) \mapsto L_\text{reg}$ — still uses turning-point parameterization $(r_*, v_0)$.
- Multiple probe times — data generated at a single fixed $v_0 = 0$.
- Monotonicity constraint on $m(v)$ — let the unconstrained NN learn it; add a penalty only if needed.
- Adjoint ODE method for memory — using `jax.checkpoint` instead.
- Reducing $\delta$ (soft cutoff width) — keep $\delta = 1.0$ as-is.
- `stop_gradient` on $h_\text{pred}$ — keep it removed (current state); revisit in a later slice if convergence issues appear.

## Notes

- Integration step size changed from `DT=0.002, N_STEPS=20000` to `DT=0.01, N_STEPS=5000` (same affine coverage). The original settings caused JIT compilation + execution to exceed 10 minutes on CPU with the NN; the larger step size makes each training step ~3s.
- Pre-training raised from 500 to 2000 steps at `lr=3e-3` to bring max profile error under 0.01.
- `plt.show()` removed — hangs in headless environments.

## Outcome

- What actually happened: Pipeline works end-to-end. `MassProfile` (2×32 tanh, softplus output) replaces the parametric $m(v)$. `jax.grad` through the NN for $\partial f/\partial r$ and $\partial f/\partial v$ works automatically. `jax.checkpoint` on the RK4 scan step keeps memory bounded. After 2000 pre-training steps and 100 geodesic training steps (Adam, lr=1e-3), the NN recovers the tanh mass profile: shell transition matches, $m(0) \approx 0.49$, $m(5) \approx 1.03$. Geodesic loss: $2.54 \times 10^{-5} \to 2.46 \times 10^{-5}$.
- Surprises: (1) The main bottleneck is runtime, not memory — each gradient step with 20 geodesics × 5000 RK4 steps takes ~3s on CPU. (2) The geodesic loss is already near its floor after pre-training ($2.5 \times 10^{-5}$), so geodesic training provides only mild refinement; the pre-training does the heavy lifting. (3) Adam has a transient loss bump at step 10 before settling — not a bug, just momentum warmup.
- Follow-up (2026-08-21, same day): stress-tested with a *wrong* pre-training profile ($v_s=2.0$, true $0.5$). Geodesic training sharpened the shell transition correctly but $m(v)$ overshot at late times ($m(5)\approx1.77$) because $v_0=0$ geodesics don't probe $v>3$. Fixed by adding an explicit asymptotic loss term anchoring $m_\theta$ to the known $m_f$ at $v\in[4,6]$ (`V_ASYMPTOTIC`, `LAMBDA_ASYM=1.0`). Re-running the wrong-pretrain test with this term: $m(5)\to1.0002$. $m(0)\approx0.61$ (true $0.50$) still shows residual shape error from the wide pretrain — not fully corrected in 100 steps; left for a later slice (multiple probe times $v_0$). See JOURNAL.md, 2026-08-21 entries.
- Deferred: Formal pytest tests for the acceptance criteria (verified manually). Reducing $\delta$. Correcting mid-transition shape error under wrong pretraining (needs multiple probe times).
- Promoted to `decisions.md` / `CLAUDE.md`: None yet.

# Slice 01 — Neural-network mass profile in the differentiable fitting pipeline

Status: **open**
Branch: `slice/01-nn-mass-profile`

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

- [ ] `test_mass_profile_output_nonneg`: `MassProfile(v)` returns $m \geq 0$ for $v \in [-10, 10]$ with random initialization.
- [ ] `test_mass_profile_pretrain_recovers_tanh`: after pre-training on the tanh profile, $\max |m_\theta(v) - m_\text{true}(v)| < 0.01$ on a 100-point grid $v \in [-5, 5]$.
- [ ] `test_f_metric_grad_through_nn`: `jax.grad(f_metric, argnums=0)` and `jax.grad(f_metric, argnums=1)` return finite values when `model` is a `MassProfile`.
- [ ] `test_geodesic_loss_decreases`: training loss decreases monotonically for the first 50 Adam steps starting from the pre-trained initialization.
- [ ] Visual check (not automatable): saved plot of $m_\theta(v)$ vs $m_\text{true}(v)$ shows qualitative agreement after training.

## Out of scope

- Boundary-controlled data $(\ell, t) \mapsto L_\text{reg}$ — still uses turning-point parameterization $(r_*, v_0)$.
- Multiple probe times — data generated at a single fixed $v_0 = 0$.
- Monotonicity constraint on $m(v)$ — let the unconstrained NN learn it; add a penalty only if needed.
- Adjoint ODE method for memory — using `jax.checkpoint` instead.
- Reducing $\delta$ (soft cutoff width) — keep $\delta = 1.0$ as-is.
- `stop_gradient` on $h_\text{pred}$ — keep it removed (current state); revisit in a later slice if convergence issues appear.

## Notes

Working notes during the slice. Disposable.

## Outcome

*Filled at close. Leave empty while the slice is open — an empty Outcome is what marks the slice
as the current one.*

- What actually happened:
- Surprises:
- Deferred:
- Promoted to `decisions.md` / `CLAUDE.md`:

# RESEARCH JOURNAL

> **FILE ROLE:** Chronological log of research discoveries, experimental results, and design decisions.
> **USAGE:** Use this to document *why* certain paths were taken or why specific numerical methods failed.
> **DO NOT:** Use for checklists or architecture specs.

## Log: [YYYY-MM-DD] - [Subject]
**Context:** *What were we trying to investigate?*
**Findings:** *What happened?*
**Decision:** *How did this change our approach?*

---
## Log: 2026-05-13 — L_reg discrete-cutoff error discovered during known-limits validation

**Context:** Ran `scripts/validate_known_limits.py` comparing the Vaidya RK4 solver (m=0 and m=1) against exact analytic results.

**Findings:** Boundary separation $\ell = 2x(r_{\mathrm{cut}})$ is extracted via linear interpolation to exactly $r_{\mathrm{cut}}$ (`interp_to_rcut`), giving sub-nanosecond accuracy ($|\Delta\ell|_{\max} < 5\times10^{-10}$ for empty AdS). However, the regularized length $L_{\mathrm{reg}}$ is computed by `geodesic_length_from_traj`, which truncates the trapezoid integral at the **first discrete trajectory step** satisfying $r \ge r_{\mathrm{cut}}$, not at the interpolated position. This means the length integrand is evaluated at $r_{\mathrm{hit}} \ge r_{\mathrm{cut}}$ rather than exactly $r_{\mathrm{cut}}$, introducing a systematic error of $\sim 4\times10^{-3}$ in $L_{\mathrm{reg}}$ for empty AdS at $r_{\mathrm{cut}}=200$, $\Delta\lambda=0.002$.

**Decision:** The inconsistency between how $\ell$ and $L_{\mathrm{reg}}$ are extracted from the discrete trajectory must be fixed before generating training data. The fix is to: (1) identify the interpolated state at $r_{\mathrm{cut}}$, (2) append it as an extra point to the segment passed to the trapezoid integrator, and (3) use the partial last step $\Delta\lambda_{\mathrm{partial}} = \alpha \cdot \Delta\lambda$. This does not require changing the RK4 integrator itself. Tracked in STATUS.md medium-priority list.

---
## Log: 2026-05-13 — Level 1 turning-grid fit is a bulk-label sanity check, not a boundary inverse problem

**Context:** Implemented `scripts/fit_parametric_vaidya_turning_grid.py` to test whether $(v_c, v_s)$ can be recovered from $L_{\mathrm{reg}}$ data on a grid of turning-point coordinates $(r_\star, v_\star)$.

**Findings:** The optimizer (Nelder-Mead, 110 evaluations) recovered $v_c = -3.37\times10^{-4}$ (true: 0) and $v_s = 0.4998$ (true: 0.5) with a final MSE loss of $3.95\times10^{-28}$ — machine zero. All 72 geodesics were accepted (no cutoff failures). The ~$10^{-4}$ residuals in $(v_c, v_s)$ reflect the discrete-cutoff offset in $L_{\mathrm{reg}}$ that makes the loss slightly non-zero away from the true parameters at the discrete level; the approximate cancellation of this offset in the MSE difference explains why the recovery is so good.

**Decision:** This experiment is a **bulk-label sanity check**, not a true boundary inverse problem. The turning-point coordinates $(r_\star, v_\star)$ are internal bulk quantities — a real observer only has access to the boundary data $(\ell, t_{\mathrm{bdy}}) \mapsto L_{\mathrm{reg}}$. Before this can be called an inverse problem, the workflow must be extended with a shooting/interpolation layer that maps the native solver output $(r_\star, v_\star) \to (\ell, t_{\mathrm{bdy}})$, so that the training grid can be specified in boundary coordinates. The turning-grid result is nonetheless useful: it confirms that the forward model is differentiable enough in $(v_c, v_s)$ for gradient-free optimization, and that there are no spurious local minima over this parameter range.

---
## Log: 2026-08-21 — NN mass profile: wrong pre-training reveals unconstrained overshoot

**Context:** Tested the NN pipeline (`MassProfile` in `scripts/fit_metric_from_lreg.py`) with pre-training on a *wrong* mass profile ($v_s=2.0$, true $v_s=0.5$) to see whether geodesic-based gradients can correct the shape.

**Findings:**
- The geodesic training **does sharpen the shell transition** from the pre-trained profile toward the true one. The $L(h)$ curve converges to the target (loss $5.8\times10^{-4} \to 3.2\times10^{-5}$), confirming that gradients through the NN and RK4 integrator carry real signal.
- However, $m(v)$ **overshoots drastically at late times**: $m(5) \approx 1.77$ (true: 1.0). The NN compensates for the sharper shell by pushing the asymptotic mass up — there is no loss signal to prevent this because geodesics at $v_0=0$ do not probe $v > 3$.
- $m(0) \approx 0.55$ (true: 0.50) — the probed region is close but not exact.

**Decision:** The unconstrained softplus output cannot enforce $m(v) \to m_f$ at late times. The architecture needs a structural constraint. Options: (1) `m(v) = m_max * sigmoid(NN(v))` to cap the output; (2) asymptotic anchoring `m(v) = sigma(v) * NN_bounded(v)` so $m(-\infty)=0$ by construction; (3) integral of a positive function for monotonicity + zero base. This should be addressed in a follow-up slice.

---
---
## Log: 2026-06-03 — Differentiable parametric fitting pipeline: design decisions and bugs

**Context:** Built `scripts/fit_metric_from_lreg.py` to fit a parametric metric $f(r,v;\theta)$ to geodesic data at a fixed probe time $v_0$ using `jax.grad` end-to-end.

**Key design decisions:**

*Soft sigmoid cutoff.* The RK4 integrator runs for a fixed number of steps past $r_\mathrm{cut}$. To extract $L$ and $h$ differentiably, a sigmoid mask $\sigma(-(r-r_\mathrm{cut})/\delta)$ is applied:

$$
L = 2\Delta\lambda\sum_k \dot s_k \cdot \sigma\!\left(-\frac{r_k - r_\mathrm{cut}}{\delta}\right),
\qquad
h = 2\,\frac{\sum_k x_k\,(w_k - w_{k+1})}{\sum_k (w_k - w_{k+1})},
$$

where $w_k = \sigma(-(r_k-r_\mathrm{cut})/\delta)$. The $h$ formula uses the forward difference of the mask as a smooth approximation to a delta function at the crossing, giving a weighted average of $x$ peaked at $r = r_\mathrm{cut}$. Currently $\delta=1.0$; this introduces a small systematic bias relative to the hard-cutoff training data.

*`stop_gradient` on $h_\mathrm{pred}$ in loss.* The loss is $\mathrm{MSE}(L_\mathrm{pred},\, L_\mathrm{target}(h_\mathrm{pred}))$ where $L_\mathrm{target}$ is interpolated from the data. Without `stop_gradient`, the gradient flows through $h_\mathrm{pred}$ into the interpolation lookup, creating a confounding term that moves parameters in the wrong direction. Specifically, when $L_\mathrm{pred} < L_\mathrm{target}$ there is a competing gradient that reduces the loss by shifting $h_\mathrm{pred}$ to a lower-$L$ region of the target curve rather than by increasing $L_\mathrm{pred}$. Applying `jax.lax.stop_gradient(h_\mathrm{pred})` eliminates this path.

*`vmap` + `@jax.jit` for the forward pass.* Replacing the Python loop over $r_\star$ values with `jax.vmap(_integrate_single, in_axes=(0, None, None))` and wrapping the whole forward pass in `@jax.jit` compiles all 20 geodesic integrations and the length/h extraction as a single XLA program. Under `jax.grad` this is also differentiated as one program rather than 20 separate VJPs.

**Bugs found and fixed:**

1. *Wrong sign on $\partial_v f$ in $\ddot r$.* The original `_rhs` had `- 0.5 * df_dv * dv²` where the correct Euler-Lagrange result (and what `Vaidya_AdS.py::get_derivs` computes) is `+ 0.5 * df_dv * dv²`. For a static metric ($\partial_v f = 0$) the error is invisible; for the Vaidya shell crossing it causes wrong trajectories in the forward model while the training data (generated via `get_derivs`) uses the correct equations. This was the primary cause of the model not fitting.

2. *Hardcoded $f_r = 2r$ in $\ddot v$.* `ddv = r*(dx² - dv²)` is only correct when $\partial f/\partial r = 2r$. The general equation is $\ddot v = r\dot x^2 - \tfrac{1}{2}(\partial f/\partial r)\dot v^2$. For the current `f_metric = r^2 - m(v)` these coincide, but the correct general form using `jax.grad` is required for future neural-network $f_\mathrm{metric}$.

3. *`h_\mathrm{data}` stored descending, passed to `jnp.interp` which requires ascending.* `h_\mathrm{data}` is generated by scanning $r_\star$ from small to large; smaller $r_\star$ gives larger $h$, so the array is naturally descending. `jnp.interp(x, xp, fp)` gives silently wrong results for non-ascending `xp`. This invalidated every loss evaluation and gradient. Fixed by sorting `h_\mathrm{data}` and `L_\mathrm{data}` together by ascending `h` in `GeodesicLoss.__init__`.

**Decision:** The pipeline is now structurally correct. Remaining known bias: the soft cutoff ($\delta=1$) underweights $\sim 10$ trajectory steps near $r_\mathrm{cut}=200$, causing a small but nonzero residual at the true parameters. Reduce $\delta$ if this is a problem in practice.

---
## Log: 2026-08-21 — First convergence test of differentiable parametric pipeline

**Context:** Ran `scripts/fit_metric_from_lreg.py` with optax SGD (LR=3.0, 20 steps) to test whether the end-to-end differentiable pipeline converges toward the true Vaidya parameters $(m_f=1, v_c=0, v_s=0.5)$ from initial guess $(m_f=0.8, v_c=0.3, v_s=0.8)$. This is the first convergence run since the three bugs were fixed on 2026-06-03.

**Changes from prior version:**
- Replaced manual gradient descent with `optax.sgd(3.0)`.
- Removed `jax.lax.stop_gradient` on `h_\mathrm{pred}` in the loss (line 189–190 in script). Without `stop_gradient`, gradients also flow through the $h$ interpolation lookup, which was previously flagged as potentially confounding (see 2026-06-03 entry). The current run suggests the effect is not catastrophic, but may be contributing to slow convergence on $v_s$.
- Initial parameters moved closer to true values: $(m_f, v_c, v_s) = (0.8, 0.3, 0.8)$ instead of previous $(0.4, -0.5, 0.8)$.

**Results (20 SGD steps, LR=3.0):**

| Parameter | True | Initial | After 20 steps | Recovery |
|-----------|------|---------|----------------|----------|
| $m_f$     | 1.0  | 0.8     | 0.9961         | ~99.6%   |
| $v_c$     | 0.0  | 0.3     | 0.1310         | ~56%     |
| $v_s$     | 0.5  | 0.8     | 0.7701         | ~10%     |

Loss: $1.48 \times 10^{-3} \to 2.77 \times 10^{-4}$ (5.3× reduction).

$m_f$ converges quickly. $v_c$ moves in the right direction but slowly. $v_s$ barely moves — this parameter controls the shell thickness and couples nonlinearly; its gradient magnitude is likely much smaller than $m_f$'s at this probe time $v_0=0$.

**Assessment:**
- The pipeline is structurally functional: gradients flow, the loss decreases monotonically, and all three parameters move in the correct direction.
- The convergence is slow, especially for $v_s$. Likely causes: (a) per-parameter gradient scale disparity — Adam or per-parameter LR scheduling would help; (b) SGD at LR=3.0 may be too conservative for $v_s$ while already near the edge for $m_f$; (c) soft cutoff bias ($\delta=1$) introduces a systematic floor; (d) the confounding $h_\mathrm{pred}$ gradient path (now active without `stop_gradient`) could compete.
- Only 20 steps were run; the loss is still decreasing and has not plateaued. Running more steps or switching to Adam should show further progress.

**Decision:** The pipeline works for parametric fitting. Next steps: (1) try Adam to handle gradient scale disparity, (2) run for more steps to check convergence to true values, (3) re-enable `stop_gradient` to compare, (4) reduce $\delta$ if residual at true params is too large. Once the parametric fit reliably converges, replace `f_metric` with an Equinox neural network.

---
## Archive of Failed Attempts
*Crucial for AI context: Listing what didn't work prevents Claude from suggesting it again.*

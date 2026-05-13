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
## Archive of Failed Attempts
*Crucial for AI context: Listing what didn't work prevents Claude from suggesting it again.*

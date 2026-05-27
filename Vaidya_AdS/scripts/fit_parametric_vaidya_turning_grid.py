"""
Parametric Vaidya fit: recover shell center v_c and thickness v_s
from a turning-grid loss on L_reg.

This is a Level 1 sanity check.  The same turning-point (r★, v★) grid
is used for both target and prediction, so bulk labels are assumed.
This is NOT boundary-only reconstruction — see JOURNAL.md for context.

Run from the project root:
    python scripts/fit_parametric_vaidya_turning_grid.py
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# ── path / JAX setup ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Vaidya_AdS import (
    integrate_geodesic,
    geodesic_length_from_traj,
    geodesic_length_reg,
    ds_dlambda,
)

# =============================================================================
# Configuration — edit here to change the experiment
# =============================================================================

# True mass profile (vacuum-to-BTZ collapse, m_i=0, m_f=1 per SPECS.md)
TRUE_V_C = 0.0
TRUE_V_S = 0.5
M_I = 0.0
M_F = 1.0

# Turning-point grid
# r★ starts above BTZ horizon (r_h = 1) to avoid trapped geodesics at late v★
R_STAR_GRID = np.linspace(1.05, 5.0, 8)
V_STAR_GRID = np.linspace(-1.2, 1.2, 9)
# Full grid alternative (slower, ~2× more geodesics):
# R_STAR_GRID = np.linspace(1.05, 6.0, 12)
# V_STAR_GRID = np.linspace(-1.5, 1.5, 13)

# Solver settings  (consistent with validate_known_limits.py)
R_CUT   = 200.0
DT      = 0.002
N_STEPS = 40000

# Optimizer initial guess — intentionally away from truth
INITIAL_V_C = 0.3
INITIAL_V_S = 0.8

# Output directory
OUT_DIR = _ROOT / "inverse_results"

# =============================================================================
# Low-level helpers (consistent with validate_known_limits.py conventions)
# =============================================================================

def _first_cutoff_index(traj: np.ndarray, r_cut: float):
    """Return first trajectory index where r >= r_cut, or None."""
    mask = traj[:, 1] >= r_cut
    return int(np.argmax(mask)) if np.any(mask) else None


def _interp_to_rcut(traj: np.ndarray, r_cut: float):
    """
    Linearly interpolate trajectory state to exactly r = r_cut.
    Returns (state, index_before_cutoff) or (None, None) if cutoff not reached.
    """
    idx = _first_cutoff_index(traj, r_cut)
    if idx is None:
        return None, None
    if idx == 0:
        return traj[0].copy(), 0
    i0, i1 = idx - 1, idx
    r0, r1 = float(traj[i0, 1]), float(traj[i1, 1])
    alpha = (r_cut - r0) / (r1 - r0)          # linear weight ∈ (0, 1]
    return traj[i0] + alpha * (traj[i1] - traj[i0]), idx


def _max_kappa_deviation(traj: np.ndarray, hit_idx: int,
                          m_i: float, m_f: float, v_c: float, v_s: float) -> float:
    """
    Max |κ − 1| along the accepted segment [0 : hit_idx+1].

    ds_dlambda returns √κ, so κ = ds_dlambda² and the deviation is |κ − 1|.
    For a correctly normalized geodesic κ = 1 everywhere; deviations are
    purely numerical (RK4 drift).
    """
    seg = jnp.array(traj[: hit_idx + 1])
    sdot = np.array(
        jax.vmap(lambda s: ds_dlambda(s, m_i=m_i, m_f=m_f, v_c=v_c, v_s=v_s))(seg)
    )
    return float(np.max(np.abs(sdot ** 2 - 1.0)))


# =============================================================================
# Dataset generation
# =============================================================================

def generate_turning_grid_dataset(
    r_star_grid,
    v_star_grid,
    *,
    m_i: float,
    m_f: float,
    v_c: float,
    v_s: float,
    r_cut: float,
    dt: float,
    n_steps: int,
    verbose: bool = True,
) -> list:
    """
    Run the geodesic solver for every (r★, v★) on the Cartesian product grid.

    Each record contains:
        r_star, v_star           — turning-point coordinates (bulk labels)
        ell                      — boundary interval  ℓ = 2 x(r_cut)
        t_boundary               — boundary time ≈ v(r_cut)
        L_reg                    — UV-regularized geodesic length
        cutoff_hit               — bool: did r reach r_cut?
        max_spacelike_norm_deviation — max |κ−1| along accepted segment
        m_i, m_f, v_c, v_s      — mass profile parameters used
    """
    grid = [(float(r), float(v)) for r in r_star_grid for v in v_star_grid]
    n_total = len(grid)
    records = []
    n_fail = 0
    t0 = time.perf_counter()

    for k, (r_star, v_star) in enumerate(grid):
        traj = np.array(integrate_geodesic(
            r_star, v_star,
            n_steps=int(n_steps), dt=float(dt),
            m_i=m_i, m_f=m_f, v_c=v_c, v_s=v_s,
        ))

        state_cut, hit_idx = _interp_to_rcut(traj, r_cut)
        cutoff_hit = state_cut is not None

        if cutoff_hit:
            ell       = 2.0 * float(state_cut[2])       # ℓ = 2 x(r_cut)
            t_bdy     = float(state_cut[0])              # v at UV boundary
            L_full    = float(geodesic_length_from_traj(
                jnp.array(traj), dt=dt, r_cut=r_cut,
                m_i=m_i, m_f=m_f, v_c=v_c, v_s=v_s,
            ))
            L_reg_val = float(geodesic_length_reg(L_full, r_cut))
            kappa_dev = _max_kappa_deviation(traj, hit_idx, m_i, m_f, v_c, v_s)
        else:
            ell = t_bdy = L_reg_val = kappa_dev = np.nan
            n_fail += 1

        records.append({
            "r_star":                    r_star,
            "v_star":                    v_star,
            "ell":                       ell,
            "t_boundary":                t_bdy,
            "L_reg":                     L_reg_val,
            "cutoff_hit":                cutoff_hit,
            "max_spacelike_norm_deviation": kappa_dev,
            "m_i": m_i, "m_f": m_f, "v_c": v_c, "v_s": v_s,
        })

        if verbose and (k + 1) % max(1, n_total // 10) == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (k + 1) * (n_total - k - 1)
            print(f"  [{k+1:3d}/{n_total}]  {elapsed:.1f}s  ETA {eta:.0f}s  "
                  f"failures: {n_fail}")

    if verbose:
        print(f"  Done: {n_total} geodesics in {time.perf_counter()-t0:.1f}s  "
              f"({n_fail} cutoff failures)")
    return records


# =============================================================================
# Parameter packing / unpacking
# =============================================================================

def pack_params(v_c: float, v_s: float) -> np.ndarray:
    """Optimizer variables: q = (v_c, log v_s) so that v_s > 0 is implicit."""
    return np.array([v_c, np.log(v_s)])


def unpack_params(q) -> tuple:
    """(v_c, log v_s) → (v_c, v_s)."""
    return float(q[0]), float(np.exp(q[1]))


# =============================================================================
# Loss function
# =============================================================================

def make_loss(target_records, *, r_cut: float, dt: float, n_steps: int,
              m_i: float, m_f: float):
    """
    Build and return a scipy-compatible loss function L(q).

    Loss:
        L(v_c, v_s) = (1/N) Σ_i [ L_reg^pred(r★_i, v★_i) − L_reg^target(r★_i, v★_i) ]²

    Sum is over all (r★, v★) where BOTH target and prediction reached r_cut.

    The discrete-cutoff error in L_reg (≈4e-3, see JOURNAL.md) approximately
    cancels in this difference because both sides use identical cutoff code.

    Returns (loss_fn, eval_counter, n_target_accepted).
    """
    # Pre-extract accepted target samples once — these are fixed throughout optimization
    accepted = [
        (r["r_star"], r["v_star"], r["L_reg"])
        for r in target_records
        if r["cutoff_hit"]
    ]
    n_target_accepted = len(accepted)
    eval_counter = [0]

    def loss(q):
        v_c, v_s = unpack_params(q)
        sse   = 0.0
        n_used = 0

        for r_star, v_star, L_reg_target in accepted:
            traj = np.array(integrate_geodesic(
                r_star, v_star,
                n_steps=int(n_steps), dt=float(dt),
                m_i=m_i, m_f=m_f, v_c=v_c, v_s=v_s,
            ))
            if not np.any(traj[:, 1] >= r_cut):
                continue   # prediction missed cutoff — exclude from loss
            L_full    = float(geodesic_length_from_traj(
                jnp.array(traj), dt=dt, r_cut=r_cut,
                m_i=m_i, m_f=m_f, v_c=v_c, v_s=v_s,
            ))
            L_reg_pred = float(geodesic_length_reg(L_full, r_cut))
            sse   += (L_reg_pred - L_reg_target) ** 2
            n_used += 1

        eval_counter[0] += 1
        val = sse / n_used if n_used > 0 else 1e10
        print(
            f"  eval={eval_counter[0]:04d}  v_c={v_c:+.4f}  v_s={v_s:.4f}  "
            f"loss={val:.3e}  n_used={n_used}/{n_target_accepted}"
        )
        return val

    return loss, eval_counter, n_target_accepted


# =============================================================================
# Plots
# =============================================================================

def save_plots(target_records, fit_records, truth: dict, fit_params: dict,
               out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v_arr = np.linspace(-3.5, 3.5, 400)

    def mass_profile(v, v_c, v_s):
        return 0.5 * (1.0 + np.tanh((v - v_c) / v_s))

    # ── mass profile ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v_arr, mass_profile(v_arr, truth["v_c"],       truth["v_s"]),
            "k-",  lw=2.5,
            label=f"true   $v_c={truth['v_c']:.3f}$, $v_s={truth['v_s']:.3f}$")
    ax.plot(v_arr, mass_profile(v_arr, fit_params["v_c"],  fit_params["v_s"]),
            "r--", lw=2.5,
            label=f"fitted $v_c={fit_params['v_c']:.3f}$, $v_s={fit_params['v_s']:.3f}$")
    ax.axhline(0.5, color="gray", lw=1, ls=":")
    ax.set_xlabel("$v$")
    ax.set_ylabel("$m(v)$")
    ax.set_title("Mass profile: truth vs fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mass_profile_fit.png", dpi=150)
    plt.close(fig)

    # ── build matched (target, prediction) L_reg pairs ───────────────────────
    key_to_Lreg_target = {
        (r["r_star"], r["v_star"]): r["L_reg"]
        for r in target_records if r["cutoff_hit"]
    }
    matched_target = []
    matched_pred   = []
    for r in fit_records:
        if not r["cutoff_hit"]:
            continue
        key = (r["r_star"], r["v_star"])
        if key in key_to_Lreg_target:
            matched_target.append(key_to_Lreg_target[key])
            matched_pred.append(r["L_reg"])

    t_Lr = np.array(matched_target)
    f_Lr = np.array(matched_pred)

    # ── L_reg scatter ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(t_Lr, f_Lr, s=25, alpha=0.8)
    lo = min(t_Lr.min(), f_Lr.min()) - 0.15
    hi = max(t_Lr.max(), f_Lr.max()) + 0.15
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal (y=x)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("$L_{\\rm reg}$ (target)")
    ax.set_ylabel("$L_{\\rm reg}$ (fitted prediction)")
    ax.set_title("$L_{\\rm reg}$: target vs fitted prediction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "target_vs_prediction_Lreg.png", dpi=150)
    plt.close(fig)

    # ── residuals ─────────────────────────────────────────────────────────────
    res = f_Lr - t_Lr
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.scatter(t_Lr, res, s=25, alpha=0.8)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("$L_{\\rm reg}$ (target)")
    ax.set_ylabel("residual (fit $-$ target)")
    ax.set_title(f"$L_{{\\rm reg}}$ residuals  "
                 f"(max $|\\Delta|$ = {np.max(np.abs(res)):.2e})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "residuals_Lreg.png", dpi=150)
    plt.close(fig)

    print(f"  mass_profile_fit.png, target_vs_prediction_Lreg.png, "
          f"residuals_Lreg.png  →  {out_dir}/")


# =============================================================================
# CSV helper
# =============================================================================

def _save_csv(records: list, path: Path):
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


# =============================================================================
# Main
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_grid = len(R_STAR_GRID) * len(V_STAR_GRID)
    print(f"Grid: {len(R_STAR_GRID)} × {len(V_STAR_GRID)} = {n_grid} turning points")
    print(f"Solver: n_steps={N_STEPS}, dt={DT}, r_cut={R_CUT}")

    # ── 1. JIT warmup ─────────────────────────────────────────────────────────
    # First call to integrate_geodesic triggers JAX compilation (~10-30 s once).
    print("\nWarming up JAX JIT (first geodesic integration) …")
    t_warm = time.perf_counter()
    _ = integrate_geodesic(
        float(R_STAR_GRID[0]), 0.0,
        n_steps=int(N_STEPS), dt=float(DT),
        m_i=M_I, m_f=M_F, v_c=TRUE_V_C, v_s=TRUE_V_S,
    )
    print(f"  JIT warmup done in {time.perf_counter()-t_warm:.1f}s")

    # ── 2. Generate target data ───────────────────────────────────────────────
    print(f"\n=== Generating target data  "
          f"(true v_c={TRUE_V_C}, v_s={TRUE_V_S}) ===")
    t0 = time.perf_counter()
    target_records = generate_turning_grid_dataset(
        R_STAR_GRID, V_STAR_GRID,
        m_i=M_I, m_f=M_F, v_c=TRUE_V_C, v_s=TRUE_V_S,
        r_cut=R_CUT, dt=DT, n_steps=N_STEPS,
    )
    n_target_fail = sum(1 for r in target_records if not r["cutoff_hit"])
    n_target_ok   = len(target_records) - n_target_fail
    print(f"  {n_target_ok} accepted, {n_target_fail} cutoff failures, "
          f"elapsed {time.perf_counter()-t0:.1f}s")

    csv_target = OUT_DIR / "parametric_vaidya_turning_grid_target.csv"
    _save_csv(target_records, csv_target)
    print(f"  Target CSV → {csv_target}")

    if n_target_ok == 0:
        print("ERROR: all target geodesics failed to reach r_cut. "
              "Increase N_STEPS or reduce R_CUT.")
        return

    # ── 3. Optimize ───────────────────────────────────────────────────────────
    loss_fn, eval_counter, n_target_accepted = make_loss(
        target_records,
        r_cut=R_CUT, dt=DT, n_steps=N_STEPS, m_i=M_I, m_f=M_F,
    )

    q0 = pack_params(INITIAL_V_C, INITIAL_V_S)
    print(f"\n=== Optimization  "
          f"(initial v_c={INITIAL_V_C}, v_s={INITIAL_V_S}) ===")
    print(f"  Using {n_target_accepted} target samples in loss")
    t_opt = time.perf_counter()
    result = minimize(
        loss_fn, q0,
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-9, "maxiter": 600, "disp": False},
    )
    opt_elapsed = time.perf_counter() - t_opt

    v_c_fit, v_s_fit = unpack_params(result.x)
    dv_c = abs(v_c_fit - TRUE_V_C)
    dv_s = abs(v_s_fit - TRUE_V_S)

    print(f"\n=== Fit result ===")
    print(f"  True:    v_c = {TRUE_V_C:+.5f}   v_s = {TRUE_V_S:.5f}")
    print(f"  Fit:     v_c = {v_c_fit:+.5f}   v_s = {v_s_fit:.5f}")
    print(f"  Error:   Δv_c = {dv_c:.2e}   Δv_s = {dv_s:.2e}")
    print(f"  Loss:    {result.fun:.4e}")
    print(f"  Evals:   {eval_counter[0]}")
    print(f"  Success: {result.success}  ({result.message})")
    print(f"  Time:    {opt_elapsed:.1f}s")

    # ── 4. Generate prediction dataset at fitted parameters ───────────────────
    print("\n=== Generating prediction dataset at fitted parameters ===")
    fit_records = generate_turning_grid_dataset(
        R_STAR_GRID, V_STAR_GRID,
        m_i=M_I, m_f=M_F, v_c=v_c_fit, v_s=v_s_fit,
        r_cut=R_CUT, dt=DT, n_steps=N_STEPS,
        verbose=False,
    )
    n_pred_fail = sum(1 for r in fit_records if not r["cutoff_hit"])

    csv_pred = OUT_DIR / "parametric_vaidya_turning_grid_prediction_fit.csv"
    _save_csv(fit_records, csv_pred)
    print(f"  {len(fit_records) - n_pred_fail} accepted, "
          f"{n_pred_fail} cutoff failures")
    print(f"  Prediction CSV → {csv_pred}")

    # Number of samples used in the final loss (both target and prediction accepted)
    target_ok_keys = {(r["r_star"], r["v_star"]) for r in target_records if r["cutoff_hit"]}
    pred_ok_keys   = {(r["r_star"], r["v_star"]) for r in fit_records   if r["cutoff_hit"]}
    n_used_final   = len(target_ok_keys & pred_ok_keys)

    # ── 5. Save JSON report ───────────────────────────────────────────────────
    report = {
        "truth":         {"m_i": M_I, "m_f": M_F, "v_c": TRUE_V_C, "v_s": TRUE_V_S},
        "initial_guess": {"v_c": INITIAL_V_C, "v_s": INITIAL_V_S},
        "fit": {
            "v_c":              v_c_fit,
            "v_s":              v_s_fit,
            "error_v_c":        dv_c,
            "error_v_s":        dv_s,
            "loss":             float(result.fun),
            "success":          bool(result.success),
            "message":          str(result.message),
            "n_optimizer_evals": int(eval_counter[0]),
            "optimization_time_s": opt_elapsed,
        },
        "grid": {
            "n_r_star":   len(R_STAR_GRID),
            "n_v_star":   len(V_STAR_GRID),
            "r_star_min": float(R_STAR_GRID[0]),
            "r_star_max": float(R_STAR_GRID[-1]),
            "v_star_min": float(V_STAR_GRID[0]),
            "v_star_max": float(V_STAR_GRID[-1]),
        },
        "solver": {"r_cut": R_CUT, "dt": DT, "n_steps": N_STEPS},
        "diagnostics": {
            "n_target_samples":                   len(target_records),
            "n_target_accepted":                  n_target_ok,
            "n_cutoff_failures_target":           n_target_fail,
            "n_used_in_loss":                     n_used_final,
            "n_cutoff_failures_prediction_at_fit": n_pred_fail,
        },
    }

    json_path = OUT_DIR / "parametric_vaidya_turning_grid_fit.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report → {json_path}")

    # ── 6. Save plots ─────────────────────────────────────────────────────────
    try:
        save_plots(
            target_records, fit_records,
            truth={"v_c": TRUE_V_C, "v_s": TRUE_V_S},
            fit_params={"v_c": v_c_fit, "v_s": v_s_fit},
            out_dir=OUT_DIR,
        )
    except Exception as exc:
        print(f"Warning: plot generation failed ({exc})")


if __name__ == "__main__":
    main()

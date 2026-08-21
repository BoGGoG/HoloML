import os, sys, json
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import optax

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Vaidya_AdS import integrate_geodesic, geodesic_length_from_traj

# ── Constants ──────────────────────────────────────────────────────────────────

V0 = 0.0
M_I, M_F = 0.0, 1.0
V_C, V_S = 0.0, 0.5
R_STARS = np.linspace(1.1, 4.0, 20)
R_CUT = 200.0
N_STEPS = 20000
DT = 0.002

OUT_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Data I/O ───────────────────────────────────────────────────────────────────


def generate_data(r_stars, v0, m_i, m_f, v_c, v_s, r_cut, n_steps, dt):
    r_out, h_out, L_out = [], [], []
    for r_star in r_stars:
        traj = np.array(
            integrate_geodesic(
                float(r_star),
                v0,
                n_steps=n_steps,
                dt=dt,
                m_i=m_i,
                m_f=m_f,
                v_c=v_c,
                v_s=v_s,
            )
        )
        hit = next((k for k in range(1, len(traj)) if traj[k, 1] >= r_cut), None)
        if hit is None:
            print(f"skipping r_star={r_star:.2f}: did not reach r_cut")
            continue
        L = float(
            geodesic_length_from_traj(
                traj, dt, r_cut=r_cut, m_i=m_i, m_f=m_f, v_c=v_c, v_s=v_s
            )
        )
        r_out.append(float(r_star))
        h_out.append(float(2.0 * traj[hit, 2]))
        L_out.append(L)
    return np.array(r_out), np.array(h_out), np.array(L_out)


def save_data(path, v0, m_i, m_f, v_c, v_s, r_cut, n_steps, dt, r_data, h_data, L_data):
    with open(path, "w") as f:
        json.dump(
            {
                "v0": v0,
                "m_i": m_i,
                "m_f": m_f,
                "v_c": v_c,
                "v_s": v_s,
                "r_cut": r_cut,
                "n_steps": n_steps,
                "dt": dt,
                "r_star": r_data.tolist(),
                "h": h_data.tolist(),
                "L": L_data.tolist(),
            },
            f,
            indent=2,
        )


def load_data(path):
    with open(path) as f:
        d = json.load(f)
    return (
        d["v0"],
        d["m_i"],
        d["m_f"],
        d["v_c"],
        d["v_s"],
        d["r_cut"],
        d["n_steps"],
        d["dt"],
        np.array(d["r_star"]),
        np.array(d["h"]),
        np.array(d["L"]),
    )


# ── Forward model ──────────────────────────────────────────────────────────────
#
# Metric: ds² = -f(r,v; params) dv² + 2 dv dr + r² dx²
#
# integrate_geodesic hard-codes f = r²-m(v) with analytic derivatives.
# Here we use jax.grad to get metric derivatives for any differentiable f.


def f_metric(r, v, params):
    """f(r,v) in the metric. Replace the body with a neural network later."""
    m = 0.5 * params["m_f"] * (1.0 + jnp.tanh((v - params["v_c"]) / params["v_s"]))
    return r**2 - m


_df_dr = jax.grad(f_metric, argnums=0)
_df_dv = jax.grad(f_metric, argnums=1)


def _rhs(state, params):
    v, r, x, dv, dr, dx = state
    f = f_metric(r, v, params)
    df_dr = _df_dr(r, v, params)
    df_dv = _df_dv(r, v, params)
    ddv = r * dx**2 - 0.5 * df_dr * dv**2
    ddr = f * ddv + df_dr * dr * dv + 0.5 * df_dv * dv**2
    ddx = -2.0 / r * dr * dx
    return jnp.array([dv, dr, dx, ddv, ddr, ddx])


def _integrate_single(r_star, v0, params):
    """Integrate one geodesic. N_STEPS and DT captured from module scope."""
    s0 = jnp.array([v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star])

    def step(s, _):
        k1 = _rhs(s, params)
        k2 = _rhs(s + 0.5 * DT * k1, params)
        k3 = _rhs(s + 0.5 * DT * k2, params)
        k4 = _rhs(s + DT * k3, params)
        s2 = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s2, s2

    _, traj = jax.lax.scan(step, s0, None, length=N_STEPS)
    return traj  # (N_STEPS, 6): v, r, x, dv, dr, dx


_integrate_batch = jax.vmap(_integrate_single, in_axes=(0, None, None))


def length_and_h_from_traj(traj, params, r_cut=R_CUT, delta=1.0):
    vs, rs, xs, dv, dr, dx = traj.T
    f = jax.vmap(lambda vi, ri: f_metric(ri, vi, params))(vs, rs)
    sdot = jnp.sqrt(jnp.maximum(-f * dv**2 + 2 * dv * dr + rs**2 * dx**2, 0.0))

    # Smooth cutoff: ~1 before r_cut, ~0 after, transition width delta
    mask = jax.nn.sigmoid(-(rs - r_cut) / delta)
    L = 2.0 * DT * jnp.sum(sdot * mask)

    # h: x at r=r_cut, approximated as mask-difference-weighted average of x
    weight = mask[:-1] - mask[1:]  # smooth bump peaked at the crossing
    h = 2.0 * jnp.dot(xs[:-1], weight) / jnp.sum(weight)

    return L, h


@jax.jit
def forward(params, r_stars, v0):
    traj_batch = _integrate_batch(r_stars, v0, params)  # (n_r, N_STEPS, 6)
    L_arr, h_arr = jax.vmap(lambda traj: length_and_h_from_traj(traj, params))(
        traj_batch
    )
    return h_arr, L_arr


class GeodesicLoss:
    """MSE loss comparing predicted L(h) against the target curve.

    Builds an interpolant of (h_data, L_data) once at construction.
    Each call evaluates it at h_pred and returns the MSE against L_pred.
    h_data must be sorted ascending.
    """

    def __init__(self, h_data, L_data):
        order = jnp.argsort(jnp.array(h_data))
        self._h_data = jnp.array(h_data)[order]
        self._L_data = jnp.array(L_data)[order]

    def __call__(self, h_pred, L_pred):
        # L_target = jnp.interp(jax.lax.stop_gradient(h_pred), self._h_data, self._L_data)
        L_target = jnp.interp(h_pred, self._h_data, self._L_data)
        return jnp.mean((L_pred - L_target) ** 2)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "l_vs_rstar.json"

    r_data, h_data, L_data = generate_data(
        R_STARS, V0, M_I, M_F, V_C, V_S, R_CUT, N_STEPS, DT
    )
    print(f"Generated {len(r_data)} geodesics")
    save_data(
        out_path, V0, M_I, M_F, V_C, V_S, R_CUT, N_STEPS, DT, r_data, h_data, L_data
    )
    print(f"Saved to {out_path}")

    v0, m_i, m_f, v_c, v_s, r_cut, n_steps, dt, r_data, h_data, L_data = load_data(
        out_path
    )

    params_init = {"m_f": jnp.array(0.8), "v_c": jnp.array(0.3), "v_s": jnp.array(0.8)}
    print("True parameters:   ", {"m_f": m_f, "v_c": v_c, "v_s": v_s})
    print("Initial parameters:", params_init)

    h_pred, L_pred = forward(params_init, r_data, v0)
    print("Target h:    ", np.round(h_data, 4))
    print("Predicted h: ", np.round(h_pred, 4))
    print("Target L:    ", np.round(L_data, 4))
    print("Predicted L: ", np.round(L_pred, 4))
    loss = GeodesicLoss(h_data, L_data)
    print("Initial loss:", loss(h_pred, L_pred))

    def scalar_loss(params):
        h_p, L_p = forward(params, r_data, v0)
        return loss(h_p, L_p)

    grad_fn = jax.jit(jax.grad(scalar_loss))
    print("Warming up grad_fn (JIT compilation)...")
    _ = grad_fn(params_init)
    print("Done.")

    # ── Optimisation ──────────────────────────────────────────────────────────

    LR = 3e-0
    N_GD_STEPS = 20

    optimizer = optax.sgd(LR)  # vanilla gradient descent
    # optimizer = optax.adam(LR)       # Adam
    params = params_init
    opt_state = optimizer.init(params)

    print(f"\n{'step':>4}  {'loss':>12}  {'m_f':>8}  {'v_c':>8}  {'v_s':>8}")
    print(f"{'true':>4}  {'-':>12}  {m_f:>8.4f}  {v_c:>8.4f}  {v_s:>8.4f}")
    print("-" * 50)
    for step in range(N_GD_STEPS):
        loss_val = float(scalar_loss(params))
        grads = grad_fn(params)
        print(
            f"{step:>4}  {loss_val:>12.6f}"
            f"  {float(params['m_f']):>8.4f}"
            f"  {float(params['v_c']):>8.4f}"
            f"  {float(params['v_s']):>8.4f}"
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    print(f"\nTrue parameters: m_f={m_f:.4f}, v_c={v_c:.4f}, v_s={v_s:.4f}")

    # ── Plot: initial vs final prediction ─────────────────────────────────────

    h_final, L_final = forward(params, r_data, v0)

    fig, ax = plt.subplots()
    ax.plot(h_data, L_data, "o-", label="Target")
    ax.plot(np.array(h_pred), np.array(L_pred), "o--", label="Initial guess")
    ax.plot(
        np.array(h_final), np.array(L_final), "o--", label=f"After {N_GD_STEPS} steps"
    )
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$L$")
    ax.set_title("Gradient descent progress")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "l_vs_h_gd.png", dpi=150)
    plt.show()

import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from Vaidya_AdS import (
    integrate_geodesic,
    geodesic_length_from_traj,
    geodesic_length_reg,
)

# ── Parameters ─────────────────────────────────────────────────────────────────

V0 = 0.0
M_I, M_F = 0.0, 1.0
V_C, V_S = 0.0, 0.5
R_STARS = np.linspace(1.1, 4.0, 20)
R_CUT = 200.0
N_STEPS = 20000
DT = 0.002

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Data generation ─────────────────────────────────────────────────────────

r_out, L_out = [], []
for r_star in R_STARS:
    traj = np.array(
        integrate_geodesic(
            float(r_star),
            V0,
            n_steps=N_STEPS,
            dt=DT,
            m_i=M_I,
            m_f=M_F,
            v_c=V_C,
            v_s=V_S,
        )
    )
    hit = next((k for k in range(1, len(traj)) if traj[k, 1] >= R_CUT), None)
    if hit is None:
        print(f"skipping r_star={r_star:.2f}: did not reach r_cut")
        continue
    L = float(
        geodesic_length_from_traj(
            traj, DT, r_cut=R_CUT, m_i=M_I, m_f=M_F, v_c=V_C, v_s=V_S
        )
    )
    r_out.append(float(r_star))
    # L_out.append(float(geodesic_length_reg(L, R_CUT)))
    L_out.append(L)

r_data = np.array(r_out)
L_data = np.array(L_out)
print(f"Generated {len(r_data)} geodesics")

# ── 2. Export ──────────────────────────────────────────────────────────────────

out_path = OUT_DIR / "l_vs_rstar.json"
with open(out_path, "w") as f:
    json.dump(
        {
            "v0": V0,
            "m_i": M_I,
            "m_f": M_F,
            "v_c": V_C,
            "v_s": V_S,
            "r_cut": R_CUT,
            "n_steps": N_STEPS,
            "dt": DT,
            "r_star": r_out,
            "L": L_out,
        },
        f,
        indent=2,
    )
print(f"Saved to {out_path}")

# ── 3. Plot ────────────────────────────────────────────────────────────────────

# fig, ax = plt.subplots()
# ax.plot(r_data, L_data, "o-")
# ax.set_xlabel(r"$r_\star$")
# ax.set_ylabel(r"$L$")
# ax.set_title(rf"$v_0={V0},\; m_i={M_I},\; m_f={M_F},\; v_c={V_C},\; v_s={V_S}$")
# plt.tight_layout()
# plt.savefig(OUT_DIR / "l_vs_rstar.png", dpi=150)
# plt.show()


# -- 4. Read in data again (let's do it like this to later not have to figure out how to do it)
V0, M_I, M_F, V_C, V_S, R_CUT, N_STEPS, DT = (
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
)
with open(out_path, "r") as f:
    data = json.load(f)
    V0 = data["v0"]
    M_I = data["m_i"]
    M_F = data["m_f"]
    V_C = data["v_c"]
    V_S = data["v_s"]
    R_CUT = data["r_cut"]
    N_STEPS = data["n_steps"]
    DT = data["dt"]


# -- 5. Forward model: Geodesic integration with a parameterized metric
#
# Metric: ds² = -f(r,v; params) dv² + 2 dv dr + r² dx²
#
# integrate_geodesic in Vaidya_AdS.py has the same RK4/lax.scan loop,
# but its RHS hard-codes f = r²-m(v) with analytic derivatives df/dr=2r,
# df/dv=-dm/dv.  Here we keep the identical loop and replace only the RHS,
# using jax.grad to get the metric derivatives for any f.

from functools import partial


def f_metric(r, v, params):
    """f(r,v) in the metric.  Replace the body with a neural network later."""
    m = 0.5 * params["m_f"] * (1.0 + jnp.tanh((v - params["v_c"]) / params["v_s"]))
    return r**2 - m


_df_dr = jax.grad(f_metric, argnums=0)
_df_dv = jax.grad(f_metric, argnums=1)


def _rhs(state, params):
    v, r, x, dv, dr, dx = state
    f = f_metric(r, v, params)
    ddv = r * (dx**2 - dv**2)
    ddr = f * ddv + _df_dr(r, v, params) * dr * dv - 0.5 * _df_dv(r, v, params) * dv**2
    ddx = -2.0 / r * dr * dx
    return jnp.array([dv, dr, dx, ddv, ddr, ddx])


@partial(jax.jit, static_argnums=(3, 4))
def integrate_param(r_star, v0, params, n_steps=N_STEPS, dt=DT):
    s0 = jnp.array([v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star])

    def step(s, _):
        k1 = _rhs(s, params)
        k2 = _rhs(s + 0.5 * dt * k1, params)
        k3 = _rhs(s + 0.5 * dt * k2, params)
        k4 = _rhs(s + dt * k3, params)
        s2 = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s2, s2

    _, traj = jax.lax.scan(step, s0, None, length=n_steps)
    return traj  # (n_steps, 6)  columns: v, r, x, dv, dr, dx


def length_from_traj(traj, params, dt=DT, r_cut=R_CUT):
    r = traj[:, 1]
    hit = next((k for k in range(len(r)) if r[k] >= r_cut), len(r) - 1)
    seg = traj[: hit + 1]
    vs, rs, _, dv, dr, dx = seg.T
    f = jax.vmap(lambda vi, ri: f_metric(ri, vi, params))(vs, rs)
    sdot = jnp.sqrt(jnp.maximum(-f * dv**2 + 2 * dv * dr + rs**2 * dx**2, 0.0))
    return float(2.0 * dt * jnp.sum(sdot))


def forward(params, r_stars):
    return np.array(
        [length_from_traj(integrate_param(r, V0, params), params) for r in r_stars]
    )


params_init = {"m_f": jnp.array(0.2), "v_c": jnp.array(-0.5), "v_s": jnp.array(0.6)}
print("True parameters:   ", {"m_f": M_F, "v_c": V_C, "v_s": V_S})
print("Initial parameters:", params_init)
L_pred = forward(params_init, r_data)
print("Target:    ", np.round(L_data, 4))
print("Predicted: ", np.round(L_pred, 4))

# plot goal vs initial prediction
fig, ax = plt.subplots()
plt.plot(r_data, L_data, "o-", label="Target")
plt.plot(r_data, L_pred, "o-", label="Initial guess")
plt.xlabel(r"$r_\star$")
plt.ylabel(r"$L$")
plt.title("True vs initial predicted geodesic lengths")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "l_vs_rstar_initial_guess.png", dpi=150)
plt.show()

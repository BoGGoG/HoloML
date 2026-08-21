import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import optax
import equinox as eqx

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
N_STEPS = 5000
DT = 0.01

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


# ── Neural network ─────────────────────────────────────────────────────────────


class MassProfile(eqx.Module):
    """MLP mapping v (scalar) → m(v) ≥ 0 via softplus output."""

    layers: list

    def __init__(self, hidden_dims=(32, 32), *, key):
        dims = [1] + list(hidden_dims) + [1]
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for k, d_in, d_out in zip(keys, dims[:-1], dims[1:])
        ]

    def __call__(self, v):
        x = jnp.array([v])
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return jax.nn.softplus(self.layers[-1](x))[0]


def pretrain_mass_profile(model, m_f, v_c, v_s, n_steps=500, lr=1e-3):
    v_grid = jnp.linspace(-5.0, 5.0, 200)
    m_target = m_f * 0.5 * (1.0 + jnp.tanh((v_grid - v_c) / v_s))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def pretrain_loss(model):
        m_pred = jax.vmap(model)(v_grid)
        return jnp.mean((m_pred - m_target) ** 2)

    for i in range(n_steps):
        loss, grads = pretrain_loss(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        if (i + 1) % 100 == 0:
            print(f"  pretrain step {i + 1}/{n_steps}: loss={float(loss):.6e}")

    return model


# ── Forward model ──────────────────────────────────────────────────────────────
#
# Metric: ds² = -f(r,v) dv² + 2 dv dr + r² dx²
# f(r,v) = r² - m_θ(v)  where m_θ is the MassProfile network.


def f_metric(r, v, model):
    return r**2 - model(v)


_df_dr = jax.grad(f_metric, argnums=0)
_df_dv = jax.grad(f_metric, argnums=1)


def _rhs(state, model):
    v, r, x, dv, dr, dx = state
    f = f_metric(r, v, model)
    df_dr = _df_dr(r, v, model)
    df_dv = _df_dv(r, v, model)
    ddv = r * dx**2 - 0.5 * df_dr * dv**2
    ddr = f * ddv + df_dr * dr * dv + 0.5 * df_dv * dv**2
    ddx = -2.0 / r * dr * dx
    return jnp.array([dv, dr, dx, ddv, ddr, ddx])


def _integrate_single(r_star, v0, model):
    s0 = jnp.array([v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star])

    @jax.checkpoint
    def step(s, _):
        k1 = _rhs(s, model)
        k2 = _rhs(s + 0.5 * DT * k1, model)
        k3 = _rhs(s + 0.5 * DT * k2, model)
        k4 = _rhs(s + DT * k3, model)
        s2 = s + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s2, s2

    _, traj = jax.lax.scan(step, s0, None, length=N_STEPS)
    return traj  # (N_STEPS, 6)


_integrate_batch = jax.vmap(_integrate_single, in_axes=(0, None, None))


def length_and_h_from_traj(traj, model, r_cut=R_CUT, delta=1.0):
    vs, rs, xs, dv, dr, dx = traj.T
    f = jax.vmap(lambda vi, ri: f_metric(ri, vi, model))(vs, rs)
    sdot = jnp.sqrt(jnp.maximum(-f * dv**2 + 2 * dv * dr + rs**2 * dx**2, 0.0))

    mask = jax.nn.sigmoid(-(rs - r_cut) / delta)
    L = 2.0 * DT * jnp.sum(sdot * mask)

    weight = mask[:-1] - mask[1:]
    h = 2.0 * jnp.dot(xs[:-1], weight) / jnp.sum(weight)

    return L, h


@eqx.filter_jit
def forward(model, r_stars, v0):
    traj_batch = _integrate_batch(r_stars, v0, model)
    L_arr, h_arr = jax.vmap(lambda traj: length_and_h_from_traj(traj, model))(
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

    # ── Create and pre-train the model ────────────────────────────────────

    key = jax.random.PRNGKey(42)
    model = MassProfile(hidden_dims=(32, 32), key=key)

    print("Pre-training on tanh mass profile...")
    model = pretrain_mass_profile(model, m_f, v_c, v_s, n_steps=2000, lr=3e-3)

    # ── Initial predictions ───────────────────────────────────────────────

    print("Compiling forward model (first call)...")
    h_init, L_init = forward(model, r_data, v0)
    loss_fn = GeodesicLoss(h_data, L_data)
    print(f"Initial geodesic loss (after pretrain): {loss_fn(h_init, L_init):.6e}")
    print("Target h:    ", np.round(h_data, 4))
    print("Predicted h: ", np.round(np.array(h_init), 4))
    print("Target L:    ", np.round(L_data, 4))
    print("Predicted L: ", np.round(np.array(L_init), 4))

    # ── Geodesic-based training ───────────────────────────────────────────

    LR = 1e-3
    N_GD_STEPS = 100

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def compute_loss(model):
        h_p, L_p = forward(model, r_data, v0)
        return loss_fn(h_p, L_p)

    print("Compiling gradient computation (first call)...")
    loss_val, grads = compute_loss(model)
    print("Done.")

    m_true_0 = m_f * 0.5 * (1 + np.tanh((0.0 - v_c) / v_s))
    m_true_5 = m_f * 0.5 * (1 + np.tanh((5.0 - v_c) / v_s))

    print(f"\n{'step':>4}  {'loss':>12}  {'m(0)':>8}  {'m(5)':>8}")
    print(f"{'true':>4}  {'-':>12}  {m_true_0:>8.4f}  {m_true_5:>8.4f}")
    print("-" * 42)

    for step in range(N_GD_STEPS):
        loss_val, grads = compute_loss(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        if step % 10 == 0 or step == N_GD_STEPS - 1:
            m_at_0 = float(model(0.0))
            m_at_5 = float(model(5.0))
            print(
                f"{step:>4}  {float(loss_val):>12.6e}"
                f"  {m_at_0:>8.4f}  {m_at_5:>8.4f}"
            )

    # ── Plots ─────────────────────────────────────────────────────────────

    h_final, L_final = forward(model, r_data, v0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(h_data, L_data, "o-", label="Target")
    ax.plot(np.array(h_init), np.array(L_init), "s--", label="After pretrain")
    ax.plot(
        np.array(h_final),
        np.array(L_final),
        "^--",
        label=f"After {N_GD_STEPS} steps",
    )
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$L$")
    ax.set_title("Geodesic length vs boundary separation")
    ax.legend()

    ax = axes[1]
    v_plot = jnp.linspace(-5.0, 5.0, 200)
    m_true_arr = m_f * 0.5 * (1.0 + np.tanh((np.array(v_plot) - v_c) / v_s))
    m_learned = np.array(jax.vmap(model)(v_plot))
    ax.plot(np.array(v_plot), m_true_arr, "k-", label=r"$m_{\mathrm{true}}(v)$", lw=2)
    ax.plot(np.array(v_plot), m_learned, "r--", label=r"$m_\theta(v)$", lw=2)
    ax.set_xlabel(r"$v$")
    ax.set_ylabel(r"$m(v)$")
    ax.set_title("Mass profile recovery")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "nn_mass_profile_fit.png", dpi=150)
    print(f"Saved plot to {OUT_DIR / 'nn_mass_profile_fit.png'}")

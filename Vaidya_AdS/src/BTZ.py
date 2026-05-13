"""
Data generation for BTZ black hole — JAX/Equinox version
"""

import os
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


# ── Analytic functions ─────────────────────────────────────────────────────────


def f_true(z: jax.Array) -> jax.Array:
    """(4.27) in paper"""
    return 1.0 - z**2


def h_true(z: jax.Array) -> jax.Array:
    return jnp.ones_like(z)


def SData(c: float, beta: float, v: float, l: jax.Array) -> jax.Array:
    """(4.33) in https://arxiv.org/abs/2406.07395"""
    return (c / 3.0) * jnp.log(jnp.sinh(jnp.pi * l / (beta * v))) + jnp.log(
        beta * v / jnp.pi
    )


def l_func(zstar: jax.Array) -> jax.Array:
    """(4.28) in https://arxiv.org/abs/2406.07395"""
    return 2 * jnp.arctanh(zstar)


def h_func(z: jax.Array) -> jax.Array:
    return jnp.ones_like(z)


def get_thermal_entropy(h, zh: jax.Array) -> jax.Array:
    """s = 4π√h(z_h) / z_h  (with c = 3L/2G_N absorbed)"""
    return 4 * jnp.pi * jnp.sqrt(h(zh)) / zh


def get_event_horizon() -> jax.Array:
    return jnp.array(1.0)


# ── Data generation ────────────────────────────────────────────────────────────


def generate_BTZ_data(cvbeta: np.ndarray, Nzstar: int = 1000) -> dict:
    """
    cvbeta: array of (c, beta, v) rows.
    Returns a dict keyed by (c, beta, v).
    """
    eps = 1e-3
    zstar_arr = jnp.linspace(eps, 1 - eps, Nzstar)
    l_arr = l_func(zstar_arr)

    print("Generating data...")
    data = {}
    for cvb in cvbeta:
        S_arr = SData(c=cvb[0], beta=cvb[1], v=cvb[2], l=l_arr)
        s_thermal = get_thermal_entropy(h_func, get_event_horizon())
        data[(cvb[0], cvb[1], cvb[2])] = {
            "zstar": np.array(zstar_arr),
            "l": np.array(l_arr),
            "SFinite": np.array(S_arr),
            "s_thermal": np.array(s_thermal),
        }
    return data


# ── Neural network ─────────────────────────────────────────────────────────────


class BTZ_NN(eqx.Module):
    """
    Two-headed network predicting f(z) and h(z).

    Call signature: model(x) where x has shape (input_dim,).
    Returns (f, h), each of shape (output_dim,).

    f is constrained to vanish at z = 1 by construction.
    """

    layers_f: list
    layers_h: list
    a: jax.Array

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list,
        a: float = 0.0,
        *,
        key: jax.Array,
    ):
        layer_dims = [input_dim] + hidden_layers + [output_dim]
        key1, key2 = jax.random.split(key)

        def make_layers(k):
            keys = jax.random.split(k, len(layer_dims) - 1)
            return [
                eqx.nn.Linear(in_d, out_d, key=ki)
                for ki, in_d, out_d in zip(keys, layer_dims[:-1], layer_dims[1:])
            ]

        self.layers_f = make_layers(key1)
        self.layers_h = make_layers(key2)
        self.a = jnp.array(a)

    def _apply(self, layers: list, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(layers):
            x = layer(x)
            if i < len(layers) - 1:
                x = jax.nn.relu(x)
        return x

    def __call__(self, x: jax.Array):
        """x: shape (input_dim,)  →  (f, h) each of shape (output_dim,)"""
        f_net = self._apply(self.layers_f, x)
        h_net = self._apply(self.layers_h, x)
        f_out = (1 - x) * (1 + (self.a + 1) * x - x**2 * f_net)
        h_out = 1 + self.a * x - x**2 * h_net
        return f_out, h_out


# ── Model evaluation helpers ───────────────────────────────────────────────────


def _h(model: BTZ_NN, z: jax.Array) -> jax.Array:
    """Evaluate h(z).  z: scalar or shape (N,).  Returns shape (N,)."""
    z = jnp.atleast_1d(jnp.asarray(z))
    _, h = jax.vmap(lambda zi: model(zi[None]))(z)
    return h[:, 0]


def _f(model: BTZ_NN, z: jax.Array) -> jax.Array:
    """Evaluate f(z).  z: scalar or shape (N,).  Returns shape (N,)."""
    z = jnp.atleast_1d(jnp.asarray(z))
    f, _ = jax.vmap(lambda zi: model(zi[None]))(z)
    return f[:, 0]


# ── Gauss-Legendre quadrature ──────────────────────────────────────────────────


def _gl_nodes_weights(n: int):
    """GL nodes and weights on [-1, 1], computed once with numpy."""
    xi, wi = np.polynomial.legendre.leggauss(n)
    return jnp.array(xi), jnp.array(wi)


# ── Integrands (scalar z, scalar zstar) ───────────────────────────────────────


def SFiniteIntegrant(z, model: BTZ_NN, zstar) -> jax.Array:
    """BTZ version of (4.25).  z, zstar: scalars."""
    f_z, h_z = model(z[None])
    _, h_zstar = model(zstar[None])
    f_z, h_z, h_zstar = f_z[0], h_z[0], h_zstar[0]

    integrand = jnp.sqrt(1.0 / ((1 - z**2 * h_zstar / (zstar**2 * h_z)) * f_z))
    return jnp.minimum((integrand - 1) / z, 1e8)


def lIntegrand_NN(alpha, zstar, model: BTZ_NN) -> jax.Array:
    """BTZ version of (4.26).  alpha, zstar: scalars."""
    f_a, h_a = model(alpha[None])
    _, h_zstar = model(zstar[None])
    f_a, h_a, h_zstar = f_a[0], h_a[0], h_zstar[0]

    out = 1.0 / jnp.sqrt(h_a * f_a * (h_a * zstar**2 / h_zstar / alpha**2 - 1))
    return jnp.minimum(out, 1e8)


# ── Integrals ──────────────────────────────────────────────────────────────────


def S_integral_NN(model: BTZ_NN, zstar: jax.Array, N_GL: int = 12) -> jax.Array:
    """Returns S_finite for every element of zstar (shape (N,))."""
    xi, wi = _gl_nodes_weights(N_GL)

    def integrate_one(zstar_i):
        eps = 1e-8
        a, b = 0.0, zstar_i - eps
        t = 0.5 * (b - a) * xi + 0.5 * (a + b)
        vals = jax.vmap(lambda z: SFiniteIntegrant(z, model, zstar_i))(t)
        result = 0.5 * (b - a) * jnp.sum(wi * vals)
        return 0.5 * result + jnp.log(zstar_i)

    return jax.vmap(integrate_one)(zstar)


def l_integral_NN(model: BTZ_NN, zstar: jax.Array, N_GL: int = 12) -> jax.Array:
    """Returns l for every element of zstar (shape (N,))."""
    xi, wi = _gl_nodes_weights(N_GL)

    def integrate_one(zstar_i):
        eps = 1e-8
        a, b = 0.0, zstar_i - eps
        t = 0.5 * (b - a) * xi + 0.5 * (a + b)
        vals = jax.vmap(lambda alpha: lIntegrand_NN(alpha, zstar_i, model))(t)
        return 2 * 0.5 * (b - a) * jnp.sum(wi * vals)

    return jax.vmap(integrate_one)(zstar)


# ── Main ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    NZstar = 1000
    cvbeta = np.array([[24 * np.pi, 1, 2 * np.pi]])
    data = generate_BTZ_data(cvbeta=cvbeta, Nzstar=NZstar)

    l = data[(cvbeta[0, 0], cvbeta[0, 1], cvbeta[0, 2])]["l"]
    S = data[(cvbeta[0, 0], cvbeta[0, 1], cvbeta[0, 2])]["SFinite"]
    print(f"{l[0:10]=} {S[0:10]=}")

    data_dir = Path("data")
    path = data_dir / "data_BTZ.pkl"
    os.makedirs(data_dir, exist_ok=True)
    pickle.dump(data, open(path, "wb"))
    print(f"Data saved to {path}")

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(
        l, S, label=f"c={cvbeta[0, 0]:.3g}, beta={cvbeta[0, 1]}, v={cvbeta[0, 2]:.3g}"
    )
    plt.xlabel("l")
    plt.ylabel("S")
    plt.title("BTZ Black Hole Entropy")
    plt.legend()
    plt.grid()
    plt.show()

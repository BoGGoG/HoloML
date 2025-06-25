"""
Ahn, Byoungjoon, Hyun-Sik Jeong, Keun-Young Kim, and Kwan Yun. 2025.
“Holographic Reconstruction of Black Hole Spacetime: Machine Learning and Entanglement Entropy.”
_Journal of High Energy Physics_ 2025 (1): 25. [https://doi.org/10.1007/JHEP01(2025)025](https://doi.org/10.1007/JHEP01\(2025\)025).
"""

import numpy as np
import torch
from torch import Tensor
import sys, os
import pickle
import scipy
from scipy.optimize import curve_fit

from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import (
    Trapezoid,
    Simpson,
    Boole,
    MonteCarlo,
    VEGAS,
    GaussLegendre,
)  # The available integrators
from torchquad.utils.set_precision import set_precision
import torchquad
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sympy as sp
from sympy import symbols, Eq, solve, sqrt
from sympy import symbols, sqrt, I, sympify, srepr
from sympy.utilities.lambdify import lambdify
import warnings
from typing import Callable

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning)
# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float32")
torch.set_default_dtype(torch.float32)
# torch.set_default_device("cuda:0")
torch.set_default_device("cpu")


def get_Q_func(Q_expr_path: str) -> Callable:
    beta, mu = symbols("beta mu", real=True, positive=True)
    with open(Q_expr_path) as f:
        Q_expr = sympify(f.read())
    Q_numeric = lambdify((mu, beta), Q_expr, modules="sympy")

    def Q_func(mu: float, beta: float) -> float:
        """Returns the value of Q for given mu and beta."""
        return float(Q_numeric(mu, beta).evalf().as_real_imag()[0])

    return Q_func


def f_true(z: Tensor, beta: float, Q: float) -> Tensor:
    """(4.13) in paper"""
    a = 1 + (1 + 3 * Q) * z + (1 + 3 * Q * (1 + Q) - beta**2 / 2.0) * torch.pow(z, 2)
    b = torch.pow(1 + Q * z, 3.0 / 2.0)
    out = (1 - z) * a / b
    return out


def h_true(z: Tensor, Q: float) -> Tensor:
    """(4.13) in paper"""
    return torch.pow(1 + Q * z, 3.0 / 2.0)


def SFiniteIntegrant_true(z, zstar, beta, mu, Q) -> torch.Tensor:
    """(2.5) in paper"""
    integrand = torch.sqrt(
        h_true(z, Q)
        / (
            (
                1
                - torch.pow(z, 4)
                * h_true(zstar, Q) ** 2
                / (zstar**4 * torch.pow(h_true(z, Q), 2))
            )
            * f_true(z, beta, Q)
        )
    )
    integrand = torch.clamp(
        (integrand - 1) / torch.pow(z, 2), max=1e8
    )  # avoid numerical issues
    return integrand


def SFiniteA(zstar: Tensor, beta: float, mu: float, Q: float, N_GL: int = 12) -> Tensor:
    """Integrate SFiniteIntegrant from 0 to zstar"""
    eps = 1e-8  # need to avoid singularity at z=0. Tried also with 1e-7 and not much changes. With 1e-9 get eror, but VEGAS problem I think.
    if len(zstar.shape) == 0:
        zstar = torch.tensor([zstar])
        print(f"{zstar.shape=}")
    # integrator = MonteCarlo()
    # integrator = Simpson()
    # integrator = Trapezoid()
    # integrator = Boole()
    # integrator = VEGAS()
    integrator = GaussLegendre()
    out = []
    for zstar_i in zstar:
        integration_domain = torch.tensor([[0, zstar_i - eps]])
        func = partial(SFiniteIntegrant_true, zstar=zstar_i, beta=beta, mu=mu, Q=Q)
        # result = integrator.integrate(func, integration_domain=integration_domain, N=2000, dim=1, max_iterations=1)
        result = integrator.integrate(
            func, integration_domain=integration_domain, N=N_GL, dim=1
        )
        out.append(result)

    return torch.stack(out)


def SFinite(zstar: Tensor, beta: float, mu: float, Q: float, N_GL: int = 12) -> Tensor:
    """(2.5) in paper"""
    SFiniteA_value = SFiniteA(zstar, beta, mu, Q, N_GL)
    print("-----")
    print(f"{SFiniteA_value.shape=}")
    print("-----")
    out = SFiniteA_value - 1.0 / zstar
    return out


def lIntegrand(
    alpha: Tensor, zstar: Tensor, beta: float, mu: float, Q: float
) -> Tensor:
    """(2.4) in paper"""
    if type(zstar) is not Tensor:
        zstar = torch.tensor(zstar, dtype=alpha.dtype, device=alpha.device)
    out = 1.0 / torch.sqrt(
        h_true(alpha, Q)
        * f_true(alpha, beta, Q)
        * (
            torch.pow(h_true(alpha, Q), 2)
            * zstar**4
            / h_true(zstar, Q) ** 2
            / torch.pow(alpha, 4)
            - 1
        )
    )
    return torch.clamp(out, max=1e8)


def lfunc(z: Tensor, beta: float, mu: float, Q: float, N_GL: int = 12) -> Tensor:
    eps = 1e-8
    # integrator = VEGAS()
    # integrator = MonteCarlo()
    # integrator = Simpson()
    if len(z.shape) == 0:
        z = torch.tensor([z])
    integrator = GaussLegendre()
    out = []
    for z_i in z:
        integration_domain = torch.tensor([[0, z_i - eps]])
        func = partial(lIntegrand, zstar=z_i, beta=beta, mu=mu, Q=Q)
        # result = 2 * integrator.integrate(func, integration_domain=integration_domain, N=2000, dim=1, max_iterations=1)
        result = 2 * integrator.integrate(
            func, integration_domain=integration_domain, N=N_GL, dim=1
        )
        out.append(result)
    out = torch.stack(out)
    return out


def get_event_horizon(f_func, other_params, max_iter: int = 100, z_start=0.5) -> Tensor:
    """find the place where f_func(z) = 0, z > 0, z < 1
    This has to be done numerically and we need to have gradients.

    For this case here, we actually know that the horizon is at z=1.
    First argument of f_func should be z, and other parameters should be passed as additional arguments.
    """

    # use Newton-Raphson to find z such that f_func(z, other_params) = 0
    # z_current = z_start
    # for i in range(max_iter):
    #     f_value = f_func(z_current, *other_params)
    #     if f_value.abs() < 1e-6:
    #         return z_current
    #     grad = torch.autograd.grad(f_value, z_current)[0]
    #     z_current = z_current - f_value / grad

    # lukily we know that our f has a zero at z=1, so we can just return it
    return torch.tensor(1.0, requires_grad=False)


def get_thermal_entropy(h, zh: Tensor, Q) -> Tensor:
    """s = L^2 h(z) / (4 G_N z^2) at z=zh (horizon)
    Here we set the constant L^2 / (4 G_N) = 1.
    """
    out = h(zh, Q) / torch.pow(zh, 2)
    return out


if __name__ == "__main__":
    beta_val = 0.6
    mu_val = 1.2
    Q_func = get_Q_func("Q_expr.txt")
    z_test = torch.linspace(0.1, 1, 6, dtype=torch.float32)
    z_star = torch.tensor([0.645])
    Q = Q_func(mu_val, beta_val)

    print(f"Q for mu={mu_val}, beta={beta_val}: {Q_func(mu_val, beta_val)}")
    print(f"{z_test=}")
    print(f"{z_star=}")
    print(f"{z_star.shape=}")
    print(f"{f_true(z_test, beta_val, Q)=}")
    print(f"{h_true(z_test, Q)}=")
    print(f"{SFiniteIntegrant_true(z_test, z_star, beta_val, mu_val, Q)=}")
    print(f"{SFiniteA(z_star, beta_val, mu_val, Q)=}")
    print(f"{SFinite(z_star, beta_val, mu_val, Q)=}")
    print(f"{lIntegrand(z_test, z_star, beta_val, mu_val, Q)=}")
    print(f"{lfunc(z_star, beta_val, mu_val, Q)=}")
    print(f"{get_event_horizon(f_true, (beta_val, Q))=}")
    print(
        f"{get_thermal_entropy(h_true, get_event_horizon(f_true, (beta_val, Q)), Q)=}"
    )

    # # plot f
    # z_arr = np.linspace(0.01, 0.99, 100)
    # plt.figure(figsize=(10, 6))
    # f_values = f_true(torch.tensor(z_arr), beta_val, Q)
    # plt.plot(z_arr, f_values.numpy(), label=f"f(z) for Q={Q:.2f}, beta={beta_val}")
    # plt.show()

    # Sfinite as a function of l
    colors = ["red", "green", "blue"]
    mu_beta_vals = [[0.5, 0.5], [0.5, 1.0], [1.0, 1.5]]
    z_star_arr = torch.linspace(0.1, 0.999, 50)
    plt.figure(figsize=(10, 6))
    for i, (mu, beta) in enumerate(mu_beta_vals):
        Q = Q_func(mu, beta)
        Sfinite_values = SFinite(z_star_arr, beta, mu, Q)
        l_values = lfunc(z_star_arr, beta, mu, Q)
        plt.plot(
            l_values.numpy(),
            Sfinite_values.numpy(),
            label=f"SFinite (mu={mu}, beta={beta})",
            c=colors[i],
        )
        plt.scatter(
            l_values.numpy(),
            Sfinite_values.numpy(),
            c=colors[i],
        )
        plt.xlabel("l")
        plt.ylabel("SFinite")
        plt.title(f"SFinite vs l for mu={mu}, beta={beta}")
        plt.legend()
        plt.xlim(0, 3)
        plt.ylim(-4.5, 2.1)
    plt.show()

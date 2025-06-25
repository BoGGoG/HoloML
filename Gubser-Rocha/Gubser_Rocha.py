"""
Ahn, Byoungjoon, Hyun-Sik Jeong, Keun-Young Kim, and Kwan Yun. 2025.
“Holographic Reconstruction of Black Hole Spacetime: Machine Learning and Entanglement Entropy.”
_Journal of High Energy Physics_ 2025 (1): 25. [https://doi.org/10.1007/JHEP01(2025)025](https://doi.org/10.1007/JHEP01\(2025\)025).
"""

import os
import pickle
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sympy as sp
import torch
import torch.nn as nn
import torchquad
from scipy.optimize import curve_fit
from sympy import Eq, I, solve, sqrt, srepr, symbols, sympify
from sympy.utilities.lambdify import lambdify
from torch import Tensor
from torchquad import (
    VEGAS,
    Boole,
    GaussLegendre,
    MonteCarlo,
    Simpson,
    Trapezoid,
    set_up_backend,  # Necessary to enable GPU support
)  # The available integrators
from torchquad.utils.set_precision import set_precision
from tqdm import tqdm

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning)
# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float64")
torch.set_default_dtype(torch.float64)
# torch.set_default_device("cuda:0")
torch.set_default_device("cuda:0")


def calc_loss(S_pred, l_pred, s_thermal_pred, s_thermal_true, Sl_fit_func, popt):
    """
    Loss function for the model. (4.11) in paper
    """
    interpolated_S = Sl_fit_func(l_pred, *popt)

    # mae
    S_loss = torch.mean(torch.abs(S_pred - interpolated_S))
    s_thermal_loss = torch.mean(torch.abs(s_thermal_pred - s_thermal_true))
    loss = S_loss + s_thermal_loss
    return {"S_loss": S_loss, "s_thermal_loss": s_thermal_loss, "loss": loss}


def get_Q_func(Q_expr_path: str) -> Callable:
    """Q(mu, beta)"""
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


def test_functions():
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


def test_S_of_l():
    Q_func = get_Q_func("Q_expr.txt")

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
            l_values.cpu().numpy(),
            Sfinite_values.cpu().numpy(),
            label=f"SFinite (mu={mu}, beta={beta})",
            c=colors[i],
        )
        plt.scatter(
            l_values.cpu().numpy(),
            Sfinite_values.cpu().numpy(),
            c=colors[i],
        )
        plt.xlabel("l")
        plt.ylabel("SFinite")
        plt.title(f"SFinite vs l for mu={mu}, beta={beta}")
        plt.legend()
        plt.xlim(0, 3)
        plt.ylim(-4.5, 2.1)
    plt.show()


class GubserRocha_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, a: float = 0.0):
        super(GubserRocha_NN, self).__init__()

        # Combine input layer, hidden layers, and output layer into one list
        layer_dims = [input_dim] + hidden_layers + [output_dim]

        # Create fully connected layers
        layers_f = []
        for i in range(len(layer_dims) - 1):
            layers_f.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            # Add ReLU activation after each hidden layer, but not after the output layer
            if i < len(layer_dims) - 2:
                layers_f.append(nn.ReLU())
        self.network_f = nn.Sequential(*layers_f)

        layers_h = []
        for i in range(len(layer_dims) - 1):
            layers_h.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            # Add ReLU activation after each hidden layer, but not after the output layer
            if i < len(layer_dims) - 2:
                layers_h.append(nn.ReLU())
        self.network_h = nn.Sequential(*layers_h)

        self.a = nn.Parameter(torch.tensor(a))

    def forward(self, x):
        """
        f always has a zero at x=1
        """
        f_out = (1 - x) * (1 + (self.a + 1) * x - torch.pow(x, 2) * self.network_f(x))
        h_out = 1 + self.a * x - torch.pow(x, 2) * self.network_h(x)
        return f_out, h_out


def _h(model, z):
    """watch out, needs model to be defined"""
    if len(z.shape) == 0:
        z = torch.unsqueeze(z, 0)
    f_out, h_out = model(z)
    return h_out


def _f(model, z):
    """watch out, needs model to be defined.
    This _f always has a zero at z=1, because this is how it is defined in model.
    For more complicated f, this will not be the case.
    """
    if len(z.shape) == 0:
        z = torch.unsqueeze(z, 0)
    f_out, h_out = model(z)
    return f_out


def SFiniteIntegrant(z, model, zstar) -> torch.Tensor:
    integrand = torch.sqrt(
        _h(model, z)
        / (
            (
                1
                - torch.pow(z, 4)
                * _h(model, zstar) ** 2
                / (zstar**4 * torch.pow(_h(model, z), 2))
            )
            * _f(model, z)
        )
    )
    integrand = torch.clamp(integrand, max=1e8)  # avoid numerical issues
    integrand = torch.clamp(
        (integrand - 1) / torch.pow(z, 2), max=1e8
    )  # avoid numerical issues
    return integrand


def S_integral_NN(model, zstar: Tensor, N_GL: int = 12) -> Tensor:
    eps = 1e-8  # need to avoid singularity at z=0. Tried also with
    integrator = GaussLegendre()

    out = []
    for zstar_i in zstar:
        integration_domain = torch.tensor([[0, zstar_i - eps]])
        func = partial(SFiniteIntegrant, model=model, zstar=zstar_i)
        result = integrator.integrate(
            func, integration_domain=integration_domain, N=N_GL, dim=1
        )
        result = result - 1.0 / zstar_i
        out.append(result)
    return torch.stack(out)


def lIntegrand_NN(alpha: Tensor, zstar: Tensor, model) -> Tensor:
    out = 1.0 / torch.sqrt(
        _h(model, alpha)
        * _f(model, alpha)
        * (
            torch.pow(_h(model, alpha), 2)
            * zstar**4
            / _h(model, zstar) ** 2
            / torch.pow(alpha, 4)
            - 1
        )
    )
    return torch.clamp(out, max=1e8)


def l_integral_NN(model, zstar: Tensor, N_GL: int = 12) -> Tensor:
    eps = 1e-8
    integrator = GaussLegendre()
    out = []
    for zstar_i in zstar:
        integration_domain = torch.tensor([[0, zstar_i - eps]])
        func = partial(lIntegrand_NN, model=model, zstar=zstar_i)
        result = 2 * integrator.integrate(
            func, integration_domain=integration_domain, N=N_GL, dim=1
        )
        out.append(result)
    return torch.stack(out)


def generate_data(beta_mu: list, Nzstar: int = 1000, N_GL: int = 12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    eps = 1e-1
    data = dict()
    print("Generating data...")
    Q_func = get_Q_func("Q_expr.txt")

    zstar_arr = torch.linspace(eps, 1 - eps, Nzstar)
    zstar_arr = zstar_arr.to(device)
    for i, (mu, beta) in tqdm(enumerate(beta_mu)):
        print(f"Generating data for mu={mu}, beta={beta} ({i+1}/{len(beta_mu)})")
        Q = Q_func(mu, beta)
        Sfinite_values = SFinite(zstar_arr, beta, mu, Q)
        l_values = lfunc(zstar_arr, beta, mu, Q)

        # sort S and l such that l is increasing
        # idxs = np.argsort(l)
        # l = l[idxs]
        # S = S[idxs]

        thermal_entropy = get_thermal_entropy(
            h_true, get_event_horizon(f_true, (beta, Q)), Q
        )

        # append to data
        data[(mu, beta)] = {
            "Q": Q,
            "zstar": zstar_arr.cpu().numpy(),
            "SFinite": Sfinite_values.cpu().numpy(),
            "l": l_values.cpu().numpy(),
            "s_thermal": thermal_entropy.cpu().numpy(),
        }

    return data


if __name__ == "__main__":
    do_tests = True
    if do_tests:
        test_functions()
        test_S_of_l()

    beta_val = 0.6
    mu_val = 1.2
    Q_func = get_Q_func("Q_expr.txt")
    z_test = torch.linspace(0.1, 1, 6, dtype=torch.float32)
    z_star = torch.tensor([0.645])
    Q = Q_func(mu_val, beta_val)

    dir = Path("data")
    path = dir / "data_Gubser_Rocha.pkl"
    os.makedirs(dir, exist_ok=True)
    do_data_generation = True
    if do_data_generation:
        beta_mu = [[0.5, 0.5], [0.5, 1.0], [1.0, 1.5]]
        data = generate_data(beta_mu, Nzstar=5000, N_GL=20)
        pickle.dump(data, open(path, "wb"))
        print(f"Data saved to {path}")

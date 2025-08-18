"""
Data generation for BTZ black hole
"""

import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchquad import GaussLegendre
from functools import partial


def SData(c: float, beta: float, v: float, l: Tensor) -> Tensor:
    """
    (4.33) in https://arxiv.org/abs/2406.07395
    """
    return (c / 3.0) * torch.log(torch.sinh(np.pi * l / (beta * v)))


def l_func(zstar: Tensor) -> Tensor:
    """
    (4.28) in https://arxiv.org/abs/2406.07395
    """
    return 2 * torch.arctanh(zstar)


def h_func(zstar: Tensor) -> Tensor:
    return torch.tensor(1)


def get_thermal_entropy(h, zh: Tensor) -> Tensor:
    """s = L^2 h(z) / (4 G_N z^2) at z=zh (horizon)
    Here we set the constant L^2 / (4 G_N) = 1.
    """
    out = torch.sqrt(h(zh)) / zh
    return out


def get_event_horizon() -> Tensor:
    """Returns the event horizon z_h = 1"""
    return torch.tensor(1.0)


def generate_BTZ_data(cvbeta: np.ndarray, Nzstar: int = 1000):
    """
    cvbeta: array of arrays (c, beta, v) for which to generate data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    eps = 1e-3
    data = dict()
    print("Generating data...")

    zstar_arr = torch.linspace(eps, 1 - eps, Nzstar)
    zstar_arr = zstar_arr.to(device)
    l_arr = l_func(zstar_arr)

    data = dict()
    for cvb in cvbeta:
        S_arr = SData(c=cvb[0], beta=cvb[1], v=cvb[2], l=l_arr)
        z_h = get_event_horizon()  # 1
        thermal_entropy = get_thermal_entropy(h_func, z_h)
        data[(cvb[0], cvb[1], cvb[2])] = {
            "zstar": zstar_arr.cpu().numpy(),
            "l": l_arr.cpu().numpy(),
            "SFinite": S_arr.cpu().numpy(),
            "s_thermal": thermal_entropy.cpu().numpy(),
        }
    return data


class BTZ_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, a: float = 0.0):
        super(BTZ_NN, self).__init__()

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


if __name__ == "__main__":
    NZstar = 1000
    cvbeta = np.array([[24 * np.pi, 1, 2 * np.pi]])
    data = generate_BTZ_data(cvbeta=cvbeta, Nzstar=NZstar)

    l = data[(cvbeta[0, 0], cvbeta[0, 1], cvbeta[0, 2])]["l"]
    S = data[(cvbeta[0, 0], cvbeta[0, 1], cvbeta[0, 2])]["SFinite"]
    print(f"{l[0:10]=} {S[0:10]=}")

    dir = Path("data")
    path = dir / "data_BTZ.pkl"
    os.makedirs(dir, exist_ok=True)

    pickle.dump(data, open(path, "wb"))
    print(f"Data saved to {path}")

    # plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(l, S, label=f"c={cvbeta[0, 0]}, beta={cvbeta[0, 1]}, v={cvbeta[0, 2]}")
    plt.xlabel("l")
    plt.ylabel("S")
    plt.title("BTZ Black Hole Entropy")
    plt.legend()
    plt.grid()
    plt.show()

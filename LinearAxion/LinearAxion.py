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
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS, GaussLegendre # The available integrators
from torchquad.utils.set_precision import set_precision
import torchquad
from functools import partial
import matplotlib.pyplot as plt
from pathlib import  Path

# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float32")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def f(z: Tensor, beta: float, mu: float) -> Tensor:
    """(3.21) in paper"""
    return 1 - beta**2 * torch.pow(z, 2) / 2.0 - (1 - beta**2 / 2. + mu**2 / 4.) * torch.pow(z, 3) + mu**2 * torch.pow(z, 4) / 4.

def h(z: Tensor) -> Tensor:
    """ Just 1, but might get extended. Return 1 in same shape as z """
    # if isinstance(z, Tensor):
    #     return torch.ones_like(z)
    # else:
    return torch.ones_like(z)

def SFiniteIntegrant(z, zstar, beta, mu) -> torch.Tensor:
    integrand = torch.sqrt(h(z) / ( ( 1 - torch.pow(z, 4) * h(zstar)**2 / (zstar**4 * torch.pow(h(z),2)) ) * f(z, beta, mu) ))
    integrand = torch.clamp(integrand, max=1e8)  # avoid numerical issues
    integrand = torch.clamp((integrand - 1) / torch.pow(z, 2), max=1e8)  # avoid numerical issues
    return integrand

# def SFiniteIntegrant(z: torch.Tensor, zstar, beta, mu) -> torch.Tensor:
#     h_z = h(z)
#     h_zstar_sq = h(zstar)**2
#     denom = (1 - (z**4 * h_zstar_sq) / (zstar**4 * h_z**2)) * f(z, beta, mu)
#     integrand = torch.sqrt(h_z / denom)
#     integrand = torch.clamp(integrand, max=1e8)
#     return torch.clamp((integrand - 1) / z**2, max=1e8)

def SFiniteA(zstar:float, beta:float, mu:float, N_GL:int=12) -> Tensor:
    """ Integrate SFiniteIntegrant from 0 to zstar """
    eps = 1e-8 # need to avoid singularity at z=0. Tried also with 1e-7 and not much changes. With 1e-9 get eror, but VEGAS problem I think.
    integration_domain = torch.tensor([[0, zstar-eps]])
    if type(zstar) is not Tensor:
        zstar = torch.tensor(zstar)
    # integrator = MonteCarlo()
    # integrator = Simpson()
    # integrator = Trapezoid()
    # integrator = Boole()
    # integrator = VEGAS()
    integrator = GaussLegendre()
    func = partial(SFiniteIntegrant, zstar=zstar, beta=beta, mu=mu)
    # result = integrator.integrate(func, integration_domain=integration_domain, N=2000, dim=1, max_iterations=1)
    result = integrator.integrate(func, integration_domain=integration_domain, N=N_GL, dim=1)
    if type(result) is not Tensor:
        result = torch.tensor(result)
    return result

def SFinite(zstar:float, beta:float, mu:float, N_GL:int=12) -> Tensor:
    SFiniteA_value = SFiniteA(zstar, beta, mu, N_GL)
    out = SFiniteA_value - 1. / zstar
    return out

def lIntegrand(alpha:Tensor, zstar:float, beta:float, mu:float) -> Tensor:
    if type(zstar) is not Tensor:
        zstar = torch.tensor(zstar, dtype=alpha.dtype, device=alpha.device)
    out = 1. / torch.sqrt(h(alpha)*f(alpha, beta, mu) * (torch.pow(h(alpha), 2) * zstar**4 / h(zstar)**2 / torch.pow(alpha, 4) - 1))
    return torch.clamp(out, max=1e8)

def lfunc(z, beta, mu, N_GL:int=12) -> Tensor:
    eps = 1e-8
    integration_domain = torch.tensor([[0, z-eps]])
    # integrator = VEGAS()
    # integrator = MonteCarlo()
    # integrator = Simpson()
    integrator = GaussLegendre()
    func = partial(lIntegrand, zstar=z, beta=beta, mu=mu)
    # result = 2 * integrator.integrate(func, integration_domain=integration_domain, N=2000, dim=1, max_iterations=1)
    result = 2 * integrator.integrate(func, integration_domain=integration_domain, N=N_GL, dim=1)
    if type(result) is not Tensor:
        result = torch.tensor(result)
    return result

def generate_data(beta:float, mu:float, Nzstar:int=1000, N_GL:int=12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1e-6
    data = dict()
    for i, (mu, beta) in enumerate([[0.5, 0.5], [0.5, 1.0], [1.0, 1.5]]):
        zstar = torch.linspace(eps, 1-eps, Nzstar)
        zstar = zstar.to(device)
        l = np.array([lfunc(zz, beta, mu, N_GL=N_GL).cpu() for zz in zstar])
        S = np.array([SFinite(zz, beta, mu, N_GL=N_GL).cpu() for zz in zstar])
        # sort S and l such that l is increasing
        idxs = np.argsort(l)
        l = l[idxs]
        S = S[idxs]
        # append to data
        data[(mu, beta)] = {
            "zstar": zstar.cpu().numpy(),
            "SFinite": S,
            "l": l,
        }
    return data

def figure_2():
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["red", "green", "blue"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps = 1e-8
    for i, (mu, beta) in enumerate([[0.5, 0.5], [0.5, 1.0], [1.0, 1.5]]):
        zstar = torch.linspace(eps, 1-eps, 1000)
        zstar = zstar.to(device)
        l = np.array([lfunc(zz, beta, mu).cpu() for zz in zstar])
        S = np.array([SFinite(zz, beta, mu).cpu() for zz in zstar])
        # sort S and l such that l is increasing
        idxs = np.argsort(l)
        l = l[idxs]
        S = S[idxs]
        ax.plot(l, S, label=f"$\\mu={mu}, \\beta={beta}$", c=colors[i])
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$S_{\text{finite}}$")
    ax.set_title(r"$S_{\text{finite}}$ vs $\ell$")
    ax.grid()
    ax.legend()
    ax.set_ylim(-5.5, 1.5)
    ax.set_xlim(0, 3.1)
    plt.show()


def f_for_fit(x, cn1, c0, c1, c2, c3, c4, c5):
    return cn1 * x**(-1) + c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5


def power_law_fit(data):
    S = data["SFinite"]
    l = data["l"]
    zstar = data["zstar"]

    idxs = np.where((zstar > 0.1) & (zstar < 0.99))[0]
    S = S[idxs]
    l = l[idxs]

    popt, pcov = curve_fit(f_for_fit, l, S, maxfev=100000)
    return popt, pcov

if __name__ == "__main__":
    beta = 0.5
    mu = 0.5
    zstar = 0.4
    print(f"{SFinite(zstar, beta, mu)=}")
    print(f"{lfunc(0.4, beta, mu)=}")

    figure_2()

    # dir = Path("data")
    # path = dir / "data.pkl"
    # do_data_generation = False
    # if do_data_generation:
    #     data = generate_data(beta, mu, Nzstar=5000, N_GL=20)
    #     pickle.dump(data, open(path, "wb"))
    #     print(f"Data saved to {path}")
    #
    # # load data
    # print(f"Loading data from {path}")
    # data_loaded = pickle.load(open(path, "rb"))
    #
    #
    # # curve fitting S(l)
    # mu = 1.0
    # beta = 1.5
    # print(data_loaded[(mu, beta)]["SFinite"])
    # popt, pcov = power_law_fit(data_loaded[(mu, beta)])
    # print(f"Fitted parameters for mu={mu}, beta={beta}:")
    # print(popt)
    #
    # plt.plot(data_loaded[(mu, beta)]["l"], data_loaded[(mu, beta)]["SFinite"], label="Data")
    # xrange = np.linspace(0, 3.1, 100)
    # plt.plot(xrange, f_for_fit(xrange, *popt), label="Fit", ls="--")
    # plt.legend()
    # plt.xlabel(r"$\ell$")
    # plt.ylabel(r"$S_{\text{finite}}$")
    # plt.ylim(-6, 2)
    # plt.show()

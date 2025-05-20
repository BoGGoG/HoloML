"""
Ahn, Byoungjoon, Hyun-Sik Jeong, Keun-Young Kim, and Kwan Yun. 2025.
“Holographic Reconstruction of Black Hole Spacetime: Machine Learning and Entanglement Entropy.”
_Journal of High Energy Physics_ 2025 (1): 25. [https://doi.org/10.1007/JHEP01(2025)025](https://doi.org/10.1007/JHEP01\(2025\)025).
"""

import numpy as np
import torch
from torch import Tensor

from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS # The available integrators
from torchquad.utils.set_precision import set_precision
import torchquad
from functools import partial
import matplotlib.pyplot as plt

# Use this to enable GPU support and set the floating point precision
# set_up_backend("torch", data_type="float32")

def f(z: Tensor, beta: float, mu: float) -> Tensor:
    """(3.21) in paper"""
    return 1 - beta**2 * torch.pow(z, 2) / 2.0 - (1 - beta**2 / 2. + mu**2 / 4.) * torch.pow(z, 3) + mu**2 * torch.pow(z, 4) / 4.

def h(z: float|np.ndarray|Tensor) -> np.ndarray|Tensor:
    """ Just 1, but might get extended. Return 1 in same shape as z """
    if isinstance(z, Tensor):
        return torch.ones_like(z)
    else:
        return np.ones_like(z)

def SFiniteIntegrant(z, zstar, beta, mu) -> torch.Tensor:

    integrand = torch.sqrt(h(z) / ( ( 1 - torch.pow(z, 4) * h(zstar)**2 / (zstar**4 * torch.pow(h(z),2)) ) * f(z, beta, mu) ))
    integrand = torch.clamp(integrand, max=1e8)  # avoid numerical issues
    integrand = torch.clamp((integrand - 1) / torch.pow(z, 2), max=1e8)  # avoid numerical issues
    return integrand

def SFiniteA(zstar:float, beta:float, mu:float) -> Tensor:
    """ Integrate SFiniteIntegrant from 0 to zstar """
    eps = 1e-6 # need to avoid singularity at z=0. Tried also with 1e-7 and not much changes. With 1e-9 get eror, but VEGAS problem I think.
    integration_domain = torch.tensor([[0, zstar-eps]])
    # integrator = MonteCarlo()
    integrator = VEGAS()
    # integrator = Simpson()
    # integrator = Trapezoid()
    # integrator = Boole()
    func = partial(SFiniteIntegrant, zstar=zstar, beta=beta, mu=mu)
    result = torch.tensor(integrator.integrate(func, integration_domain=integration_domain, N=1200, dim=1, max_iterations=1))
    return result

def SFinite(zstar:float, beta:float, mu:float) -> Tensor:
    SFiniteA_value = SFiniteA(zstar, beta, mu)
    out = SFiniteA_value - 1. / zstar
    return out

def lIntegrand(alpha:Tensor, zstar:float, beta:float, mu:float) -> Tensor:
    out = 1. / torch.sqrt(h(alpha)*f(alpha, beta, mu) * (torch.pow(h(alpha), 2) * zstar**4 / h(zstar)**2 / torch.pow(alpha, 4) - 1))
    return torch.clamp(out, max=1e8)

def lfunc(z, beta, mu) -> Tensor:
    eps = 1e-6
    integration_domain = torch.tensor([[0, z-eps]])
    integrator = VEGAS()
    # integrator = MonteCarlo()
    # integrator = Simpson()
    func = partial(lIntegrand, zstar=z, beta=beta, mu=mu)
    result = 2 * torch.tensor(integrator.integrate(func, integration_domain=integration_domain, N=1200, dim=1, max_iterations=1))
    return result

def figure_2():
    eps = 1e-3
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = ["red", "green", "blue"]
    for i, (mu, beta) in enumerate([[0.5, 0.5], [0.5, 1.0], [1.0, 1.5]]):
        zstar = torch.linspace(eps, 1.0-eps, 1000)
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

if __name__ == "__main__":
    beta = 0.5
    mu = 0.5
    zstar = 0.4
    print(f"{SFinite(zstar, beta, mu)=}")
    print(f"{lfunc(0.4, beta, mu)=}")

    figure_2()

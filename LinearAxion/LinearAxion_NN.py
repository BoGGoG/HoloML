import numpy as np
from sympy import Line
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
import torch.nn as nn

from LinearAxion import get_event_horizon, get_thermal_entropy

# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float64")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

class LinearAxion_NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, a:float=0.0):
        super(LinearAxion_NN, self).__init__()

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
        f_out = (1 - x) * ( 1 + (self.a + 1) * x - torch.pow(x, 2) * self.network_f(x)) 
        h_out = 1 + self.a * x - torch.pow(x, 2) * self.network_h(x)
        return f_out, h_out

def f_true(z: Tensor, beta: float, mu: float) -> Tensor:
    """(3.21) in paper"""
    return 1 - beta**2 * torch.pow(z, 2) / 2.0 - (1 - beta**2 / 2. + mu**2 / 4.) * torch.pow(z, 3) + mu**2 * torch.pow(z, 4) / 4.

def h_true(z: Tensor) -> Tensor:
    """ Just 1, but might get extended. Return 1 in same shape as z """
    return torch.ones_like(z)

def _h(model, z):
    """ watch out, needs model to be defined """
    if len(z.shape) == 0:
        z = torch.unsqueeze(z, 0)
    f_out, h_out = model(z)
    return h_out

def _f(model, z):
    """ watch out, needs model to be defined.
    This _f always has a zero at z=1, because this is how it is defined in model.
    For more complicated f, this will not be the case.
    """
    if len(z.shape) == 0:
        z = torch.unsqueeze(z, 0)
    f_out, h_out = model(z)
    return f_out

def SFiniteIntegrant(z, model, zstar) -> torch.Tensor:
    integrand = torch.sqrt(_h(model, z) / ( ( 1 - torch.pow(z, 4) * _h(model, zstar)**2 / (zstar**4 * torch.pow(_h(model, z),2)) ) * _f(model, z) ))
    integrand = torch.clamp(integrand, max=1e8)  # avoid numerical issues
    integrand = torch.clamp((integrand - 1) / torch.pow(z, 2), max=1e8)  # avoid numerical issues
    return integrand

def S_integral(model, zstar:Tensor, N_GL:int=12) -> Tensor:
    eps = 1e-8 # need to avoid singularity at z=0. Tried also with
    integrator = GaussLegendre()

    out = []
    for zstar_i in zstar:
        integration_domain = torch.tensor([[0, zstar_i-eps]])
        func = partial(SFiniteIntegrant, model=model, zstar=zstar_i)
        result = integrator.integrate(func, integration_domain=integration_domain, N=N_GL, dim=1)
        out.append(result)
    return torch.stack(out)

def lIntegrand(alpha:Tensor, zstar:Tensor, model) -> Tensor:
    out = 1. / torch.sqrt(_h(model, alpha)*_f(model, alpha) * (torch.pow(_h(model, alpha), 2) * zstar**4 / _h(model, zstar)**2 / torch.pow(alpha, 4) - 1))
    return torch.clamp(out, max=1e8)

def l_integral(model, zstar:Tensor, N_GL:int=12) -> Tensor:
    eps = 1e-8
    integrator = GaussLegendre()
    out = []
    for zstar_i in zstar:
        integration_domain = torch.tensor([[0, zstar_i-eps]])
        func = partial(lIntegrand, model=model, zstar=zstar_i)
        result = 2 * integrator.integrate(func, integration_domain=integration_domain, N=N_GL, dim=1)
        out.append(result)
    return torch.stack(out)

if __name__ == "__main__":
    # load data
    dir = Path("data")
    path = dir / "data.pkl"
    print(f"Loading data from {path}")
    data_loaded = pickle.load(open(path, "rb"))
    print(data_loaded[(0.5, 0.5)]["s_thermal"])

    model = LinearAxion_NN(input_dim=1, output_dim=1, hidden_layers=[32, 32])

    print(f"{get_event_horizon(_f)=}")
    print(f"{get_thermal_entropy(_h, get_event_horizon(_f))=}")

    print(f"{get_event_horizon(f_true)=}")
    print(f"{get_thermal_entropy(h_true, get_event_horizon(f_true))=}")


    f_out, h_out = model(torch.tensor([0.5]))
    print(f_out, h_out)

    zstar_list = torch.tensor(np.linspace(0.1, 0.999, 50), requires_grad=False)

    # with torch.no_grad():
    f_out, h_out = model(zstar_list.unsqueeze(1))

    print(f"f_out: {f_out[:3]}")
    print(f"h_out: {h_out[:3]}")

    s = S_integral(model, zstar_list, N_GL=12)
    l = l_integral(model, zstar_list, N_GL=12)
    print(f"S_integral: {s}")

    # try gradients
    f_out[0].backward()
    s[0].backward()

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    plt.sca(axs[0])
    plt.plot(zstar_list.detach().cpu().numpy(), f_out.detach().cpu().numpy(), label="f")
    plt.plot(zstar_list.detach().cpu().numpy(), h_out.detach().cpu().numpy(), label="h")
    plt.xlabel("zstar")
    plt.ylabel("f, h")
    plt.title("f and h functions for randomly initialized model")
    plt.legend()

    plt.sca(axs[1])
    plt.plot(zstar_list.detach().cpu().numpy(), s.detach().cpu().numpy(), label="S")
    plt.plot(zstar_list.detach().cpu().numpy(), l.detach().cpu().numpy(), label="l")
    plt.xlabel("zstar")
    plt.ylabel("S, l")
    plt.legend()

    plt.sca(axs[2])
    plt.plot(l.detach().cpu().numpy(), s.detach().cpu().numpy(), label="S vs l")
    plt.plot(data_loaded[(0.5, 0.5)]["l"], data_loaded[(0.5, 0.5)]["SFinite"], label="S vs l (data)")
    plt.ylim(-3, 5.5)
    plt.ylabel("S")
    plt.xlabel("l")
    plt.legend()
    plt.show()


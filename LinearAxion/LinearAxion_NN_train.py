import numpy as np
from sympy import Line
import torch
from torch import Tensor
import sys, os
import pickle
import scipy
from scipy.optimize import curve_fit
import torch.optim as optim

from torchquad import set_up_backend  # Necessary to enable GPU support
from torchquad import Trapezoid, Simpson, Boole, MonteCarlo, VEGAS, GaussLegendre # The available integrators
from torchquad.utils.set_precision import set_precision
import torchquad
from functools import partial
import matplotlib.pyplot as plt
from pathlib import  Path
import torch.nn as nn

from LinearAxion_NN import LinearAxion_NN, S_integral, l_integral, _h, _f
from LinearAxion import power_law_fit, f_for_fit, get_thermal_entropy

# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float64")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

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

def interpolate_S_l(data):
    popt, pcov = power_law_fit(data)
    return popt


if __name__ == "__main__":

    # load data
    dir = Path("data")
    path = dir / "data.pkl"
    print(f"Loading data from {path}")
    data_loaded = pickle.load(open(path, "rb"))
    mu = 1.0
    beta = 1.5
    N_zstar_points = 100
    S_true = data_loaded[(mu, beta)]["SFinite"]
    l_true = data_loaded[(mu, beta)]["l"]
    zstar = data_loaded[(mu, beta)]["zstar"]
    s_thermal_true = torch.tensor(data_loaded[(mu, beta)]["s_thermal"], requires_grad=False)
    print(f"{zstar.shape=}, {S_true.shape=}, {s_thermal_true.shape=}")

    print(data_loaded[(mu, beta)]["SFinite"])
    print(data_loaded[(mu, beta)]["l"])
    print(data_loaded[(mu, beta)]["zstar"])
    # S_l_interpolated = interpolate_S_l(data_loaded[(mu, beta)])
    popt = interpolate_S_l(data_loaded[(mu, beta)])

    zstar_list = torch.tensor(np.linspace(0.10, 0.99, N_zstar_points), requires_grad=False)
    # model = LinearAxion_NN(input_dim=1, output_dim=1, hidden_layers=[32, 32])
    model = LinearAxion_NN(input_dim=1, output_dim=1, hidden_layers=[20, 20, 20])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 200, 300], gamma=0.1)

    def h_helper(z):
        return _h(model, z)

    epochs = 350
    loss_hist = []
    s_thermal_loss_hist = []
    S_loss_hist = []
    lr_hist = []
    for epoch in range(350):
        S_pred = S_integral(model, zstar_list, N_GL=12)
        l_pred = l_integral(model, zstar_list, N_GL=12)
        s_thermal_pred = get_thermal_entropy(h_helper, zh=torch.tensor(1.0))
        losses = calc_loss(S_pred, l_pred, s_thermal_pred, s_thermal_true, f_for_fit, popt)
        loss = losses["loss"]
        lr = optimizer.param_groups[0]["lr"]
        print(f"e={epoch}/{epochs}, s_thermal={s_thermal_pred.item():.3f}, lr={lr}, loss={loss.item():.3f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())
        s_thermal_loss_hist.append(losses["s_thermal_loss"].item())
        S_loss_hist.append(losses["S_loss"].item())
        lr_hist.append(lr)

    plt.plot(l_pred.detach().cpu().numpy(), S_pred.detach().cpu().numpy(), label="NN")
    plt.plot(l_true, S_true, label="True")
    l_list = np.linspace(0, 3.5, 100)
    interpolated_S = np.array([f_for_fit(l, *popt) for l in l_list])
    plt.plot(l_list, interpolated_S, label="Interpolated", ls="--")
    plt.xlabel("l")
    plt.ylabel("S")
    plt.title(f"beta={beta}, mu={mu}")
    plt.legend()
    plt.ylim(-5, 5.5)
    plt.show()





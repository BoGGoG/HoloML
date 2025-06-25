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
import torch.nn as nn

# from LinearAxion_NN import LinearAxion_NN, S_integral, l_integral, _h, _f
# from LinearAxion import power_law_fit, f_for_fit, get_thermal_entropy
from Gubser_Rocha import (
    GubserRocha_NN,
    S_integral_NN,
    l_integral_NN,
    _h,
    _f,
    get_event_horizon,
    get_thermal_entropy,
)

# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float64")
torch.set_default_dtype(torch.float64)
# torch.set_default_tensor_type("torch.cuda.FloatTensor")


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


def f_for_fit(x, cn1, c0, c1, c2, c3, c4, c5):
    return cn1 * x ** (-1) + c0 + c1 * x + c2 * x**2 + c3 * x**3 + c4 * x**4 + c5 * x**5


def power_law_fit(data):
    S = np.array(data["SFinite"]).flatten()
    l = np.array(data["l"]).flatten()
    zstar = np.array(data["zstar"]).flatten()

    print("power law fit")
    print(f"{S.shape=}, {l.shape=}, {zstar.shape=}")

    idxs = np.where((zstar > 0.1) & (zstar < 0.99))[0]
    S = S[idxs]
    l = l[idxs]
    print(f"{S=}")
    print(f"{l=}")

    popt, pcov = curve_fit(f_for_fit, l, S, maxfev=100000)
    return popt, pcov


def interpolate_S_l(data):
    popt, pcov = power_law_fit(data)
    return popt


if __name__ == "__main__":
    # load data
    dir = Path("data")
    path = dir / "data_Gubser_Rocha.pkl"
    print(f"Loading data from {path}")
    data_loaded = pickle.load(open(path, "rb"))
    mu = 0.5
    beta = 0.5
    Q = data_loaded[(mu, beta)]["Q"]
    N_zstar_points = 200

    clipping_value = 0.1  # arbitrary value of your choosing
    S_true = data_loaded[(mu, beta)]["SFinite"]
    l_true = data_loaded[(mu, beta)]["l"]
    zstar = data_loaded[(mu, beta)]["zstar"]
    s_thermal_true = torch.tensor(
        data_loaded[(mu, beta)]["s_thermal"], requires_grad=False
    )
    print(f"{zstar.shape=}, {S_true.shape=}, {s_thermal_true.shape=}")

    print(data_loaded[(mu, beta)]["SFinite"])
    print(data_loaded[(mu, beta)]["l"])
    print(data_loaded[(mu, beta)]["zstar"])

    popt = interpolate_S_l(data_loaded[(mu, beta)])
    print(f"Fitted parameters for mu={mu}, beta={beta}:")
    print(popt)

    zstar_list = torch.tensor(
        np.linspace(0.10, 0.999, N_zstar_points),
        requires_grad=False,
        dtype=torch.float64,
    )
    print(f"{zstar_list.dtype=}")
    model = GubserRocha_NN(input_dim=1, output_dim=1, hidden_layers=[32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 120, 200, 300, 500, 700, 900], gamma=0.5
    )

    def h_helper(z, Q):
        return _h(model, z)

    epochs = 30
    loss_hist = []
    s_thermal_loss_hist = []
    S_loss_hist = []
    lr_hist = []
    for epoch in range(epochs):
        zstar_list_noise = (
            zstar_list + torch.randn_like(zstar_list) * 0.005
        )  # add some noise
        zstar_list_noise = torch.clamp(
            zstar_list_noise, min=0.10, max=0.9999
        )  # avoid numerical issues
        if epoch == epochs:
            N_zstar_points = 1000
            zstar_list = torch.tensor(
                np.linspace(0.10, 0.999, N_zstar_points),
                requires_grad=False,
                dtype=torch.float64,
            )
        S_pred = S_integral_NN(model, zstar_list_noise, N_GL=12)
        l_pred = l_integral_NN(model, zstar_list_noise, N_GL=12)
        s_thermal_pred = get_thermal_entropy(h_helper, zh=torch.tensor(1.0), Q=Q)
        losses = calc_loss(
            S_pred, l_pred, s_thermal_pred, s_thermal_true, f_for_fit, popt
        )
        loss = losses["loss"]
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"e={epoch}/{epochs}, s_thermal={s_thermal_pred.item():.3f}, lr={lr}, loss={loss.item():.3f}"
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())
        s_thermal_loss_hist.append(losses["s_thermal_loss"].item())
        S_loss_hist.append(losses["S_loss"].item())
        lr_hist.append(lr)
        if loss.item is None or np.isnan(loss.item()):
            print("Loss is None or NaN, stopping training.")
            break

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

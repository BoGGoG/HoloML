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
from BTZ import (
    BTZ_NN,
    S_integral_NN,
    l_integral_NN,
    _h,
    _f,
    f_true,
    h_true,
    # get_event_horizon,
    get_thermal_entropy,
)

set_up_backend("torch", data_type="float64")
torch.set_default_dtype(torch.float64)


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

    popt, pcov = curve_fit(f_for_fit, l, S, maxfev=100000)
    return popt, pcov


def interpolate_S_l(data):
    popt, pcov = power_law_fit(data)
    return popt


if __name__ == "__main__":
    # Set up the backend for torchquad
    set_up_backend("torch")

    # Define the parameters for the BTZ black hole
    c, v, beta = 24 * np.pi, 1.0, 2 * np.pi
    N_zstar_points = 200
    clipping_value = 1.0

    # load data
    dir = Path("data")
    path = dir / "data_BTZ.pkl"
    print(f"Loading data from {path}")
    data_loaded = pickle.load(open(path, "rb"))
    S_true = data_loaded[(c, v, beta)]["SFinite"]
    l_true = data_loaded[(c, v, beta)]["l"]
    zstar_true = data_loaded[(c, v, beta)]["zstar"]
    s_thermal_true = torch.tensor(
        data_loaded[(c, v, beta)]["s_thermal"], dtype=torch.float64
    )
    print(f"{s_thermal_true=}")

    popt = interpolate_S_l(data_loaded[(c, v, beta)])
    print(f"Fitted parameters for {(c, v, beta)=}:")
    print(popt)

    zstar_list = torch.tensor(
        # np.linspace(0.10, 0.999, N_zstar_points),
        np.linspace(0.10, 0.999, N_zstar_points),
        requires_grad=False,
        dtype=torch.float64,
    )

    model = BTZ_NN(input_dim=1, output_dim=1, hidden_layers=[16, 16])
    model.to(torch.float64)  # Ensure model uses float64 precision
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 120, 200, 300, 500, 700, 900], gamma=0.5
    )

    def h_helper(z):
        return _h(model, z)

    epochs = 500
    loss_hist = []
    s_thermal_loss_hist = []
    S_loss_hist = []
    lr_hist = []

    for epoch in range(epochs):
        zstar_list_noise = (
            # zstar_list + torch.randn_like(zstar_list) * 0.005
            zstar_list + torch.randn_like(zstar_list) * 0.00001
        )  # add some noise
        zstar_list_noise = torch.clamp(
            zstar_list_noise, min=0.10, max=0.9999
        )  # avoid numerical issues
        if epoch == epochs:
            N_zstar_points = 1000
            zstar_list_noise = torch.tensor(
                np.linspace(0.10, 0.999, N_zstar_points),
                requires_grad=False,
                dtype=torch.float64,
            )
        S_pred = S_integral_NN(model, zstar_list_noise, N_GL=12)
        S_pred = (c / 3.0) * S_pred
        l_pred = l_integral_NN(model, zstar_list_noise, N_GL=12)
        s_thermal_pred = get_thermal_entropy(
            h_helper, zh=torch.tensor(1.0, dtype=torch.float64)
        )

        losses = calc_loss(
            S_pred, l_pred, s_thermal_pred, s_thermal_true, f_for_fit, popt
        )
        loss = losses["loss"]
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"e={epoch}/{epochs}, thermal_loss={losses['s_thermal_loss'].item():.4f}, lr={lr}, loss={loss.item():.3f}"
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

    fig, ax = plt.subplots(5, 1, figsize=(10, 20))

    # print(f"{S_pred=}")
    # S_pred = (c / 6.0) * S_pred

    print(f"{model.fac1=}")
    print(f"{model.fac2=}")
    print(f"{model.fac1/model.fac2=}")
    print(f"{model.fac2/model.fac1=}")

    # S(l)
    plt.sca(ax[0])
    plt.plot(
        l_pred.detach().cpu().numpy(),
        S_pred.detach().cpu().numpy(),
        label="NN",
        c="blue",
        lw=2,
    )
    plt.scatter(l_true, S_true, label="True", c="red", s=1)
    l_list = np.linspace(0, 3.5, 100)
    interpolated_S = np.array([f_for_fit(l, *popt) for l in l_list])
    plt.plot(l_list, interpolated_S, label="Interpolated", ls="--")
    plt.xlabel(r"$\ell$")
    plt.ylabel("S")
    plt.legend()
    # plt.ylim(-5, 2.5)

    # l(zstar)
    plt.sca(ax[1])
    plt.plot(
        zstar_list.detach().cpu().numpy(),
        l_pred.detach().cpu().numpy(),
        label=r"$\ell$ (NN)",
    )
    plt.scatter(zstar_true, l_true, label=r"$\ell$ (True)", c="red", s=1)
    plt.xlabel(r"$z_\star$")
    plt.ylabel(r"$\ell$")
    plt.legend()

    # S(zstar)
    plt.sca(ax[2])
    plt.plot(
        zstar_list.detach().cpu().numpy(),
        S_pred.detach().cpu().numpy(),
        label=r"$S$ (NN)",
    )
    plt.scatter(zstar_true, S_true, label=r"$S$ (True)", c="red", s=1)
    plt.xlabel(r"$z_\star$")
    plt.ylabel(r"$S$")
    plt.legend()

    # h(z)
    zstar_list = torch.tensor(
        np.linspace(0.10, 0.999, N_zstar_points),
        requires_grad=False,
        dtype=torch.float64,
    )
    h_pred = np.array([_h(model, zs).detach().cpu().numpy() for zs in zstar_list])
    h_true = h_true(zstar_list.detach().cpu()).numpy()
    zstar_list = zstar_list.detach().cpu().numpy()
    plt.sca(ax[3])
    plt.plot(zstar_list, h_pred, label=r"$h$ (NN)", c="blue")
    plt.plot(zstar_list, h_true, label=r"$h$ (True)", c="red")
    plt.xlabel(r"$z_\star$")
    plt.ylabel(r"$h$")
    plt.legend()

    # f(z)
    zstar_list = torch.tensor(
        np.linspace(0.10, 0.999, N_zstar_points),
        requires_grad=False,
        dtype=torch.float64,
    )
    f_pred = np.array([_f(model, zs).detach().cpu().numpy() for zs in zstar_list])
    f_true = f_true(zstar_list.detach().cpu()).numpy()
    zstar_list = zstar_list.detach().cpu().numpy()
    plt.sca(ax[4])
    plt.plot(zstar_list, f_pred, label=r"$f$ (NN)", c="blue")
    plt.plot(zstar_list, f_true, label=r"$f$ (True)", c="red")
    plt.xlabel(r"$z_\star$")
    plt.ylabel(r"$f$")
    plt.legend()
    plt.show()

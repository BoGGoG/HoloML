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
    # get_thermal_entropy,
)

set_up_backend("torch", data_type="float64")
torch.set_default_dtype(torch.float64)


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

    popt = interpolate_S_l(data_loaded[(c, v, beta)])
    print(f"Fitted parameters for (c, v, beta) = {(c, v, beta)}:")
    print(popt)

    zstar_list = torch.tensor(
        np.linspace(0.10, 0.999, N_zstar_points),
        requires_grad=False,
        dtype=torch.float64,
    )

    model = BTZ_NN(input_dim=1, output_dim=1, hidden_layers=[32, 32])
    model.to(torch.float64)  # Ensure model uses float64 precision
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 120, 200, 300, 500, 700, 900], gamma=0.5
    )

    def h_helper(z, Q):
        return _h(model, z)

    epochs = 2
    loss_hist = []
    s_thermal_loss_hist = []
    S_loss_hist = []
    lr_hist = []

    print(f"{zstar_list.dtype=}")
    asdf = model(zstar_list[0:1])
    print(f"{asdf.shape=}")

    # for epoch in range(epochs):
    #     zstar_list_noise = (
    #         zstar_list + torch.randn_like(zstar_list) * 0.005
    #     )  # add some noise
    #     zstar_list_noise = torch.clamp(
    #         zstar_list_noise, min=0.10, max=0.9999
    #     )  # avoid numerical issues
    #     if epoch == epochs:
    #         N_zstar_points = 1000
    #         zstar_list = torch.tensor(
    #             np.linspace(0.10, 0.999, N_zstar_points),
    #             requires_grad=False,
    #             dtype=torch.float64,
    #         )
    #     S_pred = S_integral_NN(model, zstar_list_noise, N_GL=12)
    #     l_pred = l_integral_NN(model, zstar_list_noise, N_GL=12)

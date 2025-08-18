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
    print(f"{S=}")
    print(f"{l=}")

    popt, pcov = curve_fit(f_for_fit, l, S, maxfev=100000)
    return popt, pcov


def interpolate_S_l(data):
    popt, pcov = power_law_fit(data)
    return popt


if __name__ == "__main__":
    # Set up the backend for torchquad
    set_up_backend("torch")

    # Define the parameters for the BTZ black hole
    c, v, beta = 24 * np.pi, 2 * np.pi, 1.0

    # load data
    dir = Path("data")
    path = dir / "data_BTZ.pkl"
    print(f"Loading data from {path}")
    data_loaded = pickle.load(open(path, "rb"))
    S_true = data_loaded[(c, beta, v)]["SFinite"]
    l_true = data_loaded[(c, beta, v)]["l"]
    zstar_true = data_loaded[(c, beta, v)]["zstar"]
    s_thermal_true = torch.tensor(
        data_loaded[(c, beta, v)]["s_thermal"], dtype=torch.float64
    )

    popt = interpolate_S_l(data_loaded[(c, beta, v)])
    print(f"Fitted parameters for {(c, beta, v)}:")
    print(popt)

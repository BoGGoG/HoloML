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

from LinearAxion_NN import LinearAxion_NN

# Use this to enable GPU support and set the floating point precision
set_up_backend("torch", data_type="float64")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == "__main__":

    # load data
    dir = Path("data")
    path = dir / "data.pkl"
    print(f"Loading data from {path}")
    data_loaded = pickle.load(open(path, "rb"))

    model = LinearAxion_NN(input_dim=1, output_dim=1, hidden_layers=[32, 32])


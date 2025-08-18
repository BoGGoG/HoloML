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
from torch import Tensor


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
        data[(cvb[0], cvb[1], cvb[2])] = {
            "zstar": zstar_arr.cpu().numpy(),
            "l": l_arr.cpu().numpy(),
            "S": S_arr.cpu().numpy(),
        }
    return data


if __name__ == "__main__":
    NZstar = 1000
    cvbeta = np.array([[24 * np.pi, 1, 2 * np.pi]])
    data = generate_BTZ_data(cvbeta=cvbeta, Nzstar=NZstar)

    l = data[(cvbeta[0, 0], cvbeta[0, 1], cvbeta[0, 2])]["l"]
    S = data[(cvbeta[0, 0], cvbeta[0, 1], cvbeta[0, 2])]["S"]
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

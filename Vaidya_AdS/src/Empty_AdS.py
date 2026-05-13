import numpy as np
import matplotlib.pyplot as plt


def empty_ads_geodesic_exact(r_star, t_boundary=0.0, r_cut=200.0, n_points=1000):
    """
    Exact half-geodesic in empty AdS3 in ingoing EF coordinates.

    Metric:
        ds^2 = -r^2 dv^2 + 2 dv dr + r^2 dx^2

    Parameters
    ----------
    r_star : float
        Turning-point radius.
    t_boundary : float
        Boundary time t.
    r_cut : float
        UV cutoff radius.
    n_points : int
        Number of samples along the half-geodesic.

    Returns
    -------
    v, r, x : np.ndarray
        Half-geodesic from turning point to cutoff.
    """
    lam_cut = np.arccosh(r_cut / r_star)
    lam = np.linspace(0.0, lam_cut, n_points)

    r = r_star * np.cosh(lam)
    x = (1.0 / r_star) * np.tanh(lam)
    v = t_boundary - 1.0 / r

    return v, r, x


if __name__ == "__main__":
    r_stars = [0.5, 1.0, 2.0, 5.0]
    t_boundary = 0.0
    r_cut = 200.0

    fig, ax = plt.subplots(figsize=(7, 5))

    for r_star in r_stars:
        v, r, x = empty_ads_geodesic_exact(
            r_star,
            t_boundary=t_boundary,
            r_cut=r_cut,
        )

        ax.plot(x, r, label=rf"$r_\ast={r_star}$")
        ax.plot(-x, r)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$r$")
    ax.set_title(r"Exact spacelike geodesics in empty AdS$_3$")
    ax.set_ylim(0, 20)
    ax.legend()
    plt.show()

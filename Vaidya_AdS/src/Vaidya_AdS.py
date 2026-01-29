import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


@jit
def get_mass(v, m_0=1.0, v_s=1.0):
    # Profile from the paper: m(v) = tanh(v) for v_s=1, m_0=1
    # Generally: (m_0 + 1.0)/2.0 *tanh(v/v_s) + (m_0 - 1.0)/2.0
    m = (m_0 + 1.0) / 2.0 * jnp.tanh(v / v_s) + (m_0 - 1.0) / 2.0
    return m


@jit
def get_derivs(state, lam):
    """
    Calculates derivatives [dv, dr, dx, d_dv, d_dr, d_dx]
    based on the geodesic equations for 3D Vaidya-AdS.
    Metric: ds^2 = -f dv^2 + 2 dv dr + r^2 dx^2
    """
    v, r, x, dv, dr, dx = state

    m = get_mass(v)
    dm_dv = 1.0 - jnp.tanh(v) ** 2  # Derivative of tanh is sech^2

    f = r**2 - m
    df_dr = 2 * r
    df_dv = -dm_dv

    # Geodesic Equations derived from the metric:
    # d_dv (v'') = r*dx^2 - r*dv^2
    d_dv = r * dx**2 - r * dv**2

    # d_dr (r'') = f*v'' + (df_dr)*r'*v' + 0.5*(df_dv)*v'^2
    d_dr = f * d_dv + df_dr * dr * dv + 0.5 * df_dv * dv**2

    # d_dx (x'') = -2/r * r' * x'
    d_dx = -2.0 / r * dr * dx

    return jnp.array([dv, dr, dx, d_dv, d_dr, d_dx])


# --- 2. The Integrator (RK4 with jax.lax.scan) ---


@jit
def rk4_step(state, dt):
    k1 = get_derivs(state, 0.0)
    k2 = get_derivs(state + 0.5 * dt * k1, 0.0)
    k3 = get_derivs(state + 0.5 * dt * k2, 0.0)
    k4 = get_derivs(state + dt * k3, 0.0)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jit
def integrate_geodesic(r_star, v0, n_steps=2000, dt=0.005):
    """
    Integrates a single geodesic starting from the turning point (r_star).
    """
    # Initial Conditions at turning point (lambda=0):
    # v = v0
    # r = r_star (minimal radius)
    # x = 0 (symmetry axis)
    # dv = 0 (symmetry)
    # dr = 0 (turning point condition)
    # dx = 1/r_star (from normalization condition: g_uv u^u u^v = 1)

    initial_state = jnp.array([v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star])

    def step_fn(state, _):
        new_state = rk4_step(state, dt)
        return new_state, new_state

    # Integrate forward
    final_state, trajectory = jax.lax.scan(step_fn, initial_state, None, length=n_steps)

    return trajectory


def plot_figure_5():
    v0_values = [-2, -1, 0, 0.1, 0.5, 1]
    r_stars = list(np.linspace(0.0001, 0.0009, 10))
    r_stars = list(np.linspace(0.001, 0.009, 10))
    r_stars += list(np.linspace(0.01, 0.09, 50))
    r_stars += list(np.linspace(0.1, 0.9, 50))
    r_stars += list(np.linspace(1.1, 20.0, 50))

    # Setup Colormap
    norm = Normalize(vmin=min(r_stars), vmax=max(r_stars))
    cmap = plt.cm.cool
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Create figure with specific layout adjustment
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), subplot_kw={"projection": "polar"})

    # Adjust subplots to leave room on the right for the colorbar
    # left, bottom, right, top parameters define the plotting area
    plt.subplots_adjust(right=0.85, wspace=0.3, hspace=0.3)

    axes_flat = axes.flatten()

    for i, v0 in enumerate(v0_values):
        ax = axes_flat[i]

        # Boundary
        theta_boundary = np.linspace(-np.pi / 2, np.pi / 2, 100)
        ax.plot(theta_boundary, [np.pi / 2] * 100, "k-", linewidth=2)

        # Horizon
        m_val = np.tanh(v0)
        if m_val > 0:
            r_horizon = np.sqrt(m_val)  # apparent horizon
            R_horizon = np.arctan(r_horizon)
            theta_h = np.linspace(0, 2 * np.pi, 200)
            ax.plot(theta_h, [R_horizon] * 200, color="brown", linewidth=2)

        # Geodesics
        for r_star in r_stars:
            if m_val > 0 and r_star <= np.sqrt(m_val):
                continue

            traj = integrate_geodesic(r_star, v0)
            v_t, r_t, x_t = traj[:, 0], traj[:, 1], traj[:, 2]
            R_plot = np.arctan(r_t)
            color = cmap(norm(r_star))

            ax.plot(x_t, R_plot, color=color, linewidth=1)
            ax.plot(-x_t, R_plot, color=color, linewidth=1)

        ax.set_title(r"$v_0$" + f"= {v0}", fontsize=12)
        ax.set_ylim(0, np.pi / 2 + 0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines["polar"].set_visible(False)

    # Manually add axes for the colorbar to the right of the subplots
    # [left, bottom, width, height] in figure coordinate fractions
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Turning Point ($r_*$)", fontsize=12)

    plt.show()


if __name__ == "__main__":
    plot_figure_5()

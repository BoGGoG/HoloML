import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import jit
from functools import partial


@jit
def get_mass_and_dmdv(v, m_0=1.0, v_s=1.0):
    """
    Returns m(v) and dm/dv for
        m(v) = (m0+1)/2 * tanh(v/vs) + (m0-1)/2
    """
    z = v / v_s
    t = jnp.tanh(z)
    m = (m_0 + 1.0) / 2.0 * t + (m_0 - 1.0) / 2.0
    sech2 = 1.0 - t**2
    dm_dv = (m_0 + 1.0) / 2.0 * (1.0 / v_s) * sech2
    return m, dm_dv


@jit
def get_derivs(state, lam, m_0=1.0, v_s=1.0):
    """
    Calculates derivatives [dv, dr, dx, d_dv, d_dr, d_dx]
    based on the geodesic equations for 3D Vaidya-AdS.
    Metric: ds^2 = -f dv^2 + 2 dv dr + r^2 dx^2
    """
    v, r, x, dv, dr, dx = state

    m, dm_dv = get_mass_and_dmdv(v, m_0=m_0, v_s=v_s)
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


@jit
def ds_dlambda(state):
    """
    Proper-length element for spacelike curve in Vaidya-AdS:
    ds = sqrt(-f dv^2 + 2 dv dr + r^2 dx^2) dλ
    """
    v, r, x, dv, dr, dx = state
    m, _ = get_mass_and_dmdv(v)
    f = r**2 - m
    inside = -f * dv**2 + 2.0 * dv * dr + (r**2) * dx**2
    return jnp.sqrt(jnp.maximum(inside, 0.0))


def geodesic_length_reg(L, r_cut):
    r"""
    The paper defines a regularized length by subtracting the universal divergence
    $2 \log(2 r_\infty)$
    (see around eq. (6.17); also consistent with the BTZ expression eq. (5.33)).
    """
    return L - 2.0 * jnp.log(2.0 * r_cut)


def speed_stats(traj):
    sdot = np.array(jax.vmap(ds_dlambda)(traj))
    return {
        "min": float(sdot.min()),
        "mean": float(sdot.mean()),
        "max": float(sdot.max()),
        "std": float(sdot.std()),
    }


def lengths_vs_rstar(rstars, v0, n_steps=40000, dt=0.002, r_cut=200.0):
    """
    Compute (regularized) spacelike geodesic lengths in Vaidya–AdS as a function of
    turning point radius r_star.

    For each `r_star` in `rstars`, this function:
      1) Numerically integrates the geodesic from the turning point (r=r_star) outward
         using `integrate_geodesic(...)` for a fixed number of steps `n_steps` and step
         size `dt` (affine parameter increment).
      2) Computes the total proper length L by integrating the line element along the
         trajectory up to a UV cutoff radius `r_cut` via `geodesic_length_from_traj(...)`
         (typically doubling the “half-geodesic” length by symmetry).
      3) Computes the UV-regularized length L_reg via `geodesic_length_reg(L, r_cut)`.
      4) Records diagnostics at the cutoff: the boundary half-width h = x(r_cut) and
         the corresponding advanced time v_inf = v(r_cut).

    Parameters
    ----------
    rstars : array-like
        Iterable of turning point radii r_star (the minimal radius reached by each
        spacelike geodesic).
    v0 : float
        Initial advanced time at the turning point, v(λ=0) = v0.
    n_steps : int, optional
        Number of integration steps for the geodesic solver. Must be a Python `int`
        when passed into JAX-jitted code (hence the explicit `int(n_steps)` cast).
    dt : float, optional
        Step size in the affine parameter used by the geodesic integrator.
    r_cut : float, optional
        UV cutoff radius that defines where the trajectory is “read out” at the
        boundary and where the length regularization is performed.

    Returns
    -------
    Ls : np.ndarray, shape (len(rstars),)
        Total (unregularized) proper lengths L for each r_star.
    Lregs : np.ndarray, shape (len(rstars),)
        UV-regularized lengths L_reg for each r_star.
    hs : np.ndarray, shape (len(rstars),)
        Boundary half-widths h = x(r_cut) extracted at the cutoff.
    vins : np.ndarray, shape (len(rstars),)
        Advanced time values v_inf = v(r_cut) extracted at the cutoff.

    Notes
    -----
    - The cutoff index `hit` is taken as the first index where r >= r_cut. If the
      trajectory never reaches r_cut within `n_steps`, `hit` falls back to the final
      integration step; in that case, the reported (L, L_reg, h, v_inf) correspond to
      the endpoint reached rather than the intended UV cutoff.
    - For performance, keep `n_steps` fixed across sweeps to avoid repeated JAX
      recompilation if `integrate_geodesic` treats `n_steps` as a static argument.
    """
    Ls = []
    Lregs = []
    hs = []  # boundary half-width h = x(r_cut), useful diagnostic
    vins = []  # boundary v value at cutoff, also useful

    for r_star in rstars:
        traj = integrate_geodesic(
            r_star, v0, n_steps=int(n_steps), dt=float(dt), m_0=1.0, v_s=1.0
        )

        # length
        L = geodesic_length_from_traj(traj, dt=dt, r_cut=r_cut)
        Lreg = geodesic_length_reg(L, r_cut=r_cut)

        # where we hit the cutoff
        r = traj[:, 1]
        hit = np.argmax(np.array(r >= r_cut))
        if not np.any(np.array(r >= r_cut)):
            hit = traj.shape[0] - 1

        h = float(traj[hit, 2])  # x at the cutoff
        v_inf = float(traj[hit, 0])

        Ls.append(float(L))
        Lregs.append(float(Lreg))
        hs.append(h)
        vins.append(v_inf)

    return np.array(Ls), np.array(Lregs), np.array(hs), np.array(vins)


def geodesic_length_from_traj(traj, dt, r_cut=200.0):
    """
    traj: (N, 6) array [v, r, x, dv, dr, dx]
    Returns total length L (full geodesic, both sides).
    """
    r = traj[:, 1]
    # first index where r exceeds cutoff
    hit = jnp.argmax(r >= r_cut)
    # if never hits, use full trajectory
    hit = jnp.where(jnp.any(r >= r_cut), hit, traj.shape[0] - 1)

    seg = traj[: hit + 1]
    sdot = jax.vmap(ds_dlambda)(seg)  # ds/dλ along segment

    # trapezoid integral for half-geodesic
    L_half = dt * (0.5 * sdot[0] + jnp.sum(sdot[1:-1]) + 0.5 * sdot[-1])

    return 2.0 * L_half


def length_profile_vs_x(traj, dt, r_cut=200.0):
    """
    Return cumulative proper length along the *half*-geodesic as a function of x.

    Parameters
    ----------
    traj : array, shape (N, 6)
        (v, r, x, dv, dr, dx) samples along affine parameter λ.
    dt : float
        Step size in λ.
    r_cut : float
        UV cutoff radius. We truncate at the first index where r >= r_cut.

    Returns
    -------
    x : np.ndarray, shape (M,)
        x-values along the half-geodesic (from 0 outward).
    L_half : np.ndarray, shape (M,)
        Cumulative proper length from λ=0 to the point with coordinate x.
        (So L_half[0]=0 and L_half increases outward.)
    """
    r = np.array(traj[:, 1])
    # first index where r exceeds cutoff (or last point if never hits)
    hit = int(np.argmax(r >= r_cut))
    if not np.any(r >= r_cut):
        hit = traj.shape[0] - 1

    seg = traj[: hit + 1]

    # ds/dλ along segment (use your existing ds_dlambda)
    sdot = np.array(jax.vmap(ds_dlambda)(seg))

    # cumulative trapezoid integral:
    # L[k] = ∫_0^{λ_k} sdot(λ) dλ
    incr = 0.5 * (sdot[:-1] + sdot[1:]) * float(dt)
    L = np.concatenate([[0.0], np.cumsum(incr)])

    x = np.array(seg[:, 2], dtype=float)

    # For safety: enforce monotonic x (should be monotone on the half-geodesic)
    order = np.argsort(x)
    return x[order], L[order]


@jit
def rk4_step(state, dt, m_0=1.0, v_s=1.0):
    k1 = get_derivs(state, 0.0, m_0=m_0, v_s=v_s)
    k2 = get_derivs(state + 0.5 * dt * k1, 0.0, m_0=m_0, v_s=v_s)
    k3 = get_derivs(state + 0.5 * dt * k2, 0.0, m_0=m_0, v_s=v_s)
    k4 = get_derivs(state + dt * k3, 0.0, m_0=m_0, v_s=v_s)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@partial(jax.jit, static_argnames=("n_steps",))
def integrate_geodesic(r_star, v0, n_steps=2000, dt=0.005, m_0=1.0, v_s=1.0):
    """
    Integrate a single *spacelike* geodesic in 3D Vaidya–AdS starting from its turning point.

    This routine evolves the first-order state vector

        state(λ) = (v(λ), r(λ), x(λ), v'(λ), r'(λ), x'(λ))

    where prime denotes derivative w.r.t. the affine parameter λ, using a fixed-step
    fourth-order Runge–Kutta (RK4) integrator inside a `jax.lax.scan` loop.

    Geometry / equations
    --------------------
    The background is the (2+1)-dimensional Vaidya–AdS metric in ingoing EF-like
    coordinates:

        ds^2 = -f(r,v) dv^2 + 2 dv dr + r^2 dx^2,   with  f(r,v) = r^2 - m(v).

    The function `get_derivs(...)` implements the corresponding geodesic equations
    as a first-order system returning (v', r', x', v'', r'', x'').

    Initial conditions (turning point)
    ----------------------------------
    The integration starts at λ = 0 at the symmetric turning point:

        v(0)  = v0
        r(0)  = r_star            (minimal radius)
        x(0)  = 0                 (symmetry axis)
        v'(0) = 0                 (symmetry)
        r'(0) = 0                 (turning point condition)
        x'(0) = 1 / r_star        (normalization choice for a unit-speed spacelike curve at λ=0)

    Only the “forward” half-geodesic (from r=r_star outward) is integrated; when
    computing observables like the full geodesic length, the other half is typically
    obtained by symmetry (doubling).

    Parameters
    ----------
    r_star : float
        Turning point radius (minimal r reached by the geodesic).
    v0 : float
        Advanced time at the turning point, v(0)=v0.
    n_steps : int, optional
        Number of RK4 steps to take. **Must be a Python int** because this function is
        JIT-compiled and `lax.scan(..., length=n_steps)` requires a static loop length.
        Changing `n_steps` will trigger recompilation.
    dt : float, optional
        Step size Δλ for the affine parameter.

    Returns
    -------
    trajectory : jax.Array, shape (n_steps, 6)
        The integrated state at each step (v, r, x, v', r', x').

    Notes
    -----
    - This function does not implement an event-based stop condition (e.g. stop when
      r reaches a UV cutoff r_cut). If you need that, truncate the returned trajectory
      afterward, or switch to a `lax.while_loop` integrator.
    - Because the routine is JIT-compiled, passing non-Python scalars for `n_steps`
      (e.g. numpy/jax int types) will raise a ConcretizationTypeError.
    """

    initial_state = jnp.array([v0, r_star, 0.0, 0.0, 0.0, 1.0 / r_star])

    def step_fn(state, _):
        new_state = rk4_step(state, dt, m_0=m_0, v_s=v_s)
        return new_state, new_state

    # Integrate forward
    final_state, trajectory = jax.lax.scan(step_fn, initial_state, None, length=n_steps)

    return trajectory

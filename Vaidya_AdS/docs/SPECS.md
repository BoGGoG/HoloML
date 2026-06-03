# PROJECT SPECS

> **FILE ROLE:** Primary source of truth for physics conventions, numerical architecture, validation targets, and ML scope.
> **USAGE:** Read this before implementing or changing simulation, data-generation, or reconstruction code.
> **DO NOT:** Put daily progress, transient TODOs, or long-form research reasoning here. Use `docs/STATUS.md` for active status and `docs/JOURNAL.md` for research decisions.

---

## 1. Project Overview

The long-term goal is to reconstruct time-dependent asymptotically AdS spacetime, especially AdS$_3$-Vaidya, from boundary entanglement data using machine learning.

The near-term goal is more constrained:

1. Build and validate a forward model that computes HRT geodesics in AdS$_3$-Vaidya.
2. Generate reliable synthetic boundary data of the form

   $$
   (\ell,t) \mapsto L_{\mathrm{reg}}(\ell,t),
   $$

   where $\ell$ is the boundary interval size, $t$ is the boundary time, and $L_{\mathrm{reg}}$ is the UV-regularized bulk geodesic length.
3. Start inverse learning with a constrained parametric reconstruction of the Vaidya metric for constant $v$ slices. 
4. Only after that attempt more general metric reconstruction.

The project should not jump directly to unconstrained neural metric reconstruction before the forward model and inverse identifiability are tested.

---

## 2. Coordinate and Unit Conventions

### Units

Use dimensionless AdS units unless stated otherwise:

$$
L_{\mathrm{AdS}} = 1.
$$

Newton's constant and central-charge factors are usually omitted during numerical training. The primary geometric observable is the regularized geodesic length $L_{\mathrm{reg}}$. Entanglement entropy can be restored through

$$
S_A = \frac{L_{\mathrm{reg}}}{4G_N}
$$

for AdS$_3$/CFT$_2$ single-interval HRT geodesics.

### Radial Coordinates

Two radial coordinates are used:

- Vaidya numerics use the inverse-Poincaré radial coordinate $r$.
- Static BTZ / ML comparison code may use

  $$
  z = \frac{1}{r}.
  $$

The AdS boundary is

$$
r \to \infty
\qquad\text{or equivalently}\qquad
z \to 0.
$$

The BTZ horizon with final mass $m_f=1$ is

$$
r_h = 1,
\qquad
z_h = 1.
$$

Any new code must state clearly whether it uses $r$ or $z$.

---

## 3. Static BTZ Baseline

The static non-rotating BTZ black hole is the late-time limit of the Vaidya geometry when $m(v)\to 1$.

In $r$ coordinates:

$$
ds^2 = -\left(r^2-1\right)dt^2
+ \frac{dr^2}{r^2-1}
+ r^2 dx^2.
$$

In $z=1/r$ coordinates, use the convention

$$
ds^2 = \frac{1}{z^2}
\left[
-f(z)dt^2
+\frac{dz^2}{f(z)}
+h(z)dx^2
\right],
$$

with

$$
f(z)=1-z^2,
\qquad
h(z)=1.
$$

For a geodesic with turning point $z_\ast$, equivalently $r_\ast=1/z_\ast$, the boundary interval length is

$$
\ell(z_\ast)=2\,\operatorname{arctanh}(z_\ast)
=
2\,\operatorname{arctanh}\left(\frac{1}{r_\ast}\right).
$$

This relation is a primary validation benchmark.

---

## 4. AdS$_3$-Vaidya Geometry

The main dynamical geometry is AdS$_3$-Vaidya in ingoing Eddington-Finkelstein-like coordinates:

$$
ds^2
=
-f(r,v)dv^2
+2\,dv\,dr
+r^2 dx^2,
$$

with

$$
f(r,v)=r^2-m(v).
$$

Here:

- $v$ is the ingoing null / advanced time coordinate.
- At the boundary, $v$ coincides with the field-theory time $t$ up to the usual asymptotic identification.
- $x$ is the boundary spatial coordinate.
- $m(v)$ is the time-dependent mass profile of the collapsing shell.

### Preferred Physical Mass Profile

Use the following convention for new work unless explicitly testing legacy code:

$$
m(v)
=
m_i
+
\frac{m_f-m_i}{2}
\left[
1+\tanh\left(\frac{v-v_c}{v_s}\right)
\right].
$$

Parameters:

- $m_i$: initial mass.
- $m_f$: final mass.
- $v_c$: shell center.
- $v_s$: shell thickness.

For vacuum-to-BTZ collapse:

$$
m_i=0,
\qquad
m_f=1.
$$

The apparent horizon is located at

$$
r_{\mathrm{AH}}(v)=\sqrt{m(v)}
$$

when $m(v)>0$.

### Legacy Note

Older code may use a different profile. Before using legacy data for ML, check whether the early-time mass becomes negative. If it does, record this clearly and do not treat that run as a physical vacuum-to-BTZ quench.

### Early-Time Limit: Exact Geodesics in Empty AdS$_3$

Setting $m=0$ gives exact Poincaré AdS$_3$. Static spacelike geodesics at boundary time $t_{\mathrm{bdy}}$ admit the exact closed-form solution

$$
r(\lambda) = r_\ast \cosh\lambda,
\qquad
x(\lambda) = \frac{\tanh\lambda}{r_\ast},
\qquad
v(\lambda) = t_{\mathrm{bdy}} - \frac{1}{r(\lambda)}.
$$

The $v$-relation follows exactly from the Poincaré–EF coordinate identity $v = t - 1/r$. The geodesic is unit-speed ($\kappa=1$) throughout, so the affine parameter at the UV cutoff and the resulting lengths are

$$
\lambda_{\mathrm{cut}} = \operatorname{arccosh}\!\left(\frac{r_{\mathrm{cut}}}{r_\ast}\right),
\qquad
L = 2\lambda_{\mathrm{cut}},
\qquad
L_{\mathrm{reg}} = 2\operatorname{arccosh}\!\left(\frac{r_{\mathrm{cut}}}{r_\ast}\right) - 2\log(2r_{\mathrm{cut}}).
$$

This solution is implemented in `src/Empty_AdS.py` and serves as the primary validation reference for the Vaidya solver in the early-time ($v \ll v_c$) limit.

---

## 5. HRT Geodesics

Boundary entanglement entropy for a single interval in AdS$_3$/CFT$_2$ is computed from the length of a spacelike HRT geodesic anchored at the boundary endpoints.

Use an affine parameter $\lambda$ and state vector

$$
y(\lambda)
=
\left(
v(\lambda),
r(\lambda),
x(\lambda),
\dot v(\lambda),
\dot r(\lambda),
\dot x(\lambda)
\right),
$$

where dot denotes $d/d\lambda$.

The spacelike norm is

$$
\kappa
=
-f(r,v)\dot v^2
+2\dot v\dot r
+r^2\dot x^2.
$$

For the normalized initial conditions used in the current solver, target

$$
\kappa = 1.
$$

### Symmetric Turning-Point Initial Conditions

For a half-geodesic integrated from the midpoint outward:

$$
v(0)=v_\ast,
\qquad
r(0)=r_\ast,
\qquad
x(0)=0,
$$

and

$$
\dot v(0)=0,
\qquad
\dot r(0)=0,
\qquad
\dot x(0)=\frac{1}{r_\ast}.
$$

This gives $\kappa=1$ at the turning point.

The full geodesic is obtained by reflection symmetry, so the full boundary separation and full length are

$$
\ell = 2x(r_{\mathrm{cut}}),
$$

$$
L = 2L_{\mathrm{half}}.
$$

### Geodesic Equations Used by the Solver

Derived from the Euler-Lagrange equations for $\mathcal{L} = g_{\mu\nu}\dot x^\mu \dot x^\nu$, the general geodesic equations for the metric $ds^2 = -f\,dv^2+2\,dv\,dr+r^2dx^2$ with arbitrary differentiable $f(r,v)$ are

$$
\ddot v
=
r\dot x^2 - \frac{1}{2}\frac{\partial f}{\partial r}\dot v^2,
$$

$$
\ddot r
=
f\,\ddot v
+
\frac{\partial f}{\partial r}\dot r\dot v
+
\frac{1}{2}\frac{\partial f}{\partial v}\dot v^2,
$$

$$
\ddot x
=
-\frac{2}{r}\dot r\dot x.
$$

For the specific profile $f = r^2 - m(v)$, substituting $\partial f/\partial r = 2r$ and $\partial f/\partial v = -m'(v)$ recovers:

$$
\ddot v = r(\dot x^2 - \dot v^2),
\qquad
\ddot r = f\,\ddot v + 2r\dot r\dot v - \tfrac{1}{2}m'(v)\dot v^2.
$$

The general form is implemented in `scripts/fit_metric_from_lreg.py` via `jax.grad` applied to an arbitrary `f_metric(r, v, params)`. The hardcoded equations above are used in `src/Vaidya_AdS.py` for data generation.

---

## 6. Length and UV Regularization

The proper-length element is

$$
\frac{ds}{d\lambda}
=
\sqrt{
-f(r,v)\dot v^2
+2\dot v\dot r
+r^2\dot x^2
}.
$$

The half-geodesic length is

$$
L_{\mathrm{half}}
=
\int_0^{\lambda_{\mathrm{cut}}}
\frac{ds}{d\lambda}\,d\lambda.
$$

The full length is

$$
L = 2L_{\mathrm{half}}.
$$

The Vaidya numerical regularization convention is

$$
L_{\mathrm{reg}}
=
L
-
2\log(2r_{\mathrm{cut}}).
$$

All comparisons to BTZ analytic formulas must track constant-offset differences between regularization conventions.

Accepted samples must record:

- $r_{\mathrm{cut}}$,
- whether the geodesic reached the cutoff,
- interpolation method used at the cutoff,
- regularization convention.

---

## 7. Boundary Data Map

The native solver produces data in turning-point variables:

$$
(r_\ast,v_\ast)
\mapsto
\left(
\ell,
t_{\mathrm{bdy}},
L_{\mathrm{reg}}
\right).
$$

The ML problem should usually consume boundary-controlled data:

$$
(\ell,t_{\mathrm{bdy}})
\mapsto
L_{\mathrm{reg}}.
$$

Therefore the map

$$
(r_\ast,v_\ast)
\to
(\ell,t_{\mathrm{bdy}})
$$

must be inverted or interpolated.

This inversion/interpolation layer is a core part of the project, not an implementation detail. It should be implemented separately from ML training code.

---

## 8. Numerical Architecture

### Language and Libraries

- Python
- JAX for differentiable numerics and JIT acceleration
- Equinox for neural-network models
- NumPy/SciPy for utilities and validation
- Matplotlib / Plotly / Marimo for visualization and exploration

### Current Numerical Method

The current Vaidya solver uses fixed-step RK4 integration.

Important constraints:

- `n_steps` is static under JAX JIT, so changing it may trigger recompilation.
- Fixed-step integration can miss $r_{\mathrm{cut}}$.
- Production data should interpolate to exactly $r_{\mathrm{cut}}$ rather than using the first discrete point satisfying $r\ge r_{\mathrm{cut}}$.

### Recommended Directory Layout

```text
src/
  btz.py
  vaidya_ads.py
  boundary_map.py
  data_generation.py
  validation.py
notebooks/
  geodesics_plots.py
  vaidya_btz_comparison.py
data/
docs/
  SPECS.md
  STATUS.md
  JOURNAL.md
  literature/
tests/

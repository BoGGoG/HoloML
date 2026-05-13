# CLAUDE.md (AI Rules)

## Project Guidelines
- **Stack:** JAX/Equinox for numerics, Marimo for notebooks, NumPy/SciPy for utilities.
- **Math First:** Always derive or verify physics in `docs/SPECS.md` before writing code.
- **Naming:** Use descriptive physical names (e.g., `angular_velocity`) rather than single letters (`w`).
- **Units:** Explicitly state units in docstrings (e.g., `mass: float # kg`).

## Workflow Rules
1. Before starting a task, read `docs/STATUS.md`.
2. After finishing a task, update `docs/STATUS.md`.
3. If a complex decision is made, document the reasoning in `docs/JOURNAL.md`.
4. Always use LaTeX for math in documentation.

## Hard Rules
- Do not accept geodesic data unless the trajectory reaches `r_cut`.
- Do not save training data without solver metadata: `dt`, `n_steps`, `r_cut`, mass-profile parameters, cutoff-hit status, and git commit if available.
- Do not introduce new coordinate conventions without updating `docs/SPECS.md`.

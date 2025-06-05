import numpy as np
import sympy as sp
from sympy import symbols, Eq, solve, sqrt
from sympy import symbols, sqrt, I, sympify, srepr
from sympy.utilities.lambdify import lambdify

# Define the symbols
Q, beta, mu = symbols("Q beta mu", real=True, positive=True)

if False:
    # Define the expression for μ(Q, β)
    mu_expr = sqrt(3 * Q * (1 + Q) * (1 - beta**2 / (2 * (1 + Q) ** 2)))

    # Solve for Q given mu
    solutions = solve(Eq(mu_expr, mu), Q)

    with open("Q_expr.txt", "w") as f:
        f.write(srepr(solutions[2]))  # this is the positive one

with open("Q_expr.txt") as f:
    Q_expr = sympify(f.read())

beta_val = 0.6
mu_val = 1.2

# # Substitute and evaluate each solution numerically
# numeric_solutions = [
#     sol.subs({beta: beta_val, mu: mu_val}).evalf() for sol in solutions
# ]
# # Display the solutions
# for i, sol in enumerate(numeric_solutions):
#     print(f"Solution {i+1}: Q = {sol}")
#
numeric_solution_loaded = Q_expr.subs({beta: beta_val, mu: mu_val}).evalf()
print(f"Loaded solution: Q = {numeric_solution_loaded}")

Q_numeric = lambdify((mu, beta), Q_expr, modules="sympy")

# check that solution is almost real
print(Q_numeric(mu_val, beta_val).evalf().as_real_imag()[0])
# check that imaginary part is zero
print(Q_numeric(mu_val, beta_val).evalf().as_real_imag()[1] < 1e-10)

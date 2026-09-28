# computational-pdes
Repository containing Python code used for master's thesis research involving numerical solutions to partial differential equations.

Current contents:
- euler.py:
  Implements Method of Lines for the heat equation u_t = u_xx, discretizing x into time steps of length h, and running Euler's method with repeated time steps k,  to obtain approximate solution u(x) for 0 <= x <= 1 and 0 <= t <= 1. Initial boundary conditions are set to 0. u_xx is approximated using central difference scheme. The program produces a graph of the solution.

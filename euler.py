import math
import numpy 
import matplotlib.pyplot as plt

def f(x, y):
    return y * (1 - y) 


# method to solve u_t = u_xx on domain 0 <= x <= 1 and 0 <= t <= 1 using h spatial step size (N number of grid points depends on h)
h = 0.1  
k = h**2 / 2  # time step k <= h**2 / 2
N = int(1/h)
n = N-1
A = numpy.zeros((n, n))
for i in range(n):  # build the matrix A using 1 -2 1 tri-diagonal pattern used for u_xx
    A[i, i] = -2
    if i > 0:
        A[i, i-1] = 1
    if i < n-1:
        A[i, i+1] = 1

print(A)
b = numpy.zeros(n)
b[0] = 0   # set u_0j
b[n-1] = 0   # set u_Nj
# this problem assumes domain 0 <= x <= 1 and 0 <= t <= 1
# for x = 0, u = 0, and for x = 1, u = 0
# for t = 0, u = x * (1-x)
# these are the initial boundary conditions

def g(x):  # define the function along t = 0
    return x * (1-x)

u = numpy.zeros(n)  # initialize the first u vector using the t=0 initial condition g(x)
for i in range(n):
    x = (i+1) * h
    u[i] = g(x)

def F(u):
    return 1 / h**2 * (A @ u + b)

def euler(k, initial_u, tolerance):  # time step k = delta t, initial u, tolerance
    m = int(1/k)
    u = initial_u
    w = initial_u.copy()
    k2 = k / 2
    solutions = [u.copy()]
    solutions_refined = [w.copy()]
    for j in range(5):
        # print(f'u_{j} = {u}')  # j is essentially what time step iteration it's on. this prints [u_1j, u_2j, ..., u_{n-1}j]
        u = u + k * F(u) 
        w = w + k2 * F(w)
        solutions_refined.append(w.copy())
        w = w + k2 * F(w)
        solutions_refined.append(w.copy())
        solutions.append(u.copy())
    print(u, w)  # the approximate solutions after 5 steps of initial k (10 steps of halved k) (interior points)
    k = k / 2
    k2 = k2 / 2
    refinements = 1

    u_old = solutions[-1]
    w_old = solutions_refined[-1]
    while numpy.linalg.norm(u_old - w_old, ord = 2) > tolerance:
        u = initial_u 
        w = initial_u.copy()
        k2 = k / 2
        m = int(1/k)
        solutions = [u.copy()]
        solutions_refined = [w.copy()]
        
        for j in range(m):
            # print(f'u_{j} = {u}')
            u = u + k * F(u)
            w = w + k2 * F(w)
            solutions_refined.append(w.copy())
            w = w + k2 * F(w)
            solutions_refined.append(w.copy())
            solutions.append(u.copy())
        print(u, w)  # the final u and w values, which represents the solutions along t = 1 (interior points) using k and k2
        u_old = solutions[-1]
        w_old = solutions_refined[-1]
        
        refinements += 1
        k = k / 2
        k2 = k2 / 2

    print(f"Tolerance reached in {refinements} refinements of the time step k")
    print(f"Final time step size k = {k}")

    m = int(1/k)
    w = initial_u.copy()
    solutions_refined = [w.copy()]
    for j in range(m):
        w = w + k * F(w)
        solutions_refined.append(w.copy())
    print(w)  # the final approximation at t = 1


    return solutions_refined, k



def euler_simple(f, x_0, y_0, h, tolerance):
    x_initial = x_0 
    y_initial = y_0  
    n = int(2 / h)

    for i in range(n):
        y_0 = y_0 + h * f(x_0, y_0)
        x_0 += h 
    print(y_0)
    y_prev = y_0 
    h = h / 2
    refinements = 1

    while True:
        n = int(2 / h)  # target x is 2
        x_0 = x_initial
        y_0 = y_initial
        for i in range(n): 
            y_0 = y_0 + h * f(x_0, y_0)
            x_0 = x_initial + (i+1) * h
            # exact = math.exp(x_0) - x_0 - 1 
        refinements += 1
        print(y_0)
        if abs(y_0 - y_prev) < tolerance:
            break
        y_prev = y_0 
        h = h / 2
    print(f'Final solution: ({x_0}, {y_0}), h = {h}, refinements = {refinements}')

x = 0
y = 0.1
# h = 0.1
tol = 1e-3
# initial value problem with y(0) = 0.1
# exact solution for the above ODE f(x, y) is 1 / (1 + 9e**(-x))
# y(1) = 0.2319693166...
# y(2) = 0.4508530603...

# euler_simple(f, x, y, h, tol)

solutions, final_k = euler(k, u, tol) 
solutions = numpy.array(solutions)
x = numpy.arange(1, N) * h
t = numpy.arange(solutions.shape[0]) * final_k

X, T = numpy.meshgrid(x, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, T, solutions)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')

plt.show()

# initial idea: introduce PDE section with a simple PDE e.g. 1D heat equation u_t = u_xx
# numerically solve the simple PDE by "splitting it up" into ODEs and use above Euler method
# "method of lines"
# discretize x i.e. u(x_i, t) = u_i(t)
# at x = x_i, u_t(x_i, t) = u_xx(x_i, t) but since x_i is fixed, u_t(x_i, t) = du_i/dt
# u_xx ~ (u_{i+1} - 2u_i + u_{i-1}) / h^2 from central difference 
# du_i/dt = (u_{i+1} - 2u_i + u_{i-1}) / h^2 is ODE system

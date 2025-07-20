import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.8  

def n_pendulum_derivs(t, y, n, L, m):
    theta = y[:n]
    omega = y[n:]
    dydt = np.zeros(2 * n)

    M = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        for j in range(n):
            M[i, j] = m * L * L * sum(
                np.cos(theta[i] - theta[k]) * (1 if k >= max(i, j) else 0)
                for k in range(max(i, j), n)
            )
        b[i] = -m * g * L * sum(
            np.sin(theta[i]) * (1 if k >= i else 0)
            for k in range(i, n)
        )
        b[i] -= sum(
            m * L * L * omega[j] * omega[k] * np.sin(theta[i] - theta[k]) * (1 if k >= max(i, j) else 0)
            for j in range(n)
            for k in range(n)
        )

    domega = np.linalg.solve(M, b)
    dydt[:n] = omega
    dydt[n:] = domega
    return dydt

def get_positions(y, L):
    n = len(y) // 2
    theta = y[:n]
    x = np.cumsum([L * np.sin(t) for t in theta])
    y = -np.cumsum([L * np.cos(t) for t in theta])
    return x, y

n = 3           
L = 1.0           
m = 1.0           
t_max = 10
fps = 60          

theta0 = np.pi / 2 * np.ones(n) + 0.01 * np.random.randn(n)
omega0 = np.zeros(n)
y0 = np.concatenate((theta0, omega0))

t_eval = np.linspace(0, t_max, t_max * fps)

solution = solve_ivp(n_pendulum_derivs, [0, t_max], y0, args=(n, L, m), t_eval=t_eval, method='RK45', rtol=1e-10)

positions = [get_positions(solution.y[:, i], L) for i in range(len(solution.t))]
x_data = [p[0] for p in positions]
y_data = [p[1] for p in positions]

fig, ax = plt.subplots()
ax.set_xlim(-n*L, n*L)
ax.set_ylim(-n*L, n*L)
line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    x = [0] + list(x_data[i])
    y = [0] + list(y_data[i])
    line.set_data(x, y)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(solution.t), interval=1000/fps)
plt.title("Pendulum Simulation")
plt.show()

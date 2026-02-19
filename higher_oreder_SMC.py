import numpy as np
import matplotlib.pyplot as plt


# Simulation Parameters
T =  60.0
dt = 0.01
N = int(T/dt) + 1
t = np.arange(0, T + dt, dt)



# Initial Condotion 
y = np.zeros(N)
vy = np.zeros(N)
psi = np.zeros(N)
r = np.zeros(N)

y[0] = 1.0 # Initial Lateral Error

# Super-Twisting internal state
u2 = 0.0

# Vehicle Parameters
m = 1719  # kg
Cf = 170550 # N/rad
Cr = 137844 # N/rad
Iz = 3300 # kgm**2
Lf = 1.195 # m
Lr = 1.513 # m

Vx = 13.5 # m/s

# Control Parameters
lam = 1.0
alpha = 0.002
beta = 0.0001


for k in range(N - 1):
    # Sliding variable (Eq. 10)
    s = vy[k] + lam * y[k]

    # Curvature Raduis
    R = 1e6

    # phi(t, s)  eq12
    phi = (-(Cf + Cr)/(m * Vx)) * vy[k] \
            - ((Lf * Cf - Lr * Cr)/(m * Vx)) * r[k] \
            - (Vx**2 / R) \
            + lam * vy[k]
    
    # varphi eq. 12
    varphi = Cf / m

    # Equivalent Control eq.14
    delta_eq = -(m / Cf) * phi

    # Super-Twisting Part
    u1 = -alpha * np.sqrt(abs(s)) * np.sign(s)
    du2 = -beta * np.sign(s)
    u2 = u2 + dt * du2

    delta_st = u1 + u2
    delta = delta_st + delta_eq


    # Vehicle dynamics (Eq. 1)

    # ÿ  (which is vy_dot)
    vy_dot = (-(Cf + Cr)/(m * Vx)) * vy[k] \
             - ((Lf * Cf - Lr * Cr)/(m * Vx)) * r[k] \
             + (Cf / m) * delta

    # ψ (which is r_dot)
    r_dot = (-(Lf * Cf - Lr * Cr)/(Iz * Vx)) * vy[k] \
            - ((Lf**2 * Cf + Lr**2 * Cr)/(Iz * Vx)) * r[k] \
            + (Lf * Cf / Iz) * delta

    # Euler Integration
    vy[k+1] = vy[k] + dt * vy_dot
    r[k+1]  = r[k]  + dt * r_dot

    y[k+1]   = y[k]   + dt * vy[k]
    psi[k+1] = psi[k] + dt * r[k]


plt.figure()
plt.plot(t, y)
plt.xlabel("Time (s)")
plt.ylabel("Lateral Error (m)")
plt.grid()
plt.show()
 
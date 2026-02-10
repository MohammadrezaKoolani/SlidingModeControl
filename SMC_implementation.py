import numpy as np
import matplotlib.pyplot as plt

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

# Simulation parameters
dt = 0.01
T = 63.0 * 2.0
N = int(T/dt)

# Vehicle parameters
L = 2.5
v = 5.0  # assumed constant forward speed (m/s)



# ==== Path definition (circle example) ====
R = 50.0
def circle_path_point(s):
    # s: path length along circle
    theta = s / R               # angle parameter
    x_r = R * np.sin(theta)
    y_r = R * (1 - np.cos(theta))
    psi_r = theta               # tangent angle
    kappa = 1.0 / R
    return x_r, y_r, psi_r, kappa



# ==== Path definition (sinusoidal example) ====
A = 10.0          # amplitude [m]
omega = 0.05     # spatial frequency [rad/m]

def sinusoidal_path_point(s):
    # Approximate x ≈ s
    x_r = s
    y_r = A * np.sin(omega * s)

    # First derivative dy/dx
    dy_dx = A * omega * np.cos(omega * s)

    # Second derivative d2y/dx2
    d2y_dx2 = -A * omega**2 * np.sin(omega * s)

    # Reference heading
    psi_r = np.arctan(dy_dx)

    # Curvature formula for y(x)
    kappa = d2y_dx2 / (1 + dy_dx**2)**(3/2)

    return x_r, y_r, psi_r, kappa

# ==== Random smooth path definition ====
np.random.seed(1)   # reproducible

N_modes = 5
A_list = np.random.uniform(1.0, 5.0, N_modes)
omega_list = np.random.uniform(0.02, 0.08, N_modes)
phi_list = np.random.uniform(0, 2*np.pi, N_modes)

def random_path_point(s):
    x_r = s

    # y(x)
    y_r = 0.0
    dy_dx = 0.0
    d2y_dx2 = 0.0

    for A, w, phi in zip(A_list, omega_list, phi_list):
        y_r += A * np.sin(w * s + phi)
        dy_dx += A * w * np.cos(w * s + phi)
        d2y_dx2 += -A * w**2 * np.sin(w * s + phi)

    # heading
    psi_r = np.arctan(dy_dx)

    # curvature
    kappa = d2y_dx2 / (1 + dy_dx**2)**(3/2)

    return x_r, y_r, psi_r, kappa


# If you want a different path: implement path_point(s) accordingly.
path_point = random_path_point


# State init
s = 0
x, y, psi, _ = path_point(s)

# Sliding mode / super-twisting parameters (tune these)
lambda_y = 0.6
phi = 0.3
k1 = 0.4
k2 = 0.3

def sat(x):
    return np.clip(x, -1.0, 1.0)

# histories
x_hist, y_hist, psi_hist = [], [], []
e_y_hist, e_psi_hist, s_hist = [], [], []
s_ref_hist = []  # store s (path coordinate)
delta_hist = []

def project_to_path(x, y, s_prev, path_point,
                    ds=0.5, window=10):
    """
    Project (x,y) onto path by local nearest-point search.
    ds: resolution of s search
    window: number of steps forward/backward
    """
    s_candidates = s_prev + ds * np.arange(-window, window+1)
    min_dist = np.inf
    s_best = s_prev

    for s_c in s_candidates:
        x_r, y_r, _, _ = path_point(s_c)
        d = (x - x_r)**2 + (y - y_r)**2
        if d < min_dist:
            min_dist = d
            s_best = s_c

    return s_best


# initial path coordinate s (start of path)
s = 0.0

# super-twisting state
z = 0.0

for i in range(N):
    # 1) get reference at current s
    x_r, y_r, psi_r, kappa_r = path_point(s)

    # 2) compute errors in Frenet frame
    dx = x - x_r
    dy = y - y_r
    # lateral error (standard sign)
    e_y = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
    e_psi = wrap_angle(psi - psi_r)

    # 3) sliding surface
    s_y = v * e_psi + lambda_y * e_y

    # 4) Equivalent control + feedforward (use curvature at s)
    # feedforward for nominal curvature (bicycle): delta_ff = atan(L * kappa)
    # small-angle approx would be L * kappa, but better to use atan for realistic model
    delta_ff = np.arctan(L * kappa_r)
    delta_eq = -(L / v) * (lambda_y * e_psi + (lambda_y**2 / v) * e_y)

    # 5) Super-twisting (boundary layer via sat)
    sigma = sat(s_y / phi)
    z_dot = -k2 * sigma
    z += z_dot * dt
    saturation = sat(s_y / phi)  # already defined
    delta_sta = -k1 * np.sqrt(np.maximum(np.abs(s_y), 1e-12)) * saturation + z


    # total steering (clip to physical limits)
    delta = delta_ff + delta_eq + delta_sta
    delta = np.clip(delta, -0.4, 0.4)

    # 6) vehicle kinematics (use tan(delta) for bicycle)
    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    psi_dot = (v / L) * np.tan(delta)

    # integrate states (Euler)
    x += x_dot * dt
    y += y_dot * dt
    psi = wrap_angle(psi + psi_dot * dt)

    # 7) update path coordinate s using projection-based update:
    # s_dot = v * cos(e_psi) / (1 - kappa * e_y)
    denom = 1.0 - kappa_r * e_y
    denom = np.sign(denom) * max(abs(denom), 1e-3)  # guard against small denom
    # s_dot = v * np.cos(e_psi) / denom
    # # Optional: limit s_dot to avoid huge jumps if huge error
    # s_dot = np.clip(s_dot, -10*v, 10*v)
    # s += s_dot * dt
    # s = x 

    s = project_to_path(x, y, s, path_point)


    # store history
    x_hist.append(x); y_hist.append(y); psi_hist.append(psi)
    e_y_hist.append(e_y); e_psi_hist.append(e_psi); s_hist.append(s_y)
    s_ref_hist.append(s); delta_hist.append(delta)

# create reference curve for plotting
# generate many s along several rotations
s_vals = np.linspace(0, s, 1000)
x_ref = np.zeros_like(s_vals); y_ref = np.zeros_like(s_vals)
for k, ss in enumerate(s_vals):
    xr, yr, _, _ = path_point(ss)
    x_ref[k] = xr
    y_ref[k] = yr


# Time vector
t = np.arange(len(e_y_hist)) * dt

# === 1) Trajectory ===
plt.figure(figsize=(8,6))
plt.plot(x_hist, y_hist, label='Vehicle', linewidth=3)
plt.plot(x_ref, y_ref, '--', label='Reference (path)', linewidth=1)
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Vehicle Trajectory')
plt.show()


# === 2) Tracking errors ===
plt.figure(figsize=(8,6))

plt.subplot(2,1,1)
plt.plot(t, e_y_hist, linewidth=1.5)
plt.ylabel('Lateral error $e_y$ [m]')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t, e_psi_hist, linewidth=1.5)
plt.ylabel('Heading error $e_\\psi$ [rad]')
plt.xlabel('Time [s]')
plt.grid(True)

plt.suptitle('Tracking Errors')
plt.tight_layout()
plt.show()


# === 3) Sliding surface ===
plt.figure(figsize=(7,4))
plt.plot(t, s_hist, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Sliding surface $s_y$')
plt.title('Sliding Surface')
plt.grid(True)
plt.show()


# === 4) Steering angle (VERY IMPORTANT) ===
plt.figure(figsize=(7,4))
plt.plot(t, delta_hist, linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Steering angle $\\delta$ [rad]')
plt.title('Steering Input')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def sat(x):
    return np.clip(x, -1.0, 1.0)

# ==========================================
# 1. PARAMETERS (From Paper/Repo)
# ==========================================
dt = 0.01
T = 63.0 * 2.0
N = int(T/dt)

# Vehicle parameters
L = 2.5
v = 5.0  # assumed constant forward speed (m/s)

# --- Paper Controller Parameters ---
# Gains from the repository config
lambda_gain = 24.0      # Sliding surface gain
alpha = 0.8             # STA sqrt gain
beta = 0.04             # STA integral gain
phi_param = 1.0         # Variable boundary layer gain
min_phi = 2.0           # Minimum boundary layer
n_pred = 14             # Prediction horizon (steps)

# Limits
steer_lim = 0.6         # Rad
steer_rate_lim = 0.6    # Rad/s

# =======================
# 2. PATH DEFINITIONS 
# =======================
R = 50.0
def circle_path_point(s):
    theta = s / R
    x_r = R * np.sin(theta)
    y_r = R * (1 - np.cos(theta))
    psi_r = theta
    kappa = 1.0 / R
    return x_r, y_r, psi_r, kappa

A = 10.0
omega = 0.05
def sinusoidal_path_point(s):
    x_r = s
    y_r = A * np.sin(omega * s)
    dy_dx = A * omega * np.cos(omega * s)
    d2y_dx2 = -A * omega**2 * np.sin(omega * s)
    psi_r = np.arctan(dy_dx)
    kappa = d2y_dx2 / (1 + dy_dx**2)**(3/2)
    return x_r, y_r, psi_r, kappa

np.random.seed(1)
N_modes = 5
A_list = np.random.uniform(1.0, 5.0, N_modes)
omega_list = np.random.uniform(0.02, 0.08, N_modes)
phi_list = np.random.uniform(0, 2*np.pi, N_modes)

def random_path_point(s, curve_gain=1.0):
    x_r = s
    y_r = 0.0
    dy_dx = 0.0
    d2y_dx2 = 0.0
    for A, w, phi in zip(A_list, omega_list, phi_list):
        w_eff = w * curve_gain
        y_r += A * np.sin(w_eff * s + phi)
        dy_dx += A * w_eff * np.cos(w_eff * s + phi)
        d2y_dx2 += -A * w_eff**2 * np.sin(w_eff * s + phi)
    psi_r = np.arctan(dy_dx)
    kappa = d2y_dx2 / (1 + dy_dx**2)**(3/2)
    return x_r, y_r, psi_r, kappa

class OrganicPath:
    def __init__(self, length=1000.0, ds=0.1):
        """
        Generates a single closed loop path (like a racetrack).
        It uses a base circle and adds smooth organic variations.
        """
        # 1. Define the Loop Parameters
        # Circumference approx equal to length, but we force it to close
        self.ds = ds
        n_points = int(length / ds)
        self.s_values = np.linspace(0, length, n_points)
        
        # 2. Generate Curvature for a Closed Loop
        # A perfect circle has constant curvature k = 2*pi / Length
        base_kappa = 2 * np.pi / length
        
        # Add smooth variations (sines) that sum to zero over the loop
        # This ensures the total turning angle remains 2*pi (a closed circle)
        # We use integer frequencies (1x, 2x, 3x the loop length) to ensure continuity at the close
        
        k_variation = np.zeros(n_points)
        
        # Mode 1: Elliptical / Oval shape (2 waves per loop)
        k_variation += 0.015 * np.sin(2 * 2 * np.pi * self.s_values / length)
        
        # Mode 2: Organic "wiggles" (3-5 waves per loop)
        k_variation += 0.01 * np.sin(3 * 2 * np.pi * self.s_values / length + 1.0)
        k_variation += 0.005 * np.sin(5 * 2 * np.pi * self.s_values / length + 2.5)
        
        self.kappa = base_kappa + k_variation

        # 3. Limit Curvature (Feasibility Check)
        # Max curvature for L=2.5m is ~0.18. We clamp to be safe.
        self.kappa = np.clip(self.kappa, -0.15, 0.15)

        # 4. Integrate to get Heading (Psi) and Position (X,Y)
        self.psi = np.cumsum(self.kappa * self.ds)
        
        # Force closure: The path might drift slightly due to integration error.
        # We don't force X/Y closure hard here to keep it simple, 
        # but the curvature math above creates a naturally closing shape.
        self.x = np.cumsum(np.cos(self.psi) * self.ds)
        self.y = np.cumsum(np.sin(self.psi) * self.ds)

        # 5. Create Interpolation Functions
        self.f_x = interp1d(self.s_values, self.x, kind='cubic', fill_value="extrapolate")
        self.f_y = interp1d(self.s_values, self.y, kind='cubic', fill_value="extrapolate")
        self.f_psi = interp1d(self.s_values, self.psi, kind='cubic', fill_value="extrapolate")
        self.f_k = interp1d(self.s_values, self.kappa, kind='cubic', fill_value="extrapolate")

    def get_point(self, s):
        """ Returns reference state at distance s """
        # Wrap s around the track length so it loops forever
        s_loop = s % self.s_values[-1]
        return float(self.f_x(s_loop)), float(self.f_y(s_loop)), float(self.f_psi(s_loop)), float(self.f_k(s_loop))

    def __call__(self, s):
        return self.get_point(s)

# SELECT PATH HERE
# Note: T=120s * 5m/s = 600m. We make the track 600m long so it completes exactly one lap.
path_point = OrganicPath(length=v*T)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def project_to_path(x, y, s_prev, path_point, ds=0.5, window=10):
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

# ==========================================
# 4. INITIALIZATION
# ==========================================
s = 0.0
x, y, psi, _ = path_point(s)

# Controller internal states
u_lat = 0.0     # Switching term lateral
u_yaw = 0.0     # Switching term yaw
u2_lat = 0.0    # Integral term lateral
u2_yaw = 0.0    # Integral term yaw
u_prev = 0.0    # Previous steering command

# Histories
x_hist, y_hist, psi_hist = [], [], []
e_y_hist, e_psi_hist = [], []
delta_hist = []
s_lat_hist, s_yaw_hist = [], []

# Previous state for derivative calc
x_err_prev = np.zeros(4) # [ey, dey, epsi, depsi]

print("Starting Simulation with Paper's Controller...")

# ==========================================
# 5. MAIN LOOP
# ==========================================
for i in range(N):
    # ----------------------------------------
    # 1) Get Reference
    # ----------------------------------------
    x_r, y_r, psi_r, kappa_r = path_point(s)

    # ----------------------------------------
    # 2) Compute Errors (Frenet Frame)
    # ----------------------------------------
    dx = x - x_r
    dy = y - y_r
    e_y = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
    e_psi = wrap_angle(psi - psi_r)
    
    # Compute derivatives (simple finite difference for simulation)
    de_y = (e_y - x_err_prev[0]) / dt
    de_psi = (e_psi - x_err_prev[2]) / dt
    x_err_prev = np.array([e_y, de_y, e_psi, de_psi])

    # ----------------------------------------
    # 3) PREDICTION STEP (Algorithm 1)
    # ----------------------------------------
    # We predict the error state n_pred steps ahead using the 
    # "Stabilized Kinematic Model" matrix from the C++ code
    
    # State: [ey, dey, epsi, depsi]
    x_pred = np.array([e_y, de_y, e_psi, de_psi])
    
    # Pre-calculate matrix terms (Linearized around reference)
    # Ref steering for curvature
    delta_r = np.arctan(L * kappa_r)
    cos_dr_sq_inv = 1.0 / (np.cos(delta_r)**2 + 1e-6)
    
    # Construct Continuous Matrix A (stabilized) and B
    # See kinematic.cpp :: calculateDiscreteMatrix
    A_mat = np.zeros((4,4))
    A_mat[0, 2] = v
    A_mat[1, 3] = v
    
    # Stability injection (Crucial part of the paper's robustness)
    gain_stab = -max(2.0, 0.5 * v)
    A_mat[0, 0] = gain_stab
    A_mat[1, 1] = gain_stab
    A_mat[2, 2] = -2.0
    A_mat[3, 3] = -2.0
    
    B_mat = np.zeros((4,1))
    B_mat[2, 0] = v / L * cos_dr_sq_inv
    
    W_mat = np.zeros(4)
    W_mat[2] = -v / L * cos_dr_sq_inv * delta_r

    # Discretize (Euler for simplicity in this loop)
    # x_{k+1} = x_k + (A x_k + B u + W) * dt
    # Assumption: Steering stays constant during prediction horizon
    u_vec = np.array([u_prev]) 
    
    for _ in range(n_pred):
        x_dot = A_mat @ x_pred + B_mat @ u_vec + W_mat
        x_pred += x_dot * dt

    # Extract predicted errors
    ey_n = x_pred[0]
    edy_n = x_pred[1]
    epsi_n = x_pred[2]
    edpsi_n = x_pred[3]

    # ----------------------------------------
    # 4) SMC CALCULATION (Dual Surface)
    # ----------------------------------------
    
    # --- Variable Boundary Layer (Eq 5) ---
    phi_val = max(min_phi, abs(phi_param * v))
    
    # --- Surface 1: Lateral ---
    s_lat = edy_n + lambda_gain * ey_n
    sigma_lat = np.tanh(s_lat / phi_val)
    
    # Adaptive Integral (Eq 7c)
    # dot_u2 = -beta * v * sigma
    u2_lat += (-beta * v * sigma_lat * dt)
    
    # Switching Term (Eq 7a)
    u1_lat = -alpha * np.sqrt(np.abs(sigma_lat)) * sigma_lat + u2_lat
    
    # --- Surface 2: Yaw ---
    s_yaw = edpsi_n + lambda_gain * epsi_n
    sigma_yaw = np.tanh(s_yaw / phi_val)
    
    u2_yaw += (-beta * v * sigma_yaw * dt)
    u1_yaw = -alpha * np.sqrt(np.abs(sigma_yaw)) * sigma_yaw + u2_yaw
    
    # Store surfaces for plotting
    s_lat_hist.append(s_lat)
    s_yaw_hist.append(s_yaw)

    # ----------------------------------------
    # 5) CONTROL OUTPUT
    # ----------------------------------------
    # Feedforward
    delta_ff = np.arctan(L * kappa_r)
    
    # Total Command
    delta_raw = u1_lat + u1_yaw + delta_ff
    
    # Rate Limiting
    delta_rate = (delta_raw - u_prev) / dt
    delta_rate = np.clip(delta_rate, -steer_rate_lim, steer_rate_lim)
    delta = u_prev + delta_rate * dt
    
    # Angle Limiting
    delta = np.clip(delta, -steer_lim, steer_lim)
    
    # Store for next step
    u_prev = delta

    # ----------------------------------------
    # 6) VEHICLE KINEMATICS (Same as before)
    # ----------------------------------------
    x_dot = v * np.cos(psi)
    y_dot = v * np.sin(psi)
    psi_dot = (v / L) * np.tan(delta)

    x += x_dot * dt
    y += y_dot * dt
    psi = wrap_angle(psi + psi_dot * dt)

    # ----------------------------------------
    # 7) UPDATE PATH S
    # ----------------------------------------
    s = project_to_path(x, y, s, path_point)

    # Store history
    x_hist.append(x); y_hist.append(y); psi_hist.append(psi)
    e_y_hist.append(e_y); e_psi_hist.append(e_psi)
    delta_hist.append(delta)

# ==========================================
# 6. PLOTTING
# ==========================================
# create reference curve
s_vals = np.linspace(0, s, 1000)
x_ref = np.zeros_like(s_vals); y_ref = np.zeros_like(s_vals)
for k, ss in enumerate(s_vals):
    xr, yr, _, _ = path_point(ss)
    x_ref[k] = xr
    y_ref[k] = yr

t = np.arange(len(e_y_hist)) * dt
step = 30

# 1) Trajectory
plt.figure(figsize=(8,6))
plt.plot(x_hist[::step], y_hist[::step], label='Paper SMC', linewidth=3)
plt.plot(x_ref, y_ref, '--', label='Reference', linewidth=1)
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title('Vehicle Trajectory (Paper Implementation)')
plt.show()

# 2) Tracking errors
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(t[::step], e_y_hist[::step], linewidth=1.5)
plt.ylabel('Lateral error [m]')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t[::step], e_psi_hist[::step], linewidth=1.5)
plt.ylabel('Heading error [rad]')
plt.xlabel('Time [s]')
plt.grid(True)
plt.suptitle('Tracking Errors')
plt.show()

# 3) Steering
plt.figure(figsize=(7,4))
plt.plot(t[::step], delta_hist[::step], linewidth=1.5)
plt.xlabel('Time [s]')
plt.ylabel('Steering angle [rad]')
plt.title('Steering Input')
plt.grid(True)
plt.show()

# 4) Sliding Surface
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t[::step], s_lat_hist[::step], 'r', linewidth=1.5)
plt.ylabel('Lateral Surface $s_y$')
plt.grid(True)
plt.title('Sliding Surfaces (Should converge to 0)')

plt.subplot(2,1,2)
plt.plot(t[::step], s_yaw_hist[::step], 'g', linewidth=1.5)
plt.ylabel(r'Yaw Surface $s_\psi$')
plt.xlabel('Time [s]')
plt.grid(True)
plt.show()



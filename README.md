# SlidingModeControl

Dual-Surface Super-Twisting Sliding Mode Control (SMC) for robust autonomous vehicle path tracking.

This project is a Python implementation of the controller presented in:

**Sliding Mode Control for Robust Path Tracking of Automated Vehicles in Rural Environments**  
IEEE Open Journal of Vehicular Technology, 2024  
https://ieeexplore.ieee.org/document/10669799

---

## Project Description

This repository reproduces the sliding mode path-tracking controller described in the paper, including:

- Kinematic bicycle vehicle model
- Stabilized linear prediction model
- Forward prediction (Algorithm 1)
- Dual sliding surfaces (lateral + yaw)
- Super-Twisting Algorithm (STA)
- Curvature-based feedforward steering
- Steering rate and steering angle limits
- Multiple reference path types

The controller predicts tracking errors forward in time to compensate steering delay and improve robustness.

---

## Control Structure

At each control step:

1. Compute Frenet-frame errors:
   - Lateral error `e_y`
   - Heading error `e_psi`

2. Predict future error state using the stabilized kinematic model.

3. Construct sliding surfaces:
   - `s_y = de_y + λ e_y`
   - `s_psi = de_psi + λ e_psi`

4. Apply Super-Twisting control:
   - `δ1 = -α |s|^(1/2) sign(s)`
   - `δ2_dot = -β v s`

5. Total steering:
   - Feedback (SMC)
   - + Curvature feedforward
   - + Rate and angle saturation

---

## Reference Paths

The simulation supports:

- Circular path
- Sinusoidal path
- Random multi-frequency path
- Organic closed-loop racetrack (default)

---

## Simulation Outputs

The script generates:

- Vehicle trajectory vs reference
- Lateral error
- Heading error
- Steering angle history
- Sliding surface convergence



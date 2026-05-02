"""
CTBR (Collective Thrust + Body Rates) low-level controller.

Policy outputs: [ω_x^d, ω_y^d, ω_z^d, c]
  - ω_{x,y,z}^d: desired body rates [rad/s], range [-π, π]
  - c: normalized collective thrust [0, 1], mapping to [0, 1.6 * m * g]

This controller converts CTBR commands to 4 motor thrust commands via:
  1. Rate PD controller: body rate error → desired torques
  2. Mixer: [total_thrust, τ_x, τ_y, τ_z] → per-motor Ω²
  3. Motor dynamics (first-order lag): Ω_cmd → Ω_actual

Crazyflie X-configuration motor layout (body frame: x forward, y left, z up):
  M0: front-left  (+d, +d),  CCW (spin +1)
  M1: front-right (+d, -d),  CW  (spin -1)
  M2: back-right  (-d, -d),  CCW (spin +1)
  M3: back-left   (-d, +d),  CW  (spin -1)
  d = 0.046 / sqrt(2) = 0.03253 m
"""

from __future__ import annotations
import jax.numpy as jnp


# ── Physical constants ────────────────────────────────────────────────────────

MASS = 0.0321         # kg
GRAVITY = 9.81        # m/s²
INERTIA = jnp.array([1.4e-5, 1.4e-5, 2.17e-5])  # kg·m² (Ixx, Iyy, Izz)

KF = 2.350347298350041e-08   # thrust coefficient [N / (rad/s)²]
KM = 7.24e-10                # moment coefficient [N·m / (rad/s)²]
ARM = 0.046                  # arm length center-to-rotor [m]
D = ARM / jnp.sqrt(2.0)      # rotor position offset [m]

MOTOR_TAU = 0.025            # motor time constant [s]
MAX_ROTOR_SPEED = 2315.0     # [rad/s]
MAX_THRUST_TOTAL = KF * MAX_ROTOR_SPEED**2 * 4  # [N] ~0.504 N
HOVER_THRUST = MASS * GRAVITY  # [N] ~0.315 N
# Thrust-to-weight at max = ~1.6g

# Collective thrust mapping: c ∈ [0, 1] → [0, 1.6 * m * g]
MAX_COLLECTIVE_THRUST = 1.6 * MASS * GRAVITY  # [N]

# Body rate limits [rad/s]
MAX_BODY_RATE = jnp.pi  # per axis

# Rate PD gains (proportional gain; no derivative for now)
# Tuned to give reasonable response in sim; not matched to Crazyflie firmware.
KP_RATE = jnp.array([250.0, 250.0, 120.0])  # [1/s] (roll, pitch, yaw)
# Physical torque = I * Kp * ω_error:
# roll: 1.4e-5 * 250 = 3.5e-3 N·m / (rad/s) — within motor torque authority

# ── Mixer matrix ─────────────────────────────────────────────────────────────
# Maps [Ω0², Ω1², Ω2², Ω3²] → [Fz, τ_x, τ_y, τ_z]
# Derived from rotor positions and spin directions.
#
# Fz  = kf*(Ω0² + Ω1² + Ω2² + Ω3²)
# τ_x = d*kf*(+Ω0² - Ω1² - Ω2² + Ω3²)  [roll: front-left & back-left vs rest]
# τ_y = d*kf*(-Ω0² - Ω1² + Ω2² + Ω3²)  [pitch: back vs front]
# τ_z = km*(+Ω0² - Ω1² + Ω2² - Ω3²)    [yaw: CCW vs CW]

def _build_mixer():
    B = jnp.array([
        [KF,    KF,    KF,    KF  ],   # Fz
        [D*KF,  -D*KF, -D*KF, D*KF],  # τ_x (roll)
        [-D*KF, -D*KF, D*KF,  D*KF],  # τ_y (pitch)
        [KM,    -KM,   KM,    -KM ],   # τ_z (yaw)
    ])
    return B

MIXER = _build_mixer()           # (4, 4): wrench components from motor sq-speeds
MIXER_INV = jnp.linalg.inv(MIXER)  # (4, 4): motor sq-speeds from desired wrench


# ── Controller functions ──────────────────────────────────────────────────────

def rate_controller(
    omega_desired: jnp.ndarray,  # (3,) desired body rates [rad/s]
    omega_current: jnp.ndarray,  # (3,) current body rates [rad/s]
) -> jnp.ndarray:
    """
    Proportional rate controller: desired torques in body frame.
    Returns τ_desired (3,) [N·m].
    """
    omega_error = omega_desired - omega_current
    tau = INERTIA * KP_RATE * omega_error
    return tau


def allocate_motors(
    total_thrust: float,
    tau_body: jnp.ndarray,  # (3,) desired torques [N·m]
) -> jnp.ndarray:
    """
    Inverse mixer: [Fz, τ_x, τ_y, τ_z] → desired Ω² per motor.
    Returns omega_sq (4,) [rad²/s²], clipped to [0, Ω_max²].
    """
    wrench = jnp.array([total_thrust, tau_body[0], tau_body[1], tau_body[2]])
    omega_sq = MIXER_INV @ wrench
    return jnp.clip(omega_sq, 0.0, MAX_ROTOR_SPEED**2)


def motor_dynamics(
    rotor_speeds: jnp.ndarray,  # (4,) current Ω [rad/s]
    omega_cmd: jnp.ndarray,     # (4,) commanded Ω [rad/s]
    dt: float,
) -> jnp.ndarray:
    """
    First-order motor lag: Ω_{t+1} = Ω_t + dt/T_m * (Ω_cmd - Ω_t)
    Returns new rotor_speeds (4,) [rad/s].
    """
    return rotor_speeds + (dt / MOTOR_TAU) * (omega_cmd - rotor_speeds)


def compute_wrench(
    rotor_speeds: jnp.ndarray,  # (4,) current Ω [rad/s]
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute total force and torque in body frame from current rotor speeds.
    Returns (force_body (3,), torque_body (3,)).
    """
    omega_sq = rotor_speeds ** 2
    wrench = MIXER @ omega_sq  # [Fz, τ_x, τ_y, τ_z]
    force_body = jnp.array([0.0, 0.0, wrench[0]])
    torque_body = wrench[1:4]
    return force_body, torque_body


def ctbr_to_rotor_speeds(
    action: jnp.ndarray,         # (4,) [ω_x^d, ω_y^d, ω_z^d, c]
    rotor_speeds: jnp.ndarray,   # (4,) current Ω [rad/s]
    omega_current: jnp.ndarray,  # (3,) current body rates [rad/s]
    dt: float,
) -> jnp.ndarray:
    """
    Full CTBR → new rotor speeds pipeline.
    Returns next rotor_speeds (4,) after one timestep.
    """
    # Decode action
    omega_desired = action[:3] * MAX_BODY_RATE      # scale to [-π, π] rad/s
    c = jnp.clip((action[3] + 1.0) / 2.0, 0.0, 1.0)  # tanh → [0,1]
    total_thrust = c * MAX_COLLECTIVE_THRUST

    # Rate controller → desired torques
    tau_desired = rate_controller(omega_desired, omega_current)

    # Allocate to motors → desired Ω²
    omega_sq_cmd = allocate_motors(total_thrust, tau_desired)
    omega_cmd = jnp.sqrt(omega_sq_cmd)

    # Apply motor dynamics
    new_rotor_speeds = motor_dynamics(rotor_speeds, omega_cmd, dt)
    return jnp.clip(new_rotor_speeds, 0.0, MAX_ROTOR_SPEED)

# Fixing the unpacking error (extra list variable declared)
# Corrected list declarations for logging data
import numpy as np
time_log, x_log, y_log = [], [], []
vx_log, vy_log = [], []
speed_log, drag_log, thrust_log = [], [], []

# Re-initialize state variables
theta_rad = np.radians(LAUNCH_ANGLE)
vx, vy = INITIAL_VELOCITY * np.cos(theta_rad), INITIAL_VELOCITY * np.sin(theta_rad)
x, y = 0.0, 0.0
t = 0.0
fuel = FUEL_MASS
mass = MISSILE_MASS + fuel

# --- SIMULATION LOOP ---
while t <= MAX_TIME and y >= 0:
    # Air density and drag
    air_density = AIR_DENSITY_SEA_LEVEL * np.exp(-y / 8500)
    velocity = np.sqrt(vx**2 + vy**2)
    drag_force = 0.5 * air_density * DRAG_COEFF_GROOVED * MISSILE_CROSS_SECTION * velocity**2

    # Thrust phase (fuel + scramjet)
    if fuel > 0:
        current_thrust = THRUST
        fuel_used = min(FUEL_BURN_RATE * TIME_STEP, fuel)
        fuel -= fuel_used
        mass -= fuel_used
    elif y >= SCRAMJET_START_ALT:
        current_thrust = SCRAMJET_THRUST
    else:
        current_thrust = 0

    # Acceleration components
    ax = (current_thrust * vx / velocity - drag_force * vx / velocity) / mass
    ay = (current_thrust * vy / velocity - drag_force * vy / velocity) / mass - GRAVITY

    # Update velocity and position
    vx += ax * TIME_STEP
    vy += ay * TIME_STEP
    x += vx * TIME_STEP
    y += vy * TIME_STEP
    t += TIME_STEP

    # Log data
    time_log.append(t)
    x_log.append(x)
    y_log.append(y)
    vx_log.append(vx)
    vy_log.append(vy)
    speed_log.append(velocity)
    drag_log.append(drag_force)
    thrust_log.append(current_thrust)

# --- PLOT RESULTS ---
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(x_log, y_log)
plt.title("Missile Trajectory")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(time_log, speed_log, label="Speed")
plt.plot(time_log, thrust_log, label="Thrust (N)", linestyle="--")
plt.title("Speed and Thrust Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s) / Thrust (N)")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(time_log, drag_log)
plt.title("Drag Force Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Drag Force (N)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(time_log, y_log, label="Altitude")
plt.axhline(SCRAMJET_START_ALT, color='red', linestyle='--', label="Scramjet Start Altitude")
plt.title("Altitude Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constants
GRAVITY = 9.81  # m/s^2
AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3
DRAG_COEFF_GROOVED = 0.2
MISSILE_MASS = 1000  # kg dry mass
MISSILE_CROSS_SECTION = 0.3  # m^2
INITIAL_VELOCITY = 1000  # m/s
LAUNCH_ANGLE = 45  # degrees

FUEL_MASS = 200  # kg
FUEL_BURN_RATE = 5  # kg/s
THRUST = 30000  # N
SCRAMJET_THRUST = 20000  # N
SCRAMJET_START_ALT = 15000  # m

SPEED_OF_SOUND = 340  # m/s approx at sea level

TIME_STEP = 0.1
MAX_TIME = 200

# Initial conditions
theta_rad = np.radians(LAUNCH_ANGLE)
vx = INITIAL_VELOCITY * np.cos(theta_rad)
vy = INITIAL_VELOCITY * np.sin(theta_rad)
x, y = 0.0, 0.0
fuel = FUEL_MASS
mass = MISSILE_MASS + fuel

t = 0.0

# Data logs
time_log = []
x_log = []
y_log = []
vx_log = []
vy_log = []
speed_log = []
ax_log = []
ay_log = []
accel_log = []
drag_log = []
thrust_log = []
mass_log = []
angle_log = []
mach_log = []
kinetic_energy_log = []
potential_energy_log = []

while t <= MAX_TIME and y >= 0:
    air_density = AIR_DENSITY_SEA_LEVEL * np.exp(-y / 8500)
    velocity = np.sqrt(vx**2 + vy**2)
    if velocity == 0:
        velocity = 1e-6
    
    drag_force = 0.5 * air_density * DRAG_COEFF_GROOVED * MISSILE_CROSS_SECTION * velocity**2
    
    if fuel > 0:
        current_thrust = THRUST
        fuel_used = min(FUEL_BURN_RATE * TIME_STEP, fuel)
        fuel -= fuel_used
        mass -= fuel_used
    elif y >= SCRAMJET_START_ALT:
        current_thrust = SCRAMJET_THRUST
    else:
        current_thrust = 0
    
    ax = (current_thrust * vx / velocity - drag_force * vx / velocity) / mass
    ay = (current_thrust * vy / velocity - drag_force * vy / velocity) / mass - GRAVITY
    
    vx += ax * TIME_STEP
    vy += ay * TIME_STEP
    x += vx * TIME_STEP
    y += vy * TIME_STEP
    t += TIME_STEP
    
    # Calculated values
    accel = np.sqrt(ax**2 + ay**2)
    angle = np.degrees(np.arctan2(vy, vx))
    mach = velocity / SPEED_OF_SOUND
    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * GRAVITY * y
    
    # Log data
    time_log.append(t)
    x_log.append(x)
    y_log.append(y)
    vx_log.append(vx)
    vy_log.append(vy)
    speed_log.append(velocity)
    ax_log.append(ax)
    ay_log.append(ay)
    accel_log.append(accel)
    drag_log.append(drag_force)
    thrust_log.append(current_thrust)
    mass_log.append(mass)
    angle_log.append(angle)
    mach_log.append(mach)
    kinetic_energy_log.append(kinetic_energy)
    potential_energy_log.append(potential_energy)

# Plotting 15 graphs
plt.figure(figsize=(18, 30))

plt.subplot(5, 3, 1)
plt.plot(x_log, y_log)
plt.title("Trajectory (x vs y)")
plt.xlabel("Horizontal Distance (m)")
plt.ylabel("Altitude (m)")
plt.grid()

plt.subplot(5, 3, 2)
plt.plot(time_log, y_log)
plt.title("Altitude vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid()

plt.subplot(5, 3, 3)
plt.plot(time_log, x_log)
plt.title("Horizontal Distance vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.grid()

plt.subplot(5, 3, 4)
plt.plot(time_log, speed_log)
plt.title("Speed vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.grid()

plt.subplot(5, 3, 5)
plt.plot(time_log, vx_log)
plt.title("Horizontal Velocity (vx) vs Time")
plt.xlabel("Time (s)")
plt.ylabel("vx (m/s)")
plt.grid()

plt.subplot(5, 3, 6)
plt.plot(time_log, vy_log)
plt.title("Vertical Velocity (vy) vs Time")
plt.xlabel("Time (s)")
plt.ylabel("vy (m/s)")
plt.grid()

plt.subplot(5, 3, 7)
plt.plot(time_log, ax_log, label='ax')
plt.plot(time_log, ay_log, label='ay')
plt.title("Acceleration Components vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.legend()
plt.grid()

plt.subplot(5, 3, 8)
plt.plot(time_log, accel_log)
plt.title("Acceleration Magnitude vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.grid()

plt.subplot(5, 3, 9)
plt.plot(time_log, drag_log)
plt.title("Drag Force vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Drag (N)")
plt.grid()

plt.subplot(5, 3, 10)
plt.plot(time_log, thrust_log)
plt.title("Thrust vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Thrust (N)")
plt.grid()

plt.subplot(5, 3, 11)
plt.plot(time_log, mass_log)
plt.title("Mass vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Mass (kg)")
plt.grid()

plt.subplot(5, 3, 12)
plt.plot(time_log, angle_log)
plt.title("Flight Path Angle vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.grid()

plt.subplot(5, 3, 13)
plt.plot(time_log, mach_log)
plt.title("Mach Number vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Mach Number")
plt.grid()

plt.subplot(5, 3, 14)
plt.plot(time_log, kinetic_energy_log)
plt.title("Kinetic Energy vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Energy (Joules)")
plt.grid()

plt.subplot(5, 3, 15)
plt.plot(time_log, potential_energy_log)
plt.title("Potential Energy vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Energy (Joules)")
plt.grid()

plt.tight_layout()
plt.show()

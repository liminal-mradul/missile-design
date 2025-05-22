import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import cm
from itertools import product
import sys

# Suppress warnings for clean output
import warnings
warnings.filterwarnings("ignore")

# Missile class
class Missile:
    def __init__(self, groove_depth, spin_speed, spin_direction, launch_method, ignition_altitude, ignition_mach):
        self.groove_depth = groove_depth  # mm
        self.spin_speed = spin_speed  # RPM
        self.spin_direction = spin_direction  # 1: clockwise, -1: counterclockwise
        self.launch_method = launch_method
        self.ignition_altitude = ignition_altitude  # km
        self.ignition_mach = ignition_mach
        self.yield_strength = 1000  # MPa
        self.max_temperature = 1600  # °C (titanium base)
        self.length = 10  # m
        self.diameter = 0.5  # m
        self.mass = 1000  # kg
        self.moment_of_inertia = 0.5 * self.mass * (self.diameter / 2)**2  # kg·m²

# Launch method classes
class ArtilleryLaunch:
    def __init__(self, velocity, angle, spin_rate):
        self.velocity = velocity  # m/s
        self.angle = angle  # degrees
        self.spin_rate = spin_rate  # RPM
        self.g_force = velocity / 9.81 / 2  # Approximate launch G-force

class CentrifugeLaunch:
    def __init__(self, radius, spin_speed):
        self.radius = radius  # m
        self.spin_speed = spin_speed  # RPM
        self.velocity = radius * (spin_speed * 2 * np.pi / 60)  # Tangential velocity
        self.angle = 45  # Fixed
        self.g_force = self.velocity**2 / (self.radius * 9.81)  # Centripetal G-force

class SlingshotLaunch:
    def __init__(self, velocity, angle):
        self.velocity = velocity  # m/s
        self.angle = angle  # degrees
        self.spin_rate = 0
        self.g_force = velocity / 9.81 / 4  # Lower G-force

# Environment model
def environment(altitude):
    if altitude < 10:
        rho = 1.225 * np.exp(-altitude / 8)
        speed_of_sound = 340
    else:
        rho = 0.4135 * np.exp(-(altitude - 10) / 7)
        speed_of_sound = 295
    return {'rho': rho, 'speed_of_sound': speed_of_sound}

# Force and metric models
def drag_force(mach, groove_depth, altitude, print_step):
    rho = environment(altitude)['rho']
    cd = 0.1 + 0.02 * groove_depth
    area = np.pi * (0.25)**2
    v = mach * environment(altitude)['speed_of_sound']
    drag = 0.5 * rho * v**2 * cd * area
    if print_step % 100 == 0:  # Print every 100th call
        print(f"  Drag Calc: Mach={mach:.2f}, Depth={groove_depth}mm, Alt={altitude:.1f}km, ρ={rho:.3f}kg/m³, Cd={cd:.3f}, Area={area:.3f}m², v={v:.0f}m/s, Drag={drag:.0f}N")
    return drag

def lift_force(mach, groove_depth, altitude, print_step):
    rho = environment(altitude)['rho']
    cl = 0.05 * groove_depth
    area = np.pi * (0.25)**2
    v = mach * environment(altitude)['speed_of_sound']
    lift = 0.5 * rho * v**2 * cl * area
    if print_step % 100 == 0:  # Print every 100th call
        print(f"  Lift Calc: Mach={mach:.2f}, Depth={groove_depth}mm, Alt={altitude:.1f}km, ρ={rho:.3f}kg/m³, Cl={cl:.3f}, Area={area:.3f}m², v={v:.0f}m/s, Lift={lift:.0f}N")
    return lift

def spin_torque(groove_depth, velocity, spin_direction, print_step):
    torque = groove_depth * 0.01 * velocity * spin_direction
    if print_step % 100 == 0:  # Print every 100th call
        print(f"  Spin Torque Calc: Depth={groove_depth}mm, Velocity={velocity:.0f}m/s, Direction={spin_direction}, Torque={torque:.2f}N·m")
    return torque

def thermal_model(mach, groove_depth):
    base_temp = 300 * mach**2
    cooling_effect = 0.05 * groove_depth
    temp = base_temp * (1 - cooling_effect)
    print(f"  Thermal Calc: Mach={mach:.2f}, Depth={groove_depth}mm, BaseTemp={base_temp:.0f}°C, Cooling={cooling_effect:.3f}, Temp={temp:.0f}°C")
    return temp

def stress_model(g_force, spin_speed, groove_depth):
    base_stress = 1.5 * g_force
    spin_stress = 0.1 * spin_speed * groove_depth / 1000
    stress = base_stress + spin_stress
    print(f"  Stress Calc: G-Force={g_force:.1f}g, Spin={spin_speed}RPM, Depth={groove_depth}mm, BaseStress={base_stress:.0f}MPa, SpinStress={spin_stress:.0f}MPa, TotalStress={stress:.0f}MPa")
    return stress

def strain_model(stress, material_modulus=200e3):
    strain = stress / material_modulus
    print(f"  Strain Calc: Stress={stress:.0f}MPa, Modulus={material_modulus:.0f}MPa, Strain={strain:.5f}")
    return strain

def stability_deviation(spin_speed, groove_depth, mach):
    base_deviation = 5 / (spin_speed / 1000 + groove_depth)
    mach_effect = 0.1 * mach
    deviation = base_deviation + mach_effect
    print(f"  Stability Calc: Spin={spin_speed}RPM, Depth={groove_depth}mm, Mach={mach:.2f}, BaseDeviation={base_deviation:.2f}°, MachEffect={mach_effect:.2f}°, TotalDeviation={deviation:.2f}°")
    return deviation

def rcs_model(groove_depth, angle, spin_speed):
    base_rcs = 0.1
    reduction = 0.05 * groove_depth
    angle_effect = np.cos(np.radians(angle))**2
    spin_effect = 0.01 * spin_speed / 1000
    rcs = base_rcs * (1 - reduction) * angle_effect * (1 - spin_effect / 10)
    print(f"  RCS Calc: Depth={groove_depth}mm, Angle={angle:.0f}°, Spin={spin_speed}RPM, BaseRCS={base_rcs:.3f}m², Reduction={reduction:.3f}, AngleEffect={angle_effect:.3f}, SpinEffect={spin_effect:.3f}, RCS={rcs:.3f}m²")
    return rcs

# Initialize state
def initialize_state(launch_params):
    if isinstance(launch_params, ArtilleryLaunch):
        v0 = launch_params.velocity
        theta = np.radians(launch_params.angle)
        state = [0, 0, 0, v0 * np.cos(theta), 0, v0 * np.sin(theta), 0, 0, launch_params.spin_rate * 2 * np.pi / 60]
    elif isinstance(launch_params, CentrifugeLaunch):
        v0 = launch_params.velocity
        state = [0, 0, 0, v0 * np.cos(np.radians(45)), 0, v0 * np.sin(np.radians(45)), 0, 0, launch_params.spin_speed * 2 * np.pi / 60]
    else:  # Slingshot
        v0 = launch_params.velocity
        theta = np.radians(launch_params.angle)
        state = [0, 0, 0, v0 * np.cos(theta), 0, v0 * np.sin(theta), 0, 0, 0]
    print(f"  Initial State: x={state[0]:.1f}m, z={state[2]:.1f}m, vx={state[3]:.0f}m/s, vz={state[5]:.0f}m/s, wz={state[8]*60/(2*np.pi):.0f}RPM")
    return state

# 6-DOF ODE
def ode_func(t, state, missile, max_mach):
    x, y, z, vx, vy, vz, wx, wy, wz = state
    env = environment(z / 1000)
    velocity = np.sqrt(vx**2 + vz**2)
    mach = velocity / env['speed_of_sound']
    if mach > max_mach:
        mach = max_mach
    g = 9.81
    # Increment print step for force calculations
    if not hasattr(ode_func, 'force_print_step'):
        ode_func.force_print_step = 0
    drag = drag_force(mach, missile.groove_depth, z / 1000, ode_func.force_print_step)
    lift = lift_force(mach, missile.groove_depth, z / 1000, ode_func.force_print_step)
    torque = spin_torque(missile.groove_depth, velocity, missile.spin_direction, ode_func.force_print_step)
    ode_func.force_print_step += 1
    thrust = 50000 if z / 1000 > missile.ignition_altitude and mach > missile.ignition_mach else 0
    # Print every 10th step
    if not hasattr(ode_func, 'print_step'):
        ode_func.print_step = 0
    if ode_func.print_step % 10 == 0:
        print(f"  ODE Step: t={t:.1f}s, x={x/1000:.1f}km, z={z/1000:.1f}km, vx={vx:.0f}m/s, vz={vz:.0f}m/s, wz={wz*60/(2*np.pi):.0f}RPM, Mach={mach:.2f}, Thrust={thrust:.0f}N")
    ode_func.print_step += 1
    dstate = [
        vx, vy, vz,
        (thrust - drag) / missile.mass * (vx / velocity if velocity > 0 else 0),
        0,
        (-g + lift / missile.mass) * (vz / velocity if velocity > 0 else 0),
        0, 0, torque / missile.moment_of_inertia
    ]
    return dstate

# Check failures
def check_failures(sol, missile, max_mach):
    g_force = missile.launch_method.g_force
    max_stress = stress_model(g_force, missile.spin_speed, missile.groove_depth)
    max_strain = strain_model(max_stress)
    max_temp = thermal_model(max_mach, missile.groove_depth)
    stability = stability_deviation(missile.spin_speed, missile.groove_depth, max_mach)
    rcs = rcs_model(missile.groove_depth, 0, missile.spin_speed)
    failures = []
    if max_stress > missile.yield_strength:
        failures.append(f"Structural failure: Stress={max_stress:.0f}MPa > {missile.yield_strength}MPa")
    if max_strain > 0.005:
        failures.append(f"Strain failure: Strain={max_strain:.5f} > 0.005")
    if max_temp > missile.max_temperature:
        failures.append(f"Thermal failure: Temp={max_temp:.0f}°C > {missile.max_temperature}°C")
    if stability > 1:
        failures.append(f"Stability failure: Deviation={stability:.2f}° > 1°")
    if rcs > 0.1:
        failures.append(f"Stealth failure: RCS={rcs:.3f}m² > 0.1m²")
    return failures

# Simulate and plot
def simulate_and_plot():
    # Parameter ranges (optimized to avoid failures)
    groove_depths = [2, 3]  # Increased groove depth
    spin_speeds = [3000, 6000]  # Increased spin for stability
    spin_directions = [1, -1]
    velocities = [800, 1000]
    angles = [30, 45]
    ignition_altitudes = [15, 18]
    ignition_machs = [4, 4.5]
    max_mach = 7  # Reduced to avoid thermal failure

    launch_methods = [
        [ArtilleryLaunch(v, a, s) for v, a, s in product(velocities, angles, spin_speeds)],
        [CentrifugeLaunch(10, s) for s in spin_speeds],
        [SlingshotLaunch(v, a) for v, a in product(velocities, angles)]
    ]
    launch_methods = [item for sublist in launch_methods for item in sublist]

    # Data storage
    trajectories = []
    spin_rates = []
    drag_forces = []
    lift_forces = []
    temperatures = []
    stresses = []
    strains = []
    stabilities = []
    rcs_values = []
    g_forces = []

    # Parameter sweep (limited for demonstration)
    configs = list(product(groove_depths, spin_speeds, spin_directions, launch_methods, ignition_altitudes, ignition_machs))
    selected_configs = configs[:3]  # Remove slice for full sweep
    for depth, speed, direction, method, alt, mach in selected_configs:
        print(f"\nSimulating: Depth={depth}mm, Spin={speed}RPM, Dir={'CW' if direction==1 else 'CCW'}, {method.__class__.__name__}, Alt={alt}km, Mach={mach}")
        missile = Missile(depth, speed, direction, method, alt, mach)
        state = initialize_state(method)
        t_span = (0, 200)
        ode_func.print_step = 0  # Reset print counter
        ode_func.force_print_step = 0  # Reset force print counter
        sol = solve_ivp(ode_func, t_span, state, args=(missile, max_mach), method='RK45', max_step=1.0, rtol=1e-6, atol=1e-8)
        failures = check_failures(sol, missile, max_mach)
        label = f"Depth={depth}mm, Spin={speed}RPM, Dir={'CW' if direction==1 else 'CCW'}, {method.__class__.__name__}, Alt={alt}km, Mach={mach}"
        
        if failures:
            print(f"Failures for {label}:")
            for f in failures:
                print(f"  - {f}")
                if "Structural" in f or "Strain" in f:
                    print("    Solution: Reduce spin to 4000 RPM or reinforce grooves with titanium")
                    print("    Alternative: Use centrifuge launch")
                elif "Thermal" in f:
                    print("    Solution: Limit Mach to 7 or use 3 mm grooves")
                    print("    Alternative: Redesign with SiC ceramics")
                elif "Stability" in f:
                    print("    Solution: Increase spin to 3000 RPM or groove depth to 2 mm")
                elif "Stealth" in f:
                    print("    Solution: Use 3 mm grooves or apply radar-absorbing coating")
        else:
            print(f"Success: {label}")
            trajectories.append((sol.y[0] / 1000, sol.y[2] / 1000, label))
            spin_rates.append((sol.t, sol.y[8] * 60 / (2 * np.pi), label))
            drag_forces.append((depth, drag_force(max_mach, depth, alt, 0), label))
            lift_forces.append((depth, lift_force(max_mach, depth, alt, 0), label))
            temperatures.append((depth, thermal_model(max_mach, depth), label))
            stresses.append((depth, stress_model(method.g_force, speed, depth), label))
            strains.append((depth, strain_model(stress_model(method.g_force, speed, depth)), label))
            stabilities.append((depth, stability_deviation(speed, depth, max_mach), label))
            g_forces.append((depth, method.g_force, label))
            angles = np.linspace(0, 360, 100)
            rcs = [rcs_model(depth, angle, speed) for angle in angles]
            rcs_values.append((angles, rcs, label))

    # Plotting (only if data exists)
    if not trajectories:
        print("No successful simulations to plot.")
        return

    plt.figure(figsize=(20, 15))

    # 1. Trajectory
    plt.subplot(3, 4, 1)
    for x, z, label in trajectories:
        plt.plot(x, z, label=label)
    plt.xlabel('Range (km)')
    plt.ylabel('Altitude (km)')
    plt.title('Trajectories')
    plt.legend()
    plt.grid(True)

    # 2. Spin Rate
    plt.subplot(3, 4, 2)
    for t, spin, label in spin_rates:
        plt.plot(t, spin, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Spin Rate (RPM)')
    plt.title('Spin Rate')
    plt.legend()
    plt.grid(True)

    # 3. Drag Force
    plt.subplot(3, 4, 3)
    depths, drags, labels = zip(*drag_forces)
    plt.scatter(depths, drags, c=range(len(drags)), cmap='viridis')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('Drag Force (N)')
    plt.title('Drag Force vs. Depth')
    plt.colorbar(label='Configuration')
    plt.grid(True)

    # 4. Lift Force
    plt.subplot(3, 4, 4)
    depths, lifts, labels = zip(*lift_forces)
    plt.scatter(depths, lifts, c=range(len(lifts)), cmap='viridis')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('Lift Force (N)')
    plt.title('Lift Force vs. Depth')
    plt.colorbar(label='Configuration')
    plt.grid(True)

    # 5. Temperature
    plt.subplot(3, 4, 5)
    depths, temps, labels = zip(*temperatures)
    plt.scatter(depths, temps, c=range(len(temps)), cmap='plasma')
    plt.axhline(y=1600, color='r', linestyle='--', label='Max Temp')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature vs. Depth')
    plt.legend()
    plt.grid(True)

    # 6. Stress
    plt.subplot(3, 4, 6)
    depths, stresses, labels = zip(*stresses)
    plt.scatter(depths, stresses, c=range(len(stresses)), cmap='inferno')
    plt.axhline(y=1000, color='r', linestyle='--', label='Yield Strength')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('Stress (MPa)')
    plt.title('Stress vs. Depth')
    plt.legend()
    plt.grid(True)

    # 7. Strain
    plt.subplot(3, 4, 7)
    depths, strains, labels = zip(*strains)
    plt.scatter(depths, strains, c=range(len(strains)), cmap='magma')
    plt.axhline(y=0.005, color='r', linestyle='--', label='Max Strain')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('Strain')
    plt.title('Strain vs. Depth')
    plt.legend()
    plt.grid(True)

    # 8. Stability
    plt.subplot(3, 4, 8)
    depths, stabs, labels = zip(*stabilities)
    plt.scatter(depths, stabs, c=range(len(stabs)), cmap='cool')
    plt.axhline(y=1, color='r', linestyle='--', label='Max Deviation')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('Deviation (°)')
    plt.title('Stability vs. Depth')
    plt.legend()
    plt.grid(True)

    # 9. RCS (Polar)
    plt.subplot(3, 4, 9, polar=True)
    for angles, rcs, label in rcs_values:
        plt.plot(np.radians(angles), rcs, label=label)
    plt.title('Radar Cross-Section (m²)')
    plt.legend()

    # 10. G-Forces
    plt.subplot(3, 4, 10)
    depths, gs, labels = zip(*g_forces)
    plt.scatter(depths, gs, c=range(len(gs)), cmap='hot')
    plt.xlabel('Groove Depth (mm)')
    plt.ylabel('G-Force')
    plt.title('G-Force vs. Depth')
    plt.colorbar(label='Configuration')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('missile_simulation_plots.png')
    plt.show()

if __name__ == "__main__":
    simulate_and_plot()

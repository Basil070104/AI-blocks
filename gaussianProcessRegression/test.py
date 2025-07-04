import numpy as np
import matplotlib.pyplot as plt

# Kinematic bicycle model
class KinematicBicycleModel:
    def __init__(self, L=2.5):
        self.L = L  # wheelbase

    def step(self, state, control, dt):
        x, y, yaw, v = state
        delta, a = control
        x += v * np.cos(yaw) * dt
        y += v * np.sin(yaw) * dt
        yaw += v / self.L * np.tan(delta) * dt
        v += a * dt
        return np.array([x, y, yaw, v])

# Reference trajectory (straight line)
def generate_reference_trajectory(T, dt):
    t = np.arange(0, T, dt)
    x_ref = t
    y_ref = np.zeros_like(t)
    yaw_ref = np.zeros_like(t)
    v_ref = np.ones_like(t) * 5.0  # 5 m/s
    return np.stack([x_ref, y_ref, yaw_ref, v_ref], axis=1)

# Simple MPC loop (mock: brute-force over a small set of actions)
def mpc_control(model, state, ref_traj, dt, N):
    # Discretize control space (mock)
    delta_space = np.deg2rad(np.linspace(-20, 20, 5))  # steering
    a_space = np.linspace(-1, 1, 3)  # acceleration
    best_cost = float('inf')
    best_u = (0.0, 0.0)
    for delta in delta_space:
        for a in a_space:
            pred_state = state.copy()
            cost = 0.0
            for i in range(N):
                pred_state = model.step(pred_state, (delta, a), dt)
                ref = ref_traj[i]
                cost += np.sum((pred_state[:2] - ref[:2]) ** 2)  # position error
                cost += 0.1 * (pred_state[3] - ref[3]) ** 2  # speed error
                cost += 0.01 * (delta ** 2 + a ** 2)  # control effort
            if cost < best_cost:
                best_cost = cost
                best_u = (delta, a)
    return best_u

def main():
    dt = 0.1
    T = 10.0
    N = 5  # prediction horizon
    model = KinematicBicycleModel()
    ref_traj = generate_reference_trajectory(int(T/dt)+N, dt)
    state = np.array([0.0, 0.0, 0.0, 0.0])  # x, y, yaw, v
    states = [state.copy()]
    controls = []
    for t in range(int(T/dt)):
        ref_window = ref_traj[t:t+N]
        u = mpc_control(model, state, ref_window, dt, N)
        state = model.step(state, u, dt)
        states.append(state.copy())
        controls.append(u)
    states = np.array(states)
    # Plot results
    plt.figure()
    plt.plot(ref_traj[:,0], ref_traj[:,1], 'r--', label='Reference')
    plt.plot(states[:,0], states[:,1], 'b-', label='MPC Path')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.axis('equal')
    plt.title('Mock MPC for Car Path Tracking')
    plt.show()

if __name__ == "__main__":
    main()

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from .cartpole import CartPoleParams, rk4_step, wrap_angle
except ImportError:
    from cartpole import CartPoleParams, rk4_step, wrap_angle


SCRIPT_DIR = Path(__file__).resolve().parent
N_X = 4
N_U = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Run receding-horizon MPC with iLQR on week4/cartpole.py.")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--force-limit", type=float, default=10.0)
    parser.add_argument("--x0", type=float, nargs=4, default=[0.0, 0.0, np.pi - 0.2, 0.0])
    parser.add_argument("--initial-guess", choices=["zero", "sine"], default="sine")
    parser.add_argument("--sine-amplitude", type=float, default=8.0)
    parser.add_argument("--sine-frequency", type=float, default=0.8)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


class Cost:
    def __init__(self):
        self.x_goal = np.zeros(N_X)
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1])
        self.R = np.diag([0.001])
        self.Qf = np.diag([10.0, 1.0, 100.0, 10.0])

    def state_error(self, x, x_ref=None):
        x_ref = self.x_goal if x_ref is None else np.asarray(x_ref, dtype=float)
        err = np.asarray(x, dtype=float) - x_ref
        err[2] = wrap_angle(err[2])
        return err

    def stage(self, x, u):
        err = self.state_error(x)
        u = np.asarray(u, dtype=float).reshape(N_U)
        return float(err.T @ self.Q @ err + u.T @ self.R @ u)

    def final(self, x):
        err = self.state_error(x)
        return float(err.T @ self.Qf @ err)

    def stage_derivatives(self, x, u):
        err = self.state_error(x)
        u = np.asarray(u, dtype=float).reshape(N_U)
        l_x = 2.0 * self.Q @ err
        l_u = 2.0 * self.R @ u
        l_xx = 2.0 * self.Q
        l_uu = 2.0 * self.R
        l_ux = np.zeros((N_U, N_X))
        return l_x, l_u, l_xx, l_uu, l_ux

    def final_derivatives(self, x):
        err = self.state_error(x)
        return 2.0 * self.Qf @ err, 2.0 * self.Qf


def discrete_dynamics(x, u, params):
    force = float(np.clip(np.asarray(u, dtype=float).reshape(N_U)[0], -params.force_limit, params.force_limit))
    return rk4_step(x, force, params=params, dt=params.dt, wrap=True)


def rollout(x0, u_trj, params):
    u_trj = np.asarray(u_trj, dtype=float)
    x_trj = np.zeros((u_trj.shape[0] + 1, N_X))
    x_trj[0] = np.asarray(x0, dtype=float)
    x_trj[0, 2] = wrap_angle(x_trj[0, 2])
    for t in range(u_trj.shape[0]):
        x_trj[t + 1] = discrete_dynamics(x_trj[t], u_trj[t], params)
    return x_trj


def trajectory_cost(x_trj, u_trj, cost):
    total = 0.0
    for t in range(u_trj.shape[0]):
        total += cost.stage(x_trj[t], u_trj[t])
    return total + cost.final(x_trj[-1])


def dynamics_derivatives(x, u, params, eps=1e-5):
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float).reshape(N_U)

    f_x = np.zeros((N_X, N_X))
    f_u = np.zeros((N_X, N_U))

    for i in range(N_X):
        dx = np.zeros(N_X)
        dx[i] = eps
        f_x[:, i] = (
            discrete_dynamics(x + dx, u, params)
            - discrete_dynamics(x - dx, u, params)
        ) / (2.0 * eps)

    for i in range(N_U):
        du = np.zeros(N_U)
        du[i] = eps
        f_u[:, i] = (
            discrete_dynamics(x, u + du, params)
            - discrete_dynamics(x, u - du, params)
        ) / (2.0 * eps)

    return f_x, f_u


def is_positive_definite(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def backward_pass(x_trj, u_trj, cost, params, regu):
    horizon = u_trj.shape[0]
    k_trj = np.zeros((horizon, N_U))
    K_trj = np.zeros((horizon, N_U, N_X))

    V_x, V_xx = cost.final_derivatives(x_trj[-1])

    for t in reversed(range(horizon)):
        x = x_trj[t]
        u = u_trj[t]
        f_x, f_u = dynamics_derivatives(x, u, params)
        l_x, l_u, l_xx, l_uu, l_ux = cost.stage_derivatives(x, u)

        Q_x = l_x + f_x.T @ V_x
        Q_u = l_u + f_u.T @ V_x
        Q_xx = l_xx + f_x.T @ V_xx @ f_x
        Q_ux = l_ux + f_u.T @ V_xx @ f_x
        Q_uu = l_uu + f_u.T @ V_xx @ f_u
        Q_xx = 0.5 * (Q_xx + Q_xx.T)
        Q_uu = 0.5 * (Q_uu + Q_uu.T)

        Q_uu_reg = Q_uu + regu * np.eye(N_U)
        if not is_positive_definite(Q_uu_reg):
            return k_trj, K_trj, False

        k = -np.linalg.solve(Q_uu_reg, Q_u)
        K = -np.linalg.solve(Q_uu_reg, Q_ux)
        k_trj[t] = k
        K_trj[t] = K

        V_x = Q_x + K.T @ Q_u + Q_ux.T @ k + K.T @ Q_uu_reg @ k
        V_xx = Q_xx + K.T @ Q_uu_reg @ K + K.T @ Q_ux + Q_ux.T @ K
        V_xx = 0.5 * (V_xx + V_xx.T)

    return k_trj, K_trj, True


def clip_control(u, params):
    return np.clip(np.asarray(u, dtype=float).reshape(N_U), -params.force_limit, params.force_limit)


def forward_pass(x_trj, u_trj, k_trj, K_trj, alpha, cost, params):
    horizon = u_trj.shape[0]
    x_new = np.zeros_like(x_trj)
    u_new = np.zeros_like(u_trj)
    x_new[0] = x_trj[0]

    for t in range(horizon):
        dx = cost.state_error(x_new[t], x_trj[t])
        u = u_trj[t] + alpha * k_trj[t] + K_trj[t] @ dx
        u = clip_control(u, params)
        u_new[t] = u
        x_new[t + 1] = discrete_dynamics(x_new[t], u, params)

    return x_new, u_new


def ilqr(x0, u_init, cost, params, max_iter=8, tol=1e-3, regu_init=1e-4, verbose=False):
    alphas = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    regu = regu_init
    regu_factor = 10.0
    regu_min = 1e-8
    regu_max = 1e8

    u_trj = np.asarray(u_init, dtype=float).copy()
    x_trj = rollout(x0, u_trj, params)
    J = trajectory_cost(x_trj, u_trj, cost)

    K_trj = np.zeros((u_trj.shape[0], N_U, N_X))
    accepted_iterations = 0

    for iteration in range(max_iter):
        k_candidate, K_candidate, success = backward_pass(x_trj, u_trj, cost, params, regu)
        if not success:
            regu = min(regu * regu_factor, regu_max)
            if regu >= regu_max:
                break
            continue

        accepted = False
        best = (x_trj, u_trj, J)
        for alpha in alphas:
            x_new, u_new = forward_pass(x_trj, u_trj, k_candidate, K_candidate, alpha, cost, params)
            J_new = trajectory_cost(x_new, u_new, cost)
            if J_new < J:
                best = (x_new, u_new, J_new)
                accepted = True
                break

        if accepted:
            improvement = J - best[2]
            x_trj, u_trj, J = best
            K_trj = K_candidate
            regu = max(regu / regu_factor, regu_min)
            accepted_iterations += 1
            if verbose:
                print(f"  ilqr iter={iteration:02d} cost={J:.3f} improvement={improvement:.3f}")
            if improvement < tol:
                break
        else:
            regu = min(regu * regu_factor, regu_max)
            if regu >= regu_max:
                break

    return x_trj, u_trj, K_trj, {"final_cost": J, "accepted_iterations": accepted_iterations}


def make_initial_controls(horizon, params, mode, amplitude, frequency):
    if mode == "zero":
        return np.zeros((horizon, N_U))
    time_grid = np.arange(horizon) * params.dt
    controls = amplitude * np.sin(2.0 * np.pi * frequency * time_grid)
    controls = np.clip(controls, -params.force_limit, params.force_limit)
    return controls.reshape(horizon, N_U)


def shift_controls(u_trj):
    shifted = np.zeros_like(u_trj)
    if u_trj.shape[0] > 1:
        shifted[:-1] = u_trj[1:]
    shifted[-1] = u_trj[-1]
    return shifted


class Renderer:
    def __init__(self, params):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        self.plt = plt
        self.params = params
        self.cart_width = 0.5
        self.cart_height = 0.25
        self.pole_length = params.pole_length

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_ylim(-0.35, self.pole_length + 0.6)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_xlabel("x")
        self.ax.set_title("iLQR MPC CartPole")

        (self.rail,) = self.ax.plot([], [], color="black", linewidth=2)
        self.cart = Rectangle(
            (-self.cart_width / 2.0, -self.cart_height / 2.0),
            self.cart_width,
            self.cart_height,
            facecolor="#4C78A8",
            edgecolor="black",
            linewidth=1.5,
        )
        self.ax.add_patch(self.cart)
        (self.pole_line,) = self.ax.plot([], [], color="#F58518", linewidth=5, solid_capstyle="round")
        self.pivot = Circle((0.0, 0.0), radius=0.045, facecolor="black", zorder=4)
        self.tip = Circle((0.0, 0.0), radius=0.04, facecolor="#E45756", zorder=4)
        self.ax.add_patch(self.pivot)
        self.ax.add_patch(self.tip)
        self.time_text = self.ax.text(0.02, 0.94, "", transform=self.ax.transAxes)

    def render(self, state, force, step):
        x, _, theta, theta_dot = np.asarray(state, dtype=float)
        theta = wrap_angle(theta)
        margin = max(2.4, abs(x) + self.pole_length + 0.8)
        x_min = -margin
        x_max = margin

        self.ax.set_xlim(x_min, x_max)
        rail_y = -self.cart_height / 2.0
        self.rail.set_data([x_min, x_max], [rail_y, rail_y])
        self.cart.set_xy((x - self.cart_width / 2.0, -self.cart_height / 2.0))

        pivot_x = x
        pivot_y = self.cart_height / 2.0
        tip_x = pivot_x + self.pole_length * np.sin(theta)
        tip_y = pivot_y + self.pole_length * np.cos(theta)

        self.pole_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        self.pivot.center = (pivot_x, pivot_y)
        self.tip.center = (tip_x, tip_y)
        self.time_text.set_text(
            f"t = {step * self.params.dt:.2f}s\n"
            f"theta = {theta:.3f} rad\n"
            f"theta_dot = {theta_dot:.3f} rad/s\n"
            f"force = {force:.2f} N"
        )

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def run_mpc(args):
    params = CartPoleParams(dt=args.dt, force_limit=args.force_limit)
    cost = Cost()
    x_current = np.asarray(args.x0, dtype=float)
    x_current[2] = wrap_angle(x_current[2])
    u_guess = make_initial_controls(
        args.horizon,
        params,
        mode=args.initial_guess,
        amplitude=args.sine_amplitude,
        frequency=args.sine_frequency,
    )
    renderer = Renderer(params) if args.render else None

    states = [x_current.copy()]
    controls = []
    plan_costs = []
    max_steps = None if args.forever else args.steps
    step = 0

    try:
        while max_steps is None or step < max_steps:
            x_plan, u_plan, _, info = ilqr(
                x_current,
                u_guess,
                cost,
                params,
                max_iter=args.max_iter,
                verbose=False,
            )
            u_apply = clip_control(u_plan[0], params)
            x_next = discrete_dynamics(x_current, u_apply, params)

            states.append(x_next.copy())
            controls.append(u_apply.copy())
            plan_costs.append(info["final_cost"])

            if args.log_interval > 0 and step % args.log_interval == 0:
                print(
                    f"step={step:04d} cost={info['final_cost']:.2f} "
                    f"accepted={info['accepted_iterations']} "
                    f"x={x_current[0]: .3f} theta={wrap_angle(x_current[2]): .3f} "
                    f"theta_dot={x_current[3]: .3f} u={u_apply[0]: .3f}"
                )

            if renderer is not None:
                renderer.render(x_next, u_apply[0], step)

            u_guess = shift_controls(u_plan)
            x_current = x_next
            step += 1

            if args.real_time:
                time.sleep(params.dt)
    except KeyboardInterrupt:
        print("\nStopped.")

    states = np.asarray(states)
    controls = np.asarray(controls)
    plan_costs = np.asarray(plan_costs)
    if args.save:
        np.savez(args.save, states=states, controls=controls, plan_costs=plan_costs, dt=params.dt)
        print(f"Saved rollout to {args.save}")

    return states, controls, plan_costs


def main():
    args = parse_args()
    run_mpc(args)


if __name__ == "__main__":
    main()

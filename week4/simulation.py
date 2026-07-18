import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

try:
    from .cartpole import CartPole, CartPoleParams
except ImportError:
    from cartpole import CartPole, CartPoleParams


def simulate_cartpole(state0=None, total_time=6.0, params=None, forces=None, seed=1):
    params = CartPoleParams(dt=0.02) if params is None else params
    env = CartPole(params=params, seed=seed)

    if state0 is None:
        state0 = np.array([0.0, 0.0, 0.35, 0.0])
    else:
        state0 = np.asarray(state0, dtype=float)

    steps = int(total_time / params.dt)
    if forces is None:
        forces = np.zeros(steps)
    else:
        forces = np.asarray(forces, dtype=float)

    states = env.rollout(state0, forces, wrap=False)
    return states, forces, params


def create_cartpole_animation(states, params, frame_stride=2):
    plt.rcParams["animation.html"] = "jshtml"

    states = np.asarray(states, dtype=float)
    frame_ids = np.arange(0, len(states), frame_stride)

    cart_width = 0.5
    cart_height = 0.25
    pole_render_length = params.pole_length

    x_values = states[:, 0]
    x_min = min(-2.4, x_values.min() - 0.8)
    x_max = max(2.4, x_values.max() + 0.8)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-pole_render_length - 0.35, pole_render_length + 0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_title("CartPole physical simulation")

    rail_y = -cart_height / 2.0
    ax.plot([x_min, x_max], [rail_y, rail_y], color="black", linewidth=2)

    cart = Rectangle(
        (-cart_width / 2.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        facecolor="#4C78A8",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.add_patch(cart)

    (pole_line,) = ax.plot([], [], color="#F58518", linewidth=5, solid_capstyle="round")
    pivot = Circle((0.0, 0.0), radius=0.045, facecolor="black", zorder=4)
    tip = Circle((0.0, 0.0), radius=0.04, facecolor="#E45756", zorder=4)
    ax.add_patch(pivot)
    ax.add_patch(tip)

    time_text = ax.text(0.02, 0.94, "", transform=ax.transAxes)

    def update(frame_id):
        x, x_dot, theta, theta_dot = states[frame_id]
        cart.set_xy((x - cart_width / 2.0, -cart_height / 2.0))

        pivot_x = x
        pivot_y = cart_height / 2.0
        tip_x = pivot_x + pole_render_length * np.sin(theta)
        tip_y = pivot_y + pole_render_length * np.cos(theta)

        pole_line.set_data([pivot_x, tip_x], [pivot_y, tip_y])
        pivot.center = (pivot_x, pivot_y)
        tip.center = (tip_x, tip_y)
        time_text.set_text(f"t = {frame_id * params.dt:.2f}s, theta = {theta:.2f} rad")

        return cart, pole_line, pivot, tip, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=frame_ids,
        interval=1000 * params.dt * frame_stride,
        blit=True,
    )
    plt.close(fig)
    return anim


if __name__ == "__main__":
    states, forces, params = simulate_cartpole()
    anim = create_cartpole_animation(states, params)
    print(f"Simulated {len(forces)} steps from state {states[0]} to {states[-1]}")

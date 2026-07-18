from dataclasses import dataclass

import numpy as np


@dataclass
class CartPoleParams:
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 1.0
    gravity: float = 9.81
    dt: float = 0.02
    force_limit: float = 10.0
    x_limit: float = 5.0


def wrap_angle(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def cartpole_dynamics(state, force, params=CartPoleParams()):
    """Continuous-time dynamics: ds/dt = f(s, u).

    State order: [x, x_dot, theta, theta_dot].
    theta = 0 is the upright equilibrium.
    """
    state = np.asarray(state, dtype=float)
    if state.shape[-1] != 4:
        raise ValueError("state must have shape (..., 4)")

    x, x_dot, theta, theta_dot = np.moveaxis(state, -1, 0)

    u = np.asarray(force, dtype=float)
    if u.shape[-1:] == (1,):
        u = np.squeeze(u, axis=-1)
    u = np.broadcast_to(u, state.shape[:-1])

    M = params.cart_mass
    m = params.pole_mass
    L = params.pole_length
    g = params.gravity

    r = 0.5 * L
    I_cm = (1.0 / 12.0) * m * L**2
    J = I_cm + m * r**2

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    a11 = M + m
    a12 = m * r * cos_theta
    a22 = J
    rhs1 = u + m * r * sin_theta * theta_dot**2
    rhs2 = m * g * r * sin_theta
    determinant = a11 * a22 - a12**2

    x_ddot = (rhs1 * a22 - a12 * rhs2) / determinant
    theta_ddot = (a11 * rhs2 - a12 * rhs1) / determinant

    return np.stack([x_dot, x_ddot, theta_dot, theta_ddot], axis=-1).astype(float)


def euler_step(state, force, params=CartPoleParams(), dt=None, wrap=True):
    dt = params.dt if dt is None else dt
    next_state = np.asarray(state, dtype=float) + dt * cartpole_dynamics(state, force, params)
    if wrap:
        next_state = np.array(next_state, dtype=float, copy=True)
        next_state[..., 2] = wrap_angle(next_state[..., 2])
    return next_state


def rk4_step(state, force, params=CartPoleParams(), dt=None, wrap=True):
    dt = params.dt if dt is None else dt
    state = np.asarray(state, dtype=float)

    k1 = cartpole_dynamics(state, force, params)
    k2 = cartpole_dynamics(state + 0.5 * dt * k1, force, params)
    k3 = cartpole_dynamics(state + 0.5 * dt * k2, force, params)
    k4 = cartpole_dynamics(state + dt * k3, force, params)

    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if wrap:
        next_state = np.array(next_state, dtype=float, copy=True)
        next_state[..., 2] = wrap_angle(next_state[..., 2])
    return next_state


class CartPole:
    def __init__(self, params=CartPoleParams(), state=None, seed=None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(4, dtype=float)
        self.state_dim = self.state.size
        self.reset(state)

    def reset(self, state=None, noise_scale=0.05):
        if state is None:
            self.state = self.rng.uniform(low=-noise_scale, high=noise_scale, size=4)
        else:
            self.state = np.asarray(state, dtype=float).copy()
        self.state[2] = wrap_angle(self.state[2])
        return self.state.copy()

    def dynamics(self, state, force):
        return cartpole_dynamics(state, force, self.params)

    def step(self, force, dt=None, method="rk4", clip_force=True, wrap=True, return_done=False):
        u = float(np.asarray(force).reshape(()))
        if clip_force:
            u = np.clip(u, -self.params.force_limit, self.params.force_limit)

        if method == "euler":
            self.state = euler_step(self.state, u, self.params, dt=dt, wrap=wrap)
        elif method == "rk4":
            self.state = rk4_step(self.state, u, self.params, dt=dt, wrap=wrap)
        else:
            raise ValueError("method must be 'euler' or 'rk4'")

        if return_done:
            return self.state.copy(), self.is_done()

        return self.state.copy()

    def rollout(self, state0, forces, method="rk4", clip_force=True, wrap=True):
        states = [self.reset(state0)]
        for force in forces:
            states.append(self.step(force, method=method, clip_force=clip_force, wrap=wrap))
        return np.asarray(states)
    
    def rollout_for_rl(self,actor,num_steps, state0 = None,method="rk4", clip_force=True, wrap=True):
        states = [self.reset(state0)]
        log_probs = []
        for t in range(num_steps):
            force, log_prob = actor(states[-1])
            log_probs.append(log_prob)
            states.append(self.step(force, method=method, clip_force=clip_force, wrap=wrap))
        return np.asarray(states), np.asarray(log_probs)
    
    def is_done(self, state=None):
        """Episode ends when cart leaves [-x_limit, x_limit]."""
        state = self.state if state is None else np.asarray(state, dtype=float)
        return abs(float(state[0])) > self.params.x_limit


def cartpole_energy(state, params=CartPoleParams()):
    """Mechanical energy, only used as a physics sanity check."""
    x, x_dot, theta, theta_dot = np.asarray(state, dtype=float)
    M = params.cart_mass
    m = params.pole_mass
    L = params.pole_length
    g = params.gravity

    r = 0.5 * L
    I_cm = (1.0 / 12.0) * m * L**2
    J = I_cm + m * r**2

    kinetic = 0.5 * (M + m) * x_dot**2
    kinetic += m * r * x_dot * theta_dot * np.cos(theta)
    kinetic += 0.5 * J * theta_dot**2
    potential = m * g * r * np.cos(theta)
    return kinetic + potential

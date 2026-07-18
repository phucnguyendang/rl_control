## 1. Optimal Control Problem

Consider a continuous-time dynamical system

$$
\dot{x}=f(x,u,t)
$$

where:

- $x \in \mathbb{R}^n$ is the state,
- $u \in \mathbb{R}^m$ is the control input,
- $f(x,u,t)$ describes the system dynamics.

The goal of optimal control is to choose a control trajectory $u(\cdot)$ that minimizes a cost function.

A general finite-horizon cost has the form

$$
J
=
\ell_f(x(t_f))
+
\int_{0}^{t_f}
\ell(x(\tau),u(\tau),\tau)\,d\tau.
$$

Here:

- $\ell(x,u,t)$ is the **running cost** or **instantaneous cost**,
- $\ell_f(x(t_f))$ is the **terminal cost**,
- $t_f$ is the terminal time.

The running cost measures how much cost is accumulated during the motion, while the terminal cost measures how good or bad the final state is.

So the finite-horizon optimal control problem is

$$
\min_{u(\cdot)}
\left[
\ell_f(x(t_f))
+
\int_{t}^{t_f}
\ell(x(\tau),u(\tau),\tau)\,d\tau
\right]
$$

subject to

$$
\dot{x}=f(x,u,t).
$$

The infinite-horizon problem is a special case where there is no terminal time:

$$
J
=
\int_0^\infty
\ell(x(t),u(t))\,dt.
$$

If the dynamics and cost do not explicitly depend on time,

$$
\dot{x}=f(x,u),
\qquad
\ell=\ell(x,u),
$$

then the infinite-horizon problem is time-invariant.

---

## 2. Value Function

The value function represents the best possible future cost starting from a given state.

For the finite-horizon problem, the value function is

$$
V(x,t)
=
\min_{u(\cdot)}
\left[
\ell_f(x(t_f))
+
\int_{t}^{t_f}
\ell(x(\tau),u(\tau),\tau)\,d\tau
\right].
$$

In words, $V(x,t)$ is the minimum cost-to-go if the system is at state $x$ at time $t$.

The value function depends on both $x$ and $t$ because, in a finite-horizon problem, the remaining time matters.

For example, being at the same state $x$ with a lot of time remaining is different from being at the same state $x$ very close to the final time.

At the terminal time, there is no future running cost left. Therefore, the value function must satisfy the terminal condition

$$
V(x,t_f)=\ell_f(x).
$$

For an infinite-horizon time-invariant problem, there is no terminal time, and the problem looks the same regardless of the current time. Therefore, the value function depends only on the state:

$$
V(x,t)=V(x).
$$

---

## 3. Bellman Optimality Principle

The Bellman optimality principle says:

> The optimal cost from the current state and time equals the cost accumulated over a short time interval plus the optimal cost-to-go from the next state.

Consider a small time interval $h>0$.

Starting from state $x$ at time $t$, we can split the total cost into two parts:

1. the cost accumulated during $[t,t+h]$,
2. the optimal cost-to-go after reaching $x(t+h)$.

Therefore,

$$
V(x,t)
=
\min_u
\left[
\int_t^{t+h}
\ell(x(\tau),u(\tau),\tau)\,d\tau
+
V(x(t+h),t+h)
\right].
$$

This is the continuous-time version of the Bellman recursion.

The important idea is that after the first short interval, we do not need to remember the whole past trajectory. Once the system reaches $x(t+h)$, the remaining optimal cost is simply

$$
V(x(t+h),t+h).
$$

This is the key dynamic programming idea behind the HJB equation.

---

## 4. Hamilton–Jacobi–Bellman Equation

We now derive the Hamilton–Jacobi–Bellman equation from the Bellman optimality principle.

Start from

$$
V(x,t)
=
\min_u
\left[
\int_t^{t+h}
\ell(x(\tau),u(\tau),\tau)\,d\tau
+
V(x(t+h),t+h)
\right].
$$

For small $h$, assume the control input is approximately constant over $[t,t+h]$.

Because

$$
\dot{x}=f(x,u,t),
$$

we have

$$
x(t+h)
=
x+h f(x,u,t)+o(h).
$$

Also, the cost accumulated over the short interval is

$$
\int_t^{t+h}
\ell(x(\tau),u(\tau),\tau)\,d\tau
=
h\ell(x,u,t)+o(h).
$$

Substituting these approximations into the Bellman equation gives

$$
V(x,t)
=
\min_u
\left[
h\ell(x,u,t)
+
V(x+h f(x,u,t),t+h)
+
o(h)
\right].
$$

Assume that $V(x,t)$ is differentiable. Using a first-order Taylor expansion,

$$
V(x+h f,t+h)
=
V(x,t)
+
h\frac{\partial V}{\partial t}
+
h\nabla_x V(x,t)^T f(x,u,t)
+
o(h).
$$

Therefore,

$$
V(x,t)
=
\min_u
\left[
h\ell(x,u,t)
+
V(x,t)
+
h\frac{\partial V}{\partial t}
+
h\nabla_x V(x,t)^T f(x,u,t)
+
o(h)
\right].
$$

Subtract $V(x,t)$ from both sides:

$$
0
=
\min_u
\left[
h\ell(x,u,t)
+
h\frac{\partial V}{\partial t}
+
h\nabla_x V(x,t)^T f(x,u,t)
+
o(h)
\right].
$$

Divide by $h$:

$$
0
=
\min_u
\left[
\ell(x,u,t)
+
\frac{\partial V}{\partial t}
+
\nabla_x V(x,t)^T f(x,u,t)
+
\frac{o(h)}{h}
\right].
$$

Taking the limit as $h \to 0$, we obtain

$$
\boxed{
0
=
\min_u
\left[
\ell(x,u,t)
+
\nabla_x V(x,t)^T f(x,u,t)
+
\frac{\partial V}{\partial t}
\right].
}
$$

This is the general continuous-time Hamilton–Jacobi–Bellman equation.

Equivalently,

$$
\boxed{
-\frac{\partial V}{\partial t}
=
\min_u
\left[
\ell(x,u,t)
+
\nabla_x V(x,t)^T f(x,u,t)
\right].
}
$$

---

### 4.1 Finite-horizon HJB equation

For a finite-horizon problem, the value function depends on both state and time:

$$
V=V(x,t).
$$

Therefore, the HJB equation keeps the time derivative term:

$$
\boxed{
0
=
\min_u
\left[
\ell(x,u,t)
+
\nabla_x V(x,t)^T f(x,u,t)
+
\frac{\partial V}{\partial t}
\right].
}
$$

The terminal condition is

$$
\boxed{
V(x,t_f)=\ell_f(x).
}
$$

So in finite-horizon optimal control, the HJB equation is solved backward in time from the terminal condition.

---

### 4.2 Infinite-horizon HJB equation

Now consider the infinite-horizon time-invariant problem:

$$
J
=
\int_0^\infty
\ell(x(t),u(t))\,dt
$$

subject to

$$
\dot{x}=f(x,u).
$$

Because there is no terminal time and the problem does not explicitly depend on time, the value function is time-invariant:

$$
V(x,t)=V(x).
$$

Therefore,

$$
\frac{\partial V}{\partial t}=0.
$$

Substituting this into the general HJB equation gives

$$
\boxed{
0
=
\min_u
\left[
\ell(x,u)
+
\nabla_x V(x)^T f(x,u)
\right].
}
$$

This is the stationary HJB equation for the infinite-horizon continuous-time optimal control problem.

---

### 4.3 Summary

The general HJB equation is

$$
\boxed{
0
=
\min_u
\left[
\ell
+
V_x f
+
V_t
\right].
}
$$

For finite horizon:

$$
\boxed{
V_t \neq 0,
\qquad
V(x,t_f)=\ell_f(x).
}
$$

For infinite horizon with time-invariant dynamics and cost:

$$
\boxed{
V_t=0.
}
$$

Therefore, the infinite-horizon HJB equation becomes

$$
\boxed{
0
=
\min_u
\left[
\ell(x,u)
+
V_x f(x,u)
\right].
}
$$
## 6. LQR problem

The Linear Quadratic Regulator is a special optimal control problem where the dynamics are linear and the cost is quadratic:

$$ \dot{x}=Ax+Bu $$

and

$$ J = \int_0^\infty \left( x^TQx+u^TRu \right)dt $$

In this case,

$$ f(x,u)=Ax+Bu $$

and

$$ \ell(x,u)=x^TQx+u^TRu $$

while $x^TQx$ phạt state lệch khỏi điểm cân bằng và $u^TRu$ phạt việc sử dụng control quá mạnh

Substituting these into the HJB equation gives

$$ 0 = \min_u \left[ x^TQx + u^TRu + \frac{\partial V}{\partial x}(Ax+Bu) \right] $$

For LQR, the optimal value function has the quadratic form (easy to verify)

$$ V(x)=x^TSx $$

where $S=S^T \succeq 0$ (tức S là ma trận đối xứng và nửa xác dịnh dương). Substituting this quadratic form into the HJB equation leads to the algebraic Riccati equation, which determines the optimal feedback controller.

Thus, HJB provides the theoretical bridge between dynamic programming and the LQR solution.

The gradient of $J*$ given by:
$$ \frac{\partial J*}{\partial x} = 2x^TS $$

Replacing it into the HJB equation, we have
$$ 0 = \min_u \left[ x^TQx + u^TRu + 2x^TS(Ax+Bu) \right] $$

Put F(u) = $x^TQx + u^TRu + 2x^TS(Ax+Bu)$. Because $F(u)$ is quadractic and convex so we can take the minimum explicitly by finding the solution where the gradient of those terms vanishes:
$$ \frac{\partial F}{\partial u} = 2Ru + 2B^TSx = 0 $$

This yields the optimal policy
$$ \mathbf{u}^* = \pi^*(\mathbf{x}) = -\mathbf{R}^{-1}\mathbf{B}^T\mathbf{S}\mathbf{x} = -\mathbf{K}\mathbf{x}. $$

Inserting this back into the HJB and simplifying yields
$$ 0 = \mathbf{x}^T \left[ \mathbf{Q} - \mathbf{S}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{S} + 2\mathbf{S}\mathbf{A} \right]\mathbf{x}. $$

All of the terms here are symmetric except for the $2\mathbf{S}\mathbf{A}$, but since $\mathbf{x}^T\mathbf{S}\mathbf{A}\mathbf{x} = \mathbf{x}^T\mathbf{A}^T\mathbf{S}\mathbf{x}$, we can write
$$ 0 = \mathbf{x}^T \left[ \mathbf{Q} - \mathbf{S}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{S} + \mathbf{S}\mathbf{A} + \mathbf{A}^T\mathbf{S} \right]\mathbf{x}. $$

and since this condition must hold for all $\mathbf{x}$, it is sufficient to consider the matrix equation
$$ 0 = \mathbf{S}\mathbf{A} + \mathbf{A}^T\mathbf{S} - \mathbf{S}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^T\mathbf{S} + \mathbf{Q}. $$

This euqation is called the **algebraic Riccati equation**. Solving it for $S$ gives us the optimal value function, and from that we can derive the optimal policy.

## 8.1.1 Local stabilization of nonlinear systems

LQR is extremely relevant to us even though our primary interest is in nonlinear dynamics, because it can provide a local approximation of the optimal control solution for the nonlinear system. Given the nonlinear system $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$, and a stabilizable operating point, $(\mathbf{x}_0, \mathbf{u}_0)$, with $f(\mathbf{x}_0, \mathbf{u}_0) = 0$. We can define a relative coordinate system

$$ \bar{\mathbf{x}} = \mathbf{x} - \mathbf{x}_0, \quad \bar{\mathbf{u}} = \mathbf{u} - \mathbf{u}_0, $$

and observe that

$$ \dot{\bar{\mathbf{x}}} = \dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u}), $$

which we can approximate with a first-order Taylor expansion to

$$ \dot{\bar{\mathbf{x}}} \approx f(\mathbf{x}_0, \mathbf{u}_0) + \frac{\partial f(\mathbf{x}_0, \mathbf{u}_0)}{\partial \mathbf{x}} (\mathbf{x} - \mathbf{x}_0) + \frac{\partial f(\mathbf{x}_0, \mathbf{u}_0)}{\partial \mathbf{u}} (\mathbf{u} - \mathbf{u}_0) = \mathbf{A}\bar{\mathbf{x}} + \mathbf{B}\bar{\mathbf{u}}. $$

Similarly, we can define a quadratic cost function in the error coordinates, or take a (positive-definite) second-order approximation of a nonlinear cost function about the operating point (linear and constant terms in the cost function can be easily incorporated into the derivation by parameterizing a full quadratic form for $J^*$, as seen in the Linear Quadratic Tracking derivation below).

The resulting controller takes the form $\bar{\mathbf{u}}^* = -\mathbf{K}\bar{\mathbf{x}}$ or

$$ \mathbf{u}^* = \mathbf{u}_0 - \mathbf{K}(\mathbf{x} - \mathbf{x}_0). $$


# 8.2 Finite-horizon formulations
The cost-to-go fr infinite-horizon problems is time-invariant, but for finite-horizon problems, the cost-to-go is a function of both the state and time.Therefore, the HJB suffficient condition requires an additional term for $\frac{\partial J*}{\partial t}$:
$$ 0 = \min_u \left[ \ell(x,u) + \frac{\partial J*}{\partial x}f(x,u) + \frac{\partial J*}{\partial t} \right] \forall x , \forall t \in [t_0,t_f]$$

Base on these assumptions:
- The system dynamics are linear time-invariant: $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$.
- The finite-horizon cost function: $J = h\!\left(x(t_f)\right) + \int_{0}^{t_f} \ell\!\left(x(t),\,u(t)\right)\,dt$, with:
    - $h(\mathbf{x}) = \mathbf{x}^{T}\mathbf{Q}_{f}\mathbf{x}, \qquad \mathbf{Q}_{f} = \mathbf{Q}_{f}^{T} \succeq 0$ is the terminal cost, and
    - $\ell(\mathbf{x},\mathbf{u}) = \mathbf{x}^{T}\mathbf{Q}\mathbf{x} + \mathbf{u}^{T}\mathbf{R}\mathbf{u}, \qquad \mathbf{Q}=\mathbf{Q}^{T}\succeq 0, \quad \mathbf{R}=\mathbf{R}^{T}\succ 0$ is the running cost from $t_0$ to terminal time $t_f$.

We have:
- $S(t)$ must satisfy:
    - $-\dot{\mathbf{S}}(t) = \mathbf{S}(t)\mathbf{A} + \mathbf{A}^{T}\mathbf{S}(t) - \mathbf{S}(t)\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^{T}\mathbf{S}(t) + \mathbf{Q}$
    - and $\mathbf{S}(t_f) = \mathbf{Q}_f$
- $\mathbf{u}^{*} = \pi^{*}(\mathbf{x},t) = -\mathbf{R}^{-1}\mathbf{B}^{T}\mathbf{S}(t)\mathbf{x}$

## 8.2.2 Time-varying LQR
The derivation above holds even if the dynamics are given by

$\dot{\mathbf{x}} = \mathbf{A}(t)\mathbf{x} + \mathbf{B}(t)\mathbf{u}$.

Similarly, the cost functions $\mathbf{Q}$ and $\mathbf{R}$ can also be time-varying. This is quite surprising, as the class of time-varying linear systems is a quite general class of systems. It requires essentially no assumptions on how the time-dependence enters, except perhaps that if $\mathbf{A}$ or $\mathbf{B}$ is discontinuous in time then one would have to use the proper techniques to accurately integrate the differential equation.

## 8.2.3 Local trajectory stabilization for nonlinear systems

One of the most powerful applications of time-varying LQR involves linearizing around a nominal trajectory of a nonlinear system and using LQR to provide a trajectory controller. This will tie in very nicely with the algorithms we develop in the chapter on trajectory optimization.

Let us assume that we have a nominal trajectory, $\mathbf{x}_0(t), \mathbf{u}_0(t)$ defined for $t \in [t_1, t_2]$. Similar to the time-invariant analysis, we begin by defining a local coordinate system relative to the trajectory:

$\bar{\mathbf{x}}(t) = \mathbf{x}(t) - \mathbf{x}_0(t), \qquad \bar{\mathbf{u}}(t) = \mathbf{u}(t) - \mathbf{u}_0(t).$

Now we have

$\dot{\bar{\mathbf{x}}} = \dot{\mathbf{x}} - \dot{\mathbf{x}}_0 = f(\mathbf{x}, \mathbf{u}) - f(\mathbf{x}_0, \mathbf{u}_0),$

which we can again approximate with a first-order Taylor expansion to

$\dot{\bar{\mathbf{x}}} \approx f(\mathbf{x}_0,\mathbf{u}_0) + \frac{\partial f(\mathbf{x}_0,\mathbf{u}_0)}{\partial \mathbf{x}}(\mathbf{x}-\mathbf{x}_0) + \frac{\partial f(\mathbf{x}_0,\mathbf{u}_0)}{\partial \mathbf{u}}(\mathbf{u}-\mathbf{u}_0) - f(\mathbf{x}_0,\mathbf{u}_0) = \mathbf{A}(t)\bar{\mathbf{x}} + \mathbf{B}(t)\bar{\mathbf{u}}.$

This is very similar to using LQR to stabilize a fixed-point, but with some important differences. First, the linearization is time-varying. Second, our linearization is valid for any state along a feasible trajectory (not just fixed-points), because the coordinate system is moving along with the trajectory.

Similarly, we can define a quadratic cost function in the error coordinates, or take a (positive-definite) second-order approximation of a nonlinear cost function along the trajectory (linear and constant terms in the cost function can be easily incorporated into the derivation by parameterizing a full quadratic form for $J^*$, as seen in the Linear Quadratic Tracking derivation below).

The resulting controller takes the form

$\bar{\mathbf{u}}^{*} = -\mathbf{K}(t)\bar{\mathbf{x}}$

or

$\mathbf{u}^{*} = \mathbf{u}_0(t) - \mathbf{K}(t)\bigl(\mathbf{x} - \mathbf{x}_0(t)\bigr).$

Note that in section 8.2.3, we discuss in case around a nominal trajectory $x_0(t), u_0(t)$ that satisfy: 
$$\dot{x}_0(t) = f(x_0(t), u_0(t))$$

## 8.2.4 Linear Quadratic Optimal Tracking
Note that In 8.2.3, we solve for the non-linear system but it requires the trajectory to be feasible, its mean that the trajectory must satisfy the system dynamics $\dot{x}_0(t) = f(x_0(t), u_0(t))$

It's a slightly more general form of the linear quadratic regulator, the trajectory can be arbitrary. But still a linear system.

Consider the problem:

- $\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$

- $h(\mathbf{x}) = \bigl(\mathbf{x} - \mathbf{x}_d(t_f)\bigr)^T \mathbf{Q}_f \bigl(\mathbf{x} - \mathbf{x}_d(t_f)\bigr), \qquad \mathbf{Q}_f = \mathbf{Q}_f^T \succeq 0$

- $\ell(\mathbf{x},\mathbf{u},t) = \bigl(\mathbf{x} - \mathbf{x}_d(t)\bigr)^T \mathbf{Q} \bigl(\mathbf{x} - \mathbf{x}_d(t)\bigr) + \bigl(\mathbf{u} - \mathbf{u}_d(t)\bigr)^T \mathbf{R} \bigl(\mathbf{u} - \mathbf{u}_d(t)\bigr)$

- $\mathbf{Q} = \mathbf{Q}^T \succeq 0, \qquad \mathbf{R} = \mathbf{R}^T \succ 0$

Solve this problem we obtain:

$-\dot{\mathbf{S}}_{xx}(t) = \mathbf{Q} - \mathbf{S}_{xx}(t)\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^{T}\mathbf{S}_{xx}(t) + \mathbf{S}_{xx}(t)\mathbf{A} + \mathbf{A}^{T}\mathbf{S}_{xx}(t)$

$-\dot{\mathbf{s}}_{x}(t) = -\mathbf{Q}\mathbf{x}_{d}(t) + \left[\mathbf{A}^{T} - \mathbf{S}_{xx}(t)\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^{T}\right]\mathbf{s}_{x}(t) + \mathbf{S}_{xx}(t)\mathbf{B}\mathbf{u}_{d}(t)$

$-\dot{s}_{0}(t) = \mathbf{x}_{d}(t)^{T}\mathbf{Q}\mathbf{x}_{d}(t) - \mathbf{s}_{x}(t)^{T}\mathbf{B}\mathbf{R}^{-1}\mathbf{B}^{T}\mathbf{s}_{x}(t) + 2\,\mathbf{s}_{x}(t)^{T}\mathbf{B}\mathbf{u}_{d}(t)$

$\mathbf{S}_{xx}(t_f) = \mathbf{Q}_{f}$

$\mathbf{s}_{x}(t_f) = -\mathbf{Q}_{f}\mathbf{x}_{d}(t_f)$

$s_{0}(t_f) = \mathbf{x}_{d}(t_f)^{T}\mathbf{Q}_{f}\mathbf{x}_{d}(t_f)$


## 8.3.1 Discrete-time Riccati Equations

In simulation or digital control implementations, we often work with discrete-time systems. 

Consider the discrete time dynamics:

$\mathbf{x}[n+1] = \mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n],$

and we wish to minimize

$\min \sum_{n=0}^{N-1} \mathbf{x}^{T}[n]\mathbf{Q}\mathbf{x}[n] + \mathbf{u}^{T}[n]\mathbf{R}\mathbf{u}[n], \qquad \mathbf{Q}=\mathbf{Q}^{T}\succeq 0,\ \mathbf{R}=\mathbf{R}^{T}\succ 0.$

The cost-to-go is given by

$J(\mathbf{x},n-1) = \min_{\mathbf{u}} \mathbf{x}^{T}\mathbf{Q}\mathbf{x} + \mathbf{u}^{T}\mathbf{R}\mathbf{u} + J(\mathbf{A}\mathbf{x}+\mathbf{B}\mathbf{u},n).$

If we once again take

$J(\mathbf{x},n) = \mathbf{x}^{T}\mathbf{S}[n]\mathbf{x}, \qquad \mathbf{S}[n]=\mathbf{S}^{T}[n]\succ 0,$

then we have

$\mathbf{u}^{*}[n] = -\mathbf{K}[n]\mathbf{x}[n] = -\left(\mathbf{R}+\mathbf{B}^{T}\mathbf{S}[n]\mathbf{B}\right)^{-1}\mathbf{B}^{T}\mathbf{S}[n]\mathbf{A}\mathbf{x}[n],$

yielding

$\mathbf{S}[n-1] = \mathbf{Q} + \mathbf{A}^{T}\mathbf{S}[n]\mathbf{A} - \left(\mathbf{A}^{T}\mathbf{S}[n]\mathbf{B}\right)\left(\mathbf{R}+\mathbf{B}^{T}\mathbf{S}[n]\mathbf{B}\right)^{-1}\left(\mathbf{B}^{T}\mathbf{S}[n]\mathbf{A}\right), \qquad \mathbf{S}[N]=0,$

which is the famous *Riccati difference equation*. The infinite-horizon LQR solution is given by the (positive-definite) fixed-point of this equation:

$\mathbf{S} = \mathbf{Q} + \mathbf{A}^{T}\mathbf{S}\mathbf{A} - \left(\mathbf{A}^{T}\mathbf{S}\mathbf{B}\right)\left(\mathbf{R}+\mathbf{B}^{T}\mathbf{S}\mathbf{B}\right)^{-1}\left(\mathbf{B}^{T}\mathbf{S}\mathbf{A}\right).$


In reinforcement learning, it is popular to consider the infinite-horizon "discounted" cost:

$\min \sum_{n=0}^{\infty} \gamma^{n}\left(\mathbf{x}^{T}[n]\mathbf{Q}\mathbf{x}[n] + \mathbf{u}^{T}[n]\mathbf{R}\mathbf{u}[n]\right).$

The optimal controller is

$\mathbf{u}^{*} = -\gamma\left(\mathbf{R} + \gamma\mathbf{B}^{T}\mathbf{S}\mathbf{B}\right)^{-1}\mathbf{B}^{T}\mathbf{S}\mathbf{A}\mathbf{x}.$

and the corresponding Riccati equation is

$\mathbf{S} = \mathbf{Q} + \gamma\mathbf{A}^{T}\mathbf{S}\mathbf{A} - \gamma^{2}\left(\mathbf{A}^{T}\mathbf{S}\mathbf{B}\right)\left(\mathbf{R} + \gamma\mathbf{B}^{T}\mathbf{S}\mathbf{B}\right)^{-1}\left(\mathbf{B}^{T}\mathbf{S}\mathbf{A}\right).$


## 8.3.3 LQR on a Manifold

**Main idea:**
Standard LQR assumes the state lives in a flat vector space:

$$
x \in \mathbb{R}^n
$$

But many robot systems do **not** really move freely in all directions of (\mathbb{R}^n). They may have geometric or kinematic constraints, for example closed-chain mechanisms, rolling constraints, contact constraints, angles, or quaternion states.

So the key message is:

> Do not blindly apply LQR in the full coordinate space if the system actually lives on a constrained manifold.

A simple linear constraint example is:

$$
Fx = 0
$$

This means only states satisfying the constraint are valid. The correct move is to find a reduced coordinate (y) that only describes valid motions, then run LQR in that reduced coordinate.

If (P) is a basis for the valid subspace, then roughly:

$$
x = P^T y
$$

and the projected dynamics become:

$$
\dot{y} = A_y y + B_y u
$$

Then we solve LQR on $y$, not directly on $x$.

**Why it matters in robotics:**
Sometimes LQR “fails” not because LQR is bad, but because we chose the wrong coordinates. If the controller tries to stabilize directions that are physically impossible due to constraints, the controllability test or LQR design can look broken.

**Takeaway sentence:**

> LQR should be designed on the true local coordinates of the system — usually a tangent space or reduced coordinate system — not blindly on all raw state variables.

# Chapter 10: Trajectory Optimization

## 10.1 Problem Formulation
Given a state-space model,

$$\dot{\mathbf{x}} = f(\mathbf{x},\mathbf{u}),$$

an initial condition,

$$\mathbf{x}_0,$$

and an input trajectory

$$\mathbf{u}(t)$$

defined over a finite interval,

$$t \in [t_0,t_f],$$

we can simulate the dynamics forward to obtain $\mathbf{x}(t)$ over the same interval. We will define the long-term (finite-horizon) cost of executing that trajectory using the standard additive-cost optimal control objective,

$$J_{\mathbf{u}(\cdot)}(\mathbf{x}_0) = \ell_f(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} \ell(\mathbf{x}(t),\mathbf{u}(t))\,dt.$$

$\ell_f$ is a *terminal cost*, evaluated at the end of the trajectory, which we did not have in the infinite-horizon formulations we've focused on so far. We will write the trajectory optimization problem as

$$ \min_{\mathbf{u}(\cdot)} \ \ell_f(\mathbf{x}(t_f)) + \int_{t_0}^{t_f} \ell(\mathbf{x}(t),\mathbf{u}(t))\,dt $$

subject to

$$\dot{\mathbf{x}}(t) = f(\mathbf{x}(t),\mathbf{u}(t)), \qquad \forall t \in [t_0,t_f]$$

$$\mathbf{x}(t_0) = \mathbf{x}_0.$$

Many trajectory optimization problems will also include additional constraints, such as collision avoidance (e.g., where the constraint is that the signed distance between the robot's geometry and the obstacles stays positive) or input limits (e.g., $u_{min} \leq u \leq u_{max}$). Constraints can be defined for all time or some subset of the trajectory.


### Direct transcription
For time-discrete model, we can write the problem as:

$$
\min_{\mathbf{x}[\cdot],\,\mathbf{u}[\cdot]} \ \ell_f(\mathbf{x}[N]) + \sum_{n=0}^{N-1} \ell(\mathbf{x}[n],\mathbf{u}[n])
$$

subject to

$$
\mathbf{x}[n+1] = \mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n], \qquad \forall n \in [0, N-1]
$$

$$
\mathbf{x}[0] = \mathbf{x}_0
$$

$$
+\ \text{additional constraints}
$$


### Direct shooting
From 
$$
\mathbf{x}[n+1] = \mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n], \qquad \forall n \in [0, N-1]
$$
we can easily get:
$$
\mathbf{x}[n] = \mathbf{A}^{n}\mathbf{x}[0] + \sum_{k=0}^{n-1}\mathbf{A}^{\,n-1-k}\mathbf{B}\mathbf{u}[k].
$$

So, by substituting $\mathbf{x}[n]$ into the cost function, we can get rid of a bunch of decision variables, and turn a constrained optimization problem into an unconstrained optimization problem (assuming we don't have any other constraints).
This appoarch called direct shooting

### 10.2.4 Continuous Time
The dynamics equation of continuous time system is given by:
$$\dot{\mathbf{x}} = f(\mathbf{x},\mathbf{u}).$$
To solve continuous time problem with numerical optimizer, we can discretize the dynamics with a small time step $h$:
$$\mathbf{x}[n+1] = \mathbf{x}[n] + \int_{t_n}^{t_n + h} f(\mathbf{x}(t),\mathbf{u}(t)) \, dt$$


## 10.3 Noncovex trajectory optimization
### 10.3.2. Direct collocation

- Both the input trajectory and the state trajectory are represented explicitly as piecewise-polynomial functions. In particular, the sweet spot for this algorithm is taking $ u(t) $ to be a first-order polynomial and $ x(t) $ to be a cubic polynomial.
- Decision variables are sample values $ u(t) $ and $ x(t) $ at the so called "break" points of the spline.
- The spline between any 2 break points are determined because we knew $x(t_k)$, $x(t_{k+1})$, $\dot{x}(t_k)$ and $\dot{x}(t_{k+1})$.
- We choose collocation points between any 2 break points at the center of the interval, then we have:
$$
t_{c,k} = \frac{1}{2}\left(t_k + t_{k+1}\right), 
\qquad 
h_k = t_{k+1} - t_k,
$$

$$
\mathbf{u}(t_{c,k}) = \frac{1}{2}\left(\mathbf{u}(t_k) + \mathbf{u}(t_{k+1})\right),
$$

$$
\mathbf{x}(t_{c,k}) =
\frac{1}{2}\left(\mathbf{x}(t_k) + \mathbf{x}(t_{k+1})\right)
+ \frac{h}{8}\left(\dot{\mathbf{x}}(t_k) - \dot{\mathbf{x}}(t_{k+1})\right),
$$

$$
\dot{\mathbf{x}}(t_{c,k}) =
-\frac{3}{2h}\left(\mathbf{x}(t_k) - \mathbf{x}(t_{k+1})\right)
-\frac{1}{4}\left(\dot{\mathbf{x}}(t_k) + \dot{\mathbf{x}}(t_{k+1})\right).
$$

 These equations come directly from the equations that fit the cubic spline to the end points/derivatives then interpolate them at the midpoint. They give us precisely what we need to add the dynamics constraint to our optimization at the collocation times:

$$
\min_{\mathbf{x}[\cdot], \mathbf{u}[\cdot]}
\ell_f(\mathbf{x}[N]) + \sum_{n=0}^{N-1} h_n \ell(\mathbf{x}[n], \mathbf{u}[n])
$$

subject to

$$
\dot{\mathbf{x}}(t_{c,n}) = f(\mathbf{x}(t_{c,n}), \mathbf{u}(t_{c,n})),
\qquad \forall n \in [0, N-1]
$$

$$
\mathbf{x}[0] = \mathbf{x}_0
$$

$$
+ \text{additional constraints.}
$$


### **note**
Until now, we have discussed about 3 method: direct transcription, direct shooting and direct collocation. In general, all of these methods tried to replace the condition:
$$
\dot{\mathbf{x}}(t) = f(\mathbf{x}(t),\mathbf{u}(t)) \quad \forall t
$$
with a discrete approximation.

## 10.5 Local Trajectory feedback design 
Once we have obtained a locally optimal trajectory from trajectory optimization, we have found an open-loop trajectory that (at least locally) minimizes our optimal control cost. Up to numerical tolerances, this pair $\mathbf{u}_0(t), \mathbf{x}_0(t)$ represents a feasible solution trajectory of the system. But we haven't done anything, yet, to ensure that this trajectory is locally stable.

### 10.5.1 Finite-horizon LQR
We can use thís method to make the trajectory locally stable. (see 8.1.1)

### 10.5.2 Model-Predictive Control
The maturity, robustness, and speed of solving trajectory optimization using convex optimization leads to a beautiful idea: if we can optimize trajectories quickly enough, then we can use our trajectory optimization as a feedback policy. 
The recipe is simple: 
- (1) measure the current state, 
- (2) optimize a trajectory from the current state, 
- (3) execute the first action from the optimized trajectory, 
- (4) let the dynamics evolve for one step and repeat. 

This recipe is known as model-predictive control (MPC). 

#### Receding-horizon MPC
One core idea is the concept of *receding-horizon* MPC. Since our trajectory optimization problems are formulated over a finite-horizon, we can think each optimization as reasoning about the next $N$ time steps. If our true objective ==is to optimize== the performance over a horizon longer than $N$ (e.g. over the infinite horizon), then it is standard to continue solving for an $N$ step horizon on each evaluation of the controller. In this sense, the total horizon under consideration continues to move forward in time (e.g. to recede :trượt).

### 10.8.2 Iterative LQR and Differential Dynamic Programming

#### 10.8.2.1 Iterative LQR for general nonlinear systems

iLQR solves a finite-horizon nonlinear optimal control problem by repeatedly building a local time-varying LQR approximation around the current feasible trajectory.

Consider the discrete-time nonlinear system

$$
x_{t+1} = f_t(x_t,u_t),
\qquad
x_t \in \mathbb{R}^{n},
\qquad
u_t \in \mathbb{R}^{m}.
$$

The objective is

$$
\min_{u_0,\dots,u_{N-1}}
J
=
\ell_f(x_N)
+
\sum_{t=0}^{N-1}\ell_t(x_t,u_t)
$$

subject to

$$
x_{t+1}=f_t(x_t,u_t),
\qquad
x_0 \text{ given}.
$$

Here $f_t$, $\ell_t$, and $\ell_f$ are general. They do not have to come from CartPole. If the original system is continuous-time,

$$
\dot{x}=g(x,u,t),
$$

then $f_t$ is the discrete dynamics obtained by an integration scheme, for example Euler or RK4.

### Nominal trajectory

iLQR starts from an initial guess for the control trajectory:

$$
\bar{u}_{0:N-1}.
$$

Rolling out the dynamics gives a feasible nominal trajectory:

$$
\bar{x}_{0:N},
\qquad
\bar{x}_{t+1}=f_t(\bar{x}_t,\bar{u}_t).
$$

At each iteration, iLQR searches for corrections around this nominal trajectory:

$$
x_t=\bar{x}_t+\delta x_t,
\qquad
u_t=\bar{u}_t+\delta u_t.
$$

For states on a manifold, replace the simple subtraction $x-\bar{x}$ by the appropriate local error coordinate, for example a tangent-space error.

### Local linear dynamics

Around each nominal point $(\bar{x}_t,\bar{u}_t)$, linearize the dynamics:

$$
f_t(\bar{x}_t+\delta x_t,\bar{u}_t+\delta u_t)
\approx
f_t(\bar{x}_t,\bar{u}_t)
+
A_t\delta x_t
+
B_t\delta u_t,
$$

where

$$
A_t =
\left.\frac{\partial f_t}{\partial x}\right|_{\bar{x}_t,\bar{u}_t},
\qquad
B_t =
\left.\frac{\partial f_t}{\partial u}\right|_{\bar{x}_t,\bar{u}_t}.
$$

Because the nominal trajectory is feasible,

$$
\bar{x}_{t+1}=f_t(\bar{x}_t,\bar{u}_t),
$$

so the local error dynamics are

$$
\delta x_{t+1}
\approx
A_t\delta x_t+B_t\delta u_t.
$$

iLQR keeps only first-order dynamics. If we also kept second-order derivatives of the dynamics, the method would become closer to full Differential Dynamic Programming (DDP).

### Local quadratic cost

Approximate the running cost to second order:

$$
\ell_t(x_t,u_t)
\approx
\ell_t
+
\ell_x^T\delta x_t
+
\ell_u^T\delta u_t
+
\frac{1}{2}\delta x_t^T\ell_{xx}\delta x_t
+
\delta u_t^T\ell_{ux}\delta x_t
+
\frac{1}{2}\delta u_t^T\ell_{uu}\delta u_t.
$$

All derivatives are evaluated at $(\bar{x}_t,\bar{u}_t)$:

$$
\ell_x =
\left.\frac{\partial \ell_t}{\partial x}\right|_{\bar{x}_t,\bar{u}_t},
\qquad
\ell_u =
\left.\frac{\partial \ell_t}{\partial u}\right|_{\bar{x}_t,\bar{u}_t},
$$

$$
\ell_{xx} =
\left.\frac{\partial^2 \ell_t}{\partial x^2}\right|_{\bar{x}_t,\bar{u}_t},
\qquad
\ell_{uu} =
\left.\frac{\partial^2 \ell_t}{\partial u^2}\right|_{\bar{x}_t,\bar{u}_t},
\qquad
\ell_{ux} =
\left.\frac{\partial^2 \ell_t}{\partial u\,\partial x}\right|_{\bar{x}_t,\bar{u}_t}.
$$

At the terminal state:

$$
\ell_f(x_N)
\approx
\ell_f(\bar{x}_N)
+
\ell_{f,x}^T\delta x_N
+
\frac{1}{2}\delta x_N^T\ell_{f,xx}\delta x_N.
$$

The derivatives can be computed analytically, by automatic differentiation, or by finite differences.

### Local Q-function

During the backward pass, assume the value function at the next time step has the local quadratic approximation

$$
V_{t+1}(\bar{x}_{t+1}+\delta x_{t+1})
\approx
V_0'
+
{V_x'}^T\delta x_{t+1}
+
\frac{1}{2}\delta x_{t+1}^T V_{xx}'\delta x_{t+1}.
$$

The prime means "at time $t+1$", not transpose.

Define the local one-step Q-function:

$$
Q_t(\delta x_t,\delta u_t)
=
\ell_t(\delta x_t,\delta u_t)
+
V_{t+1}(A_t\delta x_t+B_t\delta u_t).
$$

Collecting first- and second-order terms gives

$$
Q_x=\ell_x+A_t^TV_x',
$$

$$
Q_u=\ell_u+B_t^TV_x',
$$

$$
Q_{xx}=\ell_{xx}+A_t^TV_{xx}'A_t,
$$

$$
Q_{ux}=\ell_{ux}+B_t^TV_{xx}'A_t,
$$

$$
Q_{uu}=\ell_{uu}+B_t^TV_{xx}'B_t.
$$

These equations are the Riccati-like core of iLQR. The nonlinear problem is locally replaced by a time-varying LQR problem.

### Local optimal control law

The quadratic model has the form

$$
Q(\delta x_t,\delta u_t)
\approx
Q_0
+
Q_x^T\delta x_t
+
Q_u^T\delta u_t
+
\frac{1}{2}
\begin{bmatrix}
\delta x_t \\
\delta u_t
\end{bmatrix}^T
\begin{bmatrix}
Q_{xx} & Q_{ux}^T \\
Q_{ux} & Q_{uu}
\end{bmatrix}
\begin{bmatrix}
\delta x_t \\
\delta u_t
\end{bmatrix}.
$$

For fixed $\delta x_t$, minimize with respect to $\delta u_t$:

$$
0 =
\frac{\partial Q}{\partial \delta u_t}
=
Q_u+Q_{uu}\delta u_t+Q_{ux}\delta x_t.
$$

If $Q_{uu}$ is positive definite, the local optimal control correction is

$$
\delta u_t^*
=
k_t+K_t\delta x_t,
$$

where

$$
k_t=-Q_{uu}^{-1}Q_u,
\qquad
K_t=-Q_{uu}^{-1}Q_{ux}.
$$

In implementation, do not compute $Q_{uu}^{-1}$ explicitly. Solve the linear systems instead.

Often we regularize

$$
Q_{uu}^{reg}
=
Q_{uu}+\lambda I
$$

and use $Q_{uu}^{reg}$ in the solve. This makes the backward pass more stable when the current nominal trajectory is poor.

### Value function update

After substituting

$$
\delta u_t=k_t+K_t\delta x_t
$$

back into the local Q-function, the value function derivatives at time $t$ become

$$
V_x
=
Q_x
+
K_t^TQ_u
+
Q_{ux}^Tk_t
+
K_t^TQ_{uu}k_t,
$$

$$
V_{xx}
=
Q_{xx}
+
K_t^TQ_{uu}K_t
+
K_t^TQ_{ux}
+
Q_{ux}^TK_t.
$$

Then $(V_x,V_{xx})$ is passed backward to the previous time step. In code, it is common to symmetrize $V_{xx}$ after each update:

$$
V_{xx}\leftarrow \frac{1}{2}(V_{xx}+V_{xx}^T).
$$

### Backward pass

Initialize at the final time:

$$
V_x[N]=\ell_{f,x}(\bar{x}_N),
\qquad
V_{xx}[N]=\ell_{f,xx}(\bar{x}_N).
$$

Then for

$$
t=N-1,N-2,\dots,0,
$$

do:

1. Compute $A_t,B_t$ from the dynamics derivatives.
2. Compute $\ell_x,\ell_u,\ell_{xx},\ell_{ux},\ell_{uu}$ from the cost derivatives.
3. Compute $Q_x,Q_u,Q_{xx},Q_{ux},Q_{uu}$.
4. Regularize $Q_{uu}$ if needed.
5. Compute $k_t$ and $K_t$.
6. Update $V_x,V_{xx}$.

The backward pass returns the feedforward corrections $k_{0:N-1}$ and feedback gains $K_{0:N-1}$.

### Forward pass with line search

The backward pass only solves the local quadratic approximation. To update the true nonlinear trajectory, roll out the real dynamics again.

For a step size $\alpha\in(0,1]$:

$$
x_0^{new}=x_0,
$$

and for $t=0,\dots,N-1$:

$$
u_t^{new}
=
\bar{u}_t
+
\alpha k_t
+
K_t(x_t^{new}-\bar{x}_t),
$$

$$
x_{t+1}^{new}
=
f_t(x_t^{new},u_t^{new}).
$$

The term $\alpha k_t$ changes the open-loop nominal control, while $K_t(x_t^{new}-\bar{x}_t)$ keeps the rollout close to the region where the local model is valid.

Try a list of step sizes such as

$$
\alpha \in \{1,\ 0.5,\ 0.25,\ 0.1,\ 0.05,\ 0.01\}.
$$

Accept the first rollout that reduces the true cost:

$$
J(x^{new},u^{new}) < J(\bar{x},\bar{u}).
$$

If the problem has input limits or other simple constraints, this is where one may project or clip $u_t^{new}$ back into the feasible set. That projection is problem-specific and is not part of unconstrained iLQR itself.

### Full iLQR algorithm

```text
Given:
    dynamics f_t(x, u)
    running costs ell_t(x, u)
    terminal cost ell_f(x)
    initial state x0
    initial control guess u[0:N-1]

Roll out x[0:N] using f_t and u[0:N-1]
J = total_cost(x, u)
lambda = lambda_init

for iteration = 0, 1, 2, ...:

    # Backward pass
    V_x  = derivative of ell_f at x[N]
    V_xx = Hessian of ell_f at x[N]

    for t = N-1, ..., 0:
        A = df_t/dx at (x[t], u[t])
        B = df_t/du at (x[t], u[t])

        compute ell_x, ell_u, ell_xx, ell_ux, ell_uu

        Q_x  = ell_x  + A.T V_x
        Q_u  = ell_u  + B.T V_x
        Q_xx = ell_xx + A.T V_xx A
        Q_ux = ell_ux + B.T V_xx A
        Q_uu = ell_uu + B.T V_xx B

        Q_uu_reg = Q_uu + lambda I

        if Q_uu_reg is not positive definite:
            increase lambda
            restart the backward pass

        k[t] = - solve(Q_uu_reg, Q_u)
        K[t] = - solve(Q_uu_reg, Q_ux)

        V_x  = Q_x + K[t].T Q_u + Q_ux.T k[t] + K[t].T Q_uu_reg k[t]
        V_xx = Q_xx + K[t].T Q_uu_reg K[t] + K[t].T Q_ux + Q_ux.T K[t]
        V_xx = 0.5 * (V_xx + V_xx.T)

    # Forward pass
    accepted = false
    for alpha in [1, 0.5, 0.25, 0.1, 0.05, 0.01]:
        x_new[0] = x0

        for t = 0, ..., N-1:
            dx = x_new[t] - x[t]
            u_new[t] = u[t] + alpha * k[t] + K[t] dx
            x_new[t+1] = f_t(x_new[t], u_new[t])

        J_new = total_cost(x_new, u_new)

        if J_new < J:
            accept x_new, u_new
            decrease lambda
            accepted = true
            break

    if not accepted:
        increase lambda
        continue

    if J - J_new < tolerance:
        stop

    x = x_new
    u = u_new
    J = J_new

Return:
    optimized trajectory x[0:N]
    optimized controls u[0:N-1]
    feedback gains K[0:N-1]
```

### Practical notes

- iLQR is a local method, so the initial control guess and horizon length matter.
- $Q_{uu}$ must be positive definite enough to solve for $k_t$ and $K_t$ safely.
- Increasing $\lambda$ makes the step more conservative; decreasing $\lambda$ makes the method closer to Newton's method.
- The feedback gains $K_t$ can also be used after optimization to track the nominal trajectory:

$$
u_t
=
u_t^\star
+
K_t(x_t-x_t^\star).
$$

- For non-Euclidean states, replace $x_t-x_t^\star$ by the correct local state error.

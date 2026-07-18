# Lyapunov Stability

Three notions for stability of a fixed point $x^*$ of a nonlinear system:

- Stability in the sense of Lyapunov (i.s.L.)

    If the system starts sufficiently close to $x^*$, then it stays close to $x^*$ for all future time.
    This only means “small perturbations remain small”; it does **not** require convergence back to $x^*$.

- Asymptotic stability

    The fixed point is stable i.s.L., and trajectories starting nearby also converge back to it:

    $$
    \lim_{t\to\infty} x(t)=x^*.
    $$

    So asymptotic stability means both “stay close” and “eventually return.”

- Exponential stability

    The fixed point is asymptotically stable, and convergence happens at least as fast as an exponential decay:

    $$
    |x(t)-x^*| \le C e^{-\alpha t}|x(0)-x^*|.
    $$

    This is stronger than asymptotic stability because it gives a guaranteed convergence rate.

### Theorem 9.1 - Lyapunov's Direct Method (for local stability)

- *!!!!!!!!!! Need to prove* 

    Given a system $\dot{\mathbf{x}} = f(\mathbf{x})$, with $f$ continuous, and for some region $\mathcal{D}$ around the origin (specifically an open subset of $\mathbb{R}^n$ containing the origin), if I can produce a scalar, continuously-differentiable function $V(\mathbf{x})$, such that

    $$
    V(\mathbf{x}) > 0, \forall \mathbf{x} \in \mathcal{D}\setminus\{0\}
    \qquad
    V(0)=0,
    $$

    and

    $$
    \dot{V}(\mathbf{x})
    =
    \frac{\partial V}{\partial \mathbf{x}}f(\mathbf{x})
    \le 0,
    \forall \mathbf{x}\in\mathcal{D}\setminus\{0\}
    \qquad
    \dot{V}(0)=0,
    $$

    then the origin ($\mathbf{x}=0$) is stable in the sense of Lyapunov (i.s.L.). [Note: the notation $A\setminus B$ represents the set $A$ with the elements of $B$ removed.]

    If, additionally, we have

    $$
    \dot{V}(\mathbf{x})
    =
    \frac{\partial V}{\partial \mathbf{x}}f(\mathbf{x})
    < 0,
    \forall \mathbf{x}\in\mathcal{D}\setminus\{0\},
    $$

    then the origin is (locally) asymptotically stable. And if we have

    $$
    \dot{V}(\mathbf{x})
    =
    \frac{\partial V}{\partial \mathbf{x}}f(\mathbf{x})
    \le -\alpha V(\mathbf{x}),
    \forall \mathbf{x}\in\mathcal{D}\setminus\{0\},
    $$

    for some $\alpha > 0$, then the origin is (locally) exponentially stable.

### Theorem 9.2 - Lyapunov analysis for global stability

- Given a system $\dot{\mathbf{x}} = f(\mathbf{x})$, with $f$ continuous, if I can produce a scalar, continuously-differentiable function $V(\mathbf{x})$, such that

    $$
    V(\mathbf{x}) > 0,
    $$

    $$
    \dot{V}(\mathbf{x})
    =
    \frac{\partial V}{\partial \mathbf{x}}f(\mathbf{x})
    < 0,
    $$

    and

    $$
    V(\mathbf{x}) \to \infty
    \text{ whenever }
    \|\mathbf{x}\| \to \infty,
    $$

    then the origin ($\mathbf{x}=0$) is globally asymptotically stable (G.A.S.).

    If additionally we have that

    $$
    \dot{V}(\mathbf{x}) \le -\alpha V(\mathbf{x}),
    $$

    for some $\alpha > 0$, then the origin is globally exponentially stable.


### Theorem 9.3 LaSalle's Invariance Principle
Given a system $\dot{\mathbf{x}} = f(\mathbf{x})$ with $f$ continuous. If we can produce a scalar function $V(\mathbf{x})$ with continuous derivatives for which we have

$$
V(\mathbf{x}) > 0,
\qquad
\dot{V}(\mathbf{x}) \le 0,
$$

and $V(\mathbf{x}) \to \infty$ as $\|\mathbf{x}\| \to \infty$, then $\mathbf{x}$ will converge to the largest *invariant set* where $\dot{V}(\mathbf{x}) = 0.$

To be clear, an *invariant set*, $\mathcal{G}$, of the dynamical system is a set for which $\mathbf{x}(0) \in \mathcal{G}\Rightarrow\forall t > 0,\;\mathbf{x}(t) \in \mathcal{G}.$ In other words, once you enter the set you never leave. 


### Theorem 9.4 - Lyapunov invariant set and region of attraction theorem

Given a system $\dot{\mathbf{x}} = f(\mathbf{x})$ with $f$ continuous, if we can find a scalar function $V(\mathbf{x}) > 0$ and a bounded sublevel set

$$
\mathcal{G} : \{\mathbf{x} \mid V(\mathbf{x}) \le \rho\}
$$

on which

$$
\forall \mathbf{x} \in \mathcal{G},
\dot{V}(\mathbf{x}) \le 0,
$$

then $\mathcal{G}$ is an invariant set. By LaSalle, $\mathbf{x}$ will converge to the largest invariant subset of $\mathcal{G}$ on which $\dot{V} = 0$.

If $\dot{V}(\mathbf{x}) < 0$ in $\mathcal{G}$, then the origin is locally asymptotically stable and furthermore the set $\mathcal{G}$ is inside the region of attraction of this fixed point.

Alternatively, if $\dot{V}(\mathbf{x}) \le 0$ in $\mathcal{G}$ and $\mathbf{x} = 0$ is the only invariant subset of $\mathcal{G}$ where $\dot{V} = 0$, then the origin is asymptotically stable and the set $\mathcal{G}$ is inside the region of attraction of this fixed point.

## 9.1.6 Barrier functions

Lyapunov functions are used to certify stability or to establish invariance of a region. But the same conditions can be used to certify that the state of a dynamical system *will not visit* some region of state space. This can be useful for e.g. verifying that your autonomous car will not crash or, perhaps, that your walking robot will not fall down. When used this way, we call the associated functions, $\mathcal{B}(\mathbf{x})$, "barrier functions".

### Theorem 9.5 - Barrier functions

Given a continuously-differentiable dynamical system described by $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x})$, if I can find a function $\mathcal{B}(\mathbf{x})$ such that $\forall \mathbf{x}, \dot{\mathcal{B}}(\mathbf{x}) \le 0$, then the system will never visit states with $\mathcal{B}(\mathbf{x}(t)) > \mathcal{B}(\mathbf{x}(0))$.

In particular, if we ensure that $\forall \mathbf{x}$ in "failure" regions of state space, we have $\mathcal{B}(\mathbf{x}) > 0$, then the level-set $\mathcal{B}(\mathbf{x}) = 0$ can serve as a *barrier* certifying that trajectories starting with initial conditions with $\mathcal{B}(\mathbf{x}) < 0$ will never enter the failure set.

Natural extensions exist for certifying these conditions over a region, considering piecewise-polynomial dynamics, worst-case robustness, etc.

--- 

## 9.2 Lyapunov analysis with convex optimization
OK, until now, we have discussed how to use Lyapunov functions to certify stability of a fixed point. The difficulty is how to find suitable Lyapunov function candidates. In this section, we'll look at some computational approaches to verifying the Lyapunov conditions, and even to searching for (the coefficients of) the Lyapunov functions themselves.

9.2.1 In a linear system $\dot{x} = Ax$ we have the explicit formulation of $V(x)$ dạng $V(x) = x^T P x$, với $P$ is positive definite. We need to find $P$ that satisfies $P$ is positive definite và $A^T P + P A < 0$ ($\dot{V} < 0$). This is an LP and can be computed directly.

9.2.2 OK, the problem is for a nonlinear system, how can we check $V(x) > 0$ and $\dot{V}(x) < 0$ for all $x$.
- For polynomial system $\dot{x} = f(x)$, $f(x)$ is a polynomial.
To make sure $V(x)$ is positive definite, we put $V(x)$ in a form: $V(x) = m(x)^T P m(x)$, where $m(x)$ is a vector of monomials, $P$ is positive definite. 
We also need $-\dot{V}$ to be SOS, which means $-\dot{V} = m(x)^T Q m(x)$, $Q$ is positive definite. So our problem becomes: find $P$, $Q$ such that $P > 0$, $Q > 0$ and the coefficients of $-\dot{V}$ fit with $V$.

If we wish to demonstrate (global) asymptotic stability, then we can write the constraint $\dot{V}$ as $-\dot{V} - \epsilon x^T x$ is SOS where epsilon is a small positive constant that needs only to be larger than the numerical tolerances of the solver.

OK, the remaining work is delegated to the solver, we only need to understand that once the problem has been properly formulated, it can be solved.

9.2.3 Problem is: most real nonlinear/robot systems are not globally stable. So instead of proving global asymptotic stability for all $x \in \mathbb{R}^n$,
we only want to certify a region of attraction.

**The S-procedure**
If we can find a polynomial multiplier $\lambda(x)$ that:
$p(x) + \lambda^T(x) g(x)$ is SOS and $\lambda(x)$ is SOS, then this is sufficient to demonstrate that:
$$p(x) \geq 0 \text{ for all } x \text{ such that } g(x) \leq 0.$$

So to prove a region is attractive in a region:
Instead of proving $$\dot V(x) < 0 \quad \forall x \in \{x|V(x) \leq \rho\}$$

We prove that there exists $\lambda$ that is SOS and satisfies:
$$
-\dot V(x) + \lambda(x)(V(x)-\rho) \quad  \text{is SOS} $$

OK, we have another way to formulate this problem:

Under the assumption that the Hessian of $\dot{V}(\mathbf{x})$ is negative-definite at the origin (which is easily checked), we can write

$$
\max_{\rho,\lambda(\mathbf{x})} \rho
$$

subject to

$$
(\mathbf{x}^T\mathbf{x})^d(V(\mathbf{x})-\rho)
+
\lambda^T(\mathbf{x})\dot{V}(\mathbf{x})
\text{ is SOS},
$$

with $d$ a fixed positive integer. As you can see, $\rho$ no longer multiplies the coefficient of $\lambda(\mathbf{x})$. (check in Tedrake chapter 9.2.3)


### Searching for $V$
Earlier sections maximize the ROA estimate for a fixed Lyapunov candidate. But the ROA of a quadratic candidate may not cover the part of state space we care about. Therefore, in “Searching for $V(x)$,” Tedrake searches over more general polynomial Lyapunov functions to potentially certify a larger or better-shaped region.

(for more information check in Tedrake chapter 9.2.4)

### Convex outer approximations
Until now, we have certified a region of attraction by finding a sublevel set of a Lyapunov function that is contained within the true region of attraction:
$$
\hat{\mathcal{R}}_{\mathrm{inner}} \subseteq \mathcal{R}_{\mathrm{true}}
$$

Outer approximation is a region that encloses the true region of attraction from the outside.
$$
\mathcal{R}_{\mathrm{true}} \subseteq \hat{\mathcal{R}}_{\mathrm{outer}}
$$

We want to find the smallest outer approximation $\hat{\mathcal{R}}_{\mathrm{outer}}$. (The specific algorithm can be found in chapter 9 Tedrake)


## 9.3 Finite-time reachability
So far we have used Lyapunov analysis to analyze stability, which is fundamentally a description of the system's behavior as time goes to infinity. But Lyapunov analysis can also be used to analyze the finite-time behavior of nonlinear systems.

OK, the question in this section is: In an interval of time $[t_1, t_2]$, known that $x(t_1) \in \mathcal{X}_1$, can we guarantee that $x(t) \in \mathcal{X}_2$ for all $t\in[t_1, t_2]$, where  $\mathcal{X}_2$ is a "safe" region of state space?

1. **Forward reachability.**  
   We choose the initial region $\mathcal{X}_1$ and try to find the smallest possible final region $\mathcal{X}_2$ that contains all states reachable at time $t_2$. This is useful for safety verification, e.g. proving that a UAV starting from its current uncertain state will not collide with obstacles in the next few seconds.

2. **Backward reachability.**  
   We choose the desired final region $\mathcal{X}_2$ and try to find the largest initial region $\mathcal{X}_1$ from which all trajectories are guaranteed to reach $\mathcal{X}_2$. Region of attraction analysis is a special case of this idea, where the final target is the equilibrium point.

# 9.4 Control Design
## 9.4.1 Control design via alternations

### Global stability
Find controller $u = K(x)$ where $K(x)$ is a polynomial vector and find $V(x)$.

Consider a control-affine system:
$\dot{x} = f_1(x) + f_2(x) u$.
Choose feedback controller $u = K(x)$, where $K(x)$ is a polynomial vector. Then we have closed-loop system:
$$\dot{x} = f_1(x) + f_2(x) K(x).
$$

OK, the problem becomes:
Find $V(x)$ and $K(x)$ such that $V(x)$ is positive definite and $\dot{V}(x) = \frac{\partial V}{\partial x} (f_1(x) + f_2(x) K(x))$ is negative definite.

(There is an alternation method to solve this problem, see in chapter 9 Tedrake)

### Maximizing the region of attraction

An extension of this approach can be used to design controllers that are certified only over a region of attraction, but this requires some additional care. We can write the search for the maximal region of attraction as:

$$
\begin{aligned}
\max_{\rho,\,K(x),\,V(x),\,\lambda(x)} \quad & \rho \\
\text{subject to} \quad
& V(x),\ \lambda(x)\ \text{are SOS}, \\
& -\frac{\partial V}{\partial x}
\left(f_1(x)+f_2(x)K(x)\right)
+\lambda(x)(V(x)-\rho)
\ \text{is SOS}, \\
& V(0)=0,\quad V(1)=1.
\end{aligned}
$$

The last line yields linear constraints which simply fixes the scaling of $V(x)$.

Once again, we find that this optimization is bilinear in the decision variables. But the conditions are linear in $\lambda(x)$ and $K(x)$ for fixed $V(x)$, and are linear in $V(x)$ for fixed $\lambda(x)$ and $K(x)$. In principle, we could use bilinear alternations, but [24] recommended a three-way alternation approach in order to include a step which explicitly optimizes $K(x)$ with an objective of maximizing $\rho$. Starting with an initial guess for $V(x)$ we can repeat the following alternations:

1. Fix $V(x)$ and $\rho$, and solve a feasibility problem for $K(x)$ and $\lambda(x)$:

   $$
   \begin{aligned}
   \text{find}_{\lambda(x),\,K(x)} \\
   \text{subject to} \quad
   & -\frac{\partial V}{\partial x}
   \left(f_1(x)+f_2(x)K(x)\right)
   +\lambda(x)(V(x)-\rho)
   \ \text{is SOS}.
   \end{aligned}
   $$

2. Fix $V(x)$ and $\lambda(x)$ and maximize $\rho$ searching over $K(x)$:

   $$
   \begin{aligned}
   \max_{\rho,\,K(x)} \quad & \rho \\
   \text{subject to} \quad
   & \lambda(x)\ \text{is SOS}, \\
   & -\frac{\partial V}{\partial x}
   \left(f_1(x)+f_2(x)K(x)\right)
   +\lambda(x)(V(x)-\rho)
   \ \text{is SOS}.
   \end{aligned}
   $$

3. Fix $\lambda(x)$ and $K(x)$ and maximize $\rho$ searching over $V(x)$:

   $$
   \begin{aligned}
   \max_{\rho,\,V(x)} \quad & \rho \\
   \text{subject to} \quad
   & V(x)\ \text{is SOS}, \\
   & -\frac{\partial V}{\partial x}
   \left(f_1(x)+f_2(x)K(x)\right)
   +\lambda(x)(V(x)-\rho)
   \ \text{is SOS}, \\
   & V(0)=0,\quad V(1)=1.
   \end{aligned}
   $$


### 9.1.3 Relationship to the Hamilton-Jacobi-Bellman equations

At this point, you might be wondering if there is any relationship between Lyapunov functions and the cost-to-go functions that we discussed in the context of dynamic programming. After all, the cost-to-go functions also captured a great deal about the long-term dynamics of the system in a scalar function. We can see the connection if we re-examine the HJB equation

$$
0 = \min_{\mathbf{u}}
\left[
\ell(\mathbf{x}, \mathbf{u})
+
\frac{\partial J^*}{\partial x} f(\mathbf{x}, \mathbf{u})
\right].
$$

Let's imagine that we can solve for the optimizing $\mathbf{u}^*(\mathbf{x})$, then we are left with $0 = \ell(\mathbf{x}, \mathbf{u}^*) + \frac{\partial J^*}{\partial x} f(\mathbf{x}, \mathbf{u}^*)$ or simply

$$
\dot{J}^*(\mathbf{x}) = -\ell(\mathbf{x}, \mathbf{u}^*)
\qquad \text{vs} \qquad
\dot{V}(\mathbf{x}) \le 0.
$$

In other words, in optimal control we must find a cost-to-go function which matches this gradient for every $x$; that's very difficult and involves solving a potentially high-dimensional partial differential equation. By contrast, Lyapunov analysis is asking for much less - any function which is going downhill (at any rate) for all states. This can be much easier, for theoretical work, but also for our numerical algorithms. Also note that if we do manage to find the optimal cost-to-go, $J^*(\mathbf{x})$, then it can also serve as a Lyapunov function so long as $\ell(\mathbf{x}, \mathbf{u}^*(\mathbf{x})) \ge 0$.


### Upper and lower bounds on cost-to-go

Asking for $\dot{V}(\mathbf{x}) \le 0$ is sufficient for proving stability. But we can also use this idea to provide rigorous certificates as upper or lower bounds of the cost-to-go. Given a control-dynamical system $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$, and a fixed controller $\pi(\mathbf{x})$ we can find a function $V(\mathbf{x})$:

- $\forall \mathbf{x}, \dot{V}^{\pi}(\mathbf{x}) \le -\ell(\mathbf{x}, \pi(\mathbf{x}))$ to provide an **upper bound**, or
- $\forall \mathbf{x}, \dot{V}^{\pi}(\mathbf{x}) \ge -\ell(\mathbf{x}, \pi(\mathbf{x}))$ to provide a **lower bound**.

To see this, take the integral of both sides along any *solution* trajectory, $\mathbf{x}(t), \mathbf{u}(t)$. For the upper-bound we get

$$
\int_0^\infty \dot{V}^{\pi}(\mathbf{x})dt
=
V^{\pi}(\mathbf{x}(\infty)) - V^{\pi}(\mathbf{x}(0))
\le
\int_0^\infty -\ell(\mathbf{x}(t), \pi(\mathbf{x}(t)))dt.
$$

Assuming $V^{\pi}(\mathbf{x}(\infty)) = 0$, we have

$$
V^{\pi}(\mathbf{x}(0))
\ge
\int_0^\infty \ell(\mathbf{x}(t), \pi(\mathbf{x}(t)))dt.
$$

The upper bound is the one that we would want to use in a certification procedure -- it provides a *guarantee* that the total cost achieved by the system started in $\mathbf{x}$ is less than $V(\mathbf{x})$. But it turns out that the lower bound is much better for control design. This is because we can write

$$
\forall \mathbf{x}, \min_{\mathbf{u}}
\left[
\ell(\mathbf{x}, \mathbf{u})
+
\frac{\partial V}{\partial x} f(\mathbf{x}, \mathbf{u})
\right]
\ge 0
\equiv
\forall \mathbf{x}, \forall \mathbf{u},
\ell(\mathbf{x}, \mathbf{u})
+
\frac{\partial V}{\partial x} f(\mathbf{x}, \mathbf{u})
\ge 0.
$$

Therefore, without having to specify a priori a controller, if we can find a function $V(\mathbf{x})$ such that $\forall \mathbf{x}, \forall \mathbf{u}, \dot{V}(\mathbf{x}, \mathbf{u}) \ge -\ell(\mathbf{x}, \mathbf{u})$, then we have a lower-bound on the *optimal* cost-to-go.

You should take a minute to convince yourself that, unfortunately, the same trick does not work for the upper-bound. Again, we would need $\exists$ as the quantifier on $\mathbf{u}$ instead of $\forall$.

## Linear Programming Dynamic Programming

In the dynamic programming chapter, we introduced the [linear programming approach to dynamic programming]. This was exactly using the idea of pushing up on a lower bound on the cost-to-go. In the case of a discrete state, then this lower bound will be tight (and we obtain the true cost-to-go function); in the case of linear function approximators, then we hope for a tight lower bound.

## Sums-of-Squares Dynamic Programming

The linear programming approach works natively in discrete (or sampled) time/state/action settings. Sums-of-squares provides a beautiful extension to continuous time/state/actions.

In the simple case, let us assume that $f(\mathbf{x}, \mathbf{u})$ and $\ell(\mathbf{x}, \mathbf{u})$ are polynomials (this assumption is discussed at length in the Lyapunov chapter). We can search for a polynomial $\hat{J}_{\alpha}(\mathbf{x})$ which satisfies the HJB inequality

$$
\forall \mathbf{x}, \forall \mathbf{u}, \quad
0 \le \ell(\mathbf{x}, \mathbf{u})
+
\frac{\partial \hat{J}_{\alpha}}{\partial x} f(\mathbf{x}, \mathbf{u}),
$$

using the following *convex* optimization:

$$
\begin{aligned}
\max_{\alpha} \quad
& \int_{\mathcal{X}} \hat{J}_{\alpha}(\mathbf{x}) dx \\
\text{subject to} \quad
& \ell(\mathbf{x}, \mathbf{u})
+
\frac{\partial \hat{J}_{\alpha}}{\partial x} f(\mathbf{x}, \mathbf{u})
\ \text{is SOS}\quad (\forall \mathbf{x}, \forall \mathbf{u}) \\
& \hat{J}_{\alpha}(0) = 0.
\end{aligned}
$$

Since $\hat{J}_{\alpha}(\mathbf{x})$ is polynomial, the integral objective (over a bounded domain of interest) can be computed exactly, and is still linear in the coefficients $\alpha$. As in the linear programming approach, here the SOS constraint ensures that $\hat{J}$ is a *lower* bound on the cost to go, and the objective "pushes up" on this lower bound. As we increase the degree of the polynomial representing $\hat{J}$, we expect better and better approximations of the true cost-to-go.

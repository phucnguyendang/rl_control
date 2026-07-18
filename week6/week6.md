

## Passivity of Memoryless Functions

### 1. Memoryless function

A memoryless function is a system of the form

$$
y = h(t,u),
$$

where $u \in \mathbb{R}^p$ is the input and $y \in \mathbb{R}^p$ is the output.

It is called **memoryless** because the output at time $t$ depends only on the current input $u(t)$ and possibly time $t$, not on past inputs or internal states.

A simple example is a resistor:

$$
y = h(u),
$$

where $u$ is voltage and $y$ is current.

---

### 2. Power/Supply rate

The key quantity is

$$
u^T y.
$$

This is interpreted as the instantaneous power flowing into the system.

* If $u^T y > 0$, the system is absorbing power.
* If $u^T y = 0$, the system is exchanging no net power.
* If $u^T y < 0$, the system is delivering power to the outside.

Passivity is based on asking whether the system can generate power by itself.

---

### 3. Passive system

The memoryless system $y=h(t,u)$ is **passive** if

$$
u^T y \ge 0
$$

for all $(t,u)$.

This means the system never produces net power at its input-output port. It can absorb energy, dissipate energy, or exchange zero energy, but it does not act like an energy source.

For a scalar system, this means the graph of $y=h(t,u)$ lies in the first and third quadrants, so $u$ and $y$ always have the same sign or one of them is zero.

---

### 4. Lossless system

The system is **lossless** if

$$
u^T y = 0
$$

for all $(t,u)$.

A lossless system does not dissipate energy, but it also does not generate energy. It only transfers or redistributes energy.

---

### 5. Input-feedforward passive

The system is **input-feedforward passive** if there exists a function $\varphi(u)$ such that

$$
u^T y \ge u^T \varphi(u)
$$

for all $(t,u)$.

This means the system may not be passive in its original form, but it can be made passive by subtracting a feedforward term from the output.

Define a new output

$$
\tilde y = y - \varphi(u).
$$

Then

$$
u^T \tilde y
= u^T y - u^T \varphi(u)
\ge 0.
$$

So the modified input-output pair $(u,\tilde y)$ is passive.

---

### 6. Input strictly passive

The system is **input strictly passive** if there exists a function $\varphi(u)$ such that

$$
u^T y \ge u^T \varphi(u)
$$

and

$$
u^T \varphi(u) > 0
\quad \text{for all } u \ne 0.
$$

This is stronger than passivity. It says the system has a positive amount of passivity measured from the input side.

A common scalar example is

$$
\varphi(u)=cu.
$$

Then

$$
u y \ge c u^2.
$$

If $c>0$, the system has an **excess of passivity**.

---

### 7. Output-feedback passive

The system is **output-feedback passive** if there exists a function $\rho(y)$ such that

$$
u^T y \ge y^T \rho(y)
$$

for all $(t,u)$.

This means the system can be made passive by modifying the input with output feedback.

Define a new input

$$
\tilde u = u - \rho(y).
$$

Then

$$
\tilde u^T y
= u^T y - y^T \rho(y)
\ge 0.
$$

So the modified input-output pair $(\tilde u,y)$ is passive.

---

### 8. Output strictly passive

The system is **output strictly passive** if there exists a function $\rho(y)$ such that

$$
u^T y \ge y^T \rho(y)
$$

and

$$
y^T \rho(y) > 0
\quad \text{for all } y \ne 0.
$$

This is stronger than passivity. It says the system has a positive amount of passivity measured from the output side.

A common scalar example is

$$
\rho(y)=\delta y.
$$

Then

$$
u y \ge \delta y^2.
$$

If $\delta>0$, the system has an excess of passivity from the output side.

---

### 9. Excess and shortage of passivity

For input-side passivity, suppose

$$
u^T y \ge u^T \varphi(u).
$$

If

$$
u^T \varphi(u)>0
$$

for all nonzero $u$, the system has **excess passivity**.

If $u^T \varphi(u)$ can be negative, the system may have a **shortage of passivity**. It is not necessarily passive, but the shortage may be compensated by feedback or feedforward transformations.

Similarly, on the output side, the sign of

$$
y^T \rho(y)
$$

determines whether there is excess or shortage of passivity.

---

### 10. Sector interpretation

For scalar systems, passivity is often described using sectors.

The condition

$$
u h(t,u) \ge 0
$$

means that $h$ belongs to the sector

$$
[0,\infty].
$$

This is exactly the scalar passivity condition.

More generally, saying that a nonlinearity belongs to a sector means its input-output graph lies between two lines. This gives a geometric way to understand passivity and strict passivity.

---

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Definition 6.1 — Summary

For the memoryless system

$$
y=h(t,u),
$$

Khalil defines:

| Concept                   | Condition                                                        |
| ------------------------- | ---------------------------------------------------------------- |
| Passive                   | $u^T y \ge 0$                                                    |
| Lossless                  | $u^T y = 0$                                                      |
| Input-feedforward passive | $u^T y \ge u^T \varphi(u)$                                       |
| Input strictly passive    | $u^T y \ge u^T \varphi(u)$ and $u^T\varphi(u)>0$ for all $u\ne0$ |
| Output-feedback passive   | $u^T y \ge y^T \rho(y)$                                          |
| Output strictly passive   | $u^T y \ge y^T\rho(y)$ and $y^T\rho(y)>0$ for all $y\ne0$        |

All inequalities must hold for every admissible $(t,u)$.

The main idea is simple:

$$
u^T y
$$

is the power entering the system. A passive system cannot generate energy by itself; it can only store, dissipate, or transmit energy supplied from outside.

</div>

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Definition 6.2

A memoryless function $h : [0,\infty) \times \mathbb{R}^p \rightarrow \mathbb{R}^p$ is said to belong to the sector

- $[0,\infty]$ if $u^T h(t,u) \geq 0$.

- $[K_1,\infty]$ if $u^T[h(t,u)-K_1u] \geq 0$.

- $[0,K_2]$ with $K_2 = K_2^T > 0$ if $h^T(t,u)[h(t,u)-K_2u] \leq 0$.

- $[K_1,K_2]$ with $K = K_2-K_1 = K^T > 0$ if

$$
[h(t,u)-K_1u]^T[h(t,u)-K_2u] \leq 0. \qquad (6.5)
$$

In all cases, the inequality should hold for all $(t,u)$. If in any case the inequality is strict, we write the sector as $(0,\infty)$, $(K_1,\infty)$, $(0,K_2)$, or $(K_1,K_2)$. In the scalar case, we write $(\alpha,\beta]$, $[\alpha,\beta)$, or $(\alpha,\beta)$ to indicate that one or both sides of the sector inequality is satisfied as a strict inequality.

</div>

## 6.2 State Models

Define:

- $V(x)$ is the energy stored in the network.
- The dynamic system is

$$
\dot{x} = f(x,u) \qquad (6.18)
$$

$$
y = h(x,u) \qquad (6.19)
$$

where $f: \mathbb{R}^n \times \mathbb{R}^p \rightarrow \mathbb{R}^n$ is locally Lipschitz and $h: \mathbb{R}^n \times \mathbb{R}^p \rightarrow \mathbb{R}^p$ is continuous, $f(0,0) = 0$, and $h(0,0) = 0$. The system has the same number of inputs and outputs.

This system is used for the definitions below.

Intuition: The system is passive if the energy stored in the network is less than or equal to the energy supplied to the network. This means

$$
\int_0^t y(\tau)^T u(\tau)\,d\tau \geq V(x(t)) - V(x(0)).
$$

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Definition 6.3

The system defined above is said to be **passive** if there exists a continuously differentiable positive semidefinite function $V(x)$, called the **storage function**, such that

$$
u^T y \geq \dot{V} = \dfrac{\partial V}{\partial x}f(x,u),
\quad \forall\,(x,u)\in \mathbb{R}^n \times \mathbb{R}^p. \qquad (6.10)
$$

Moreover, it is said to be:

- **lossless** if $u^T y = \dot{V}$.

- **input-feedforward passive** if $u^T y \geq \dot{V} + u^T\varphi(u)$ for some function $\varphi$.

- **input strictly passive** if $u^T y \geq \dot{V} + u^T\varphi(u)$ and $u^T\varphi(u) > 0,\ \forall\ u \neq 0$.

- **output-feedback passive** if $u^T y \geq \dot{V} + y^T\rho(y)$ for some function $\rho$.

- **output strictly passive** if $u^T y \geq \dot{V} + y^T\rho(y)$ and $y^T\rho(y) > 0,\ \forall\ y \neq 0$.

- **strictly passive** if $u^T y \geq \dot{V} + \psi(x)$ for some positive definite function $\psi$.

In all cases, the inequality should hold for all $(x,u)$.

Definition 6.3 reads almost the same as Definition 6.1 for memoryless functions, except for the presence of a storage function $V(x)$. If we adopt the convention that $V(x)=0$ for a memoryless function, Definition 6.3 can be used for both state models and memoryless functions.

</div>

## 6.4 $\mathcal{L}_2$ and Lyapunov Stability

Given a signal $u(t)$, we can define the $\mathcal{L}_2$ norm of $u$ as

$$
\|u\|_2 = \sqrt{\int_0^\infty u(t)^T u(t)\,dt}.
$$

If this integral is finite, we say $u \in \mathcal{L}_2$.

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Lemma 6.5

If the system (6.18)–(6.19) is output strictly passive with

$$
u^T y \geq \dot{V} + \delta y^T y,
$$

for some $\delta > 0$, then it is finite-gain $\mathcal{L}_2$ stable and its $\mathcal{L}_2$ gain is less than or equal to $1/\delta$.

<details>
  <summary><b>Proof (click for detail)</b></summary>

  <div style="margin-left: 20px; background-color: #01172c; padding: 15px; border-radius: 8px; border-left: 4px solid #1172d3; margin-top: 10px;">

  The derivative of the storage function $V(x)$ satisfies

  $$
  \dot{V} \leq u^T y - \delta y^T y
  = -\dfrac{1}{2\delta}(u-\delta y)^T(u-\delta y)
  + \dfrac{1}{2\delta}u^T u
  - \dfrac{\delta}{2}y^T y.
  $$

  Therefore,

  $$
  \dot{V} \leq \dfrac{1}{2\delta}u^T u - \dfrac{\delta}{2}y^T y.
  $$

  Integrating both sides over $[0,\tau]$ yields

  $$
  \int_0^\tau y^T(t)y(t)\,dt
  \leq \dfrac{1}{\delta^2}\int_0^\tau u^T(t)u(t)\,dt
  - \dfrac{2}{\delta}[V(x(\tau))-V(x(0))].
  $$

  Thus,

  $$
  \|y_\tau\|_{\mathcal{L}_2}
  \leq \dfrac{1}{\delta}\|u_\tau\|_{\mathcal{L}_2}
  + \sqrt{\dfrac{2}{\delta}V(x(0))}.
  $$

  Here we used the facts that $V(x) \geq 0$ and $\sqrt{a^2+b^2} \leq a+b$ for nonnegative numbers $a$ and $b$. $\square$

  </div>
</details>

</div>

“Finite-gain $\mathcal{L}_2$ stable” means that the system does not amplify the energy of the signal from the input to the output without bound.

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Lemma 6.6

If the system (6.18)–(6.19) is passive with a positive definite storage function $V(x)$, then the origin of $\dot{x}=f(x,0)$ is stable.

<details>
  <summary><b>Proof (click for detail)</b></summary>

  <div style="margin-left: 20px; background-color: #01172c; padding: 15px; border-radius: 8px; border-left: 4px solid #1172d3; margin-top: 10px;">

  Take $V$ as a Lyapunov function candidate for $\dot{x}=f(x,0)$. Since the system is passive,

  $$
  u^T y \geq \dot{V}.
  $$

  Setting $u=0$ gives

  $$
  \dot{V} \leq 0.
  $$

  Therefore, by Lyapunov stability theory, the origin of $\dot{x}=f(x,0)$ is stable. $\square$

  </div>
</details>

</div>

To show asymptotic stability of the origin of $\dot{x}=f(x,0)$, we need to either show that $\dot{V}$ is negative definite or apply the invariance principle. In the next lemma, we apply the invariance principle by considering a case where $\dot{V}=0$ when $y=0$ and then require the additional property that

$$
y(t) \equiv 0 \Rightarrow x(t) \equiv 0. \qquad (6.20)
$$

This should hold for all solutions of (6.18) when $u=0$. Equivalently, no solutions of $\dot{x}=f(x,0)$ can stay identically in $S=\{x\in \mathbb{R}^n \mid h(x,0)=0\}$, other than the trivial solution $x(t)=0$. The property (6.20) can be interpreted as an observability condition.

Recall that for the linear system

$$
\dot{x}=Ax, \qquad y=Cx,
$$

observability is equivalent to

$$
y(t)=Ce^{At}x(0)=0
\Leftrightarrow x(0)=0
\Leftrightarrow x(t)=0.
$$

For easy reference, we define (6.20) as an observability property of the system.

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Definition 6.5

The system (6.18)–(6.19) is said to be **zero-state observable** if no solution of $\dot{x}=f(x,0)$ can stay identically in

$$
S=\{x\in \mathbb{R}^n \mid h(x,0)=0\},
$$

other than the trivial solution $x(t)\equiv 0$.

</div>

<div style="background-color: #102437; padding: 16px; border-radius: 10px; border-left: 5px solid #58a6ff; margin: 18px 0;">

### Lemma 6.7

Consider the system (6.18)–(6.19). The origin of $\dot{x}=f(x,0)$ is asymptotically stable if the system is

- strictly passive, or
- output strictly passive and zero-state observable.

Furthermore, if the storage function is radially unbounded, the origin will be globally asymptotically stable. $\diamond$

<details>
  <summary><b>Proof (click for detail)</b></summary>

  <div style="margin-left: 20px; background-color: #01172c; padding: 15px; border-radius: 8px; border-left: 4px solid #1172d3; margin-top: 10px;">

  Suppose the system is strictly passive and let $V(x)$ be its storage function. Then, with $u=0$, $\dot{V}$ satisfies

  $$
  \dot{V}\leq -\psi(x),
  $$

  where $\psi(x)$ is positive definite. We can use this inequality to show that $V(x)$ is positive definite. In particular, for any $x\in \mathbb{R}^n$, the equation $\dot{x}=f(x,0)$ has a solution $\phi(t;x)$, starting from $x$ at $t=0$ and defined on some interval $[0,\delta]$. Integrating the inequality $\dot{V}\leq -\psi(x)$ yields

  $$
  V(\phi(\tau;x))-V(x)
  \leq -\int_0^\tau \psi(\phi(t;x))\,dt,
  \quad \forall\ \tau\in[0,\delta].
  $$

  Using $V(\phi(\tau;x))\geq 0$, we obtain

  $$
  V(x)\geq \int_0^\tau \psi(\phi(t;x))\,dt.
  $$

  Suppose now that there is $\bar{x}\neq 0$ such that $V(\bar{x})=0$. The foregoing inequality implies

  $$
  \int_0^\tau \psi(\phi(t;\bar{x}))\,dt=0,
  \quad \forall\ \tau\in[0,\delta]
  \Rightarrow \psi(\phi(t;\bar{x}))\equiv 0
  \Rightarrow \phi(t;\bar{x})\equiv 0
  \Rightarrow \bar{x}=0,
  $$

  which contradicts the claim that $\bar{x}\neq 0$. Thus, $V(x)>0$ for all $x\neq 0$. This qualifies $V(x)$ as a Lyapunov function candidate, and since $\dot{V}(x)\leq -\psi(x)$, we conclude that the origin is asymptotically stable.

  Suppose now the system is output strictly passive and let $V(x)$ be its storage function. Then, with $u=0$, $\dot{V}$ satisfies

  $$
  \dot{V}\leq -y^T\rho(y),
  $$

  where $y^T\rho(y)>0$ for all $y\neq 0$. By repeating the preceding argument, we can use the inequality to show that $V(x)$ is positive definite. In particular, for any $x\in \mathbb{R}^n$, we have

  $$
  V(x)\geq \int_0^\tau h^T(\phi(t;x),0)\rho(h(\phi(t;x),0))\,dt.
  $$

  Suppose now that there is $\bar{x}\neq 0$ such that $V(\bar{x})=0$. The foregoing inequality implies

  $$
  \int_0^\tau h^T(\phi(t;\bar{x}),0)\rho(h(\phi(t;\bar{x}),0))\,dt=0,
  \quad \forall\ \tau\in[0,\delta]
  \Rightarrow h(\phi(t;\bar{x}),0)\equiv 0.
  $$

  Due to zero-state observability, this implies

  $$
  \phi(t;\bar{x})\equiv 0
  \Rightarrow \bar{x}=0.
  $$

  Hence, $V(x)>0$ for all $x\neq 0$. This qualifies $V(x)$ as a Lyapunov function candidate. Since $\dot{V}(x)\leq -y^T\rho(y)$ and $y(t)\equiv 0\Rightarrow x(t)\equiv 0$, we conclude by the invariance principle that the origin is asymptotically stable.

  Finally, if $V(x)$ is radially unbounded, we can infer global asymptotic stability from Theorem 4.2 and Corollary 4.2, respectively. $\square$

  </div>
</details>

</div>


## 6.5 Feedback Systems : Passivity theorem


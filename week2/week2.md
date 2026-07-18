# Policy Gradient Methods
In this method, we learn a parameterized policy that selects actions without consulting a value function.

We update $\theta$ in $\pi(a|s,\theta)$ to maximize the performance measure $J(\theta)$ using
$$\theta_{t+1} = \theta_t + \alpha \nabla J(\theta_t)$$

But what is $J(\theta)$?

In the discounted episodic case, the performance is defined as:
$$J(\theta) = v_{\pi_\theta}(s_0)$$
where $v_{\pi_\theta}(s_0) = \mathbb{E}_{\pi_\theta}[\sum_{t=0}^{T-1}\gamma^t R_{t+1} \mid S_0=s_0]$ is the true discounted value function for $\pi_\theta$, the policy determined by $\theta$.

We proved (below in the Appendix) that 

$$\boxed{ \nabla J(\theta) \propto \sum_s \mu_\gamma(s) \sum_a q_\pi(s,a) \nabla \pi(a|s) }$$
## REINFORCE 
$\nabla J(\theta) \propto \sum_s \mu_\gamma(s) \sum_a q_\pi(s,a) \nabla \pi(a|s) = \mathbb{E}_{S_t \sim \mu_\gamma} \left[ \sum_{a} q_{\pi}(S_t, a)\,\nabla \pi(a \mid S_t, \theta) \right]$
$$\begin{align} \nabla J(\theta) &\propto \mathbb{E}_{S_t \sim \mu_\gamma} \left[ \sum_a \pi(a \mid S_t,\theta)\, q_{\pi}(S_t,a)\, \frac{\nabla \pi(a \mid S_t,\theta)} {\pi(a \mid S_t,\theta)} \right] \\ &= \mathbb{E}_{S_t \sim \mu_\gamma, A_t \sim \pi} \left[ q_{\pi}(S_t,A_t)\, \nabla \ln \pi(A_t \mid S_t,\theta) \right] \qquad \text{(replacing } a \text{ by the sample } A_t \sim \pi) \\ &= \mathbb{E}_{S_t \sim \mu_\gamma, A_t \sim \pi} \left[ G_t\, \nabla \ln \pi(A_t \mid S_t,\theta) \right], \qquad \text{(because } \mathbb{E}_{\pi}[G_t \mid S_t,A_t] = q_{\pi}(S_t,A_t)\text{)}. \end{align}$$
Because $\mu_\gamma$ weights the state at time $t$ by $\gamma^t$, the Monte-Carlo episode update uses the factor $\gamma^t$.


**REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for $\pi_*$**

- Input: a differentiable policy parameterization $\pi(a \mid s,\theta)$  
- Algorithm parameter: step size $\alpha > 0$  
- Initialize policy parameter $\theta \in \mathbb{R}^{d'}$ (e.g., to $\mathbf{0}$)

- Loop forever (for each episode):

    - Generate an episode  
    - $S_0,A_0,R_1,\ldots,S_{T-1},A_{T-1},R_T$, following $\pi(\cdot \mid \cdot,\theta)$
    - Loop for each step of the episode $t = 0,1,\ldots,T-1$:

        - $G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k \qquad (G_t)$

        - $\theta \leftarrow \theta + \alpha \gamma^t G \nabla \ln \pi(A_t \mid S_t,\theta)$

## REINFORCE with Baseline

In the Appendix, we have:
$$\nabla J(\theta) \propto \sum_s \mu_\gamma(s) \sum_a q_\pi(s,a) \nabla \pi(a|s)$$

If we add a baseline $b(s)$ to the action-value function, we have:
$$\nabla J(\theta) \propto \sum_s \mu_\gamma(s) \sum_a (q_\pi(s,a) - b(s)) \nabla \pi(a|s)$$

As long as $b(s)$ does not depend on $a$, the expected value of the policy gradient remains unchanged, because $\sum_a b(s) \nabla \pi(a|s) = b(s) \nabla \sum_a \pi(a|s) = b(s) \nabla 1 = 0$.


With the baseline, the update rule for REINFORCE becomes:
$$\theta_{t+1} \leftarrow \theta_t + \alpha \gamma^t (G - b(S_t)) \nabla \ln \pi(A_t \mid S_t,\theta)$$

The natural choice for the baseline is an estimate of the state value, $ \hat{v}(S_t,w) $, where $w \in \mathbb{R}^d$ is a learned parameter.


**Algorithm: REINFORCE with Baseline (episodic)**

- Input: a differentiable policy parameterization $\pi(a \mid s,\theta)$  
- Input: a differentiable state-value function parameterization $\hat{v}(s,w)$
- Algorithm parameter: step sizes $\alpha^\theta > 0$, $\alpha^w > 0$  
- Initialize policy parameter $\theta \in \mathbb{R}^{d'}$ (e.g., to $\mathbf{0}$)
- Initialize state-value function parameter $w \in \mathbb{R}^d$ (e.g., to $\mathbf{0}$)

- Loop forever (for each episode):

    - Generate an episode  
    - $S_0,A_0,R_1,\ldots,S_{T-1},A_{T-1},R_T$, following $\pi(\cdot \mid \cdot,\theta)$
    - Loop for each step of the episode $t = 0,1,\ldots,T-1$:

        - $G \leftarrow \sum_{k=t+1}^{T} \gamma^{k-t-1} R_k \qquad (G_t)$
        - $\delta \leftarrow G - \hat{v}(S_t,w)$
        - $w \leftarrow w + \alpha^w \delta \nabla \hat{v}(S_t,w)$

        - $\theta \leftarrow \theta + \alpha^\theta \gamma^t \delta \nabla \ln \pi(A_t \mid S_t,\theta)$




## Actor-Critic Methods
*TD error* stands for Temporal-Difference error. It measures how wrong the value prediction was after observing one more step of experience.

The basic formula is:
$$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1},w_t) - \hat{v}(S_t,w_t)$$

Instead of using the Monte-Carlo return $G_t$ to update the policy, we can use the one-step TD target $R_{t+1} + \gamma \hat{v}(S_{t+1},w)$. If $\hat{v}$ is accurate, this target is an estimate of $q_\pi(S_t,A_t)$; in practice it is biased but usually has lower variance than the full return. This leads to the actor-critic method.

In actor-critic, the actor is the policy $\pi(a \mid s,\theta)$ and the critic is the state-value estimate $\hat{v}(s,w)$. The TD error
$$\delta \leftarrow R + \gamma \hat{v}(S',w) - \hat{v}(S,w)$$
plays the role of the advantage estimate used to update both the critic and the actor.

Generalizing to the forward view of $n$-step methods and then to a $\lambda$-return algorithm is straightforward. The one-step return is replaced by $G_{t:t+n}$ or $G_t^\lambda$, respectively. The backward view of the $\lambda$-return algorithm is also straightforward, using separate eligibility traces for the actor and critic. Pseudocode for the complete algorithm is given below.


**Actor-Critic with Eligibility Traces (episodic), for estimating $\pi_\theta \approx \pi_*$**

- Input: a differentiable policy parameterization $\pi(a \mid s,\theta)$
- Input: a differentiable state-value function parameterization $\hat{v}(s,w)$
- Parameters: trace-decay rates $\lambda^\theta \in [0,1]$, $\lambda^w \in [0,1]$; step sizes $\alpha^\theta > 0$, $\alpha^w > 0$
- Initialize policy parameter $\theta \in \mathbb{R}^{d'}$ and state-value weights $w \in \mathbb{R}^{d}$ (e.g., to $\mathbf{0}$)

- Loop forever (for each episode):

    - Initialize $S$ (first state of episode)
    - $z^\theta \leftarrow \mathbf{0} \qquad (d'\text{-component eligibility trace vector})$
    - $z^w \leftarrow \mathbf{0} \qquad (d\text{-component eligibility trace vector})$
    - $I \leftarrow 1$
    - Loop while $S$ is not terminal (for each time step):

        - $A \sim \pi(\cdot \mid S,\theta)$
        - Take action $A$, observe $S'$, $R$
        - $\delta \leftarrow R + \gamma \hat{v}(S',w) - \hat{v}(S,w) \qquad$ (if $S'$ is terminal, then $\hat{v}(S',w) \doteq 0$)
        - $z^w \leftarrow \gamma \lambda^w z^w + \nabla \hat{v}(S,w)$
        - $z^\theta \leftarrow \gamma \lambda^\theta z^\theta + I \nabla \ln \pi(A \mid S,\theta)$
        - $w \leftarrow w + \alpha^w \delta z^w$
        - $\theta \leftarrow \theta + \alpha^\theta \delta z^\theta$
        - $I \leftarrow \gamma I$
        - $S \leftarrow S'$


*Why do we use the parameter updates above?*
- In Monte-Carlo methods, we estimate $v_\pi(S_t)$ by generating a complete episode and then calculating the full return $G_t$.
- In TD methods, we estimate $v_\pi(S_t)$ after one step by using the bootstrapped target $R_{t+1} + \gamma \hat{v}(S_{t+1},w)$.
- The main trade-off is bias, variance, and speed of learning. Monte-Carlo targets have low bias but high variance, while TD targets have lower variance but more bias. The $\lambda$-return lets us interpolate between these two extremes by mixing returns of different lengths. When $\lambda=0$, the target becomes the one-step TD target. When $\lambda=1$ in an episodic task, it becomes the Monte-Carlo return.
- We try to learn $\hat{v}(S_t,w)$ to approximate the expected return from state $S_t$:

$$
v_\pi(S_t) = \mathbb{E}_\pi[G_t \mid S_t].
$$

For an $n$-step backup, the target is

$$
G_{t:t+n}
=
R_{t+1}
+ \gamma R_{t+2}
+ \cdots
+ \gamma^{n-1}R_{t+n}
+ \gamma^n \hat{v}(S_{t+n},w).
$$

The $\lambda$-return is a weighted average of these $n$-step targets. In an episodic task,

$$
G_t^\lambda
=
(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_{t:t+n}
+ \lambda^{T-t-1}G_{t:T},
$$

where $G_{t:T}$ is the full Monte-Carlo return from time $t$ to the end of the episode. Equivalently, in a continuing task we often write

$$
G_t^\lambda
=
(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}.
$$

Then the critic can be updated by semi-gradient descent on the prediction error:

$$
w_{t+1}
=
w_t
+ \alpha^w
\left[
G_t^\lambda - \hat{v}(S_t,w_t)
\right]
\nabla_w \hat{v}(S_t,w_t).
$$

The term

$$
G_t^\lambda - \hat{v}(S_t,w_t)
$$

acts like an advantage estimate: it says whether the observed return was better or worse than what the critic expected from $S_t$. Therefore the actor can use the same signal to update the policy:

$$
\theta_{t+1}
=
\theta_t
+ \alpha^\theta \gamma^t
\left[
G_t^\lambda - \hat{v}(S_t,w_t)
\right]
\nabla_\theta \ln \pi(A_t \mid S_t,\theta_t).
$$

Computing $G_t^\lambda$ directly is called the forward view because it needs future rewards. The backward view implements the same idea online with eligibility traces.

The key idea is that the $\lambda$-return error for an earlier state can be decomposed into a discounted sum of future TD errors:

$$
G_t^\lambda - \hat{v}(S_t,w)
\approx
\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k.
$$

This means that a TD error observed at time $k$ should not update only the current state $S_k$. It should also update previous states $S_{k-1}, S_{k-2}, \ldots$, but with smaller weights. The farther a state is in the past, the smaller its weight becomes because it is multiplied by powers of $\gamma\lambda$.

Instead of storing the whole history and manually applying these weights, we maintain an eligibility trace $z$. For the critic, the trace at time $t$ is a decayed sum of past value-function gradients:

$$
z_t^w
=
\nabla_w \hat{v}(S_t,w_t)
+ \gamma\lambda^w \nabla_w \hat{v}(S_{t-1},w_{t-1})
+ (\gamma\lambda^w)^2 \nabla_w \hat{v}(S_{t-2},w_{t-2})
+ \cdots.
$$

So $z_t^w$ tells us which previous states are still "eligible" to be updated by the current TD error. The recursive form is just a compact way to compute this decayed sum:

$$
z_t^w
=
\gamma \lambda^w z_{t-1}^w
+ \nabla_w \hat{v}(S_t,w_t).
$$

The actor trace $z_t^\theta$ has the same meaning, but it stores past policy-gradient terms instead of value-function gradients. The factor $I_t$ is the discounted weighting factor used in episodic policy-gradient updates, usually updated as $I_{t+1} = \gamma I_t$ with $I_0=1$.

At each step, we compute the one-step TD error

$$
\delta_t
=
R_{t+1}
+ \gamma \hat{v}(S_{t+1},w_t)
- \hat{v}(S_t,w_t),
$$

then update the traces and parameters:

$$
z_t^w
=
\gamma \lambda^w z_{t-1}^w
+ \nabla_w \hat{v}(S_t,w_t),
$$

$$
z_t^\theta
=
\gamma \lambda^\theta z_{t-1}^\theta
+ I_t \nabla_\theta \ln \pi(A_t \mid S_t,\theta_t),
$$

$$
w_{t+1}
=
w_t + \alpha^w \delta_t z_t^w,
\qquad
\theta_{t+1}
=
\theta_t + \alpha^\theta \delta_t z_t^\theta.
$$

The eligibility traces make the current TD error affect not only the current state/action, but also recently visited states/actions. The effect decays by the factor $\gamma\lambda$, so more recent states receive stronger credit.



## Policy Gradient Methods for Continuing Tasks

As discussed for continuing problems without episode boundaries, we need to define performance in terms of the average rate of reward per time step:

$$
\begin{align*}
J(\boldsymbol{\theta}) \doteq r(\pi) &\doteq \lim_{h\to\infty} \frac{1}{h} \sum_{t=1}^{h} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi] \tag{13.15} \\
&= \lim_{t\to\infty} \mathbb{E}[R_t \mid S_0, A_{0:t-1} \sim \pi] \\
&= \sum_s \mu(s) \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)r,
\end{align*}
$$

where $\mu$ is the steady-state distribution under $\pi$, $\mu(s) \doteq \lim_{t\to\infty} \Pr\{S_t=s \mid A_{0:t} \sim \pi\}$, which is assumed to exist and to be independent of $S_0$ (an ergodicity assumption). Remember that this is the special distribution under which, if you select actions according to $\pi$, you remain in the same distribution:

$$
\sum_s \mu(s) \sum_a \pi(a|s, \boldsymbol{\theta}) p(s'|s,a) = \mu(s'), \quad \text{for all } s' \in \mathcal{S}. \tag{13.16}
$$

Naturally, in the continuing case, we define values, $v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t=s]$ and $q_\pi(s,a) \doteq \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$, with respect to the differential return:

$$
G_t \doteq R_{t+1} - r(\pi) + R_{t+2} - r(\pi) + R_{t+3} - r(\pi) + \cdots. \tag{13.17}
$$

With these alternative definitions, the policy gradient theorem as given for the episodic case remains true for the continuing case. A proof is given in the Appendix. The forward-view and backward-view equations also remain the same.

The complete pseudocode for the actor–critic algorithm in the continuing case (backward view) is given below.

**Actor-Critic with Eligibility Traces (continuing), for estimating $\pi_\theta \approx \pi_*$**

- Input: a differentiable policy parameterization $\pi(a|s,\boldsymbol{\theta})$
- Input: a differentiable state-value function parameterization $\hat{v}(s,\mathbf{w})$
- Algorithm parameters: $\lambda^{\mathbf{w}} \in [0,1], \lambda^{\boldsymbol{\theta}} \in [0,1], \alpha^{\mathbf{w}} > 0, \alpha^{\boldsymbol{\theta}} > 0, \alpha^{\bar{R}} > 0$
- Initialize $\bar{R} \in \mathbb{R}$ (e.g., to 0)
- Initialize state-value weights $\mathbf{w} \in \mathbb{R}^d$ and policy parameter $\boldsymbol{\theta} \in \mathbb{R}^{d'}$ (e.g., to $\mathbf{0}$)
- Initialize $S \in \mathcal{S}$ (e.g., to $s_0$)
- $\mathbf{z}^{\mathbf{w}} \leftarrow \mathbf{0} \qquad (d\text{-component eligibility trace vector})$
- $\mathbf{z}^{\boldsymbol{\theta}} \leftarrow \mathbf{0} \qquad (d'\text{-component eligibility trace vector})$
- Loop forever (for each time step):
    - $A \sim \pi(\cdot|S,\boldsymbol{\theta})$
    - Take action $A$, observe $S'$, $R$
    - $\delta \leftarrow R - \bar{R} + \hat{v}(S',\mathbf{w}) - \hat{v}(S,\mathbf{w})$
    - $\bar{R} \leftarrow \bar{R} + \alpha^{\bar{R}} \delta$
    - $\mathbf{z}^{\mathbf{w}} \leftarrow \lambda^{\mathbf{w}} \mathbf{z}^{\mathbf{w}} + \nabla \hat{v}(S,\mathbf{w})$
    - $\mathbf{z}^{\boldsymbol{\theta}} \leftarrow \lambda^{\boldsymbol{\theta}} \mathbf{z}^{\boldsymbol{\theta}} + \nabla \ln \pi(A|S,\boldsymbol{\theta})$
    - $\mathbf{w} \leftarrow \mathbf{w} + \alpha^{\mathbf{w}} \delta \mathbf{z}^{\mathbf{w}}$
    - $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha^{\boldsymbol{\theta}} \delta \mathbf{z}^{\boldsymbol{\theta}}$
    - $S \leftarrow S'$

## Policy Parameterization for Continuous Actions

For continuous action spaces, we cannot assign a separate probability to every possible action. Instead, we parameterize a probability density over actions.

A common choice is a Gaussian policy:

$$
A \sim \mathcal N(\mu(s,\theta), \sigma(s,\theta)^2)
$$

where:

- $\mu(s,\theta)$ is the mean action, i.e. the center of the policy.
- $\sigma(s,\theta)$ controls the spread of the distribution, i.e. the amount of exploration.

Actions near the mean have the highest probability density, while actions farther away are less likely to be sampled.

The policy parameters can be split into two parts:

$$
\theta = [\theta_\mu, \theta_\sigma]
$$

where $\theta_\mu$ controls the mean and $\theta_\sigma$ controls the standard deviation.

Gaussian policies are useful because they are simple, differentiable, easy to sample from, and work naturally with policy-gradient updates.

The key idea is:

$$
\text{For continuous actions, learn a differentiable distribution over actions rather than discrete action probabilities.}
$$

## GAE




### Appendix
#### Proof of the Policy Gradient Theorem — discounted episodic case

With just elementary calculus and rearranging of terms, we can prove the policy gradient theorem from first principles. To keep the notation simple, we leave it implicit in all cases that $\pi$ is a function of $\theta$, and all gradients are also implicitly with respect to $\theta$. The value functions below use the discounted return with discount factor $\gamma \in [0,1]$.

First note that the gradient of the state-value function can be written in terms of the action-value function as:

$$\nabla v_\pi(s) = \nabla \left[ \sum_a \pi(a|s)q_\pi(s,a) \right], \qquad \text{for all } s \in \mathcal{S}$$

By the product rule of calculus:

$$\nabla v_\pi(s) = \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s)\nabla q_\pi(s,a) \right]$$

Using the Bellman equation for $q_\pi(s,a)$:

$$q_\pi(s,a) = \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_\pi(s') \right]$$

Therefore:

$$\nabla v_\pi(s) = \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s) \nabla \sum_{s',r} p(s',r|s,a) \left[ r + \gamma v_\pi(s') \right] \right]$$

Because $p(s',r|s,a)$, $r$, and $\gamma$ do not depend on $\theta$:

$$\nabla v_\pi(s) = \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \gamma \pi(a|s) \sum_{s'}p(s'|s,a)\nabla v_\pi(s') \right]$$

Now unroll $\nabla v_\pi(s')$ recursively:

$$\nabla v_\pi(s) = \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \gamma \pi(a|s) \sum_{s'}p(s'|s,a) \sum_{a'} \left[ \nabla \pi(a'|s')q_\pi(s',a') + \gamma \pi(a'|s') \sum_{s''}p(s''|s',a')\nabla v_\pi(s'') \right] \right]$$

After repeated unrolling:

$$\nabla v_\pi(s) = \sum_{x \in \mathcal{S}} \sum_{k=0}^{\infty} \gamma^k \Pr(s \to x,k,\pi) \sum_a \nabla \pi(a|x)q_\pi(x,a)$$

where $\Pr(s \to x,k,\pi)$ is the probability of transitioning from state $s$ to state $x$ in $k$ steps under policy $\pi$. The factor $\gamma^k$ appears because every recursive step through the Bellman equation contributes one more factor of $\gamma$.

It is then immediate that:

$$\nabla J(\theta) = \nabla v_\pi(s_0)$$

$$\nabla J(\theta) = \sum_s \left( \sum_{k=0}^{\infty} \gamma^k \Pr(s_0 \to s,k,\pi) \right) \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

Define:

$$\eta_\gamma(s) = \sum_{k=0}^{\infty} \gamma^k \Pr(s_0 \to s,k,\pi)$$

$\eta_\gamma(s)$ denotes the discounted expected number of times the agent visits state $s$ during an episode.

Then:

$$\nabla J(\theta) = \sum_s \eta_\gamma(s) \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

Now normalize $\eta_\gamma(s)$ into the discounted on-policy distribution $\mu_\gamma(s)$:

$$\mu_\gamma(s) = \frac{\eta_\gamma(s)} {\sum_{s'}\eta_\gamma(s')}$$

$\mu_\gamma(s)$ denotes the discounted fraction of time that the agent spends in state $s$ while following policy $\pi$.

So:

$$\nabla J(\theta) = \sum_{s'}\eta_\gamma(s') \sum_s \frac{\eta_\gamma(s)} {\sum_{s'}\eta_\gamma(s')} \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

$$\nabla J(\theta) = \sum_{s'}\eta_\gamma(s') \sum_s \mu_\gamma(s) \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

Since $\sum_{s'}\eta_\gamma(s')$ is a positive constant independent of $s$ and $a$, we have:

$$\nabla J(\theta) \propto \sum_s \mu_\gamma(s) \sum_a \nabla \pi(a|s)q_\pi(s,a)$$

Therefore:

$$\boxed{ \nabla J(\theta) \propto \sum_s \mu_\gamma(s) \sum_a q_\pi(s,a) \nabla \pi(a|s) }$$

Q.E.D.

#### Proof of the Policy Gradient Theorem (continuing case)

The proof of the policy gradient theorem for the continuing case begins similarly to the episodic case. Again we leave it implicit in all cases that $\pi$ is a function of $\boldsymbol{\theta}$ and that the gradients are with respect to $\boldsymbol{\theta}$. Recall that in the continuing case $J(\boldsymbol{\theta}) = r(\pi)$ and that $v_\pi$ and $q_\pi$ denote values with respect to the differential return. The gradient of the state-value function can be written, for any $s \in \mathcal{S}$, as

$$
\begin{align*}
\nabla v_\pi(s) &= \nabla \left[ \sum_a \pi(a|s)q_\pi(s,a) \right], \qquad \text{for all } s \in \mathcal{S} \\
&= \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s)\nabla q_\pi(s,a) \right] \qquad \text{(product rule of calculus)} \\
&= \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s)\nabla \sum_{s',r} p(s',r|s,a) \left( r - r(\boldsymbol{\theta}) + v_\pi(s') \right) \right] \\
&= \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s) \left[ -\nabla r(\boldsymbol{\theta}) + \sum_{s'} p(s'|s,a)\nabla v_\pi(s') \right] \right].
\end{align*}
$$

After rearranging terms, we obtain

$$
\nabla r(\boldsymbol{\theta}) = \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s) \sum_{s'} p(s'|s,a)\nabla v_\pi(s') \right] - \nabla v_\pi(s).
$$

Notice that the left-hand side can be written $\nabla J(\boldsymbol{\theta})$, and that it does not depend on $s$. Thus the right-hand side does not depend on $s$ either, and we can safely sum it over all $s \in \mathcal{S}$, weighted by $\mu(s)$, without changing it (because $\sum_s \mu(s) = 1$):

$$
\begin{align*}
\nabla J(\boldsymbol{\theta}) &= \sum_s \mu(s) \left( \sum_a \left[ \nabla \pi(a|s)q_\pi(s,a) + \pi(a|s) \sum_{s'} p(s'|s,a)\nabla v_\pi(s') \right] - \nabla v_\pi(s) \right) \\
&= \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a) \\
&\quad + \sum_s \mu(s) \sum_a \pi(a|s) \sum_{s'} p(s'|s,a)\nabla v_\pi(s') - \sum_s \mu(s)\nabla v_\pi(s) \\
&= \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a) \\
&\quad + \sum_{s'} \underbrace{\sum_s \mu(s) \sum_a \pi(a|s) p(s'|s,a)}_{\mu(s')} \nabla v_\pi(s') - \sum_s \mu(s)\nabla v_\pi(s) \\
&= \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a) + \sum_{s'} \mu(s') \nabla v_\pi(s') - \sum_s \mu(s) \nabla v_\pi(s) \\
&= \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a). \qquad \text{Q.E.D.}
\end{align*}
$$

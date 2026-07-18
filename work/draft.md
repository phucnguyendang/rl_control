#### Off-Policy Policy Gradient

Both REINFORCE and the vanilla version of actor-critic method are on-policy: training samples are collected according to the target policy — the very same policy that we try to optimize for. Off policy methods, however, result in several additional advantages:

1. The off-policy approach does not require full trajectories and can reuse any past episodes ("experience replay") for much better sample efficiency.
2. The sample collection follows a behavior policy different from the target policy, bringing better exploration.

Now let's see how off-policy policy gradient is computed. The behavior policy for collecting samples is a known policy (predefined just like a hyperparameter), labelled as $\beta(a \mid s)$. The objective function sums up the reward over the state distribution defined by this behavior policy:

$$
J(\theta)
=
\sum_{s \in S} d^{\beta}(s)
\sum_{a \in A}
Q^{\pi}(s,a)\pi_{\theta}(a \mid s)
=
\mathbb{E}_{s \sim d^{\beta}}
\left[
\sum_{a \in A}
Q^{\pi}(s,a)\pi_{\theta}(a \mid s)
\right]
$$

where $d^{\beta}(s)$ is the stationary distribution of the behavior policy $\beta$; recall that

$$
d^{\beta}(s)
=
\lim_{t \to \infty}
P(S_t = s \mid S_0, \beta);
$$

and $Q^{\pi}$ is the action-value function estimated with regard to the target policy $\pi$ (not the behavior policy!).

**Comment**: *This surrogate performance function above is quite silly. Maximizing the surrogate objective does not guarantee the maximization of the original performance objective or even its lower bound.*

Given that the training observations are sampled by $a \sim \beta(a \mid s)$, we can rewrite the gradient as:

$$
\begin{aligned}
\nabla_{\theta}J(\theta)
&=
\nabla_{\theta}
\mathbb{E}_{s \sim d^{\beta}}
\left[
\sum_{a \in A}
Q^{\pi}(s,a)\pi_{\theta}(a \mid s)
\right]
\\
&=
\mathbb{E}_{s \sim d^{\beta}}
\left[
\sum_{a \in A}
\left(
Q^{\pi}(s,a)\nabla_{\theta}\pi_{\theta}(a \mid s)
+
\pi_{\theta}(a \mid s)\nabla_{\theta}Q^{\pi}(s,a)
\right)
\right],
\end{aligned}
$$

Derivative product rule.

$$
\overset{(i)}{\approx}
\mathbb{E}_{s \sim d^{\beta}}
\left[
\sum_{a \in A}
Q^{\pi}(s,a)\nabla_{\theta}\pi_{\theta}(a \mid s)
\right]
$$

Ignore the red part:
$\pi_{\theta}(a \mid s)\nabla_{\theta}Q^{\pi}(s,a)$.

$$
=
\mathbb{E}_{s \sim d^{\beta}}
\left[
\sum_{a \in A}
\beta(a \mid s)
\frac{\pi_{\theta}(a \mid s)}{\beta(a \mid s)}
Q^{\pi}(s,a)
\frac{\nabla_{\theta}\pi_{\theta}(a \mid s)}
{\pi_{\theta}(a \mid s)}
\right]
$$

$$
=
\mathbb{E}_{\beta}
\left[
\frac{\pi_{\theta}(a \mid s)}
{\beta(a \mid s)}
Q^{\pi}(s,a)
\nabla_{\theta}\ln\pi_{\theta}(a \mid s)
\right]
$$

The blue part is the importance weight.

where

$$
\frac{\pi_{\theta}(a \mid s)}
{\beta(a \mid s)}
$$

is the **importance weight**. Because $Q^{\pi}$ is a function of the target policy and thus a function of policy parameter $\theta$, we should take the derivative of $\nabla_{\theta}Q^{\pi}(s,a)$ as well according to the product rule. However, it is super hard to compute $\nabla_{\theta}Q^{\pi}(s,a)$ in reality. Fortunately if we use an approximated gradient with the gradient of $Q$ ignored, we still guarantee the policy improvement and eventually achieve the true local minimum. This is justified in the proof here (Degris, White & Sutton, 2012).

In summary, when applying policy gradient in the off-policy setting, we can simple adjust it with a weighted sum and the weight is the ratio of the target policy to the behavior policy,

$$
\frac{\pi_{\theta}(a \mid s)}
{\beta(a \mid s)}.
$$


### Deadly triad 

**Deadly triad** là hiện tượng trong Reinforcement Learning khi ta kết hợp cùng lúc 3 thứ sau:

1. **Function approximation**: dùng mô hình xấp xỉ như linear function approximator hoặc neural network để học $V(s)$ hoặc $Q(s,a)$ thay vì bảng tabular.

2. **Bootstrapping**: target học có chứa chính estimate hiện tại/tương lai, ví dụ:

$$
y_t = r_t + \gamma V_\theta(s_{t+1})
$$

hoặc:

$$
y_t = r_t + \gamma \max_{a'} Q_\theta(s_{t+1}, a')
$$

Tức là ta không chờ full return thật như Monte Carlo, mà dùng một estimate khác để cập nhật estimate hiện tại.

3. **Off-policy training**: dữ liệu được sinh ra bởi một policy khác với policy đang được học/evaluate. Ví dụ replay buffer chứa data từ nhiều policy cũ, hoặc Q-learning học greedy target nhưng hành động thực tế được lấy từ $\epsilon$-greedy behavior policy.

Theo Sutton & Barto, nguy cơ **instability/divergence** xuất hiện khi ba thành phần này cùng có mặt; nếu chỉ có hai trong ba thì thường có thể tránh được bất ổn hơn. Họ cũng nhấn mạnh vấn đề không phải do control hay generalized policy iteration; ngay cả prediction thuần cũng có thể diverge khi có đủ ba thành phần này. 

Trực giác là thế này: với neural network, một update ở một vùng state-action có thể làm thay đổi giá trị ở nhiều vùng khác do generalization. Nếu target lại dùng chính $Q_\theta$ hoặc $V_\theta$ tương lai, thì lỗi estimate có thể tự truyền ngược về các state trước đó. Nếu dữ liệu lại off-policy, phân phối dữ liệu dùng để sửa lỗi không khớp với phân phối mà target policy thật sự đi qua. Ba thứ này tạo thành vòng feedback xấu:

$$
Q_\theta \text{ sai}
\rightarrow
\text{bootstrap target sai}
\rightarrow
Q_\theta \text{ học theo target sai}
\rightarrow
\text{policy/target chọn vùng bị overestimate}
\rightarrow
Q_\theta \text{ càng sai hơn}
$$

Một ví dụ rất gần với deep RL hiện đại là DDPG/TD3/SAC. Critic thường học bằng target kiểu:

$$
y = r + \gamma Q_{\bar\theta}(s', \pi_{\bar\phi}(s'))
$$

Nó có đủ ba yếu tố: neural network là function approximation, target có bootstrapping, replay buffer làm cho training off-policy. Vì vậy các thuật toán này không “thoát” khỏi deadly triad; chúng dùng nhiều kỹ thuật để **giảm độ nguy hiểm**: target network, double Q, clipped double Q, delayed policy update, entropy regularization, target policy smoothing, v.v. TD3 chẳng hạn nhấn mạnh rằng function approximation error trong TD learning có thể tích lũy, gây overestimation và làm actor học theo critic sai. 

Điểm quan trọng: **deadly triad không nói rằng cứ có ba thứ này là chắc chắn diverge**, mà nói rằng khi kết hợp ba thứ này thì không còn bảo đảm ổn định đơn giản như tabular/on-policy nữa, và có thể phát sinh divergence hoặc Q-value nổ nếu không có cơ chế ổn định.


### Algorithm 1: DDPG, with actor surrogate objective

**Initialize**

Randomly initialize critic and actor:

$$
Q(s,a \mid \theta^Q), 
\qquad 
\mu(s \mid \theta^\mu)
$$

Initialize target networks:

$$
\theta^{Q'} \leftarrow \theta^Q,
\qquad
\theta^{\mu'} \leftarrow \theta^\mu
$$

Trong đó:

- $Q'$ là target critic, tức bản copy chậm của critic $Q$.
- $\mu'$ là target actor, tức bản copy chậm của actor $\mu$.
- $Q'$ và $\mu'$ chỉ dùng để tính TD target ổn định hơn, không phải policy chính dùng để inference.

Initialize replay buffer:

$$
\mathcal R
$$

---

**For episode $=1,\dots,M$ do**

&nbsp;&nbsp;&nbsp;&nbsp;Initialize exploration noise process:

$$
\mathcal N
$$

&nbsp;&nbsp;&nbsp;&nbsp;Receive initial state:

$$
s_1
$$

&nbsp;&nbsp;&nbsp;&nbsp;**For $t=1,\dots,T$ do**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Select action using behavior policy:

$$
a_t = \mu(s_t \mid \theta^\mu) + \mathcal N_t
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ở đây behavior policy là:

$$
\beta_t(s) = \mu(s \mid \theta^\mu) + \mathcal N_t
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Execute action $a_t$, observe reward and next state:

$$
r_t, \ s_{t+1}
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Store transition in replay buffer:

$$
(s_t,a_t,r_t,s_{t+1}) \in \mathcal R
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sample a random minibatch of $N$ transitions from replay buffer:

$$
(s_i,a_i,r_i,s_{i+1}) \sim \mathcal R
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Lưu ý: bước này không dùng policy nào để sample action.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Nó chỉ lấy lại transition đã lưu trong $\mathcal R$.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Các transition này có thể đến từ nhiều behavior policies cũ:

$$
a_i \sim \beta_{\text{old}}(s_i)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute target value using target actor and target critic:

$$
y_i
=
r_i
+
\gamma Q'
\left(
s_{i+1},
\mu'(s_{i+1}\mid\theta^{\mu'})
\mid
\theta^{Q'}
\right)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update critic by minimizing:

$$
L(\theta^Q)
=
\frac{1}{N}
\sum_i
\left(
Q(s_i,a_i\mid\theta^Q) - y_i
\right)^2
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Define actor surrogate objective:

$$
J_{\text{actor}}(\theta^\mu)
=
\mathbb E_{s\sim \mathcal R}
\left[
Q(s,\mu(s\mid\theta^\mu)\mid\theta^Q)
\right]
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Minibatch approximation:

$$
J_{\text{actor}}(\theta^\mu)
\approx
\frac{1}{N}
\sum_i
Q(s_i,\mu(s_i\mid\theta^\mu)\mid\theta^Q)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Khi update actor, critic $Q(s,a\mid\theta^Q)$ được xem là cố định.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Actor chỉ thay đổi action $\mu(s_i\mid\theta^\mu)$ để làm $Q$ lớn hơn.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update actor using deterministic policy gradient:

$$
\nabla_{\theta^\mu}
Q(s_i,\mu(s_i\mid\theta^\mu)\mid\theta^Q)
=
\nabla_a Q(s_i,a\mid\theta^Q)
\bigg|_{a=\mu(s_i)}
\nabla_{\theta^\mu}\mu(s_i\mid\theta^\mu)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Therefore:

$$
\nabla_{\theta^\mu} J_{\text{actor}}
\approx
\frac{1}{N}
\sum_i
\nabla_a Q(s,a\mid\theta^Q)
\bigg|_{s=s_i,\ a=\mu(s_i)}
\nabla_{\theta^\mu}\mu(s\mid\theta^\mu)
\bigg|_{s=s_i}
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Trong code, nếu dùng gradient descent, ta thường minimize actor loss:

$$
L_{\text{actor}}(\theta^\mu)
=
-
\frac{1}{N}
\sum_i
Q(s_i,\mu(s_i\mid\theta^\mu)\mid\theta^Q)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Nghĩa là actor $\mu$ được update để chọn action làm $Q(s,a)$ lớn hơn,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;không phải để imitate target actor $\mu'$.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update target networks by Polyak averaging:

$$
\theta^{Q'}
\leftarrow
\tau \theta^Q + (1-\tau)\theta^{Q'}
$$

$$
\theta^{\mu'}
\leftarrow
\tau \theta^\mu + (1-\tau)\theta^{\mu'}
$$

&nbsp;&nbsp;&nbsp;&nbsp;**End for**

**End for**



#### DDPG đã làm gì để hạn chế deadly triad
- Sử dụng target policy và target 
- Thêm noise để khuyển khích việc khám phá xung quang action giúp gradient của Q theo action tính tốt



## Algorithm 1 TD3

<u>Initialize critic networks $Q_{\theta_1}, Q_{\theta_2}$, and actor network $\pi_\phi$</u>  
with random parameters $\theta_1, \theta_2, \phi$

<u>Initialize target networks $\theta'_1 \leftarrow \theta_1,\ \theta'_2 \leftarrow \theta_2,\ \phi' \leftarrow \phi$</u>

Initialize replay buffer $\mathcal{B}$

**for** $t = 1$ to $T$ **do**

&nbsp;&nbsp;&nbsp;&nbsp;Select action with exploration noise $a \sim \pi(s) + \epsilon,$  
&nbsp;&nbsp;&nbsp;&nbsp;$\epsilon \sim \mathcal{N}(0, \sigma)$ and observe reward $r$ and new state $s'$

&nbsp;&nbsp;&nbsp;&nbsp;Store transition tuple $(s, a, r, s')$ in $\mathcal{B}$

&nbsp;&nbsp;&nbsp;&nbsp;Sample mini-batch of $N$ transitions $(s, a, r, s')$ from $\mathcal{B}$

&nbsp;&nbsp;&nbsp;&nbsp;$\tilde{a} \leftarrow \pi_{\phi'}(s) + \epsilon,\quad \epsilon \sim \operatorname{clip}(\mathcal{N}(0, \tilde{\sigma}), -c, c)$
<span style="color:red">Target policy smoothing</span>

&nbsp;&nbsp;&nbsp;&nbsp;$y \leftarrow r + \gamma \min_{i=1,2} Q_{\theta'_i}(s', \tilde{a})$
<span style="color:red">Clipped Double Q-learning</span>

&nbsp;&nbsp;&nbsp;&nbsp;Update critics  
&nbsp;&nbsp;&nbsp;&nbsp;$\theta_i \leftarrow \min_{\theta_i} N^{-1} \sum (y - Q_{\theta_i}(s, a))^2$

&nbsp;&nbsp;&nbsp;&nbsp;<mark>**if** $t \bmod d$ **then**</mark>
<span style="color:red">Delayed update of target and policy networks</span>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update $\phi$ by the deterministic policy gradient:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$
\nabla_\phi J(\phi)
=
N^{-1}
\sum
\nabla_a Q_{\theta_1}(s, a)\big|_{a=\pi_\phi(s)}
\nabla_\phi \pi_\phi(s)
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update target networks:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$
\theta'_i \leftarrow \tau \theta_i + (1-\tau)\theta'_i
$$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
$$
\phi' \leftarrow \tau \phi + (1-\tau)\phi'
$$

&nbsp;&nbsp;&nbsp;&nbsp;**end if**

**end for**

**TD3 improve**: TD3 là 1 phiên bản cải thiện của DDPG bổ sung
- Clipped Double Q-learning để tránh overestimate Q
- Delayed policy update để critic bớt nhiễu trước khi học policy
- So với DDPG chỉ thêm noise vào action khi collect data thì TD3 thêm noise cả khi tính target critic nữa, giúp cho hàm target cũng được smooth.



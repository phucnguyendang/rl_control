
# SAC
## Deriving the Boltzmann Policy from the Maximum-Entropy Objective

Consider a fixed state $s$.

Assume that the soft action-value function $Q(s,a)$ is already known.

The maximum-entropy policy improvement objective at state $s$ is:

$$
\max_{\pi(\cdot|s)}
\mathbb{E}_{a\sim \pi(\cdot|s)}
\left[
Q(s,a)
\right]
+
\alpha H(\pi(\cdot|s))
$$

where $\alpha > 0$ controls the strength of entropy regularization.

The entropy of the policy is:

$$
H(\pi(\cdot|s))
=
-\sum_a \pi(a|s)\log \pi(a|s)
$$

Therefore, the objective becomes:

$$
\max_{\pi(\cdot|s)}
\sum_a \pi(a|s)Q(s,a)
-
\alpha
\sum_a \pi(a|s)\log \pi(a|s)
$$

subject to the probability constraint:

$$
\sum_a \pi(a|s)=1
$$

---

## Step 1: Form the Lagrangian

We introduce a Lagrange multiplier $\lambda$ for the normalization constraint:

$$
\mathcal{L}(\pi,\lambda)
=
\sum_a \pi(a|s)Q(s,a)
-
\alpha
\sum_a \pi(a|s)\log \pi(a|s)
+
\lambda
\left(
\sum_a \pi(a|s)-1
\right)
$$

---

## Step 2: Take the derivative with respect to $\pi(a|s)$

For each action $a$, differentiate the Lagrangian:

$$
\frac{\partial \mathcal{L}}{\partial \pi(a|s)}
=
Q(s,a)
-
\alpha
\left(
\log \pi(a|s)+1
\right)
+
\lambda
$$

At the optimum, this derivative must be zero:

$$
Q(s,a)
-
\alpha
\left(
\log \pi(a|s)+1
\right)
+
\lambda
=0
$$

---

## Step 3: Solve for $\log \pi(a|s)$

Rearrange the equation:

$$
\alpha
\left(
\log \pi(a|s)+1
\right)
=
Q(s,a)+\lambda
$$

$$
\log \pi(a|s)+1
=
\frac{Q(s,a)}{\alpha}
+
\frac{\lambda}{\alpha}
$$

$$
\log \pi(a|s)
=
\frac{Q(s,a)}{\alpha}
+
\frac{\lambda}{\alpha}
-
1
$$

The last two terms do not depend on action $a$, so we can write:

$$
\log \pi(a|s)
=
\frac{Q(s,a)}{\alpha}
+
C
$$

where $C$ is a state-dependent constant.

---

## Step 4: Exponentiate both sides

$$
\pi(a|s)
=
\exp
\left(
\frac{Q(s,a)}{\alpha}
+
C
\right)
$$

$$
\pi(a|s)
=
\exp(C)
\exp
\left(
\frac{Q(s,a)}{\alpha}
\right)
$$

Since $\exp(C)$ is only a normalization constant, we write:

$$
\pi(a|s)
\propto
\exp
\left(
\frac{Q(s,a)}{\alpha}
\right)
$$

---

## Step 5: Normalize the distribution

Because the policy must satisfy:

$$
\sum_a \pi(a|s)=1
$$

we obtain:

$$
\pi^*(a|s)
=
\frac{
\exp
\left(
\frac{Q(s,a)}{\alpha}
\right)
}{
\sum_{a'}
\exp
\left(
\frac{Q(s,a')}{\alpha}
\right)
}
$$

This is the Boltzmann, or Gibbs, distribution.

---

## Final Result

The optimal policy for the maximum-entropy objective is:

$$
\boxed{
\pi^*(a|s)
=
\frac{
\exp(Q(s,a)/\alpha)
}{
\sum_{a'} \exp(Q(s,a')/\alpha)
}
}
$$

Equivalently:

$$
\pi^*(a|s)
\propto
\exp(Q(s,a)/\alpha)
$$

Thus, adding entropy regularization turns the hard greedy policy into a soft greedy policy.

Without entropy, the optimal policy would put all probability mass on:

$$
a^* = \arg\max_a Q(s,a)
$$

With entropy, the optimal policy assigns higher probability to actions with larger $Q(s,a)$, but it does not collapse immediately to a deterministic action.

## From the SAC KL Objective to the Practical Actor Loss

In maximum-entropy policy improvement, for each state $s$, the ideal improved policy is:

$$
\pi^*(a|s)
=
\frac{\exp(Q_\phi(s,a)/\alpha)}
{Z_\phi(s)}
$$

where

$$
Z_\phi(s)
=
\int \exp(Q_\phi(s,a)/\alpha)\, da
$$

is the partition function.

Since the actor $\pi_\theta(a|s)$ usually cannot exactly represent $\pi^*(a|s)$, SAC projects the actor onto this ideal Boltzmann policy by minimizing the KL divergence:

$$
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot|s)
\;\middle\|\;
\frac{\exp(Q_\phi(s,\cdot)/\alpha)}
{Z_\phi(s)}
\right)
\right]
$$

---

## Step 1: Expand the KL divergence

By definition,

$$
D_{\mathrm{KL}}(p\|q)
=
\mathbb{E}_{a\sim p}
\left[
\log p(a) - \log q(a)
\right]
$$

Therefore,

$$
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D}
\left[
\mathbb{E}_{a\sim \pi_\theta(\cdot|s)}
\left[
\log \pi_\theta(a|s)
-
\log
\left(
\frac{\exp(Q_\phi(s,a)/\alpha)}
{Z_\phi(s)}
\right)
\right]
\right]
$$

---

## Step 2: Expand the logarithm of the Boltzmann policy

We have:

$$
\log
\left(
\frac{\exp(Q_\phi(s,a)/\alpha)}
{Z_\phi(s)}
\right)
=
\log \exp(Q_\phi(s,a)/\alpha)
-
\log Z_\phi(s)
$$

So:

$$
\log
\left(
\frac{\exp(Q_\phi(s,a)/\alpha)}
{Z_\phi(s)}
\right)
=
\frac{Q_\phi(s,a)}{\alpha}
-
\log Z_\phi(s)
$$

Substitute this into the KL objective:

$$
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D,\ a\sim \pi_\theta}
\left[
\log \pi_\theta(a|s)
-
\frac{Q_\phi(s,a)}{\alpha}
+
\log Z_\phi(s)
\right]
$$

---

## Step 3: Remove the partition function term

The term $\log Z_\phi(s)$ depends on $Q_\phi$ and $s$, but not on the actor parameters $\theta$.

Therefore, when optimizing the actor, it is a constant with respect to $\theta$.

So we can drop it:

$$
J_\pi(\theta)
\equiv
\mathbb{E}_{s\sim \mathcal D,\ a\sim \pi_\theta}
\left[
\log \pi_\theta(a|s)
-
\frac{Q_\phi(s,a)}{\alpha}
\right]
$$

where $\equiv$ means "equivalent for optimization with respect to $\theta$."

---

## Step 4: Multiply by $\alpha$

Multiplying the objective by a positive constant does not change the minimizer.

Thus, multiply by $\alpha$:

$$
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D,\ a\sim \pi_\theta}
\left[
\alpha \log \pi_\theta(a|s)
-
Q_\phi(s,a)
\right]
$$

This is the practical SAC actor loss.

---

## Final Practical Actor Loss

In code, SAC minimizes:

$$
\boxed{
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D,\ a\sim \pi_\theta(\cdot|s)}
\left[
\alpha \log \pi_\theta(a|s)
-
Q_\phi(s,a)
\right]
}
$$

Equivalently, SAC maximizes:

$$
\mathbb{E}_{s\sim \mathcal D,\ a\sim \pi_\theta}
\left[
Q_\phi(s,a)
-
\alpha \log \pi_\theta(a|s)
\right]
$$

---

## Step 5: Reparameterized form used in code

For continuous actions, SAC usually uses a reparameterized stochastic policy:

$$
a
=
f_\theta(s,\epsilon),
\qquad
\epsilon \sim \mathcal N(0,I)
$$

So instead of writing:

$$
a\sim \pi_\theta(\cdot|s)
$$

we write:

$$
a_\theta
=
f_\theta(s,\epsilon)
$$

The actor loss becomes:

$$
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D,\ \epsilon\sim \mathcal N(0,I)}
\left[
\alpha \log \pi_\theta(f_\theta(s,\epsilon)|s)
-
Q_\phi(s,f_\theta(s,\epsilon))
\right]
$$

This is what is implemented in practice.

---

## With twin Q-networks

Modern SAC uses two Q-networks to reduce overestimation bias:

$$
Q_{\min}(s,a)
=
\min
\left(
Q_{\phi_1}(s,a),
Q_{\phi_2}(s,a)
\right)
$$

So the practical actor loss is:

$$
\boxed{
J_\pi(\theta)
=
\mathbb{E}_{s\sim \mathcal D,\ \epsilon\sim \mathcal N(0,I)}
\left[
\alpha \log \pi_\theta(a_\theta|s)
-
Q_{\min}(s,a_\theta)
\right]
}
$$

where

$$
a_\theta = f_\theta(s,\epsilon)
$$

---

## Code-level version

A typical implementation looks like:


obs = replay_buffer.sample_obs(batch_size)

action, log_prob = actor.sample(obs)
q1 = critic1(obs, action)
q2 = critic2(obs, action)
q_min = torch.min(q1, q2)

actor_loss = (alpha * log_prob - q_min).mean()


# Bài giảng: Double Q-learning

Double Q-learning không chỉ là một “trick implementation”. Nó có cơ sở lý thuyết khá rõ: **Q-learning thường bị overestimation bias vì dùng cùng một estimate để vừa chọn action tốt nhất vừa đánh giá action đó**.

Double Q-learning giải quyết vấn đề này bằng cách tách hai vai trò:

$$
\text{selection}
\neq
\text{evaluation}
$$

Nói đơn giản:

$$
\boxed{
\text{Một Q dùng để chọn action, Q còn lại dùng để đánh giá action đó.}
}
$$

---

## 1. Vấn đề gốc: maximization bias

Trong Q-learning, target thường có dạng:

$$
y
=
r
+
\gamma \max_{a'} Q(s',a')
$$

Target này nhìn rất hợp lý. Nếu agent đến state $s'$, ta giả sử sau đó agent sẽ chọn action tốt nhất, nên lấy:

$$
\max_{a'} Q(s',a')
$$

Nhưng vấn đề là $Q(s',a')$ không phải true value. Nó chỉ là một estimate.

Ta có thể viết:

$$
Q(s',a)
=
q(s',a)
+
\epsilon_a
$$

trong đó:

- $q(s',a)$ là true action-value.
- $\epsilon_a$ là sai số ước lượng.

Giả sử tại state $s'$, mọi action đều có true value bằng $0$:

$$
q(s',a_1)
=
q(s',a_2)
=
\cdots
=
q(s',a_n)
=
0
$$

Nhưng estimate thì có nhiễu:

$$
Q(s',a_i)
=
\epsilon_i
$$

Có action bị estimate cao hơn true value, có action bị estimate thấp hơn true value.

Khi Q-learning lấy:

$$
\max_i Q(s',a_i)
=
\max_i \epsilon_i
$$

giá trị này thường là số dương, dù true max thực ra là:

$$
\max_i q(s',a_i)
=
0
$$

Đây chính là **maximization bias**.

---

## 2. Tại sao lấy max lại tạo overestimation?

Nếu $Q(s,a)$ là estimate có noise, thì:

$$
\mathbb{E}
\left[
\max_a Q(s,a)
\right]
\ge
\max_a
\mathbb{E}
\left[
Q(s,a)
\right]
$$

Lý do trực giác là: khi ta lấy maximum trên nhiều estimate nhiễu, ta thường chọn đúng estimate có noise dương lớn nhất.

Nói cách khác:

$$
\text{action được chọn không chỉ tốt vì true value cao,}
$$

mà còn có thể được chọn vì:

$$
\text{noise của nó tình cờ cao.}
$$

Nếu sau đó ta lại dùng chính giá trị bị noise cao đó để làm target, lỗi overestimation sẽ được bootstrap ngược về các state trước.

---

## 3. Sai ở đâu trong Q-learning?

Q-learning dùng cùng một $Q$ cho hai việc.

Thứ nhất, dùng $Q$ để chọn action:

$$
a^*
=
\arg\max_{a'} Q(s',a')
$$

Thứ hai, cũng dùng chính $Q$ để đánh giá action đó:

$$
\text{target}
=
r
+
\gamma Q(s',a^*)
$$

Tức là:

$$
\boxed{
\text{cùng một estimator vừa chọn action vừa chấm điểm action}
}
$$

Nếu $Q$ đánh giá quá cao một action do noise, action đó sẽ được chọn. Sau đó chính giá trị quá cao đó lại được đưa vào Bellman target:

$$
Q(s,a)
\leftarrow
r
+
\gamma \max_{a'} Q(s',a')
$$

Vì vậy lỗi overestimation không chỉ nằm ở $s'$. Nó có thể lan ngược về $s$, rồi tiếp tục lan về các state trước nữa.

---

## 4. Ý tưởng chính của Double Q-learning

Double Q-learning giữ hai action-value estimate:

$$
Q_1(s,a)
\qquad
\text{và}
\qquad
Q_2(s,a)
$$

Thay vì dùng cùng một $Q$ để chọn và đánh giá, ta tách hai việc này ra.

Khi update $Q_1$, ta dùng $Q_1$ để chọn action:

$$
a^*
=
\arg\max_{a'} Q_1(s',a')
$$

nhưng dùng $Q_2$ để đánh giá action đó:

$$
y
=
r
+
\gamma Q_2(s',a^*)
$$

Do đó update của $Q_1$ là:

$$
Q_1(s,a)
\leftarrow
Q_1(s,a)
+
\alpha
\left[
r
+
\gamma Q_2
\left(
s',
\arg\max_{a'} Q_1(s',a')
\right)
-
Q_1(s,a)
\right]
$$

Ngược lại, khi update $Q_2$, ta dùng $Q_2$ để chọn action:

$$
a^*
=
\arg\max_{a'} Q_2(s',a')
$$

nhưng dùng $Q_1$ để đánh giá:

$$
Q_2(s,a)
\leftarrow
Q_2(s,a)
+
\alpha
\left[
r
+
\gamma Q_1
\left(
s',
\arg\max_{a'} Q_2(s',a')
\right)
-
Q_2(s,a)
\right]
$$

Cốt lõi là:

$$
\boxed{
\text{Q chọn action không phải Q đánh giá action đó.}
}
$$

---

## 5. Cơ sở lý thuyết: tại sao Double Q-learning giảm bias?

Xét một bài toán đơn giản với true action values:

$$
q(a)
$$

Giả sử ta có hai estimate độc lập hơn:

$$
Q_1(a)
\qquad
\text{và}
\qquad
Q_2(a)
$$

với:

$$
\mathbb{E}[Q_1(a)]
=
q(a)
$$

và:

$$
\mathbb{E}[Q_2(a)]
=
q(a)
$$

Dùng $Q_1$ để chọn action:

$$
A^*
=
\arg\max_a Q_1(a)
$$

Nếu ta đánh giá action này bằng chính $Q_1$, ta dùng:

$$
Q_1(A^*)
$$

Giá trị này dễ bị overestimate vì $A^*$ được chọn chính vì $Q_1(A^*)$ lớn nhất.

Nhưng nếu ta đánh giá bằng $Q_2$, ta dùng:

$$
Q_2(A^*)
$$

Vì $Q_2$ không trực tiếp tham gia vào quá trình chọn $A^*$, nên noise dương trong $Q_1$ không tự động đi vào giá trị đánh giá.

Cụ thể, nếu $Q_2$ là unbiased estimator, thì với một action cụ thể $a$:

$$
\mathbb{E}
[
Q_2(A^*) \mid A^* = a
]
=
q(a)
$$

Do đó:

$$
\mathbb{E}
[
Q_2(A^*)
]
=
\mathbb{E}
[
q(A^*)
]
$$

Nghĩa là $Q_2(A^*)$ không bị thổi phồng bởi chính noise đã khiến $Q_1$ chọn $A^*$.

---

## 6. Điểm tinh tế: Double Q-learning không làm target hoàn hảo

Ta có:

$$
\mathbb{E}
[
Q_2(A^*)
]
=
\mathbb{E}
[
q(A^*)
]
$$

Nhưng:

$$
\mathbb{E}
[
q(A^*)
]
\le
\max_a q(a)
$$

Điều này nghĩa là Double Q-learning không đảm bảo mỗi target là estimate không chệch của optimal value:

$$
\max_a q(a)
$$

Nó chỉ đảm bảo rằng ta không tự thưởng thêm noise dương của estimator đã chọn action.

Nói cách khác, Double Q-learning đổi từ:

$$
\text{systematic overestimation}
$$

sang:

$$
\text{có thể chọn sai action do noise, nhưng không tự cộng thêm noise dương}
$$

Đây là lý do Double Q-learning thường giảm overestimation bias.

---

## 7. Thuật toán tabular Double Q-learning

Ta duy trì hai bảng:

$$
Q_1
\qquad
\text{và}
\qquad
Q_2
$$

Behavior policy thường chọn action theo $\epsilon$-greedy dựa trên tổng hoặc trung bình hai bảng:

$$
Q_{\text{behavior}}(s,a)
=
Q_1(s,a)
+
Q_2(s,a)
$$

hoặc:

$$
Q_{\text{behavior}}(s,a)
=
\frac{1}{2}
\left(
Q_1(s,a)
+
Q_2(s,a)
\right)
$$

Mỗi lần nhận transition:

$$
(s,a,r,s')
$$

ta tung coin để quyết định update $Q_1$ hay $Q_2$.

### Trường hợp 1: update $Q_1$

Chọn action tốt nhất theo $Q_1$:

$$
a^*
=
\arg\max_{a'} Q_1(s',a')
$$

Tạo target bằng $Q_2$:

$$
y
=
r
+
\gamma Q_2(s',a^*)
$$

Update:

$$
Q_1(s,a)
\leftarrow
Q_1(s,a)
+
\alpha
[
y
-
Q_1(s,a)
]
$$

hay viết đầy đủ:

$$
Q_1(s,a)
\leftarrow
Q_1(s,a)
+
\alpha
\left[
r
+
\gamma Q_2(s',a^*)
-
Q_1(s,a)
\right]
$$

### Trường hợp 2: update $Q_2$

Chọn action tốt nhất theo $Q_2$:

$$
a^*
=
\arg\max_{a'} Q_2(s',a')
$$

Tạo target bằng $Q_1$:

$$
y
=
r
+
\gamma Q_1(s',a^*)
$$

Update:

$$
Q_2(s,a)
\leftarrow
Q_2(s,a)
+
\alpha
[
y
-
Q_2(s,a)
]
$$

hay viết đầy đủ:

$$
Q_2(s,a)
\leftarrow
Q_2(s,a)
+
\alpha
\left[
r
+
\gamma Q_1(s',a^*)
-
Q_2(s,a)
\right]
$$

---

## 8. Pseudocode

~~~text
Initialize Q1(s,a) and Q2(s,a)

For each episode:
    Initialize state s

    For each step:
        Choose action a using epsilon-greedy policy based on Q1(s,a) + Q2(s,a)

        Take action a
        Observe reward r and next state s'

        With probability 0.5:
            a_star = argmax_a' Q1(s', a')
            target = r + gamma * Q2(s', a_star)
            Q1(s,a) = Q1(s,a) + alpha * (target - Q1(s,a))

        Otherwise:
            a_star = argmax_a' Q2(s', a')
            target = r + gamma * Q1(s', a_star)
            Q2(s,a) = Q2(s,a) + alpha * (target - Q2(s,a))

        s = s'
~~~

---

## 9. Ví dụ trực giác

Giả sử ở state $s'$ có 4 action. True value của tất cả action đều bằng $0$:

$$
q(a_1)
=
q(a_2)
=
q(a_3)
=
q(a_4)
=
0
$$

Nhưng estimate của $Q_1$ hiện tại là:

$$
Q_1(a_1)
=
0.1
$$

$$
Q_1(a_2)
=
0.8
$$

$$
Q_1(a_3)
=
-0.2
$$

$$
Q_1(a_4)
=
0.3
$$

Q-learning thông thường sẽ lấy:

$$
\max_a Q_1(a)
=
0.8
$$

Target bị đẩy lên bởi $0.8$, dù true value thật là $0$.

Double Q-learning thì dùng $Q_1$ để chọn:

$$
a^*
=
a_2
$$

nhưng không dùng $Q_1(a_2)$ để đánh giá. Nó dùng:

$$
Q_2(a_2)
$$

Nếu:

$$
Q_2(a_2)
=
0.05
$$

thì target chỉ bị cộng thêm $0.05$, hợp lý hơn nhiều so với $0.8$.

Điều quan trọng là: noise khiến $Q_1(a_2)$ cao không nhất thiết khiến $Q_2(a_2)$ cũng cao.

---

## 10. Double Q-learning và Double DQN

Double Q-learning gốc dùng hai estimator tương đối đối xứng:

$$
Q_1
\qquad
\text{và}
\qquad
Q_2
$$

Trong Double DQN, ta thường có:

$$
Q_{\text{online}}
$$

và:

$$
Q_{\text{target}}
$$

Online network dùng để chọn action:

$$
a^*
=
\arg\max_{a'} Q_{\text{online}}(s',a')
$$

Target network dùng để đánh giá action đó:

$$
y
=
r
+
\gamma Q_{\text{target}}(s',a^*)
$$

Tức là Double DQN vẫn giữ cùng tinh thần:

$$
\boxed{
\text{selection network}
\neq
\text{evaluation network}
}
$$

Nhưng trong deep RL, hai network này không hoàn toàn độc lập. Target network thường là bản copy chậm của online network. Vì vậy Double DQN chỉ giảm overestimation bias, chứ không loại bỏ hoàn toàn.

---

## 11. Double Q-learning trong TD3 và SAC

Trong continuous control, ta thường không thể tính:

$$
\arg\max_a Q(s,a)
$$

trên toàn bộ action space.

Thay vào đó, ta có actor:

$$
a
=
\pi_\theta(s)
$$

Actor học cách chọn action làm critic đánh giá cao.

Critic target thường có dạng:

$$
y
=
r
+
\gamma Q_{\bar{\phi}}
(
s',
\pi_{\bar{\theta}}(s')
)
$$

Vấn đề vẫn tương tự Q-learning: nếu critic có vùng overestimate, actor có thể học chui vào vùng đó.

TD3 và SAC dùng hai critic:

$$
Q_1(s,a)
\qquad
\text{và}
\qquad
Q_2(s,a)
$$

Sau đó dùng biến thể gọi là **Clipped Double Q-learning**:

$$
y
=
r
+
\gamma
\min_{i=1,2}
Q_{\bar{\phi}_i}
(
s',
\pi_{\bar{\theta}}(s')
)
$$

hoặc trong SAC:

$$
y
=
r
+
\gamma
\left[
\min_{i=1,2}
Q_{\bar{\phi}_i}
(s',a')
-
\alpha \log \pi_\theta(a'|s')
\right]
$$

với:

$$
a'
\sim
\pi_\theta(\cdot|s')
$$

Điểm khác biệt quan trọng:

$$
\text{Double Q-learning gốc}
\neq
\min(Q_1,Q_2)
$$

Double Q-learning gốc là:

$$
\text{một estimator chọn action, estimator còn lại đánh giá action}
$$

Clipped Double Q-learning là:

$$
\text{lấy giá trị nhỏ hơn giữa hai critic}
$$

Clipped Double Q-learning bi quan hơn. Nó chấp nhận có thể underestimate một chút để tránh overestimate, vì overestimate thường nguy hiểm hơn trong actor-critic.

---

## 12. Tại sao overestimation nguy hiểm hơn underestimation?

Trong actor-critic, actor update theo hướng làm tăng critic:

$$
\nabla_\theta J(\theta)
\approx
\nabla_a Q_\phi(s,a)
\vert_{a=\pi_\theta(s)}
\nabla_\theta \pi_\theta(s)
$$

Nếu critic overestimate một vùng action nào đó, actor sẽ bị kéo vào vùng đó.

Sau đó dữ liệu mới lại được sinh ra bởi actor đã bị kéo sai. Critic tiếp tục học trên dữ liệu bị lệch. Lỗi có thể tự khuếch đại.

Ngược lại, nếu một action bị underestimate, actor thường tránh action đó. Lỗi underestimate ít có cơ hội được khai thác và lan rộng hơn.

Vì vậy trong TD3/SAC, người ta thường chấp nhận:

$$
\text{slight underestimation}
$$

để tránh:

$$
\text{dangerous overestimation}
$$

---

## 13. Double Q-learning có giải quyết triệt để không?

Không.

Double Q-learning có cơ sở lý thuyết, nhưng cơ sở đó dựa trên các giả định lý tưởng:

$$
Q_1
\text{ và }
Q_2
\text{ đủ độc lập}
$$

$$
Q_1
\text{ và }
Q_2
\text{ là unbiased estimators}
$$

$$
\text{mỗi action được estimate đủ tốt}
$$

Trong deep RL, các giả định này không hoàn toàn đúng.

Hai critic thường:

- dùng cùng replay buffer,
- có architecture giống nhau,
- dùng cùng optimizer,
- học từ target có liên quan với nhau,
- thấy cùng distribution dữ liệu.

Do đó sai số của chúng có thể correlated.

Vì vậy Double Q-learning không đảm bảo critic đúng tuyệt đối. Nó chỉ giảm một nguồn bias rất nguy hiểm:

$$
\boxed{
\text{bias do dùng cùng một estimate để chọn và đánh giá action}
}
$$

---

## 14. Tổng kết

Q-learning thông thường dùng:

$$
y
=
r
+
\gamma \max_{a'} Q(s',a')
$$

Vấn đề là:

$$
\mathbb{E}
[
\max_a Q(s,a)
]
\ge
\max_a
\mathbb{E}
[
Q(s,a)
]
$$

Do đó Q-learning dễ bị overestimation bias.

Double Q-learning tách selection và evaluation:

$$
a^*
=
\arg\max_{a'} Q_1(s',a')
$$

$$
y
=
r
+
\gamma Q_2(s',a^*)
$$

hoặc ngược lại:

$$
a^*
=
\arg\max_{a'} Q_2(s',a')
$$

$$
y
=
r
+
\gamma Q_1(s',a^*)
$$

Cốt lõi của Double Q-learning là:

$$
\boxed{
\text{Đừng dùng cùng một estimate để vừa chọn action tốt nhất vừa đánh giá action đó.}
}
$$

Trong tabular RL, đây là một ý tưởng rất sạch.

Trong deep RL, nó trở thành một nhóm kỹ thuật ổn định hóa critic:

- Double Q-learning,
- Double DQN,
- twin critics,
- clipped Double Q-learning trong TD3,
- clipped Double Q-learning trong SAC.

Bản chất chung vẫn là chống lại việc Bellman backup hoặc actor khai thác sai số dương của critic.

## 15. Một câu nhớ nhanh

$$
\boxed{
\text{Q-learning overestimates because max selects noise.}
}
$$

$$
\boxed{
\text{Double Q-learning reduces this by separating selection from evaluation.}
}
$$
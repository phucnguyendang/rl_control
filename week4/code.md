# CartPole physical dynamics

Ta mô hình hóa hệ cartpole không ma sát, gồm:

- Cart có khối lượng $M$ và vị trí ngang $x$.
- Pole là một thanh đồng chất có khối lượng $m$ và chiều dài thật $L$.
- Tâm khối lượng của pole cách khớp quay một đoạn $r = \frac{L}{2}$.
- Lực điều khiển tác dụng lên cart là $u$.
- Góc $\theta = 0$ nghĩa là pole thẳng đứng hướng lên. $\theta > 0$ là quay theo chiều kim đồng hồ trong mặt phẳng $x-y$.

State của hệ:

$$
s = \begin{bmatrix}x \\ \dot{x} \\ \theta \\ \dot{\theta}\end{bmatrix}
= \begin{bmatrix}x \\ v \\ \theta \\ \omega\end{bmatrix}
$$

Vì pole là thanh đồng chất, moment quán tính quanh tâm khối lượng là:

$$
I_{cm}=\frac{1}{12}mL^2
$$

Moment quán tính quanh khớp quay là:

$$
J = I_{cm} + mr^2 = \frac{1}{3}mL^2
$$

Hai phương trình động lực học liên tục là:

$$
(M+m)\ddot{x} + mr\cos\theta\,\ddot{\theta} - mr\sin\theta\,\dot{\theta}^2 = u
$$

$$
J\ddot{\theta} + mr\cos\theta\,\ddot{x} - mgr\sin\theta = 0
$$

Giải hệ trên theo $\ddot{x}$ và $\ddot{\theta}$:

$$
\Delta = (M+m)J - (mr\cos\theta)^2
$$

$$
\ddot{x} = \frac{J(u + mr\sin\theta\,\omega^2) - (mr\cos\theta)(mgr\sin\theta)}{\Delta}
$$

$$
\ddot{\theta} = \frac{(M+m)(mgr\sin\theta) - (mr\cos\theta)(u + mr\sin\theta\,\omega^2)}{\Delta}
$$

Do đó dạng state-space liên tục là:

$$
\dot{s} = f(s,u) =
\begin{bmatrix}
v \\
\ddot{x} \\
\omega \\
\ddot{\theta}
\end{bmatrix}
$$

Trong code, `cartpole_dynamics(state, force, params)` chính là hàm $f(s,u)$. Hàm này chỉ tính đạo hàm tức thời của state, tức là hệ đang đổi như thế nào tại state hiện tại.

Để mô phỏng rời rạc, ta cần biến phương trình liên tục $\dot{s}=f(s,u)$ thành cập nhật từng bước:

$$
s_t \rightarrow s_{t+1}
$$

Trong chương trình, `rk4_step` dùng Runge-Kutta bậc 4 để cập nhật state:

$$
s_{t+1} = s_t + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

với:

$$
\begin{aligned}
k_1 &= f(s_t,u_t) \\
k_2 &= f(s_t + \frac{\Delta t}{2}k_1,u_t) \\
k_3 &= f(s_t + \frac{\Delta t}{2}k_2,u_t) \\
k_4 &= f(s_t + \Delta t k_3,u_t)
\end{aligned}
$$

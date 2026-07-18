

## Trust Region Policy Optimization

Sau GAE, ta đã có một cách tốt hơn để estimate advantage:

$$
\hat{A}_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^V.
$$

GAE trả lời câu hỏi:

$$
\boxed{
\text{Action vừa sample tốt hay xấu đến mức nào?}
}
$$

Nhưng policy optimization vẫn còn một câu hỏi khác quan trọng không kém:

$$
\boxed{
\text{Sau khi biết action nào tốt/xấu, ta nên thay đổi policy mạnh đến đâu?}
}
$$

Đây chính là câu hỏi mà TRPO trả lời.

GAE xử lý phần **estimate direction**: nên tăng hay giảm xác suất của action nào.

TRPO xử lý phần **safe update**: tăng/giảm bao nhiêu để policy không bị phá hỏng sau một update.

Trong phần GAE ở trên, bạn đã kết thúc bằng ý rằng GAE tạo advantage signal rồi đưa vào TRPO policy update; paper GAE cũng dùng pipeline này: collect trajectories, tính TD residual, tính GAE, dùng $\hat A_t$ trong TRPO update, rồi mới update value function. 

---

### Why vanilla policy gradient can be unstable

Trong policy gradient thông thường, ta có gradient estimator:

$$
\hat{g}
=
\hat{\mathbb{E}}_t
\left[
\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t
\right].
$$

Nếu $\hat{A}_t > 0$, ta tăng xác suất action $a_t$.

Nếu $\hat{A}_t < 0$, ta giảm xác suất action $a_t$.

Nhìn rất hợp lý. Nhưng vấn đề nằm ở **step size**.

Nếu bước update quá nhỏ, học rất chậm.

Nếu bước update quá lớn, policy mới có thể khác policy cũ quá nhiều. Khi đó batch data vừa collect bằng policy cũ không còn đại diện tốt cho policy mới nữa.

Ta có thể hình dung như sau.

Ở iteration hiện tại, policy cũ là:

$$
\pi_{\theta_{\text{old}}}.
$$

Ta dùng policy này để collect trajectories:

$$
(s_0,a_0,r_0), (s_1,a_1,r_1), \ldots
$$

Sau đó ta tính advantage:

$$
\hat{A}_t.
$$

Nếu update quá mạnh, policy mới:

$$
\pi_\theta
$$

có thể chọn action rất khác, đi vào vùng state rất khác, gặp reward rất khác. Nhưng gradient estimate của ta vẫn được tính từ data của policy cũ.

Vậy vấn đề không chỉ là “gradient noisy”.

Vấn đề là:

$$
\boxed{
\text{Policy update quá lớn làm data cũ trở nên không còn đáng tin.}
}
$$

Nói cách khác, policy gradient là local information. Nó chỉ đáng tin gần policy hiện tại. Nếu ta đi quá xa, local approximation có thể sai hoàn toàn.

TRPO sinh ra từ trực giác này:

> Khi update policy, đừng chỉ hỏi update direction có tốt không. Hãy hỏi policy mới có còn đủ gần policy cũ không.

Paper TRPO mô tả thuật toán như một procedure tối ưu policy có xu hướng monotonic improvement, gần với natural policy gradient, và thực nghiệm cho thấy hoạt động tốt với policy phi tuyến lớn như neural networks. 

---

### The core idea: improve policy, but stay inside a trust region

Tên TRPO là:

$$
\text{Trust Region Policy Optimization}.
$$

“Trust region” nghĩa là vùng mà trong đó approximation của ta còn đáng tin.

Trong supervised learning, ta có thể dùng learning rate nhỏ và SGD nhiều lần vì data distribution cố định.

Nhưng trong RL, data distribution phụ thuộc vào policy. Khi policy thay đổi, distribution của states cũng thay đổi:

$$
\pi
\Rightarrow
\rho_\pi(s).
$$

Do đó update policy quá mạnh sẽ làm thay đổi cả:

$$
\text{action distribution}
$$

lẫn:

$$
\text{state visitation distribution}.
$$

TRPO nói rằng:

$$
\boxed{
\text{Ta chỉ tin surrogate objective trong vùng policy mới gần policy cũ.}
}
$$

Vậy thay vì solve:

$$
\max_\theta
\quad
\text{estimated policy improvement},
$$

TRPO solve:

$$
\max_\theta
\quad
\text{estimated policy improvement}
$$

subject to:

$$
\text{policy mới không quá xa policy cũ}.
$$

Khoảng cách giữa policy mới và policy cũ được đo bằng KL divergence.

---

### Performance difference lemma: where TRPO starts

TRPO bắt đầu từ một identity rất quan trọng.

Gọi:

$$
\eta(\pi)
$$

là expected discounted return của policy $\pi$.

Với policy cũ $\pi$ và policy mới $\tilde{\pi}$, ta có:

$$
\eta(\tilde{\pi})
=
\eta(\pi)
+
\mathbb{E}_{\tilde{\pi}}
\left[
\sum_{t=0}^{\infty}
\gamma^t
A^\pi(s_t,a_t)
\right].
$$

Công thức này nói rằng:

> Performance của policy mới bằng performance của policy cũ cộng với tổng advantage của các action mà policy mới chọn, nhưng advantage được đánh giá dưới policy cũ.

Viết theo state visitation distribution:

$$
\eta(\tilde{\pi})
=
\eta(\pi)
+
\sum_s
\rho_{\tilde{\pi}}(s)
\sum_a
\tilde{\pi}(a|s)
A^\pi(s,a).
$$

Ý nghĩa rất rõ:

Nếu ở những state mà policy mới đi qua, policy mới chọn action có expected advantage dương so với policy cũ, thì policy mới tốt hơn.

Nếu:

$$
\sum_a
\tilde{\pi}(a|s)A^\pi(s,a)
\ge 0
$$

với mọi state $s$, thì policy mới không tệ hơn policy cũ.

Đây chính là policy improvement ở dạng advantage.

Nhưng vấn đề là biểu thức này phụ thuộc vào:

$$
\rho_{\tilde{\pi}}(s).
$$

Tức là nó phụ thuộc vào state distribution của policy mới. Mà state distribution của policy mới lại phụ thuộc vào toàn bộ dynamics tương lai sau khi update policy.

Vì vậy, biểu thức đúng thì đẹp, nhưng khó optimize trực tiếp.

TRPO thay nó bằng một local approximation.

Paper TRPO trình bày identity này rồi thay distribution $\rho_{\tilde{\pi}}$ khó xử lý bằng $\rho_\pi$ của policy cũ để tạo local surrogate objective. 

---

### Local surrogate objective

TRPO định nghĩa surrogate objective:

$$
L_\pi(\tilde{\pi})
=
\eta(\pi)
+
\sum_s
\rho_\pi(s)
\sum_a
\tilde{\pi}(a|s)
A^\pi(s,a).
$$

So với objective thật:

$$
\eta(\tilde{\pi})
=
\eta(\pi)
+
\sum_s
\rho_{\tilde{\pi}}(s)
\sum_a
\tilde{\pi}(a|s)
A^\pi(s,a),
$$

surrogate thay:

$$
\rho_{\tilde{\pi}}(s)
$$

bằng:

$$
\rho_\pi(s).
$$

Tức là nó giả sử state distribution không thay đổi nhiều sau update.

Đây là approximation rất quan trọng.

Nếu policy mới gần policy cũ, thì:

$$
\rho_{\tilde{\pi}}(s)
\approx
\rho_\pi(s).
$$

Khi đó surrogate tương đối đáng tin.

Nhưng nếu policy mới khác policy cũ quá xa, approximation này có thể sai nặng.

Vậy ta có insight trung tâm:

$$
\boxed{
\text{Surrogate objective chỉ đáng tin khi policy update đủ nhỏ.}
}
$$

Đó là lý do TRPO cần trust region.

---

### Sample-based surrogate objective

Trong thực tế, ta không có tổng trên toàn bộ state/action. Ta chỉ có batch trajectories collect từ policy cũ:

$$
\pi_{\theta_{\text{old}}}.
$$

Ta muốn đánh giá policy mới:

$$
\pi_\theta.
$$

Vì data được sample từ policy cũ, nhưng objective lại nói về policy mới, TRPO dùng importance ratio:

$$
r_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}.
$$

Surrogate objective dạng sample là:

$$
L_{\theta_{\text{old}}}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
\hat{A}_t
\right].
$$

Hay viết ngắn:

$$
L_{\theta_{\text{old}}}(\theta)
=
\hat{\mathbb{E}}_t
\left[
r_t(\theta)\hat{A}_t
\right].
$$

Đây là objective rất quan trọng vì nó là cầu nối từ policy gradient sang TRPO/PPO.

Ý nghĩa của ratio:

Nếu:

$$
r_t(\theta) > 1,
$$

policy mới tăng xác suất action $a_t$ so với policy cũ.

Nếu:

$$
r_t(\theta) < 1,
$$

policy mới giảm xác suất action $a_t$ so với policy cũ.

Vậy từng term:

$$
r_t(\theta)\hat{A}_t
$$

có ý nghĩa như sau.

Nếu:

$$
\hat{A}_t > 0,
$$

thì objective muốn tăng $r_t(\theta)$, tức là tăng xác suất action tốt.

Nếu:

$$
\hat{A}_t < 0,
$$

thì objective muốn giảm $r_t(\theta)$, tức là giảm xác suất action xấu.

Do đó surrogate objective vẫn giữ đúng tinh thần policy gradient:

$$
\boxed{
\text{Increase probability of good sampled actions, decrease probability of bad sampled actions.}
}
$$

Nhưng nếu chỉ maximize objective này mà không có ràng buộc, policy có thể thay đổi quá mạnh.

Ví dụ, với action có advantage dương, objective luôn thích tăng xác suất action đó nhiều hơn nữa. Với action có advantage âm, objective luôn thích giảm xác suất action đó nhiều hơn nữa. Nếu để optimizer tự do, nó có thể overfit batch data và phá policy.

Vì vậy TRPO thêm ràng buộc KL.

---

### KL constraint

TRPO solve bài toán:

$$
\max_\theta
\quad
\hat{\mathbb{E}}_t
\left[
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
\hat{A}_t
\right]
$$

subject to:

$$
\hat{\mathbb{E}}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot|s_t)
\,\Vert\,
\pi_\theta(\cdot|s_t)
\right)
\right]
\le
\delta.
$$

Trong đó:

$$
\delta
$$

là trust-region size.

Nếu $\delta$ nhỏ, policy mới phải rất gần policy cũ. Update an toàn hơn nhưng học chậm hơn.

Nếu $\delta$ lớn, policy được phép thay đổi mạnh hơn. Học có thể nhanh hơn nhưng dễ unstable hơn.

Điểm quan trọng: TRPO không giới hạn khoảng cách giữa parameter vector theo Euclidean distance:

$$
|\theta - \theta_{\text{old}}|_2.
$$

Nó giới hạn khoảng cách giữa hai **policy distributions**:

$$
\pi_{\theta_{\text{old}}}(\cdot|s)
\quad
\text{và}
\quad
\pi_\theta(\cdot|s).
$$

Đây là khác biệt rất quan trọng.

Trong neural network policy, một thay đổi nhỏ trong parameter có thể làm output distribution thay đổi lớn ở một số state. Ngược lại, một thay đổi parameter lớn đôi khi lại chỉ làm policy distribution thay đổi nhỏ. Vì vậy đo khoảng cách trong parameter space không phản ánh đúng thay đổi hành vi.

TRPO đo trực tiếp:

$$
\boxed{
\text{Policy mới hành xử khác policy cũ đến mức nào?}
}
$$

Trong GAE paper, phần policy optimization cũng viết TRPO dưới dạng constrained optimization với surrogate dùng probability ratio và ràng buộc average KL, sau đó giải gần đúng bằng linearizing objective và quadraticizing constraint. 

---

### Why KL divergence?

KL divergence đo độ khác nhau giữa hai phân phối xác suất.

Với mỗi state $s$, ta có hai phân phối action:

$$
\pi_{\theta_{\text{old}}}(\cdot|s)
$$

và:

$$
\pi_\theta(\cdot|s).
$$

KL divergence:

$$
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot|s)
\,\Vert\,
\pi_\theta(\cdot|s)
\right)
$$

đo xem policy mới khác policy cũ bao nhiêu tại state đó.

Ràng buộc average KL:

$$
\hat{\mathbb{E}}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot|s_t)
\,\Vert\,
\pi_\theta(\cdot|s_t)
\right)
\right]
\le
\delta
$$

nói rằng trung bình trên các state trong batch, policy mới không được đi quá xa policy cũ.

Với discrete action, KL đo sự thay đổi trong categorical distribution.

Với continuous action Gaussian, KL đo sự thay đổi của mean và variance. Nếu policy là:

$$
\pi_\theta(a|s)
=
\mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2),
$$

thì KL sẽ phạt việc mean dịch quá xa hoặc variance thay đổi quá mạnh.

Vậy KL constraint không chỉ ngăn một action probability cụ thể bị thay đổi quá mạnh; nó ngăn toàn bộ action distribution bị kéo đi quá xa.

---

### Monotonic improvement bound

Lý thuyết của TRPO cho một lower bound dạng:

$$
\eta(\tilde{\pi})
\ge
L_\pi(\tilde{\pi})
-
C
D^{\max}_{\mathrm{KL}}(\pi,\tilde{\pi}).
$$

Trong đó:

$$
L_\pi(\tilde{\pi})
$$

là surrogate objective, còn:

$$
D^{\max}_{\mathrm{KL}}(\pi,\tilde{\pi})
$$

là maximum KL divergence giữa policy cũ và policy mới trên tất cả state.

Ý nghĩa:

$$
\text{true performance}
\ge
\text{surrogate improvement}
-
\text{penalty for changing policy too much}.
$$

Nếu ta tăng surrogate objective nhưng policy thay đổi quá xa, penalty lớn, lower bound có thể không tăng.

Nếu ta tăng surrogate objective trong khi giữ KL nhỏ, lower bound tăng.

Vậy TRPO có một nguyên tắc rất đẹp:

$$
\boxed{
\text{Improve the surrogate, but keep policy change small enough.}
}
$$

Về mặt lý thuyết, nếu mỗi iteration maximize lower bound này chính xác, ta có monotonic improvement.

Nhưng trong thực tế, có vài approximation:

* advantage không biết thật, phải estimate bằng $\hat{A}_t$;
* state space lớn, không thể lấy max KL trên mọi state;
* neural network optimization không solve chính xác;
* surrogate chỉ estimated từ finite batch.

Vì vậy guarantee thực tế không còn tuyệt đối. Nhưng nó vẫn tạo ra một update rule ổn định hơn vanilla policy gradient.

TRPO paper chỉ ra bound dựa trên KL và nói rằng maximizing surrogate minorizing objective có thể tạo chuỗi policy không giảm performance trong setting lý tưởng. 

---

### Why not just use a KL penalty?

Một cách tự nhiên là thay constraint bằng penalty:

$$
\max_\theta
\quad
\hat{\mathbb{E}}_t
\left[
r_t(\theta)\hat{A}_t
-
\beta
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot|s_t)
\,\Vert\,
\pi_\theta(\cdot|s_t)
\right)
\right].
$$

Ở đây $\beta$ điều khiển mức phạt KL.

Nếu $\beta$ lớn, policy update nhỏ.

Nếu $\beta$ nhỏ, policy update lớn.

Về lý thuyết, penalty form rất tự nhiên. Nhưng thực tế khó chọn một $\beta$ tốt.

Một $\beta$ có thể tốt ở task này nhưng tệ ở task khác.

Ngay trong cùng một task, early training và late training cũng có thể cần mức penalty khác nhau.

TRPO chọn cách dùng hard constraint:

$$
\bar{D}_{\mathrm{KL}}
\le
\delta.
$$

Tức là thay vì hỏi “phạt bao nhiêu là đủ?”, TRPO hỏi trực tiếp:

$$
\boxed{
\text{Policy mới được phép cách policy cũ tối đa bao nhiêu?}
}
$$

Đây là điểm khác biệt quan trọng giữa TRPO và natural policy gradient cổ điển: TRPO enforce KL constraint ở mỗi update, còn natural gradient thường dùng một step size hoặc penalty coefficient phải tune. Paper TRPO cũng nhấn mạnh rằng việc dùng fixed KL constraint giúp robust hơn trên các bài toán lớn. 

---

### From constrained optimization to natural gradient

Bài toán TRPO thực tế là:

$$
\max_\theta
\quad
L_{\theta_{\text{old}}}(\theta)
$$

subject to:

$$
\bar{D}_{\mathrm{KL}}(\theta_{\text{old}},\theta)
\le
\delta.
$$

Nhưng với neural network lớn, ta không solve bài toán này chính xác.

TRPO dùng approximation quanh $\theta_{\text{old}}$.

Đặt:

$$
s = \theta - \theta_{\text{old}}.
$$

Objective được linearize:

$$
L_{\theta_{\text{old}}}(\theta_{\text{old}} + s)
\approx
L_{\theta_{\text{old}}}(\theta_{\text{old}})
+
g^T s,
$$

trong đó:

$$
g
=
\nabla_\theta
L_{\theta_{\text{old}}}(\theta)
\bigg|_{\theta=\theta_{\text{old}}}.
$$

KL constraint được quadraticize:

$$
\bar{D}_{\mathrm{KL}}(\theta_{\text{old}},\theta_{\text{old}}+s)
\approx
\frac{1}{2}
s^T F s.
$$

Trong đó:

$$
F
$$

là Fisher information matrix, hay Hessian của KL divergence tại policy cũ.

Vậy bài toán gần đúng trở thành:

$$
\max_s
\quad
g^T s
$$

subject to:

$$
\frac{1}{2}
s^T F s
\le
\delta.
$$

Đây là một quadratic constrained linear optimization problem.

Nghiệm có hướng:

$$
s
\propto
F^{-1}g.
$$

Đây chính là natural gradient direction.

Vậy:

$$
\boxed{
\text{TRPO dùng natural gradient direction, nhưng chọn step size bằng KL constraint.}
}
$$

Nếu chuẩn hóa đúng constraint, ta có:

$$
s^*
=
\sqrt{
\frac{2\delta}
{g^T F^{-1}g}
}
F^{-1}g.
$$

Công thức này rất đáng nhớ.

Vanilla policy gradient đi theo:

$$
g.
$$

Natural policy gradient đi theo:

$$
F^{-1}g.
$$

TRPO đi theo:

$$
F^{-1}g
$$

nhưng scale sao cho KL divergence xấp xỉ bằng trust-region size $\delta$.

---

### Intuition for Fisher matrix

Tại sao lại có $F^{-1}g$?

Gradient thường:

$$
g
$$

nói rằng parameter nên đổi theo hướng nào để surrogate objective tăng nhanh trong parameter space.

Nhưng parameter space không phải không gian tự nhiên của policy.

Hai thay đổi parameter có cùng độ lớn Euclidean:

$$
|s_1|_2 = |s_2|_2
$$

có thể tạo ra hai mức thay đổi policy rất khác nhau.

Fisher matrix $F$ mã hóa độ nhạy của policy distribution đối với parameter.

Nếu một hướng parameter làm policy distribution thay đổi rất mạnh, thì:

$$
s^T F s
$$

lớn.

Nếu một hướng parameter làm policy distribution thay đổi nhẹ, thì:

$$
s^T F s
$$

nhỏ.

Vậy constraint:

$$
\frac{1}{2}s^T F s \le \delta
$$

nói rằng:

> Ta không giới hạn độ dài bước đi trong parameter space. Ta giới hạn độ dài bước đi trong policy distribution space.

Đây là lý do natural gradient thường ổn định hơn vanilla gradient.

---

### Conjugate gradient

Vấn đề là neural network policy có thể có hàng chục nghìn, hàng triệu tham số.

Không thể tính full Fisher matrix:

$$
F
$$

và càng không thể tính inverse:

$$
F^{-1}.
$$

TRPO không trực tiếp invert $F$.

Nó cần solve:

$$
Fx = g.
$$

Nghiệm là:

$$
x = F^{-1}g.
$$

TRPO dùng conjugate gradient để solve gần đúng hệ tuyến tính này.

Điểm hay là conjugate gradient không cần materialize full matrix $F$. Nó chỉ cần function tính Fisher-vector product:

$$
v \mapsto Fv.
$$

Trong neural network, Fisher-vector product có thể tính bằng automatic differentiation thông qua Hessian-vector product của KL.

Vậy practical TRPO làm như sau:

1. Tính policy gradient $g$.
2. Dùng conjugate gradient để tìm:

$$
x \approx F^{-1}g.
$$

3. Scale $x$ để KL constraint xấp xỉ bằng $\delta$.
4. Dùng line search để đảm bảo objective thật sự cải thiện và KL thật sự không vượt quá constraint.

TRPO paper mô tả rõ procedure gồm hai bước: tính search direction bằng linear approximation objective và quadratic approximation constraint, rồi line search để đảm bảo objective phi tuyến cải thiện trong khi constraint phi tuyến vẫn được thỏa mãn. Paper cũng nói nếu không line search, đôi khi update lớn có thể gây catastrophic degradation. 

---

### Line search

Tại sao cần line search nếu ta đã scale bằng KL constraint?

Vì các công thức:

$$
L(\theta_{\text{old}}+s)
\approx
L(\theta_{\text{old}})
+
g^Ts
$$

và:

$$
\bar{D}_{\mathrm{KL}}
\approx
\frac{1}{2}s^TFs
$$

chỉ là approximation quanh $\theta_{\text{old}}$.

Neural network là nonlinear. Nếu bước đi hơi xa, approximation có thể sai.

Vì vậy TRPO sau khi tìm candidate step:

$$
s
$$

sẽ thử:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
s.
$$

Nếu KL quá lớn hoặc surrogate không cải thiện, nó giảm step:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
+
\alpha s,
$$

với:

$$
\alpha < 1.
$$

Lặp lại cho đến khi thỏa:

$$
\bar{D}_{\mathrm{KL}}
\le
\delta
$$

và surrogate objective cải thiện.

Line search là phần thực dụng giúp TRPO không chỉ đẹp về lý thuyết mà còn chạy ổn trong thực tế.

---

### TRPO algorithm in practice

Khi kết hợp với GAE, một vòng lặp TRPO thường như sau.

1. Run current policy:

$$
\pi_{\theta_{\text{old}}}
$$

để collect trajectories.

2. Dùng value function hiện tại:

$$
V_\phi(s)
$$

để tính TD residual:

$$
\delta_t^V
=
r_t
+
\gamma V_\phi(s_{t+1})
-
V_\phi(s_t).
$$

3. Tính GAE advantage:

$$
\hat{A}_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^V.
$$

4. Xây surrogate objective:

$$
L_{\theta_{\text{old}}}(\theta)
=
\hat{\mathbb{E}}_t
\left[
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
\hat{A}_t
\right].
$$

5. Tính gradient:

$$
g
=

\nabla_\theta L_{\theta_{\text{old}}}(\theta)
\bigg|_{\theta=\theta_{\text{old}}}.
$$

6. Tính Fisher-vector products từ KL:

$$
\bar{D}_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}},
\pi_\theta
\right).
$$

7. Dùng conjugate gradient để lấy natural-gradient direction:

$$
x
\approx
F^{-1}g.
$$

8. Scale direction sao cho:

$$
\frac{1}{2}x^TFx
\approx
\delta.
$$

9. Backtracking line search để đảm bảo:

$$
L_{\theta_{\text{old}}}(\theta_{\text{new}})
>
L_{\theta_{\text{old}}}(\theta_{\text{old}})
$$

và:

$$
\bar{D}_{\mathrm{KL}}
\le
\delta.
$$

10. Update value function bằng regression vào return target hoặc value target.

Paper GAE trình bày gần đúng pipeline này khi dùng GAE cùng TRPO: simulate current policy, compute TD residuals, compute GAE, compute TRPO update, rồi update value function. 

---

### Why TRPO helps with nonstationarity

Trong phần GAE, ta nói deep RL có hai vấn đề lớn:

$$
\text{sample inefficiency}
$$

và:

$$
\text{unstable improvement due to nonstationary data}.
$$

GAE giúp giảm variance của gradient estimator, tức là giúp ta có tín hiệu update tốt hơn.

Nhưng GAE không ngăn policy update quá mạnh.

TRPO xử lý vấn đề thứ hai.

Trong RL, sau mỗi policy update, data distribution thay đổi:

$$
\pi_{\theta_i}
\Rightarrow
\rho_{\pi_{\theta_i}}(s).
$$

Nếu update quá mạnh:

$$
\pi_{\theta_{i+1}}
\not\approx
\pi_{\theta_i},
$$

thì batch vừa collect từ $\pi_{\theta_i}$ không còn phản ánh tốt behavior của $\pi_{\theta_{i+1}}$.

TRPO ép:

$$
\pi_{\theta_{i+1}}
\approx
\pi_{\theta_i}
$$

theo KL divergence.

Vậy data cũ vẫn còn tương đối relevant với policy mới.

Ta có thể nói:

$$
\boxed{
\text{TRPO làm policy iteration trở nên conservative hơn.}
}
$$

Nó không cố tìm policy tốt nhất trong một update. Nó cố tìm một policy tốt hơn một chút, nhưng đủ an toàn để lặp lại nhiều lần.

---

### Connection to policy iteration

TRPO có thể được hiểu như một phiên bản approximate policy iteration cho neural network policy.

Trong exact policy iteration, ta làm:

1. Policy evaluation: tính $V^\pi$, $Q^\pi$, $A^\pi$.
2. Policy improvement: chọn action tốt hơn theo $Q^\pi$ hoặc $A^\pi$.

Nếu tabular và exact, policy improvement có guarantee rõ ràng.

Nhưng với neural network policy, continuous action, finite samples, ta không thể greedy improvement trực tiếp.

TRPO thay greedy improvement bằng constrained improvement:

$$
\boxed{
\text{Improve according to advantage, but only within a KL trust region.}
}
$$

Vì vậy TRPO nằm giữa hai thế giới:

$$
\text{policy gradient}
$$

và:

$$
\text{policy iteration}.
$$

Nó dùng sample-based policy gradient objective, nhưng tinh thần là conservative policy improvement.

---

### Single-path and vine versions

TRPO paper có hai biến thể sampling.

Biến thể thứ nhất là **single path**.

Ta run policy trong environment để lấy trajectories bình thường:

$$
s_0,a_0,s_1,a_1,\ldots
$$

Sau đó dùng toàn bộ state-action pairs trong trajectories để estimate objective.

Đây là biến thể model-free và thực tế hơn.

Biến thể thứ hai là **vine**.

Ở vine method, ta collect một số “trunk trajectories”, chọn một số state, rồi từ mỗi state đó thử nhiều action/rollout khác nhau. Cách này có thể giảm variance trong việc estimate Q/advantage, nhưng cần khả năng reset simulator về state cụ thể.

Trong thế giới thực hoặc nhiều environment thông thường, reset về arbitrary state là không khả thi. Vì vậy single-path TRPO là biến thể gần với thực tế hơn.

TRPO paper nói single-path có thể dùng trong model-free setting, còn vine thường cần simulator có khả năng restore về state cụ thể. 

---

### What TRPO really contributes

TRPO không phát minh lại policy gradient.

TRPO cũng không phải là một advantage estimator.

TRPO giữ nguyên tinh thần:

$$
\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}_t.
$$

Điểm mới của TRPO là cách biến gradient đó thành một **policy update an toàn**.

Có thể tóm tắt:

$$
\boxed{
\text{Policy gradient tells us the direction.}
}
$$

$$
\boxed{
\text{GAE gives a better advantage signal.}
}
$$

$$
\boxed{
\text{TRPO decides how far we are allowed to move.}
}
$$

Một cách nói khác:

$$
\boxed{
\text{TRPO = policy gradient + trust region in policy distribution space.}
}
$$

Nó làm ba việc quan trọng:

1. Dùng surrogate objective với probability ratio để estimate policy improvement từ old-policy data.
2. Dùng KL constraint để đảm bảo policy mới không quá khác policy cũ.
3. Dùng natural-gradient-style update để giải bài toán constrained optimization hiệu quả.

---

### Why TRPO was important before PPO

TRPO là một bước rất quan trọng trong deep policy optimization vì nó cho thấy policy gradient có thể được làm ổn định hơn bằng cách kiểm soát policy update.

Trong experiments, TRPO được test trên locomotion tasks như swimmer, hopper, walker và cả Atari từ image input. Paper báo cáo rằng single-path và vine TRPO giải được các locomotion problems và cho kết quả tốt hơn nhiều baseline; natural gradient hoạt động trên bài dễ hơn nhưng không tạo được hopping/walking gait tốt ở bài khó, cung cấp bằng chứng rằng constraining KL robust hơn fixed penalty/stepsize. 

Nhưng TRPO cũng phức tạp:

* phải tính Fisher-vector product;
* phải chạy conjugate gradient;
* phải line search;
* khó kết hợp với một số architecture phức tạp;
* không tiện bằng optimizer first-order đơn giản như Adam.

Đây là lý do PPO ra đời sau đó.

PPO cố giữ spirit của TRPO:

$$
\text{do not let policy move too far}
$$

nhưng bỏ conjugate gradient và hard KL constraint, thay bằng clipped surrogate objective hoặc adaptive KL penalty.

PPO paper nói PPO giữ nhiều lợi ích của TRPO nhưng đơn giản hơn, tổng quát hơn, và empirically sample-efficient hơn; paper cũng nói TRPO tương đối phức tạp so với mục tiêu dùng first-order optimization. 

---

### Summary

TRPO có thể được hiểu bằng một câu:

$$
\boxed{
\text{TRPO is a conservative policy-gradient update that keeps the new policy close to the old policy.}
}
$$

GAE trả lời:

$$
\boxed{
\text{How should we estimate advantage?}
}
$$

TRPO trả lời:

$$
\boxed{
\text{Given those advantages, how far should we update the policy?}
}
$$

Công thức trung tâm của TRPO là:

$$
\max_\theta
\quad
\hat{\mathbb{E}}_t
\left[
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{\theta_{\text{old}}}(a_t|s_t)
}
\hat{A}_t
\right]
$$

subject to:

$$
\hat{\mathbb{E}}_t
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot|s_t)
\,\Vert\,
\pi_\theta(\cdot|s_t)
\right)
\right]
\le
\delta.
$$

Ý nghĩa của từng thành phần:

$$
\hat{A}_t
=
\text{action was better or worse than expected}
$$

$$
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
=
\text{how much the new policy changes probability of sampled action}
$$

$$
D_{\mathrm{KL}}
(\pi_{\text{old}}\,\Vert\,\pi_{\text{new}})
=
\text{how far the whole policy distribution moves}
$$

Insight cuối cùng:

> Trong deep RL, không đủ để biết gradient direction. Ta phải kiểm soát mức độ thay đổi của policy, vì policy quyết định luôn data distribution của chính nó. TRPO làm policy gradient ổn định hơn bằng cách chỉ cho phép những update nằm trong một trust region đo bằng KL divergence.

Vậy nếu GAE là cách tạo tín hiệu advantage ít nhiễu hơn, thì TRPO là cách dùng tín hiệu đó mà không phá policy sau một bước update quá lớn.

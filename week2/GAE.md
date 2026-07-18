
---

## Generalized Advantage Estimation

Sau phần policy gradient, REINFORCE, baseline, actor-critic và eligibility traces, ta có thể nhìn lại toàn bộ câu chuyện như sau.

Policy-gradient methods cố gắng tối đa hóa performance của policy bằng cách cập nhật tham số $\theta$ theo hướng:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta).
$$

Nhờ policy gradient theorem, ta không cần đạo hàm qua toàn bộ dynamics của môi trường. Gradient có thể được viết dưới dạng:

$$
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
\nabla_\theta \log \pi_\theta(a_t|s_t)\Psi_t
\right].
$$

Trong đó:

* $\nabla_\theta \log \pi_\theta(a_t|s_t)$ nói rằng ta nên thay đổi xác suất của action vừa được sample như thế nào.
* $\Psi_t$ là tín hiệu đánh giá action đó: action này tốt hay xấu, đáng tăng xác suất hay đáng giảm xác suất.

Trong paper GAE, các lựa chọn khả dĩ cho $\Psi_t$ gồm total reward, reward-to-go, reward-to-go trừ baseline, $Q^\pi(s_t,a_t)$, $A^\pi(s_t,a_t)$, hoặc TD residual. Paper nhấn mạnh rằng lựa chọn gần lý tưởng nhất là advantage function, vì advantage trực tiếp trả lời câu hỏi: action này tốt hơn hay tệ hơn hành vi trung bình của policy tại state đó. 

Advantage thật được định nghĩa là:

$$
A^\pi(s_t,a_t)
=
Q^\pi(s_t,a_t) - V^\pi(s_t).
$$

Nếu:

$$
A^\pi(s_t,a_t) > 0,
$$

thì action $a_t$ tốt hơn mức trung bình tại $s_t$, nên policy nên tăng xác suất chọn action đó.

Nếu:

$$
A^\pi(s_t,a_t) < 0,
$$

thì action đó tệ hơn mức trung bình, nên policy nên giảm xác suất chọn action đó.

Vì vậy, bản chất của policy gradient có thể được hiểu rất đơn giản:

$$
\text{Increase probability of better-than-average actions,}
$$

$$
\text{decrease probability of worse-than-average actions.}
$$

Vấn đề là $A^\pi(s_t,a_t)$ không biết trước. Ta phải ước lượng nó. Và đây chính là trọng tâm của GAE:

$$
\boxed{
\text{GAE không thay đổi bản chất policy gradient.}
}
$$

$$
\boxed{
\text{GAE trả lời câu hỏi: nên ước lượng advantage như thế nào?}
}
$$

Paper GAE xuất phát từ một vấn đề thực tế của policy-gradient methods: chúng có thể dùng được với neural network và continuous control, nhưng thường cần rất nhiều sample và rất khó cải thiện ổn định. Nguyên nhân lớn là gradient estimate có variance cao, đặc biệt khi reward đến trễ sau nhiều bước. Value function giúp giảm variance bằng cách ước lượng “goodness” của action trước khi delayed reward thật sự xuất hiện, nhưng việc dùng value function cũng đưa bias vào. 

Ta có hai cực đoan quen thuộc.

### Monte-Carlo advantage estimate

Trong REINFORCE with baseline, ta có thể dùng:

$$
\hat{A}_t
=
G_t - V(s_t).
$$

Trong đó $G_t$ là return thực tế từ timestep $t$ trở đi.

Ưu điểm: nếu lấy đủ sample, estimator này khá gần với advantage thật.

Nhược điểm: variance rất cao, vì $G_t$ phụ thuộc vào rất nhiều reward tương lai. Một action ở timestep $t$ bị trộn lẫn với ảnh hưởng của các action trước đó, các action sau đó, stochasticity của môi trường, và stochasticity của policy. Paper nói variance của gradient estimator tăng không thuận lợi theo horizon, vì effect của một action bị confounded với effects của past và future actions. 

Ta có thể nói ngắn gọn:

$$
G_t - V(s_t)
\quad
\text{low bias, high variance.}
$$

### One-step TD advantage estimate

Trong actor-critic, thay vì đợi full return, ta dùng TD residual:

$$
\delta_t^V
=
r_t + \gamma V(s_{t+1}) - V(s_t).
$$

Đại lượng này đo xem sau một bước, value prediction của critic bị sai theo hướng nào.

Nếu:

$$
r_t + \gamma V(s_{t+1}) > V(s_t),
$$

thì kết quả sau action tốt hơn critic dự đoán. Khi đó $\delta_t^V > 0$, action này nên được tăng xác suất.

Nếu:

$$
r_t + \gamma V(s_{t+1}) < V(s_t),
$$

thì kết quả sau action tệ hơn critic dự đoán. Khi đó $\delta_t^V < 0$, action này nên bị giảm xác suất.

Nếu value function là chính xác, tức là:

$$
V = V^{\pi,\gamma},
$$

thì TD residual là một estimator hợp lý cho discounted advantage:

$$
\mathbb{E}
\left[
r_t + \gamma V^{\pi,\gamma}(s_{t+1}) - V^{\pi,\gamma}(s_t)
\mid s_t,a_t
\right]
=
A^{\pi,\gamma}(s_t,a_t).
$$

Nhưng nếu $V$ chưa chính xác, TD residual một bước sẽ bị bias. 

Ta có thể nói:

$$
\delta_t^V
\quad
\text{low variance, potentially high bias.}
$$

Vậy ta có bảng sau:

$$
\begin{array}{c|c|c}
\text{Estimator} & \text{Bias} & \text{Variance} \\
\hline
G_t - V(s_t) & \text{low} & \text{high} \\
\delta_t^V & \text{higher if } V \text{ is inaccurate} & \text{low}
\end{array}
$$

GAE xuất hiện để lấy một điểm nằm giữa hai cực đoan này.

---

### From one-step TD to k-step advantage estimates

Thay vì chỉ dùng một TD error:

$$
\delta_t^V,
$$

ta có thể cộng nhiều TD error liên tiếp.

Với một bước:

$$
\hat{A}^{(1)}_t
=
\delta_t^V
=
-V(s_t) + r_t + \gamma V(s_{t+1}).
$$

Với hai bước:

$$
\hat{A}^{(2)}_t
=
\delta_t^V + \gamma \delta_{t+1}^V.
$$

Khai triển ra:

$$
\hat{A}^{(2)}_t
=
-V(s_t)
+
r_t
+
\gamma r_{t+1}
+
\gamma^2 V(s_{t+2}).
$$

Với ba bước:

$$
\hat{A}^{(3)}_t
=
\delta_t^V
+
\gamma \delta_{t+1}^V
+
\gamma^2 \delta_{t+2}^V.
$$

Khai triển ra:

$$
\hat{A}^{(3)}_t
=
-V(s_t)
+
r_t
+
\gamma r_{t+1}
+
\gamma^2 r_{t+2}
+
\gamma^3 V(s_{t+3}).
$$

Tổng quát:

$$
\hat{A}^{(k)}_t
=
\sum_{l=0}^{k-1}
\gamma^l \delta_{t+l}^V.
$$

Sau khi telescoping, ta được:

$$
\hat{A}^{(k)}_t
=
-V(s_t)
+
r_t
+
\gamma r_{t+1}
+
\cdots
+
\gamma^{k-1}r_{t+k-1}
+
\gamma^k V(s_{t+k}).
$$

Đây chính là một $k$-step return trừ baseline $V(s_t)$. Paper trình bày đúng ý này: tổng nhiều TD residual tạo thành một $k$-step estimate của return, rồi trừ đi baseline $-V(s_t)$. 

Ý nghĩa:

* $k=1$: nhìn rất ngắn, variance thấp, nhưng phụ thuộc mạnh vào độ chính xác của $V$.
* $k$ lớn: nhìn xa hơn, ít phụ thuộc vào bootstrap, bias giảm, nhưng variance tăng.
* $k \to \infty$: trở thành Monte-Carlo return trừ baseline.

Khi $k \to \infty$:

$$
\hat{A}^{(\infty)}_t
=
\sum_{l=0}^{\infty}
\gamma^l \delta_{t+l}^V
=
\sum_{l=0}^{\infty}
\gamma^l r_{t+l}
-
V(s_t).
$$

Tức là:

$$
\hat{A}^{(\infty)}_t
=
G_t^\gamma - V(s_t).
$$

Vậy ta thấy một continuum:

$$
\hat{A}^{(1)}_t
\rightarrow
\hat{A}^{(2)}_t
\rightarrow
\hat{A}^{(3)}_t
\rightarrow
\cdots
\rightarrow
G_t^\gamma - V(s_t).
$$

---

### GAE: exponentially weighted average of k-step advantage estimates

GAE không chọn cố định một $k$. Thay vào đó, nó lấy trung bình có trọng số mũ của tất cả các $k$-step advantage estimators:

$$
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
(1-\lambda)
\left(
\hat{A}^{(1)}_t
+
\lambda \hat{A}^{(2)}_t
+
\lambda^2 \hat{A}^{(3)}_t
+
\cdots
\right).
$$

Ở đây $\lambda \in [0,1]$ quyết định ta tin vào estimate ngắn hạn hay dài hạn nhiều hơn.

Nếu $\lambda$ nhỏ, trọng số tập trung vào $\hat{A}^{(1)}_t$, tức là gần one-step TD.

Nếu $\lambda$ lớn, các estimate nhiều bước được giữ lại nhiều hơn, tức là gần Monte Carlo hơn.

Khi khai triển biểu thức trên, ta có:

$$
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
\sum_{l=0}^{\infty}
(\gamma \lambda)^l
\delta_{t+l}^V.
$$

Đây là công thức trung tâm của GAE. Paper nhấn mạnh rằng công thức này rất đơn giản: advantage estimator chỉ là discounted sum của các Bellman residual / TD residual terms. 

Với:

$$
\delta_t^V
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t).
$$

Tức là:

$$
\boxed{
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
\delta_t^V
+
\gamma\lambda \delta_{t+1}^V
+
(\gamma\lambda)^2 \delta_{t+2}^V
+
\cdots
}
$$

Trực giác rất quan trọng:

> Một action không chỉ nên được đánh giá bằng reward ngay sau nó. Nó nên được đánh giá bằng chuỗi các “surprise” mà nó gây ra trong tương lai.

Ở đây mỗi TD error là một surprise:

$$
\delta_t^V
=
\text{what happened}
-
\text{what the critic expected}.
$$

Nếu sau action $a_t$, nhiều timestep tiếp theo liên tục có TD error dương, nghĩa là thế giới đang diễn ra tốt hơn critic dự đoán. Khi đó action $a_t$ nên được credit dương.

Nếu sau action $a_t$, nhiều timestep tiếp theo có TD error âm, nghĩa là action này có thể đã mở đường cho kết quả xấu. Khi đó nó nên bị giảm xác suất.

Do đó GAE là một cách làm credit assignment mềm qua thời gian:

$$
\text{current action receives credit from future TD errors,}
$$

nhưng credit này giảm dần theo hệ số:

$$
(\gamma\lambda)^l.
$$

Điều này rất giống eligibility traces trong Sutton & Barto: TD error hiện tại có thể ảnh hưởng đến các state/action trước đó, với trọng số giảm dần theo $\gamma\lambda$. Trong note trước, ta đã thấy $\lambda$-return error có thể decomposed thành tổng discounted TD errors; GAE dùng đúng tinh thần đó, nhưng thay vì dùng nó để update value function, nó dùng nó để estimate advantage cho policy gradient. 

Điểm khác biệt cốt lõi là:

$$
\text{TD}(\lambda)
\quad
\text{estimates a value function,}
$$

trong khi:

$$
\text{GAE}
\quad
\text{estimates an advantage function.}
$$

---

### Two special cases

GAE có hai trường hợp đặc biệt rất quan trọng.

Nếu:

$$
\lambda = 0,
$$

thì:

$$
\hat{A}^{\mathrm{GAE}(\gamma,0)}_t
=
\delta_t^V
=
r_t+\gamma V(s_{t+1})-V(s_t).
$$

Đây là one-step TD advantage estimate.

Nó có variance thấp, vì chỉ nhìn một bước. Nhưng nếu $V$ sai, bias có thể lớn.

Nếu:

$$
\lambda = 1,
$$

thì:

$$
\hat{A}^{\mathrm{GAE}(\gamma,1)}_t
=
\sum_{l=0}^{\infty}
\gamma^l \delta_{t+l}^V.
$$

Khai triển telescoping:

$$
\hat{A}^{\mathrm{GAE}(\gamma,1)}_t
=
\sum_{l=0}^{\infty}
\gamma^l r_{t+l}
-
V(s_t).
$$

Đây chính là discounted Monte-Carlo return trừ baseline.

Nó ít bias hơn đối với discounted policy gradient, nhưng variance cao vì dùng tổng nhiều reward tương lai. Paper nói rõ: $\lambda=1$ là $\gamma$-just bất kể $V$ chính xác hay không, nhưng có high variance; $\lambda=0$ có lower variance nhưng bị bias nếu $V$ không chính xác. 

Vậy:

$$
\lambda = 0
\Rightarrow
\text{TD-like, low variance, more bias}
$$

$$
\lambda = 1
\Rightarrow
\text{Monte-Carlo-like, low bias, high variance}
$$

$$
0 < \lambda < 1
\Rightarrow
\text{bias-variance trade-off}
$$

---

### Why does $\gamma$ introduce bias in the GAE paper?

Trong Sutton & Barto, ta thường xem $\gamma$ là một phần của problem definition. Nếu bài toán được định nghĩa là maximize discounted return:

$$
\mathbb{E}
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\right],
$$

thì $\gamma$ không phải bias. Nó là mục tiêu gốc của bài toán.

Nhưng paper GAE làm một việc tinh tế hơn. Nó bắt đầu từ một undiscounted objective:

$$
\max_\theta
\mathbb{E}
\left[
\sum_{t=0}^{\infty}
r_t
\right].
$$

Sau đó paper đưa $\gamma$ vào như một algorithm parameter để giảm variance. Tức là $\gamma$ không nhất thiết là objective thật của environment, mà là một núm điều chỉnh trong estimator. Paper nói rõ rằng họ không dùng discount như một phần của problem specification; discount xuất hiện như một parameter điều chỉnh bias-variance trade-off. 

Khi dùng $\gamma < 1$, ta giảm trọng số của reward xa trong tương lai:

$$
r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots.
$$

Điều này giảm variance, vì reward càng xa thì càng noisy và càng khó biết có phải do action hiện tại gây ra hay không.

Nhưng nó đưa bias vào, vì ta không còn estimate đúng gradient của undiscounted total reward nữa. Ta đang estimate gradient của một objective bị làm ngắn lại.

Nói ngắn gọn:

$$
\gamma < 1
\Rightarrow
\text{ignore part of long-term effects}
\Rightarrow
\text{lower variance but biased gradient.}
$$

Đây là lý do paper phân biệt hai thứ:

$$
\gamma:
\text{downweight delayed rewards}
$$

và:

$$
\lambda:
\text{mix different k-step advantage estimates}.
$$

Cả hai đều ảnh hưởng bias-variance, nhưng chúng không giống nhau.

Paper nói:

* $\gamma < 1$ đưa bias vào ngay cả khi value function chính xác.
* $\lambda < 1$ chủ yếu đưa bias vào khi value function không chính xác.
* Vì vậy trong thực nghiệm, $\lambda$ có thể thấp hơn $\gamma$ mà vẫn không gây bias quá nặng, miễn là value function tương đối tốt. 

---

### Gamma-just estimator

Paper đưa vào khái niệm $\gamma$-just estimator để nói rõ hơn về bias.

Sau khi ta đã chấp nhận dùng discounted policy gradient:

$$
g^\gamma
=
\mathbb{E}
\left[
\sum_{t=0}^{\infty}
A^{\pi,\gamma}(s_t,a_t)
\nabla_\theta \log \pi_\theta(a_t|s_t)
\right],
$$

ta cần một estimator $\hat{A}_t$ thay cho $A^{\pi,\gamma}(s_t,a_t)$.

Một estimator được gọi là $\gamma$-just nếu khi thay nó vào policy gradient, kỳ vọng của gradient estimate vẫn bằng đúng $g^\gamma$:

$$
\mathbb{E}
\left[
\hat{A}_t
\nabla_\theta \log \pi_\theta(a_t|s_t)
\right]
=
\mathbb{E}
\left[
A^{\pi,\gamma}(s_t,a_t)
\nabla_\theta \log \pi_\theta(a_t|s_t)
\right].
$$

Tức là estimator này không thêm bias so với discounted policy gradient $g^\gamma$.

Nhưng cần cẩn thận:

$$
g^\gamma
$$

bản thân đã là biased approximation nếu objective thật là undiscounted. Paper thậm chí chú thích rõ rằng bias đã được đưa vào khi dùng $A^{\pi,\gamma}$ thay cho $A^\pi$; khái niệm $\gamma$-just chỉ đảm bảo estimator không đưa thêm bias so với $g^\gamma$. 

Vậy có hai tầng bias:

$$
\text{Bias 1: dùng } \gamma < 1 \text{ thay cho undiscounted objective.}
$$

$$
\text{Bias 2: dùng estimator } \hat{A}_t \text{ không chính xác cho } A^{\pi,\gamma}.
$$

GAE cố gắng kiểm soát tầng bias thứ hai bằng cách chọn $\lambda$ hợp lý.

---

### Reward shaping interpretation

Một insight rất hay của paper là: GAE có thể được hiểu như reward shaping.

Reward shaping là biến đổi reward theo dạng:

$$
\tilde{r}(s,a,s')
=
r(s,a,s')
+
\gamma \Phi(s')
-
\Phi(s),
$$

trong đó $\Phi$ là một hàm potential trên state.

Nếu chọn:

$$
\Phi(s) = V(s),
$$

thì shaped reward trở thành:

$$
\tilde{r}_t
=
r_t + \gamma V(s_{t+1}) - V(s_t).
$$

Nhưng đây chính là TD residual:

$$
\tilde{r}_t = \delta_t^V.
$$

Vậy GAE:

$$
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^V
$$

có thể được hiểu là:

$$
\boxed{
\text{discounted return của shaped rewards}
}
$$

với discount mới là $\gamma\lambda$.

Paper chỉ ra rằng nếu shaping dùng value function thật $V^{\pi,\gamma}$, thì temporal credit assignment sẽ được làm ngắn lại rất mạnh: ảnh hưởng dài hạn của action lên reward được chuyển thành tín hiệu gần hơn, tức là immediate TD residual. Nếu $V$ chỉ xấp xỉ đúng, nó vẫn giúp co ngắn phần nào temporal spread của credit assignment. 

Trực giác:

> Value function biến delayed reward thành chuỗi prediction errors gần hơn.

Thay vì chờ 100 bước sau mới biết action hiện tại tốt hay xấu, critic tạo ra một tín hiệu trung gian ngay mỗi bước:

$$
\delta_t^V
=
\text{actual one-step outcome}
-
\text{predicted one-step outcome}.
$$

Nếu critic tốt, các TD residual này là dạng reward đã được “nén thời gian”. Khi đó ta có thể dùng thêm $\lambda$ để cắt bớt các residual quá xa, giảm noise mà vẫn giữ phần lớn thông tin quan trọng.

Đây là insight sâu hơn công thức:

$$
\boxed{
\text{GAE = reward shaping bằng value function + discount mạnh hơn bằng } \lambda.
}
$$

---

### Why GAE matters for modern policy optimization

Trong các bài toán nhỏ, ta có thể chạy Monte Carlo nhiều lần, variance cao cũng chưa quá thảm họa.

Nhưng trong deep RL, đặc biệt là continuous control với neural network policy, vấn đề trở nên nghiêm trọng hơn nhiều:

* policy có rất nhiều tham số;
* trajectory dài;
* reward đến trễ;
* action liên tục;
* data thay đổi liên tục vì policy thay đổi sau mỗi update;
* mỗi batch data đắt đỏ.

Paper GAE nhắm vào bối cảnh này. Nó không chỉ nói về một công thức advantage estimate. Nó là một phần của hệ thống học policy ổn định hơn:

$$
\text{good advantage estimator}
+
\text{trust region policy update}
+
\text{neural network value function}.
$$

Paper nói hai thách thức chính là sample requirement lớn và khó cải thiện ổn định do nonstationarity của incoming data. GAE xử lý thách thức thứ nhất bằng cách dùng value function để giảm variance của policy gradient estimate; trust region optimization xử lý thách thức thứ hai bằng cách giới hạn mức thay đổi của policy và value function. 

Trong experiments của paper, GAE được dùng cùng TRPO. Quy trình tổng quát là:

1. Run current policy để collect trajectories.
2. Dùng current value function $V_{\phi_i}$ để tính TD residual:

$$
\delta_t^V
=
r_t + \gamma V_{\phi_i}(s_{t+1}) - V_{\phi_i}(s_t).
$$

3. Tính advantage estimate:

$$
\hat{A}_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^V.
$$

4. Dùng $\hat{A}_t$ trong TRPO policy update.
5. Sau đó update value function.

Paper còn lưu ý một chi tiết thực tế quan trọng: policy update $\theta_i \to \theta_{i+1}$ dùng value function cũ $V_{\phi_i}$ để estimate advantage, không dùng value function mới $V_{\phi_{i+1}}$. Nếu update value function trước và overfit batch hiện tại, Bellman residual có thể gần bằng 0 ở mọi timestep, làm policy gradient estimate biến mất. 

---

### Connection to PPO

PPO sau này dùng lại đúng vai trò này của GAE. Trong PPO, policy update dùng surrogate objective, nhưng bên trong surrogate vẫn cần advantage estimate $\hat{A}_t$.

PPO với fixed-length trajectory segment thường dùng truncated GAE:

$$
\hat{A}_t
=
\delta_t
+
(\gamma\lambda)\delta_{t+1}
+
\cdots
+
(\gamma\lambda)^{T-t+1}\delta_{T-1},
$$

với:

$$
\delta_t
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t).
$$

Tức là PPO không thay thế GAE. PPO dùng GAE như cách chuẩn để tạo advantage signal, rồi dùng clipped surrogate objective để update policy an toàn hơn. 

Trong hyperparameters của PPO paper, các setting phổ biến cho MuJoCo, Roboschool và Atari đều dùng:

$$
\gamma = 0.99,
\qquad
\lambda = 0.95.
$$

Điều này phản ánh kinh nghiệm thực tế: $\gamma$ rất gần 1 để vẫn giữ long-term reward, còn $\lambda$ hơi thấp hơn để giảm variance từ các TD residual quá xa. 

---

### Empirical takeaway

Paper GAE kiểm tra ảnh hưởng của $\gamma$ và $\lambda$ trong cart-pole và các task locomotion 3D. Với cart-pole, kết quả tốt nhất xuất hiện ở vùng trung gian, ví dụ:

$$
\gamma \in [0.96, 0.99],
\qquad
\lambda \in [0.92, 0.99].
$$

Trong simulated robotic locomotion, paper kết luận rằng chọn $\lambda$ trung gian trong khoảng:

$$
[0.9, 0.99]
$$

thường cho performance tốt nhất. Paper cũng ghi nhận rằng $\lambda=0$, tức one-step return, gây excessive bias và poor performance trong các task họ xét.  

Vậy bài học thực nghiệm là:

$$
\lambda = 0
\quad
\text{often too biased}
$$

$$
\lambda = 1
\quad
\text{often too noisy}
$$

$$
\lambda \approx 0.9 \text{ to } 0.99
\quad
\text{often a good compromise}
$$

---

### Summary

GAE có thể được hiểu bằng một câu:

$$
\boxed{
\text{GAE is TD}(\lambda)\text{-style advantage estimation for policy gradient.}
}
$$

Nó lấy các TD residual:

$$
\delta_t^V
=
r_t
+
\gamma V(s_{t+1})
-
V(s_t)
$$

rồi cộng chúng lại với trọng số giảm dần:

$$
1,\quad
\gamma\lambda,\quad
(\gamma\lambda)^2,\quad
\ldots
$$

để tạo advantage estimate:

$$
\boxed{
\hat{A}^{\mathrm{GAE}(\gamma,\lambda)}_t
=
\sum_{l=0}^{\infty}
(\gamma\lambda)^l
\delta_{t+l}^V
}
$$

Ý nghĩa của từng thành phần:

$$
\delta_t^V
=
\text{local surprise / one-step advantage signal}
$$

$$
\gamma
=
\text{downweight delayed rewards}
$$

$$
\lambda
=
\text{decide how far future TD errors should affect current action}
$$

Nếu $\lambda=0$, GAE trở thành one-step TD advantage: variance thấp nhưng bias cao nếu critic sai.

Nếu $\lambda=1$, GAE trở thành Monte-Carlo return trừ baseline: bias thấp hơn nhưng variance cao.

Nếu $0<\lambda<1$, GAE tạo ra trade-off giữa hai cực đoan.

Insight cuối cùng của paper không chỉ là công thức. Insight thật là:

> Muốn policy gradient hoạt động tốt trong deep RL, ta cần một advantage signal đủ ít nhiễu để học nhanh, nhưng không quá biased để học sai. GAE cung cấp tín hiệu đó bằng cách biến delayed reward thành chuỗi TD residuals, rồi dùng $\lambda$ để quyết định bao nhiêu tín hiệu tương lai nên được giữ lại.

Vì vậy, GAE là cầu nối giữa các ý tưởng cổ điển trong Sutton & Barto — baseline, actor-critic, TD error, $\lambda$-return, eligibility traces — với các thuật toán hiện đại như TRPO và PPO.

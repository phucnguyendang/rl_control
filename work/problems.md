### Scope of research
Research the limitations of PPO, SAC, TD3, and REPPO in online continuous-control reinforcement learning.

The study should focus on applicability to high-dimensional robotic tasks with contact dynamics, with the goal of developing a novel algorithm that achieves higher training stability and better final performance—or at least performance comparable to strong baselines—while maintaining sample efficiency, robustness, and acceptable computational and memory overhead.

### Algorithm Claim and limitation
TRPO đã chứng minh được về inner trust region nhưng nó khó tối ưu vì ràng buộc này enforce trust region quá hẹp và việc tính max_advantage trên toàn bộ (state,action) space là bất khả thi

#### PPO
- Giải quyết ràng buộc về trust region bằng clipping.-> chọn clipping ratio là 1 hyper params, không chắc chắn rằng clipping ratio bị chặn thì đảm bảo KL divergence bị chặn (https://proceedings.mlr.press/v115/wang20b.html), chưa nói đến việc chặn KL_divergence chỉ là 1 surrogate constraint
- Variance cao với long-horizon tasks do noise khi thực hiện monte carlo sampling (REPPO cũng bị)
- PPO cho phép exploration bằng cách cộng entropy vào hàm loss -> giả định entropy không bị collapse quá sớm (assumption của đa số thuật toán đều như vậy)

**rsl_rl trick for PPO**:
- đo KL để tăng giảm lr
- rsl_rl normalize advantage về mean 0, std 1? nó có thể làm advantage đổi dấu khi trừ đi mean? tại sao lại làm vậy nhỉ, à nếu số lượng mẫu đủ nhiều thì mean thường = 0 rồi do mean(A) = mean_a~pi (Q(s,a) - V(s)) = 0 do V(s) = mean_a~pi Q(s,a)

#### REPPO

- Q phải được học đủ tốt -> sử dùng HL Gauss loss và entropy reward để khuyến khích exploration nhiều hơn -> HL Gauss loss yêu cầu phải ước lượng khoảng và độ rộng khoảng hợp lý của Q(s,a) -> 1 hyper param khác
- Q phải có thể học được sau warmup- hiện tượng bị sập có xảy ra nhiều không nhỉ?
- REPPO coi entropy-corrected reward là 1 thành phần quan trọng, dù entroipy đó là của old policy và không có gradient đi qua khi tính loss? Tại sao vậy nhỉ?
- áp dụng ràng buộc D_KL < epsilon_kl và entropy phải lớn hơn epsilon_entropy khi maximum Q(s,a) của actor

-> Tại sao REPPO có thể tốt hơn, có lẽ nó đang ép exploration mạnh hơn để học Q(s,a) đủ tốt, vậy liệu có thể ép PPO exploration mạnh hơn được không?

#### SAC
- sử dụng double Q để hạn chế over-estimation (tương tự TD3)
- các thuật toán off-policy nói chung yêu cầu sử dụng Q_target hoặc V_target để hạn chế cập nhật Q quá nhanh.
- phải tính gradient Q qua action

#### TD3

- sử dụng double Q để hạn chế over-esimation nhưng lại có thể gặp under-estimation -> giảm performance và có thể vào local minimum -> hạn chế này là hạn chế, overestimation nguy hiểm hơn
- các thuật toán off-policy nói chung thường sử dụng Q_target để hạn chế Q cập nhật quá nhanh.
- Tính gradient Q theo action nên cần học Q tốt
- Cộng thêm noise vào sampled action để thúc đẩy học các giá trị chính xác hơn quang action a, từ đó giúp đạo hàm tính đúng hơn
- update critic nhiều lần hơn so với actor 

#### CDPO
- Read the critic’s discrimination, nếu các giá trị Q(s,a) khác nhau rõ rệt thì critic phân biệt được action tốt xấu tại state đó, nếu Q gần như phẳng thì không nên tin vào pathwise. ??? cơ sở nào cho lập luận rằng **nếu các giá trị Q(s,a) khác nhau rõ rệt thì critic phân biệt được action tốt xấu tại state đó**. Đo bằng $C_D$
- 

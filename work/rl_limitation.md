**1. Deep RL bị thổi phồng quá mức**

RL là một khuôn khổ rất tổng quát, nên dễ tạo cảm giác nó có thể giải quyết mọi thứ. Nhưng theo tác giả, các kết quả đẹp thường che giấu rất nhiều công sức tinh chỉnh, thử sai và thất bại.

**2. Rất kém hiệu quả về dữ liệu**

Deep RL thường cần lượng tương tác khổng lồ với môi trường. Ví dụ, Rainbow DQN cần khoảng **18 triệu frame**, tương đương khoảng **83 giờ chơi Atari**, để đạt mức người chơi trung bình; các môi trường MuJoCo cũng cần từ (10^5) đến (10^7) bước học.

**3. Nhiều bài toán có cách giải tốt hơn RL**

Nếu mục tiêu chỉ là hiệu năng cuối cùng, các phương pháp chuyên biệt như model predictive control, search, thuật toán robotics cổ điển, hoặc tối ưu hoá miền cụ thể thường nhanh hơn và đáng tin cậy hơn RL. Tác giả nhấn mạnh rằng tính “tổng quát” của RL đi kèm cái giá rất lớn.

**4. Thiết kế reward rất khó**

RL cần một hàm thưởng, nhưng hàm thưởng phải phản ánh **chính xác** điều ta muốn. Nếu reward bị lệch, agent sẽ tối ưu theo cách kỳ quặc: ví dụ “farm” điểm thay vì hoàn thành cuộc đua, lật khối Lego thay vì nhấc nó lên, hoặc tối ưu metric ROUGE nhưng tạo bản tóm tắt tệ.

**5. Dễ mắc kẹt ở local optimum và khó exploration**

Ngay cả khi reward hợp lý, agent vẫn có thể học hành vi “tàm tạm” nhưng sai ý định, như robot HalfCheetah học cách ngã hoặc lộn thay vì chạy. Vấn đề exploration–exploitation là một trong những khó khăn nền tảng của RL.

**6. Khả năng generalization kém**

Một policy có thể rất giỏi trong đúng môi trường huấn luyện nhưng thất bại khi môi trường thay đổi nhẹ. Ví dụ, agent được huấn luyện đối đầu với một đối thủ cụ thể có thể không chơi tốt trước đối thủ khác.

**7. Kết quả không ổn định và khó tái lập**

Deep RL rất nhạy với random seed, hyperparameter, implementation và scale của reward. Tác giả đưa ví dụ cùng một cấu hình nhưng nhiều lần chạy cho kết quả khác nhau; thậm chí 25–30% run có thể thất bại chỉ vì ngẫu nhiên.

**9. Khi nào deep RL có thể phù hợp?**

RL dễ thành công hơn khi có thể sinh rất nhiều dữ liệu, bài toán được đơn giản hóa, có self-play, reward rõ ràng và khó bị “hack”, hoặc reward đủ dày để phản hồi nhanh.

**10. Tác giả bi quan ngắn hạn nhưng lạc quan dài hạn**

Ông cho rằng deep RL hiện tại còn lộn xộn, chưa đáng tin cậy, nhưng vẫn có thể tiến bộ nhờ model-based RL, thêm learning signal, imitation/inverse RL, transfer learning, meta-learning, prior tốt hơn và môi trường huấn luyện đa dạng hơn. Kết luận là: hiện tại deep RL chưa phải công nghệ plug-and-play, nhưng tương lai có thể khác.

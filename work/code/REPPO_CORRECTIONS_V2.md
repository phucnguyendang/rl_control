# REPPO port V2 - xác nhận đính chính

Gói V2 này thay thế gói ZIP cùng tên cũ.

Các câu sai đã bị loại bỏ:

- Sai: actor trong fork RSL-RL cũ không có LayerNorm ở mọi hidden layer.
- Sai: fork RSL-RL không triển khai recurrent/CNN REPPO.

Thông tin đúng:

- `cvoelcker/rsl_rl/rsl_rl/networks/mlp.py` dùng `Linear -> LayerNorm -> activation` ở mỗi hidden block; actor và critic `ActorQ` đều sử dụng `MLP` này.
- Fork có cả `ActorQRecurrent` và `ActorQCNN`; chỉ port RSL-RL 5.x hiện tại giới hạn ở feedforward MLP.
- Standalone PyTorch mới hơn dùng normalized blocks với `RMSNorm`; paper thảo luận LayerNorm. Port giữ `nn.LayerNorm` từ fork và đổi activation mặc định sang SiLU.
- Nhận định timeout leakage là suy luận từ code: fork xóa timeout khỏi `done`, nhưng TD-lambda recursion chỉ kiểm tra `done`; với MJLab auto-reset, bước kế tiếp là observation sau reset. Đây không phải câu được tác giả paper tự gọi là bug.
- `target_saturation` là metric chẩn đoán do port bổ sung, không phải technique lấy từ fork hoặc paper.

Để kiểm tra nhanh gói đúng, chạy:

```bash
unzip -p REPPO_RSLRL5_v2_verified.zip REPPO_PORT_GUIDE_VI.md \
  | grep -n "Bản V2 đã xác minh"
```

Lệnh phải in ra marker `Bản V2 đã xác minh` ở đầu tài liệu.

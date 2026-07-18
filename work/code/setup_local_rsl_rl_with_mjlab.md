# Dùng `rsl_rl` local để custom cùng `mjlab`

## 1. Mục tiêu

Thay package `rsl-rl-lib` cài từ PyPI bằng source local ở chế độ **editable**, để:

* Tiếp tục sử dụng các task, environment và script training có sẵn của `mjlab`.
* Có thể sửa trực tiếp code trong repository `rsl_rl`.
* Không cần cài lại package sau mỗi lần sửa file Python.
* Tiếp tục sử dụng Torch, CUDA và các dependency đã cài trong Conda environment `rl`.

## 2. Tổ hợp phiên bản đã kiểm tra

```text
Python          : 3.11
mjlab           : 1.5.0
rsl-rl-lib      : 5.4.0
mujoco-warp     : 3.10.0.1
mujoco          : 3.10.0
warp-lang       : 1.14.0
torch           : 2.12.0+cu132
setuptools      : >=77,<82
```

Lưu ý quan trọng:

```text
mujoco-warp==3.10.0.2
```

gây lỗi khi chạy `Mjlab-Cartpole-Swingup` với `num_envs > 1`.

Triệu chứng là tensor joint limit chỉ có world dimension bằng `1`:

```text
data.nworld                  : 2
model.jnt_range shape        : (1, 2, 2)
default_joint_pos shape      : (2, 2)
soft_joint_pos_limits shape  : (1, 2, 2)
```

Khi reset environment số `1`, CUDA báo:

```text
vectorized gather kernel index out of bounds
CUDA error: device-side assert triggered
```

Với `mjlab==1.5.0`, cần giữ:

```text
mujoco-warp==3.10.0.1
```

---

# 3. Quy trình cài `rsl_rl` local

## Bước 1: Kích hoạt Conda environment

```bash
conda activate rl
```

Kiểm tra Python:

```bash
which python
python --version
python -m pip --version
```

Python mong đợi:

```text
/home/vr/ENTER/envs/rl/bin/python
```

## Bước 2: Đi tới workspace

```bash
cd ~/phucnd/learning/rl_control/work/code
```

## Bước 3: Clone đúng phiên bản `rsl_rl`

```bash
git clone \
    --branch v5.4.0 \
    --single-branch \
    https://github.com/leggedrobotics/rsl_rl.git
```

Đi vào repository root:

```bash
cd rsl_rl
```

Kiểm tra:

```bash
git rev-parse HEAD
git describe --tags --exact-match HEAD
```

Kết quả mong đợi:

```text
afa0cb0cfd6cf471a55de992b735b7b90e3d8e89
v5.4.0
```

## Bước 4: Tạo branch để custom

```bash
git switch -c custom-rsl-rl-5.4.0
```

Kiểm tra:

```bash
git branch --show-current
```

Kết quả:

```text
custom-rsl-rl-5.4.0
```

## Bước 5: Gỡ bản `rsl_rl` cài từ PyPI

```bash
python -m pip uninstall -y rsl-rl-lib
```

Việc này chỉ gỡ package `rsl-rl-lib`, không gỡ Torch, Mjlab hoặc các dependency khác.

## Bước 6: Cài phiên bản `setuptools` tương thích

`rsl_rl v5.4.0` cần `setuptools>=77` để xử lý trường `project.license` trong `pyproject.toml`.

Tuy nhiên, `torch==2.12.0+cu132` yêu cầu:

```text
setuptools<82
```

Do đó phải dùng khoảng:

```text
setuptools>=77,<82
```

Cài bằng Conda:

```bash
conda install -n rl "setuptools>=77,<82"
```

Kiểm tra:

```bash
python - <<'PY'
import setuptools

print(setuptools.__version__)
PY
```

Version phải:

```text
>=77
<82
```

Ví dụ, `setuptools 81.x` là phù hợp.

## Bước 7: Giữ đúng phiên bản `mujoco-warp`

```bash
python -m pip install \
    --force-reinstall \
    --no-deps \
    "mujoco-warp==3.10.0.1"
```

`--no-deps` bảo đảm pip không thay đổi Torch, CUDA, MuJoCo, Warp hoặc các package đang tương thích khác.

Kiểm tra:

```bash
python - <<'PY'
from importlib.metadata import version

print("mujoco-warp:", version("mujoco-warp"))
PY
```

Kết quả:

```text
mujoco-warp: 3.10.0.1
```

## Bước 8: Cài source local ở chế độ editable

Phải chạy lệnh này tại repository root, nơi chứa `pyproject.toml`:

```bash
cd ~/phucnd/learning/rl_control/work/code/rsl_rl

ls pyproject.toml
```

Sau đó cài:

```bash
python -m pip install -e . \
    --no-deps \
    --no-build-isolation
```

Ý nghĩa:

* `-e .`: cài source hiện tại ở chế độ editable.
* `--no-deps`: không cài lại Torch và các dependency khác.
* `--no-build-isolation`: sử dụng `setuptools` đang có trong Conda environment, không tạo build environment tạm và không tải lại package.

---

# 4. Xác nhận editable install

## Kiểm tra đường dẫn import

```bash
python -c "import rsl_rl; print(rsl_rl.__file__)"
```

Kết quả phải trỏ về source local:

```text
/home/vr/phucnd/learning/rl_control/work/code/rsl_rl/rsl_rl/__init__.py
```

Không được trỏ về:

```text
/home/vr/ENTER/envs/rl/lib/python3.11/site-packages/rsl_rl/...
```

## Kiểm tra metadata của package

```bash
python -m pip show rsl-rl-lib
```

Output nên có:

```text
Name: rsl-rl-lib
Version: 5.4.0
Editable project location: /home/vr/phucnd/learning/rl_control/work/code/rsl_rl
```

## Kiểm tra `direct_url.json`

```bash
python - <<'PY'
from importlib import metadata

dist = metadata.distribution("rsl-rl-lib")
print(dist.read_text("direct_url.json"))
PY
```

Kết quả mong đợi:

```json
{
  "dir_info": {
    "editable": true
  },
  "url": "file:///home/vr/phucnd/learning/rl_control/work/code/rsl_rl"
}
```

## Kiểm tra toàn bộ phiên bản quan trọng

```bash
python - <<'PY'
from importlib.metadata import version
import sys
import mjlab
import rsl_rl
import torch

print("Python          :", sys.executable)
print("mjlab source    :", mjlab.__file__)
print("rsl_rl source   :", rsl_rl.__file__)
print("torch source    :", torch.__file__)
print("torch CUDA      :", torch.version.cuda)
print()

for package in [
    "mjlab",
    "rsl-rl-lib",
    "mujoco-warp",
    "mujoco",
    "warp-lang",
    "torch",
    "torchvision",
    "tensordict",
    "numpy",
    "setuptools",
]:
    print(f"{package:16}: {version(package)}")
PY
```

## Kiểm tra dependency

```bash
python -m pip check
```

Kết quả đúng:

```text
No broken requirements found.
```

Nếu xuất hiện:

```text
torch 2.12.0+cu132 has requirement setuptools<82,
but you have setuptools 82.0.1
```

hãy hạ riêng `setuptools`:

```bash
conda install -n rl "setuptools>=77,<82"
```

Sau đó kiểm tra lại:

```bash
python -m pip check
```

---

# 5. Chạy training bằng Mjlab

Không dùng:

```bash
uv run train ...
```

khi đang đứng trong repository `rsl_rl`.

Repository `rsl_rl` có `pyproject.toml`, nên `uv run` sẽ coi nó là một uv project riêng, tạo hoặc đồng bộ `.venv`, sau đó tải lại Torch, CUDA libraries và các dependency khác.

Thay vào đó, dùng trực tiếp Python của Conda environment:

```bash
conda activate rl

cd ~/phucnd/learning/rl_control/work/code/rsl_rl

python -m mjlab.scripts.train \
    Mjlab-Cartpole-Swingup \
    --env.scene.num-envs 4096
```

Lệnh này:

* Dùng Python của Conda environment `rl`.
* Dùng Torch và CUDA đã cài trong environment.
* Dùng task Cartpole từ package `mjlab`.
* Dùng source `rsl_rl` local ở chế độ editable.
* Không tạo `.venv`.
* Không tải lại Torch.

Có thể xác nhận ngay trước khi train:

```bash
python - <<'PY'
import sys
import mjlab
import rsl_rl
import torch

print("Python :", sys.executable)
print("Mjlab  :", mjlab.__file__)
print("rsl_rl :", rsl_rl.__file__)
print("Torch  :", torch.__version__)
print("CUDA   :", torch.version.cuda)
PY
```

---

# 6. Custom source `rsl_rl`

Có thể sửa trực tiếp các file như:

```text
rsl_rl/algorithms/ppo.py
rsl_rl/runners/on_policy_runner.py
rsl_rl/modules/actor_critic.py
```

Sau khi sửa file Python, chỉ cần chạy lại training:

```bash
python -m mjlab.scripts.train \
    Mjlab-Cartpole-Swingup \
    --env.scene.num-envs 4096
```

Không cần chạy lại:

```bash
python -m pip install -e .
```

Chỉ cần cài lại package khi thay đổi metadata hoặc build configuration, ví dụ:

```text
pyproject.toml
dependencies
console scripts
package structure
```

---

# 7. Cấu hình cuối cùng mong đợi

```text
workspace/
└── code/
    └── rsl_rl/
        ├── pyproject.toml
        ├── rsl_rl/
        │   ├── algorithms/
        │   ├── modules/
        │   ├── runners/
        │   └── __init__.py
        └── ...
```

Python environment:

```text
/home/vr/ENTER/envs/rl
```

Source editable:

```text
/home/vr/phucnd/learning/rl_control/work/code/rsl_rl
```

---

# Appendix: Xử lý lỗi

## A. Terminal không tìm thấy lệnh `python`

Lỗi:

```text
Command 'python' not found
```

Nguyên nhân: Conda environment chưa được kích hoạt.

Chạy:

```bash
conda activate rl
```

Kiểm tra:

```bash
which python
```

Kết quả:

```text
/home/vr/ENTER/envs/rl/bin/python
```

Nếu `conda activate` chưa khả dụng:

```bash
source ~/ENTER/etc/profile.d/conda.sh
conda activate rl
```

---

## B. Không tìm thấy `pyproject.toml`

Lỗi:

```text
does not appear to be a Python project
neither setup.py nor pyproject.toml found
```

Nguyên nhân thường là đang đứng trong thư mục package con:

```text
.../rsl_rl/rsl_rl
```

Lùi về repository root:

```bash
cd ~/phucnd/learning/rl_control/work/code/rsl_rl

ls pyproject.toml
```

Sau đó chạy lại:

```bash
python -m pip install -e . \
    --no-deps \
    --no-build-isolation
```

---

## C. Lỗi SSL khi build package

Lỗi:

```text
SSLCertVerificationError
CERTIFICATE_VERIFY_FAILED
```

Không dùng build isolation:

```bash
python -m pip install -e . \
    --no-deps \
    --no-build-isolation
```

Lệnh này sử dụng build tools đang có trong Conda environment thay vì tải một build environment mới.

---

## D. Lỗi `project.license`

Lỗi:

```text
configuration error: project.license
GIVEN VALUE: "BSD-3-Clause"
```

Nguyên nhân: `setuptools` quá cũ.

Cài version phù hợp với cả `rsl_rl` và Torch:

```bash
conda install -n rl "setuptools>=77,<82"
```

Kiểm tra:

```bash
python -c "import setuptools; print(setuptools.__version__)"
```

Sau đó cài lại editable package:

```bash
python -m pip install -e . \
    --no-deps \
    --no-build-isolation
```

---

## E. `pip check` báo Torch không tương thích với `setuptools`

Lỗi:

```text
torch 2.12.0+cu132 has requirement setuptools<82,
but you have setuptools 82.0.1
```

Hạ riêng `setuptools`:

```bash
conda install -n rl "setuptools>=77,<82"
```

Không cần cài lại Torch.

Kiểm tra lại:

```bash
python -m pip check
```

---

## F. `uv run` tải lại Torch và CUDA packages

Triệu chứng:

```text
Building rsl-rl-lib @ file:///.../rsl_rl
Preparing packages...
torch
nvidia-cublas
nvidia-cudnn
nvidia-cusparse
...
```

Nguyên nhân:

* Đang đứng trong repository `rsl_rl`.
* `uv` phát hiện `pyproject.toml`.
* `uv` coi `rsl_rl` là project cần quản lý.
* `uv` tạo hoặc đồng bộ một environment khác với Conda environment `rl`.

Không dùng:

```bash
uv run train ...
```

Dùng:

```bash
python -m mjlab.scripts.train \
    Mjlab-Cartpole-Swingup \
    --env.scene.num-envs 4096
```

Nếu `uv` đã tạo `.venv` ngoài ý muốn:

```bash
cd ~/phucnd/learning/rl_control/work/code/rsl_rl

ls -ld .venv 2>/dev/null
git status --short
```

Nếu `.venv` chỉ được tạo ngoài ý muốn:

```bash
rm -rf .venv
```

Nếu `uv.lock` là file untracked vừa được tạo:

```bash
rm -f uv.lock
```

Không xóa `uv.lock` nếu repository đang chủ động quản lý file đó.

---

## G. CUDA báo `vectorized gather kernel index out of bounds`

Lỗi:

```text
vectorized gather kernel index out of bounds
CUDA error: device-side assert triggered
```

Lỗi xuất hiện khi:

```bash
--env.scene.num-envs 2
```

hoặc lớn hơn, trong khi một environment vẫn chạy được.

Kiểm tra version:

```bash
python - <<'PY'
from importlib.metadata import version

print("mjlab       :", version("mjlab"))
print("mujoco-warp :", version("mujoco-warp"))
PY
```

Với tổ hợp này:

```text
mjlab==1.5.0
```

cần dùng:

```text
mujoco-warp==3.10.0.1
```

Cài lại riêng package này:

```bash
python -m pip install \
    --force-reinstall \
    --no-deps \
    "mujoco-warp==3.10.0.1"
```

Không cần cài lại:

```text
torch
torchvision
mjlab
mujoco
warp-lang
numpy
rsl-rl-lib
```

Sau đó mở process Python mới và thử:

```bash
python -m mjlab.scripts.train \
    Mjlab-Cartpole-Swingup \
    --env.scene.num-envs 2
```

Rồi:

```bash
python -m mjlab.scripts.train \
    Mjlab-Cartpole-Swingup \
    --env.scene.num-envs 4096
```

---

## H. Kiểm tra Mjlab yêu cầu version nào của `rsl_rl`

```bash
python - <<'PY'
from importlib.metadata import requires, version

print("mjlab:", version("mjlab"))
print("rsl-rl-lib:", version("rsl-rl-lib"))

for requirement in requires("mjlab") or []:
    if "rsl" in requirement.lower():
        print("mjlab requires:", requirement)
PY
```

Kết quả mong đợi:

```text
mjlab: 1.5.0
rsl-rl-lib: 5.4.0
mjlab requires: rsl-rl-lib==5.4.0
```

---

## I. Kiểm tra source local có đúng tag hay không

```bash
cd ~/phucnd/learning/rl_control/work/code/rsl_rl

git rev-parse HEAD
git describe --tags --exact-match HEAD
git status --short
git diff --stat v5.4.0
```

Kết quả mong đợi trước khi custom:

```text
afa0cb0cfd6cf471a55de992b735b7b90e3d8e89
v5.4.0
```

Sau khi tạo branch và bắt đầu sửa code, `git describe --tags --exact-match HEAD` có thể không còn trả về tag nếu đã tạo commit mới. Khi đó dùng:

```bash
git merge-base --is-ancestor v5.4.0 HEAD \
    && echo "Branch được phát triển từ v5.4.0"
```

---

## J. Kiểm tra cuối cùng

```bash
conda activate rl

cd ~/phucnd/learning/rl_control/work/code/rsl_rl

python - <<'PY'
from importlib.metadata import version
import sys
import mjlab
import rsl_rl
import torch

print("Python          :", sys.executable)
print("mjlab           :", version("mjlab"))
print("rsl-rl-lib      :", version("rsl-rl-lib"))
print("mujoco-warp     :", version("mujoco-warp"))
print("mujoco          :", version("mujoco"))
print("warp-lang       :", version("warp-lang"))
print("torch           :", version("torch"))
print("setuptools      :", version("setuptools"))
print("torch CUDA      :", torch.version.cuda)
print("mjlab source    :", mjlab.__file__)
print("rsl_rl source   :", rsl_rl.__file__)
PY

python -m pip check
```

Kết quả quan trọng:

```text
mjlab           : 1.5.0
rsl-rl-lib      : 5.4.0
mujoco-warp     : 3.10.0.1
mujoco          : 3.10.0
warp-lang       : 1.14.0
torch           : 2.12.0+cu132
setuptools      : >=77,<82
rsl_rl source   : /home/vr/phucnd/learning/rl_control/work/code/rsl_rl/rsl_rl/__init__.py
```

Và:

```text
No broken requirements found.
```

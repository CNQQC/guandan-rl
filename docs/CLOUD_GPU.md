# 云 GPU 训练（Linux / NVIDIA）

本仓库使用 CPU 进程采样、单张 GPU 更新模型，目前没有多 GPU / DDP 实现。CUDA 配置尚未在真实 NVIDIA 云 GPU 上验证；先完成冒烟测试，再开始长训。

## 获取代码与安装环境

在云机器的持久化磁盘中执行：

```bash
git clone --recurse-submodules https://github.com/CNQQC/guandan-rl.git
cd guandan-rl
nvidia-smi
# 已有 uv 可跳过这一行
python3 -m pip install --user uv
uv python install 3.12
uv sync --locked
```

若 `uv` 不在 PATH 中，将安装器提示的可执行文件目录加入 PATH。项目要求 Python 3.12。`vendor/DanLM` 包含 GPU 配置所需的纯 Python 竞赛策略，它们不依赖 DanLM 的 Mac 二进制。

锁文件当前固定 PyTorch 2.14.0，Linux 依赖包含 CUDA 13 组件。驱动须兼容所安装的 PyTorch CUDA 构建。需要其他构建时，根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 为 `.venv` 安装符合 `torch>=2.6,<3` 的版本；用 `uv pip install --python .venv/bin/python ...` 执行所选安装参数。自定义安装后直接用 `.venv/bin/python` 或 `uv run --no-sync`，避免普通 `uv run` 恢复锁文件中的版本。

```bash
.venv/bin/python -m guandan doctor
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA 不可用：检查 GPU 挂载、驱动和 PyTorch 构建'
x = torch.ones(8, device='cuda')
print('GPU:', torch.cuda.get_device_name(0))
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('CUDA compute:', (x @ x).item())
PY
```

Linux 上 `danlm_pretrained_available: false` 是预期结果，不影响本地训练。`expert-eval` 和公开模型试玩仅适用于其上游支持的 Mac 环境。

## 先验证，再长训

```bash
.venv/bin/python -m pytest -q
# 用正式 GPU 配置缩小采样量，验证 CUDA、子进程和竞赛策略
.venv/bin/python -u -m guandan train --config configs/gpu.toml \
  --out runs/gpu-smoke --iterations 1 --games-per-iteration 4 \
  --workers 2 --updates 1 --eval-every 1 --eval-pairs 1 --max-minutes 10

# 正式训练：最多 12 小时，或 100000 次迭代，以先达到的为准
.venv/bin/python -u -m guandan train --config configs/gpu.toml --out runs/gpu
```

长训放在云平台持久会话或 `tmux` 中运行。CPU 资源不足时可用 `--workers 2 --threads 2` 开始。GPU 利用率受 CPU 规则枚举和采样限制，应根据实际吞吐量调整 worker 数。

## 保存、恢复与评测

按 `Ctrl+C` 或在另一终端执行 `touch runs/gpu/STOP` 请求在批次边界保存停止。批次及评测可能超出时间预算，给云实例关机留出余量。

```bash
# 若曾用 STOP 文件停止，恢复前删除该停止标记
rm -f runs/gpu/STOP
.venv/bin/python -u -m guandan train --config configs/gpu.toml \
  --out runs/gpu --resume runs/gpu/latest.pt \
  --iterations 100000 --max-minutes 720

.venv/bin/python -m guandan eval --a runs/gpu/latest.pt --b team-rule \
  --pairs 500 --seed 9000019 --out reports/gpu-holdout.json
```

`--iterations` 是本次额外迭代次数。迁移机器时复制整个运行目录（包括 `pool/`），保留相同相对路径；不要只复制 `latest.pt`。强制终止可能遗留 `.train.lock`，确认旧进程退出后才能清理该锁。`runs/`、`reports/` 和模型权重被 Git 忽略，需自行备份到持久磁盘或对象存储。

**从本机已有模型续训**：先上传整个运行目录，使用原训练配置并覆盖 `--device cuda`。例如 Mac 模型使用 `--config configs/mac.toml --device cuda`，不能直接换为 `configs/gpu.toml`：`small` 与 `full` 结构不同，续训还会检查随机种子、学习率、经验池大小等参数一致。GPU 配置默认训练新的 `full` 模型。

仓库没有提供已验证达到人类高手水平的自训练权重。评测结果须区分开发集、保留测试集和真人对局。

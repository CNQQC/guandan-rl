# 掼蛋研究室 · GuanDan Lab

一个在 Apple Silicon Mac 上验证过的强化学习掼蛋项目：可试玩的公开预训练模型、源码可改的自我对弈训练、同牌换队评测，以及中文控制台。

**目标是明显超过普通玩家；当前交付不等于已经证明达标。** 自训练模型从零开始，短训结果很弱。DanLM 公开权重是独立参考模型，其原始训练代码未公开，不能与本项目的自训练成果混为一谈。

## GitHub 与云 GPU

```bash
git clone --recurse-submodules https://github.com/CNQQC/guandan-rl.git
cd guandan-rl
```

**Linux / NVIDIA 云 GPU：请先阅读 [云 GPU 训练指南](docs/CLOUD_GPU.md)**，包含 CUDA 检查、首次训练、断点续训和持久化目录说明。上游代码通过固定版本的 Git 子模块提供，必须使用 `--recurse-submodules`；已克隆的仓库可执行 `git submodule update --init --recursive`。

仓库不包含本机 `runs/`、`reports/` 和虚拟环境。下文 Mac 试训成绩是开发记录，对应检查点与报告需从原机器另行复制。Linux 可使用本地训练引擎和纯 Python 竞赛基线；DanLM 公开模型的 Mac 二进制不能在 Linux 运行。

## 立即使用

在本目录执行：

```bash
uv sync --locked
uv run python -m guandan doctor
uv run python -m guandan serve
```

打开 **http://127.0.0.1:7860**。已经安装好的环境也可直接运行 `.venv/bin/python -m guandan serve`。

- **对战**：与规则基线或自己的检查点对战，查看提示和出牌记录。
- **挑战公开预训练模型**：进入 `/expert/`，默认 DanLM V1；支持单副牌和从 2 打到 A 的完整对局。
- **训练**：启动有时间上限的本地训练，查看真实损失曲线，保存并停止。
- **评测**：读取 `reports/` 中的实际评测记录；明确区分两套规则引擎。

控制台只绑定本机 `127.0.0.1`。训练不会调用收费 API，也不需要大语言模型账号。

## 已实测的结果

2026-09-18，当前 12 核 / 18 GB Apple Silicon Mac：

| 项目 | 实测结果 | 如何解读 |
|---|---|---|
| 工程测试 | 36 项通过 | 覆盖规则边界、隐私输入、续训、NumPy 导出一致性与完整网页牌局 |
| MPS 自训练 | 80 次迭代，640 局，41,195 样本，640 次梯度更新，约 157 秒 | 证明训练能真实运行，不证明棋力 |
| 初始自训练模型 vs `team-rule` | 100 局，4.0%；配对区间约 1.0%–13.5% | 明显未达到目标，需继续训练与改进 |
| DanLM 公开模型 vs `fin-njupt-guandan-ai` | 40 局，75.0% | 比竞赛规则基线强的初步证据；不是本地模型成绩，也不是真人验证 |

原始文件：`runs/mac-mps/status.json`、`runs/mac-mps/metrics.jsonl`、`reports/local-vs-team-rule-100.json`、`reports/danlm-vs-competition-40.json`。评测报告含随机种子、模型 SHA-256；本地评测另含每副牌的结果。

DanLM 采用上游独立引擎和评测器，不能与本地引擎分数直接横比。上游报告的是固定队伍座位评测，本地则对同一副牌交换两队策略。40 局样本量较小。

## 与纯程序策略博弈

已将南邮掼蛋 AI 和蛋饼 / NUAA 的 Python 策略直接接入本地规则引擎。它们是固定对手，强化学习模型通过实际整局博弈获得训练样本。网页训练入口使用 `configs/league.toml`；详情、源码链接和命令见 [docs/PROGRAM_BASELINES.md](docs/PROGRAM_BASELINES.md)。

```bash
uv run python -m guandan train --config configs/league.toml --out runs/league
```

## 继续训练

```bash
# 先验证完整管线，约几秒
uv run python -m guandan train --config configs/smoke.toml --out runs/my-smoke

# Mac：自动选择 MPS，两个 CPU 采样进程，默认最多 30 分钟或 200 次迭代
uv run python -m guandan train --config configs/mac.toml --out runs/mac

# 继续已交付的模型；iterations 指本次额外迭代次数
uv run python -m guandan train --config configs/mac.toml \
  --out runs/mac-mps --resume runs/mac-mps/latest.pt \
  --iterations 10000 --max-minutes 30

# GPU 迁移：先安装与机器匹配的 CUDA PyTorch；此配置尚未在 NVIDIA GPU 上实测
uv run python -m guandan train --config configs/gpu.toml --out runs/gpu
```

`--max-minutes` 在一批采样/训练结束的边界检查，因此最后一批及评测可能稍微超时。按 `Ctrl+C` 会在当前批次结束后保存；也可执行 `touch runs/mac/STOP`。再次续训前移除自己创建的 `STOP` 文件。输出目录的 `.train.lock` 防止两个训练同时覆盖模型；进程被强制杀死时，需确认旧进程退出后清理遗留锁。

macOS 普通终端可以使用 MPS；某些应用沙箱会让 PyTorch 报告 MPS 不可用，此时自动退回 CPU。CPU 采样通过 NumPy 数组传递权重，避免依赖受限的 PyTorch 共享内存管理器。

检查点包括：模型、优化器、训练配置、随机状态、经验回放、迭代/局数、历史对手池引用。续训时保留整个运行目录，包括 `pool/`。`small` 和 `full` 是不同网络结构，不能直接互相续训。保留测试集的 seed 不应用于训练或调参。

## 训练方法

- **DMC（Deep Monte Carlo）**：用完整一副牌的团队收益训练 `Q(观察, 合法动作)`，头二游同队 `+3`，头三游同队 `+2`，头末游同队 `+1`；另一队取负值，训练时除以 3。
- **小型因果 Transformer + 手牌/动作网络**：只读取自己的手牌、公开历史、剩余张数等可见信息。花色信息进入特征，能够区分保留不同花色的选择。
- **辅助学习**：下一事件 token 预测，以及其他三家手牌点数分布预测。真实隐藏手牌只用于训练标签，决策输入不包含它们。
- **历史对手池**：纯自我对弈、团队规则对手、冻结的历史检查点混合；仅将学习方的动作写入其训练数据。
- **全面探索**：epsilon 探索覆盖全部枚举动作，不把模型永久限制在当前 top-k 以内。
- **晋级评测**：定期对 `team-rule` 和当前 champion 做同牌换队开发集评测；冠军晋级要求区间下界超过 50%。第一个 champion 仅为初始参照，不代表已达到任何棋力门槛。

这是同步 CPU actor + 单 learner 的可读实现；CUDA 能加速学习，但 CPU 规则枚举和推理会限制吞吐量。大规模训练应进一步实现集中批量推理、原生规则内核，并做吞吐量与棋力消融实验。没有声称 GPU 配置已达工业规模。

## 评测

```bash
# 真正相同的牌、级牌和进贡设置，各交换一次两队；500 组 = 1000 局
uv run python -m guandan eval --a runs/mac-mps/latest.pt --b team-rule \
  --pairs 500 --seed 9000019 --out reports/holdout-team-rule.json

# 跨版本对比，避免只会利用固定规则对手
uv run python -m guandan eval --a runs/mac-mps/latest.pt \
  --b runs/mac-mps/champion.pt --pairs 500 --seed 9100199 \
  --out reports/holdout-champion.json

# Mac 专用的公开参考模型评测；默认对南邮竞赛基线
uv run python -m guandan expert-eval --games 100 \
  --opponent bot:fin-njupt-guandan-ai --out reports/danlm-competition.json
```

本地胜率区间以同牌的两局为一个聚类单位进行 bootstrap，并用保守的 Wilson 外包络防止“少量全胜”出现虚假的零宽区间。这里的有效独立样本量是副牌组数，而非两倍局数。开发集与单次预设的保留测试集应分开；反复依据保留集挑模型会污染测试。

**“大幅超越普通人”的验收建议**：提前登记普通玩家的选取标准和经验范围；采用固定搭档、统一规则、随机座位、无提示/无透视、同牌交换与跨玩家分组的盲测；例如把队伍胜率至少 70%、95% 区间下界超过 60% 作为项目目标。数值是待验证的目标，当前没有真人实验。界面中的“你 + AI”练习不是“AI 对纯真人队伍”的验证。

## 规则范围与限制

详见 [docs/RULES.md](docs/RULES.md)。本地训练为 **Botzone 风格单副掼蛋**，含随机级牌、单贡/双贡/抗贡、接风与双下终局；进贡与还贡目前采用固定合法策略。**尚未学习进还贡选择，也没有训练从 2 到 A 的跨副长期决策。** `/expert/` 的完整升级对战来自上游 DanLM，不应归到本地训练模型能力名下。

不同赛事关于双配子、同贡、还贡及过 A 有变体。项目把使用的选择写入规则文档；并未声称通过国家竞赛规则认证。

## 工程验证与目录

```bash
uv run pytest -q
uv run ruff check guandan tests
```

```text
guandan/          本地训练、对手池、评测、CLI、网页服务
fabledan/         有明确修改记录的 FableDan 规则/编码/网络源码
configs/          smoke、Mac、程序对抗 league、CUDA 配置
web/              中文牌桌、训练与评测界面
tests/            规则与训练正确性测试
runs/             本地训练日志、检查点、历史对手
reports/          带种子和模型哈希的评测报告
vendor/FableDan/  上游固定版本源码，保留原样
vendor/DanLM/     上游固定版本预训练模型和 Mac 二进制，保留原样
```

首次接入并不是从头重新发明全部算法：本项目基于 [FableDan](https://github.com/lrx0716/FableDan) 做工程修订，并将 [DanLM](https://github.com/dashidhy/DanLM) 作为独立公开参考。算法背景参考 [DanZero 论文](https://arxiv.org/abs/2210.17087)。上游二者均附有非商业限制，适合本次个人研究；商业用途需另行获得作者许可。版本、许可证和本地修改见 [THIRD_PARTY.md](THIRD_PARTY.md)。

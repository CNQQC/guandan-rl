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

- **对战**：与规则基线、竞赛程序或任意一个训练节点（最新模型 / 当前冠军 / 每个历史快照）对战，查看提示和出牌记录。
- **挑战公开预训练模型**：进入 `/expert/`，默认 DanLM V1；支持单副牌和从 2 打到 A 的完整对局。
- **模型节点**：每个检查点都按「节点 · 迭代 · 开发集胜率 · 95% 区间 · 文件大小 · 生成时间」列出，可直接试玩或评测；训练记录可以起备注名，之后在所有下拉框里都用这个名字。
- **训练**：在网页上配置全部训练参数（基于 configs/*.toml 或继续已有训练；实时校验并给出等价 TOML 与命令行），启动有时间上限的本地训练，一键暂停/继续（不中断进程），实时查看训练进度与剩余时间、损失分解（总损失 / Q / NTP / 信念）、开发集胜率与 95% 置信区间、晋级记录、对手构成、采样吞吐、控制台日志与运行配置；可切换 EMA 平滑、对数纵轴和数据表，随时保存并停止。
- **评测**：直接发起同牌换队评测——任选模型节点与对手、设定组数与随机种子，实时显示进度与当前胜率；结果写入 `reports/` 并与历史记录一同列出，明确区分两套规则引擎。

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

# Mac：CPU learner + 4 个采样进程，采样与学习并行，默认最多 8 小时
# 结束训练的一直是 max_minutes 而不是 iterations——后者被有意设得够大
uv run python -m guandan train --config configs/mac.toml --out runs/mac

# 继续已交付的模型；iterations 指本次额外迭代次数
uv run python -m guandan train --config configs/mac.toml \
  --out runs/mac-mps --resume runs/mac-mps/latest.pt \
  --iterations 10000 --max-minutes 30

# GPU 迁移：先安装与机器匹配的 CUDA PyTorch；此配置尚未在 NVIDIA GPU 上实测
uv run python -m guandan train --config configs/gpu.toml --out runs/gpu
```

**暂停不是停止，也不是续训。**网页的「暂停 / 继续训练」按钮，或命令行的 `touch runs/mac/PAUSE` 与 `rm runs/mac/PAUSE`，只是让训练循环在迭代边界原地等待：进程不退出，模型、优化器、经验回放和随机状态全都留在内存里，继续时从下一个迭代号接着跑，不重新加载任何东西。暂停时长不计入 `max_minutes`，也不会拉低 `games_per_minute`。与之相对，`--resume`（网页上的「继续训练」下拉项）是另一回事：那是让**新进程**从 `latest.pt` 把这些状态重建出来，九项参数必须与检查点一致。`PAUSE` 是运行期状态，训练进程启动时一定会清掉遗留的标记；`STOP` 是持久意图，需要你自己删除。

`--max-minutes` 在一批采样/训练结束的边界检查，因此最后一批及评测可能稍微超时。按 `Ctrl+C` 会在当前批次结束后保存；也可执行 `touch runs/mac/STOP`。再次续训前移除自己创建的 `STOP` 文件。输出目录的 `.train.lock` 防止两个训练同时覆盖模型；进程被强制杀死时会留下遗留锁，网页「管理记录 → 清理训练锁」会先核对锁里记录的进程号，确认该进程确实已退出才允许清理，命令行下手动删除该文件同理。

macOS 普通终端可以使用 MPS；某些应用沙箱会让 PyTorch 报告 MPS 不可用，此时自动退回 CPU。CPU 采样通过 NumPy 数组传递权重，避免依赖受限的 PyTorch 共享内存管理器。

**`small` 网络在 Apple silicon 上不要用 MPS。**实测（M3 Pro，12 核）MPS 比 CPU 更慢：模型太小，kernel 启动开销压过算力收益，因此 `configs/mac.toml` 固定 `device = "cpu"`。`full` 网络或 CUDA 机器另当别论，`device = "auto"` 仍然可用。

## 吞吐量

训练管线有三个与学习目标无关、只影响速度的开关，默认全开，可在网页控制台或 TOML 里关掉：

| 开关 | 作用 |
| --- | --- |
| `bucket_batches` | 每个批次从「按 token 长度排序后的回放缓冲」里取一段连续窗口。牌局历史长度差异很大（中位 214、最长 490），随机批要 padding 到批内最长，**约 58% 的 Transformer 算力花在 PAD 上**；等长分桶把这部分基本消掉。窗口跨尾部回绕，所以每条样本被抽中的边际概率仍然严格均匀，只有「批次如何分组」变了。 |
| `pipeline` | 下一轮采样在本轮梯度更新**之前**派发，actor 在 learner 计算期间继续打牌。代价是 actor 的权重晚一轮，这是 actor-learner 架构的常规取舍。仅在 `workers > 1` 时生效。 |
| `save_every` | 完整检查点（含经验回放）的写盘间隔；体积随 `replay_size` 走，8192 条约 16 MB、131072 条约 220 MB，后者写一次约 1.8 秒。`Ctrl+C`、`STOP`、预算到期和正常结束都一定会写完整检查点；只有被 `kill -9` 强杀时才最多丢失 `save_every` 轮，此时 `metrics.jsonl` 会比 `latest.pt` 多出几行，续训后这几个迭代号会在日志里重复出现。 |

在 M3 Pro（12 核）上按相同墙钟时间实测，相对改动前的 `configs/mac.toml`：

| 配置 | 对局吞吐 | 梯度吞吐 | learner 等待 actor |
| --- | --- | --- | --- |
| 改动前（MPS，2 采样进程，bs 32） | 281 局/分 | 8 986 样本/分 | 39% |
| 改动前但换 CPU | 336 局/分 | 10 754 样本/分 | 45% |
| 只加等长分桶 | 461 局/分 | 14 758 样本/分 | 63% |
| **现在的 `configs/mac.toml`** | **666 局/分** | **56 815 样本/分** | **3%** |

即对局吞吐 2.4 倍、梯度吞吐 6.3 倍。瓶颈已经从「采样串行等待」转移到 learner 本身，所以采样进程从 6 降到 4、把核让给 `threads`，反而更快。想进一步提速需要改模型或编码（例如截断 token 上下文窗口），那会改变 `ENCODING_VERSION`，不在本次改动范围内。

检查点包括：模型、优化器、训练配置、随机状态、经验回放、迭代/局数、历史对手池引用。续训时保留整个运行目录，包括 `pool/`。**每次保存快照后会自动清理 `pool/` 中不再使用的文件**，只保留当前对手池和开发集胜率最高的那个快照；网页的「模型节点」面板也可以手动清理历史遗留快照，并显示会释放多少空间。`small` 和 `full` 是不同网络结构，不能直接互相续训。保留测试集的 seed 不应用于训练或调参。

训练记录的日常管理都在网页「训练」页完成。记录下拉框旁的**继续训练**直接从该记录的 `latest.pt` 接着训练：点下会把上方的配置切换成这条记录保存的参数（种子、网络规模、学习率等九项锁定不可改），并列出「从第几次迭代接着跑、再训练多少次、最多多少分钟」让你确认；同一时间只允许一个训练，按钮会写明当前不能续训的原因。**管理记录**可以归档（只把记录收进下拉框底部的「已归档」分组，磁盘文件不动）、清理遗留训练锁、把模型导出成仅含权重的 `.npz`（写入 `exports/`，便于复制到其他机器），以及删除整条记录——删除前会列出迭代数、快照数和占用空间并要求二次确认，正在运行或仍被进程持锁的记录会被拒绝。模型节点表格里每个检查点还可以单独「导出」或「删除」——删除前会说明这个文件具体意味着什么（`latest.pt` 删了就不能再续训、`champion.pt` 删了下次评测会重建冠军、快照会说明它是否仍在对手池里），同样需要二次确认。

## 训练方法

- **DMC（Deep Monte Carlo）**：用完整一副牌的团队收益训练 `Q(观察, 合法动作)`，头二游同队 `+3`，头三游同队 `+2`，头末游同队 `+1`；另一队取负值，训练时除以 3。
- **小型因果 Transformer + 手牌/动作网络**：只读取自己的手牌、公开历史、剩余张数等可见信息。花色信息进入特征，能够区分保留不同花色的选择。
- **辅助学习**：下一事件 token 预测，以及其他三家手牌点数分布预测。真实隐藏手牌只用于训练标签，决策输入不包含它们。
- **历史对手池**：纯自我对弈、团队规则对手、冻结的历史检查点混合；仅将学习方的动作写入其训练数据。对手池保留最近 `pool_recent` 个快照，更早的按至少 `pool_archive_every` 次迭代的间隔稀疏归档，因此回溯范围约 `(pool_size - pool_recent) × pool_archive_every` 次迭代——只留最近几个快照会让策略打得过刚才的自己、打不过几百次迭代前的自己，从而在原地绕圈。
- **全面探索**：epsilon 探索覆盖全部枚举动作，不把模型永久限制在当前 top-k 以内。
- **晋级评测**：定期对 `team-rule` 和当前 champion 做同牌换队开发集评测；冠军晋级要求胜率超过 `gate_threshold`（默认 50%）。早先的判据是置信区间下界超过 50%，但 40 局的下界要到约 72% 胜率才越得过去，冠军因此会卡在半程；要降低这个判断的噪声，该调的是 `eval_pairs` 而不是门槛。第一个 champion 仅为初始参照，不代表已达到任何棋力门槛。

这是 CPU actor + 单 learner 的可读实现（默认采样与学习流水线并行，见上文「吞吐量」）；CUDA 能加速学习，但 CPU 规则枚举和推理会限制吞吐量。大规模训练应进一步实现集中批量推理、原生规则内核，并做吞吐量与棋力消融实验。没有声称 GPU 配置已达工业规模。

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
exports/          从网页导出的仅权重 .npz，可复制到其他机器
reports/          带种子和模型哈希的评测报告
vendor/FableDan/  上游固定版本源码，保留原样
vendor/DanLM/     上游固定版本预训练模型和 Mac 二进制，保留原样
```

首次接入并不是从头重新发明全部算法：本项目基于 [FableDan](https://github.com/lrx0716/FableDan) 做工程修订，并将 [DanLM](https://github.com/dashidhy/DanLM) 作为独立公开参考。算法背景参考 [DanZero 论文](https://arxiv.org/abs/2210.17087)。上游二者均附有非商业限制，适合本次个人研究；商业用途需另行获得作者许可。版本、许可证和本地修改见 [THIRD_PARTY.md](THIRD_PARTY.md)。

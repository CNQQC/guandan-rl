# 强化学习直接对抗纯程序策略

已接入两个公开竞赛策略的 Python 源码。它们不使用神经网络、不需要训练权重，也不依赖 DanLM 的 Mac 二进制。它们和本地强化学习模型使用**同一个规则引擎、同一个观察协议、同一份合法动作集合**。

| 策略标识 | 源码 | 策略特点 | 本机同牌换队实测 |
|---|---|---|---|
| `program:njupt` | [南邮掼蛋 AI](https://github.com/dashidhy/DanLM/tree/main/baselines/fin-njupt-guandan-ai) | 候选出牌评分、拆牌代价、配子消耗、对手尾牌与搭档配合 | 对 `team-rule`：100 局胜率 80%，区间约 67.0%–88.8% |
| `program:egg-pancake` | [蛋饼 / NUAA](https://github.com/dashidhy/DanLM/tree/main/baselines/2nd-egg-pancake) | 对出牌后的剩余手牌做加权估值，考虑炸弹/同花顺和队友牌权 | 对 `team-rule`：100 局胜率 68%，区间约 54.2%–79.2% |

这只是对弱规则基线的本地对照成绩；不把目录中的历史比赛排名当作当前实力排名，也不据此推断已经超过普通人。源代码来自 DanLM 公开整理的竞赛归档，保留作者注释和原许可证条款。

## 实际训练配置

```bash
uv run python -m guandan baselines

# 从零开始，与程序策略及历史模型博弈
uv run python -m guandan train --config configs/league.toml --out runs/league

# 继续本次已经实际跑过的程序对抗模型
uv run python -m guandan train --config configs/league.toml \
  --out runs/league-first --resume runs/league-first/latest.pt \
  --iterations 10000 --max-minutes 30
```

`league.toml` 的采样分布为：

- 约 20% 完全自我对弈，四个座位均由当前学习模型控制。
- 约 40% 当前模型队伍对抗冻结历史模型。
- 约 40% 当前模型队伍对抗程序策略池；南邮、蛋饼、本地 `team-rule` 等概率选取。

程序对手的参数不会更新。强化学习模型在 0/2 与 1/3 队伍间随机切换；只有学习方自己的决策进入其经验回放，以整副牌的团队收益训练 Q 值。日志中的 `modes` 明确记录实际遇到的每类对手，不能把“配置里写了程序名”当作已完成博弈。

默认每 20 次迭代分别测一次各个程序对手和当前 champion，保存在 `eval-program-*.json`。训练网页的「开始训练」也使用这一配置。

## 独立保留集评测

```bash
uv run python -m guandan eval --a runs/league-first/latest.pt \
  --b program:njupt --pairs 500 --seed 9100027 \
  --out reports/league-vs-njupt.json

uv run python -m guandan eval --a runs/league-first/latest.pt \
  --b program:egg-pancake --pairs 500 --seed 9200033 \
  --out reports/league-vs-egg.json
```

这些公开程序现在既是训练对手，也是分项评测参照。若要评价泛化，需另保留没有进入训练池的程序/策略版本、不同随机种子，并最终安排真人盲测；击败训练对手本身不足以证明泛化。

## 适配边界

适配器为 `guandan/programs.py`，可以直接查看：

- 只传自己的手牌、公开剩余张数、当前牌权和环境生成的动作列表。
- 序列牌型按原协议传入最高点数；配子传入实际物理牌；最终动作索引始终由本地规则引擎校验。
- 各归档的同名 Python 模块通过独立加载器隔离，不改全局 `sys.path` 或全局标准输出；随机数也按策略隔离。
- 修正南邮代码的 `all_match` 配子计数逐次累加问题，每次决策重新计数。原始源码保留不变，修订只在适配器中进行。
- 非法返回会报错，不静默替换成随机动作；不存在“程序崩溃算强化学习胜利”的捷径。
- 还贡阶段沿用本地环境的固定策略，这里的比较仅针对出牌决策，不等于比较原竞赛程序的全部完整对局能力。

还可参考 [rlcard-guandan 的 base1–base8](https://github.com/Choysang/rlcard-guandan)（仓库声明 MIT）和 [guandan-arena 的规则/Monte Carlo 策略](https://github.com/shef-wang/guandan-arena)。目前没有把这些额外策略接入训练；它们适合作为后续独立泛化测试候选。

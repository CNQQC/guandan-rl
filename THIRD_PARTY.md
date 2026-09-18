# 来源、版本与本地修订

## FableDan

- 来源：https://github.com/lrx0716/FableDan
- 固定版本：`7cc5e311b9860bc44f76c082d9c1b21fc8b2d3ec`
- 原许可证：`vendor/FableDan/LICENSE`（Apache 2.0 文本及附加非商业限制）。
- `vendor/FableDan` 保留上游仓库；顶层 `fabledan/` 是其派生副本，原文件头保留。

本地修订：

- 新增 `fabledan/legal.py`，按花色保留物理出牌选择，包含可选配子使用；只合并完全同面的副本。
- `combos.py` 将原枚举保留为 `gen_moves_reduced`，默认改用新枚举；为配子补充申明花色信息。
- `encode.py` 增加自己的精确花色牌计数、候选动作耗牌、公开出牌统计、当前牌权归属；总共 248 维，标记 v2 编码。
- `engine.py` 修复还贡 A 点数判断，校验发牌、动作索引和级牌范围。
- 新增 `guandan/`、`web/`、配置、测试与中文文档；原 `train_fast` 仅作为保留的上游代码，不作为本项目推荐入口。

## DanLM

- 来源：https://github.com/dashidhy/DanLM
- 固定版本：`16ace5ada4db39086bb4c8ebb372ed16cb82fab1`
- 原许可证：`vendor/DanLM/LICENSE`（Apache 2.0 文本及附加非商业限制）。
- 模型：`ckpts/DanLM_v1/dansformer_v1_best_eval.pt`
- SHA-256：`5341d7316bac1ead4098c796ea03f7e38fb47f001cc647a52be3ed0d15cd7244`

`vendor/DanLM` 保留原样。公开版本核心为 Apple Silicon / CPython 3.12 二进制扩展，不能被称为源码完整的跨平台训练框架；本项目也没有宣称复现其作者报告的全部结果。

本地适配位于 `guandan/expert.py` 和 `guandan/web.py`：

- 修复发布二进制中竞赛基线旧目录布局与实际打包目录不一致的问题。
- 对返回的完成局数进行核验，不将未完成的评测作为有效成绩。
- 将原网页挂载于 `/expert/`，改写静态资源和全部 API URL，默认中文；关闭会退出整个宿主进程的空闲 watchdog。
- 当前本地自训练模型与 DanLM 不共享权重或规则状态。二者评测分别报告。

竞赛基线文件也来自 DanLM 打包内容，保留原有作者声明。公开权重及源代码仅用于个人研究；商业使用应按上游许可证取得作者许可。

## 纯程序策略桥接

`guandan/programs.py` 直接加载上述 DanLM 归档中的南邮、蛋饼纯 Python 源码，独立于 DanLM 二进制。它们进入本地环境的训练对手池、CLI 评测和网页对战。适配器做协议转换、模块隔离、确定性随机数以及南邮配子计数重置；原始归档不修改。详见 `docs/PROGRAM_BASELINES.md`。

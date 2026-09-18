from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="掼蛋研究室 · 自我对弈与可复现评测")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查设备与预训练模型")
    sub.add_parser("baselines", help="列出源码可读、无需神经网络的竞赛程序")
    tr = sub.add_parser("train", help="DMC 自我对弈训练")
    tr.add_argument("--config", default="configs/mac.toml")
    tr.add_argument("--out", default="runs/mac")
    tr.add_argument("--resume")
    for key, kind in [("iterations", int), ("games-per-iteration", int), ("workers", int),
                      ("batch-size", int), ("updates", int), ("eval-every", int),
                      ("eval-pairs", int), ("max-minutes", float), ("seed", int),
                      ("device", str), ("threads", int)]:
        tr.add_argument("--" + key, type=kind)
    ev = sub.add_parser("eval", help="同牌换队评测，输出置信区间和逐局记录")
    ev.add_argument("--a", required=True)
    ev.add_argument("--b", default="team-rule")
    ev.add_argument("--pairs", type=int, default=100)
    ev.add_argument("--seed", type=int, default=9_000_019)
    ev.add_argument("--out", default="reports/evaluation.json")
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--threads", type=int, default=2)
    expert = sub.add_parser("expert-eval", help="单独验证公开 DanLM 权重")
    expert.add_argument("--games", type=int, default=100)
    expert.add_argument("--seed", type=int, default=9_100_019)
    expert.add_argument("--opponent", default="bot:fin-njupt-guandan-ai")
    expert.add_argument("--out", default="reports/danlm-evaluation.json")
    serve = sub.add_parser("serve", help="启动中文控制台和对战界面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.command == "baselines":
        from .programs import REGISTRY, ROOT, source_fingerprint
        print(json.dumps([dict(spec="program:"+name, name=entry["name"],
                               source=str(ROOT/'vendor/DanLM/baselines'/entry["directory"]),
                               source_sha256=source_fingerprint(name), neural_network=False)
                          for name, entry in REGISTRY.items()], ensure_ascii=False, indent=2))
    elif args.command == "doctor":
        from .expert import available
        from .runtime import system_info
        info = system_info()
        info["danlm_pretrained_available"] = available()
        print(json.dumps(info, ensure_ascii=False, indent=2))
    elif args.command == "train":
        from .train import TrainConfig, train
        data = tomllib.loads(Path(args.config).read_text()).get("training", {})
        for key in asdict(TrainConfig()):
            value = getattr(args, key, None)
            if value is not None:
                data[key] = value
        train(TrainConfig(**data), args.out, args.resume)
    elif args.command == "eval":
        import torch

        from .evaluate import evaluate_specs
        from .runtime import device_for
        torch.set_num_threads(args.threads)
        evaluate_specs(args.a, args.b, args.pairs, args.seed, args.out, device_for(args.device))
    elif args.command == "expert-eval":
        from .expert import evaluate_expert
        evaluate_expert(args.games, args.seed, args.opponent, args.out)
    elif args.command == "serve":
        import torch
        import uvicorn

        from .web import create_app
        torch.set_num_threads(2)
        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

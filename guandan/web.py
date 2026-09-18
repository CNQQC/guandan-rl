from __future__ import annotations

import json
import secrets
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fabledan.cards import RANK_NAMES, is_wildcard, order_of, rank_of, suit_of
from fabledan.engine import GuandanRound

from .agents import make_agent
from .runtime import ROOT, system_info

TYPE_NAMES = ["过", "单张", "对子", "三张", "三带二", "顺子", "三连对", "钢板", "炸弹", "同花顺", "四王炸"]
PLAYERS = ["你", "右侧对手", "你的搭档", "左侧对手"]


def card_json(c, level):
    r = rank_of(c)
    return dict(id=c, rank=RANK_NAMES[r] if r < 13 else ("小王" if r == 13 else "大王"),
                suit=["♥", "♦", "♠", "♣"][suit_of(c)] if r < 13 else "★",
                red=r == 14 or (r < 13 and suit_of(c) < 2), wild=is_wildcard(c, level))


def move_json(m, lv, index=None):
    return dict(index=index, type=TYPE_NAMES[m.type], size=m.size,
                cards=[card_json(c, lv) for c in m.cards],
                claims=[RANK_NAMES[r] for r in m.claim_ranks],
                label=TYPE_NAMES[m.type] + (" · " + " ".join(RANK_NAMES[r] for r in m.claim_ranks) if m.size else ""))


class Session:
    def __init__(self, seed, level, agent, agent_label):
        import random
        self.id = secrets.token_hex(12)
        self.round = GuandanRound(level, random.Random(seed))
        self.gen = self.round.play_steps()
        self.obs = next(self.gen)
        self.agent = agent
        self.agent_label = agent_label
        self.version = 0
        self.result = None
        self.touched = time.monotonic()
        self.lock = threading.Lock()
        self._advance_ai()

    def _send(self, idx):
        try:
            self.obs = self.gen.send(int(idx))
        except StopIteration as e:
            rewards, ranking = e.value
            self.result = dict(rewards=rewards, ranking=ranking, won=rewards[0] > 0)
            self.obs = None

    def _advance_ai(self):
        while self.obs is not None and self.obs["player"] != 0:
            self._send(self.agent.act(self.obs))

    def play(self, index, version):
        with self.lock:
            if version != self.version:
                raise HTTPException(409, "牌局已更新，请刷新后重新出牌。")
            if self.obs is None or not 0 <= index < len(self.obs["legal"]):
                raise HTTPException(400, "无效的出牌选择。")
            self._send(index)
            self._advance_ai()
            self.version += 1
            self.touched = time.monotonic()
            return self.state()

    def state(self):
        lv = self.round.lv
        played = []
        for event in self.round.events[-32:]:
            if event[0] == "play":
                played.append(dict(player=event[1], move=move_json(event[2], lv)))
            elif event[0] == "pass":
                played.append(dict(player=event[1], move=dict(type="过", label="过", cards=[], claims=[])))
        hand = sorted(self.round.hands[0], key=lambda c: (-order_of(rank_of(c), lv), suit_of(c), c))
        # Never send other players' hidden cards to the browser.
        return dict(id=self.id, version=self.version, level=RANK_NAMES[lv],
                    hand=[card_json(c, lv) for c in hand],
                    remaining=[len(h) for h in self.round.hands],
                    done=self.round.done_order, current_player=self.obs["player"] if self.obs else None,
                    legal=[move_json(m, lv, i) for i, m in enumerate(self.obs["legal"])] if self.obs else [],
                    history=played, result=self.result, agent=self.agent_label)


class NewGame(BaseModel):
    seed: int = Field(default_factory=lambda: secrets.randbelow(2**31))
    level: int = Field(default=1, ge=0, le=12)
    agent: str = "team-rule"


class Play(BaseModel):
    session: str
    index: int = Field(ge=0)
    version: int = Field(ge=0)


class TrainRequest(BaseModel):
    minutes: Literal[5, 30] = 5


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return fallback


def create_app():
    app = FastAPI(title="掼蛋研究室")
    sessions = {}
    session_lock = threading.Lock()
    process_lock = threading.Lock()
    training = {"process": None, "out": None}
    web_root = ROOT / "web"

    @app.get("/")
    def index():
        return FileResponse(web_root / "index.html")

    @app.get("/api/overview")
    def overview():
        from .expert import available
        runs = []
        for directory in sorted((ROOT / "runs").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            if not directory.is_dir():
                continue
            status = read_json(directory / "status.json", {})
            rows = []
            try:
                for line in (directory / "metrics.jsonl").read_text().splitlines()[-100:]:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            except OSError:
                pass
            runs.append(dict(name=directory.name, status=status, metrics=rows,
                             checkpoint=(directory / "latest.pt").is_file()))
        reports = []
        for path in sorted((ROOT / "reports").glob("*.json")):
            report = read_json(path, {})
            if isinstance(report, dict):
                reports.append({"name": path.name, **{k: v for k, v in report.items() if k != "records"}})
        proc = training["process"]
        return dict(system=system_info(), expert_available=available(), runs=runs,
                    reports=reports, managed_training=bool(proc and proc.poll() is None))

    @app.post("/api/game")
    def new_game(req: NewGame):
        spec = req.agent
        from .programs import REGISTRY
        builtins = {"team-rule", "rule", "random"} | {"program:"+name for name in REGISTRY}
        if spec not in builtins:
            path = (ROOT / spec).resolve()
            if not path.is_relative_to((ROOT / "runs").resolve()) or not path.is_file() or path.suffix != ".pt":
                raise HTTPException(400, "请选择本项目 runs 目录下的模型。")
            spec = str(path)
        try:
            agent = make_agent(spec, req.seed)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        session = Session(req.seed, req.level, agent, req.agent)
        with session_lock:
            for key in list(sessions):
                if time.monotonic() - sessions[key].touched > 3600:
                    del sessions[key]
            if len(sessions) >= 16:
                del sessions[min(sessions, key=lambda k: sessions[k].touched)]
            sessions[session.id] = session
        return session.state()

    @app.post("/api/play")
    def play(req: Play):
        session = sessions.get(req.session)
        if not session:
            raise HTTPException(404, "牌局不存在或已过期，请重新开局。")
        return session.play(req.index, req.version)

    @app.post("/api/hint")
    def hint(req: Play):
        session = sessions.get(req.session)
        if not session:
            raise HTTPException(404, "牌局不存在。")
        with session.lock:
            if req.version != session.version:
                raise HTTPException(409, "牌局已更新。")
            if session.obs is None:
                raise HTTPException(400, "本局已结束。")
            return dict(index=session.agent.act(session.obs), version=session.version)

    @app.post("/api/train")
    def start_training(req: TrainRequest):
        with process_lock:
            proc = training["process"]
            if proc and proc.poll() is None:
                raise HTTPException(409, "已有训练正在运行。")
            name = "web-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
            directory = ROOT / "runs" / name
            directory.mkdir(parents=True)
            with (directory / "console.log").open("w") as log:
                proc = subprocess.Popen([sys.executable, "-u", "-m", "guandan", "train",
                                         "--config", "configs/league.toml", "--out", str(directory),
                                         "--iterations", "100000", "--max-minutes", str(req.minutes)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            training.update(process=proc, out=directory)
            return dict(name=name, pid=proc.pid, minutes=req.minutes)

    @app.post("/api/train/stop")
    def stop_training():
        with process_lock:
            proc, directory = training["process"], training["out"]
            if proc is None or proc.poll() is not None:
                raise HTTPException(409, "没有由控制台启动的活动训练。")
            (directory / "STOP").touch()
            return {"status": "stopping_after_current_iteration"}

    app.mount("/assets", StaticFiles(directory=web_root), name="assets")

    from .expert import DANLM_ROOT, available, web_app
    if available():
        # Serve the upstream UI under a prefix without altering vendored files.
        @app.get("/expert/", response_class=HTMLResponse)
        def expert_index():
            html = (DANLM_ROOT / "ui/static/index.html").read_text()
            return html.replace('"/static/', '"/expert/static/')

        @app.get("/expert/static/app.js")
        def expert_js():
            js = (DANLM_ROOT / "ui/static/app.js").read_text()
            js = js.replace("/api/", "/expert/api/")
            js = js.replace("localStorage.getItem('lang') || 'en'", "localStorage.getItem('lang') || 'zh'")
            return Response(js, media_type="application/javascript")

        app.mount("/expert", web_app())
    return app

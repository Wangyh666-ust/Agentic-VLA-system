#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地网页 + HTTP API：持久场景 v2（先载入场景，再用真实 Hermes 做视觉规划）。

仅使用 Python 标准库。默认监听 127.0.0.1:8081（用 --port 修改）。

端点（INTERFACE）：
  GET  /                         中文网页
  GET  /api/health               代理服务 /health（附前端标识）
  GET  /api/scenes               代理服务 /scenes 场景目录
  GET  /api/session/<id>         代理服务 /sessions/<id>
  POST /api/session              校验后代理 POST /sessions（选择场景 + seed + 初始状态）
  POST /api/agent                启动唯一后台 agent job（生成精确 request_id 交给 run_agent.py）
  GET  /api/agent/<request_id>   该精确 request/session 的 job 状态与结果
  POST /api/agent/<request_id>/cancel
                                 仅对精确 request 归属的 session 请求停止（代理服务取消，
                                 服务确认后才原子落盘 <run_dir>/cancel_requested.json）
  GET  /artifacts/<relative>     代理服务 artifact（仅 PNG/MP4，安全转发 Range->206）
  GET  /agent-artifacts/<request_id>/<allowedbasename>
                                 仅在 run_root 内、且与 request_id 精确关联时暴露本机日志

额外只读进度代理（绝不替代规划/执行）：
  GET  /api/plan/<request_id>    代理服务 /plans/<id>
  GET  /api/job/<job_id>         代理服务 /jobs/<id>

场景**只在用户点「载入场景」时创建一次**：后续请求复用同一 session，绝不按请求重建。
网页与 host 都不选择能力（capability）：能力只由真实 Hermes 通过 MCP 选择。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_AGENT = os.path.join(APP_DIR, "run_agent.py")
PYTHON = sys.executable

SERVICE_BASE = os.environ.get("SCENE_SERVICE_URL", "http://127.0.0.1:8767").rstrip("/")
RUNS_DIR = os.environ.get("SCENE_RUNS_DIR", "/home/yhwang/fyp/scene_demo/runs")
FRONTEND_ID = "scene_demo_frontend"

MAX_BODY = 65536
SERVICE_MEDIA_EXTS = {".png", ".mp4"}
ALLOWED_AGENT_FILES = {
    "agent_result.json",
    "hermes_initial.log",
    "hermes_repair.log",
    "request.json",
}

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
RUNNING = {"request_id": None}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def proxy_json(method: str, path: str, body=None, timeout: float = 30.0):
    url = SERVICE_BASE + path
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with _opener().open(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _valid_int(value, minimum, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    if value < minimum:
        return False
    if maximum is not None and value >= maximum:
        return False
    return True


def _run_dir_matches(run_dir: str, request_id: str) -> bool:
    """run_dir 是否与 request_id 精确关联（读 request.json / agent_result.json）。"""
    for name in ("request.json", "agent_result.json"):
        path = os.path.join(run_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and data.get("request_id") == request_id:
                return True
        except (OSError, ValueError):
            continue
    base = os.path.basename(os.path.normpath(run_dir))
    return request_id in base


def resolve_run_dir(request_id: str):
    with JOBS_LOCK:
        job = JOBS.get(request_id)
    if job and job.get("run_dir"):
        return job["run_dir"]
    if not os.path.isdir(RUNS_DIR):
        return None
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        path = os.path.join(RUNS_DIR, name)
        if os.path.isdir(path) and _run_dir_matches(path, request_id):
            return path
    return None


# 公开 job 视图：只暴露固定白名单字段，绝不泄露内部 process 对象 / Event 等。
PUBLIC_JOB_FIELDS = (
    "request_id",
    "session_id",
    "status",
    "request",
    "case_id",
    "run_dir",
    "created",
    "started",
    "finished",
    "exit_code",
    "result",
    "log_path",
    "cancel_requested",
    "cancel_note",
)


def public_job(job):
    """返回 job 的**字典快照**（新 dict，仅白名单字段），绝非 live 引用。

    调用方须在 JOBS_LOCK 内调用，使取消字段更新无法与 JSON 序列化竞争。
    """
    if not isinstance(job, dict):
        return {}
    return {key: job.get(key) for key in PUBLIC_JOB_FIELDS if key in job}


# 真实 plan 的终态集合（queued/running 视为活动）。
PLAN_TERMINAL_STATES = ("completed", "blocked", "error", "cancelled")
# cancellation_pending 时的只读轮询间隔（测试可 monkeypatch 加速）。
CANCEL_PLAN_POLL_INTERVAL = 1.0
CANCEL_PENDING_NOTE = "后端未确认取消完成：真实计划仍在执行，保持 cancelling 并等待真实终态。"
# 真实 plan 已终态、但归属本次请求的终态 job 尚未刷新到位时的明确等待说明。
JOBS_REFRESH_NOTE = "真实计划已终态，但归属本次请求的终态 job 尚未就绪：保持 cancelling 并等待真实 job 确认。"


def plan_is_terminal(plan) -> bool:
    return isinstance(plan, dict) and plan.get("state") in PLAN_TERMINAL_STATES


def _owned_terminal_plan(plan, request_id, session_id) -> bool:
    """仅当 plan 的 request_id 与 session_id **都精确相等**且处于终态时才采纳。

    省略 request_id、缺失/wrong session_id 一律视为不匹配（绝不采纳 latest 或不归属的 plan）。
    """
    if not isinstance(plan, dict):
        return False
    if plan.get("request_id") != request_id:
        return False
    if plan.get("session_id") != session_id:
        return False
    return plan_is_terminal(plan)


def refresh_owned_jobs(plan, request_id, session_id):
    """只读 GET /jobs/<exact job_id>，仅保留 job_id/request_id/session_id 全精确匹配的公开记录。

    成功返回 owned 公开记录列表（可能为空列表）；只要任一 job_id 非字符串、取不到、响应非对象、
    或 job_id/request_id/session_id 任一不精确匹配，就返回 None。调用方据此保持 cancelling 并重试，
    绝不使用 latest 回退，也不沿用 runner 的旧 10 秒窗口快照或编造 job。
    """
    job_ids = plan.get("job_ids")
    if job_ids is None:
        job_ids = []
    if not isinstance(job_ids, list):
        return None
    refreshed = []
    for job_id in job_ids:
        if not isinstance(job_id, str) or not job_id:
            return None
        path = "/jobs/" + urllib.parse.quote(job_id, safe="")
        try:
            record = proxy_json("GET", path, None, 10.0)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(record, dict):
            return None
        if record.get("job_id") != job_id:
            return None
        if record.get("request_id") != request_id:
            return None
        if record.get("session_id") != session_id:
            return None
        refreshed.append(record)
    return refreshed


def await_real_plan_terminal(request_id: str, session_id, result, job=None):
    """只读轮询 GET /plans/<exact request_id>，直到真实计划终态并刷新其 owned 终态 job。

    仅采纳 request_id 与 session_id **都精确相等**的真实 plan（省略 request_id 或 session 缺失/
    不符一律忽略，仍保持 cancelling）。真实终态后只读 GET /jobs/<exact id> 刷新 plan.job_ids 的公开
    job，只有全部精确归属才合并；服务不可达或 job 未就绪时保留 cancelling 并重试（绝不伪造确认、
    绝不编造/沿用旧 job）。返回 (merged_result, note)。
    """
    plan_path = "/plans/" + urllib.parse.quote(request_id, safe="")
    while True:
        plan = None
        try:
            plan = proxy_json("GET", plan_path, None, 10.0)
        except Exception:  # noqa: BLE001
            plan = None
        if _owned_terminal_plan(plan, request_id, session_id):
            refreshed = refresh_owned_jobs(plan, request_id, session_id)
            if refreshed is not None:
                merged = dict(result) if isinstance(result, dict) else {}
                merged["jobs"] = refreshed       # 只用刷新后的 owned 公开记录替换旧快照
                merged["plan"] = plan            # 只合并真实精确终态 plan
                merged.pop("cancellation_pending", None)
                return merged, None
            # 终态已确认但 job 未就绪/归属不符：明确说明并继续等待，绝不提前发布旧 job。
            if job is not None:
                with JOBS_LOCK:
                    job["cancel_note"] = JOBS_REFRESH_NOTE
        time.sleep(CANCEL_PLAN_POLL_INTERVAL)


CANCEL_MARKER_NAME = "cancel_requested.json"


def write_cancel_marker(run_dir, request_id, session_id, requested_at) -> str:
    """原子写入 <run_dir>/cancel_requested.json（临时文件 + os.replace）。

    run_dir 必须解析到 RUNS_DIR 之内；schema 精确为
    {request_id, session_id, requested_at}（UTF-8 JSON）。仅在服务确认取消后调用。
    """
    if not run_dir:
        raise OSError("unknown run_dir")
    root = os.path.realpath(RUNS_DIR)
    target_dir = os.path.realpath(run_dir)
    if target_dir != root and not target_dir.startswith(root + os.sep):
        raise OSError("run_dir outside RUNS")
    payload = {"request_id": request_id, "session_id": session_id,
               "requested_at": requested_at}
    final_path = os.path.join(target_dir, CANCEL_MARKER_NAME)
    handle_fd, tmp_path = tempfile.mkstemp(prefix=".cancel_requested.", suffix=".tmp",
                                           dir=target_dir)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return final_path


def run_job(request_id: str, session_id: str, request_text: str, case_id, run_dir: str) -> None:
    """后台线程：调用 run_agent.py（真实 Hermes + MCP），不做任何 Agent 逻辑复刻。

    host/UI 不选择能力；run_agent.py 会一直阻塞到本次请求的计划到达终态
    （或在总 deadline 到期后取消），因此这里 job 保持 running 直到真实终态。
    """
    job = JOBS.get(request_id)
    if job is None:
        return
    log_path = os.path.join(run_dir, "run_agent.log")
    cmd = [
        PYTHON, RUN_AGENT,
        "--session-id", session_id,
        "--request", request_text,
        "--request-id", request_id,
        "--run-dir", run_dir,
        "--cancel-file", os.path.join(run_dir, CANCEL_MARKER_NAME),
    ]
    if case_id:
        cmd += ["--case-id", str(case_id)]
    with JOBS_LOCK:
        # 已被请求取消（queued 阶段即收到 Stop）的任务保持 cancelling，
        # 绝不用 running 覆盖排队中的取消状态；真实 runner 仍会阻塞到实际退出。
        if job.get("cancel_requested"):
            job["status"] = "cancelling"
        else:
            job["status"] = "running"
        job["started"] = _now()

    exit_code = None
    stdout_text = ""
    try:
        proc = subprocess.run(
            cmd, cwd=RUNS_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        stdout_text = (proc.stdout or b"").decode("utf-8", "replace")
        exit_code = proc.returncode
    except Exception as exc:  # noqa: BLE001
        stdout_text = "启动 run_agent.py 失败: %s\n" % exc
        exit_code = -1
    try:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(stdout_text)
    except OSError:
        pass

    result = None
    for line in reversed(stdout_text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result = json.loads(line)
                break
            except ValueError:
                continue
    if result is None:
        candidate = os.path.join(run_dir, "agent_result.json")
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, ValueError):
                result = None

    # 运行器可能返回 cancellation_pending=true：其 10 秒收集窗口结束时真实 plan 仍
    # queued/running。此时绝不因为 cancelled_by_user 就标记已停止或释放 RUNNING；
    # 保持 cancelling 占用，并只读轮询精确 request 的真实 plan，直到真实终态为止。
    # 注意：只有 runner 自己的 plan 快照明确处于 queued/running 才等待；plan 缺失
    # 视为“无计划提交”，那是真实终态取消（绝不无限轮询）。
    pending_plan = result.get("plan") if isinstance(result, dict) else None
    pending_active = (
        isinstance(result, dict)
        and result.get("cancellation_pending") is True
        and isinstance(pending_plan, dict)
        and pending_plan.get("state") in ("queued", "running")
    )
    if pending_active:
        with JOBS_LOCK:
            job["cancel_requested"] = True
            if job.get("status") not in ("completed", "error", "cancelled"):
                job["status"] = "cancelling"
            job["cancel_note"] = CANCEL_PENDING_NOTE
            job["log_path"] = log_path
        result, note = await_real_plan_terminal(request_id, session_id, result, job)
        with JOBS_LOCK:
            if note:
                job["cancel_note"] = note
            else:
                job.pop("cancel_note", None)

    # cancellation_pending 至此必然已落定（无活动计划，或轮询到真实终态）：如实清除，
    # 使前端不再把它当成“仍未决”。
    if isinstance(result, dict) and result.get("cancellation_pending") is True:
        result = dict(result)
        result.pop("cancellation_pending", None)

    # 终态只依据真实结果：真实 plan 终态优先，其次才是 cancelled_by_user。
    cancelled = False
    result_plan = result.get("plan") if isinstance(result, dict) else None
    plan_state = result_plan.get("state") if isinstance(result_plan, dict) else None
    if plan_state == "cancelled":
        cancelled = True
    elif plan_state in PLAN_TERMINAL_STATES:
        cancelled = False        # 真实计划已到非取消终态：以真实计划为准
    elif plan_state in ("queued", "running"):
        cancelled = False        # 真实计划仍活动：绝不因 cancelled_by_user 误判
    elif isinstance(result, dict) and result.get("cancelled_by_user") is True:
        cancelled = True

    with JOBS_LOCK:
        job["exit_code"] = exit_code
        job["result"] = result
        job["log_path"] = log_path
        job["finished"] = _now()
        if cancelled:
            job["status"] = "cancelled"
        else:
            job["status"] = "completed" if exit_code == 0 else "error"
        if RUNNING.get("request_id") == request_id:
            RUNNING["request_id"] = None


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "SceneDemoUI/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- low level -------------------------------------------------------- #
    def _send_bytes(self, body: bytes, code=200, ctype="application/octet-stream", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"), code,
                         "application/json; charset=utf-8")

    def _send_text(self, text, code=200, ctype="text/plain; charset=utf-8"):
        self._send_bytes(text.encode("utf-8"), code, ctype)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return None, "Content-Length 无效"
        if length <= 0:
            return None, "请求体必须是 JSON 对象"
        if length > MAX_BODY:
            return None, "请求体过大"
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None, "JSON 解析失败"
        if not isinstance(payload, dict):
            return None, "请求体必须是 JSON 对象"
        return payload, None

    # ---- routing ---------------------------------------------------------- #
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path in ("/", "/index.html"):
                self._send_text(render_page(), 200, "text/html; charset=utf-8")
            elif path == "/api/health":
                self._api_health()
            elif path == "/api/scenes":
                self._api_proxy("/scenes")
            elif path.startswith("/api/session/"):
                self._api_proxy("/sessions/" + urllib.parse.quote(path[len("/api/session/"):], safe=""))
            elif path.startswith("/api/plan/"):
                self._api_proxy("/plans/" + urllib.parse.quote(path[len("/api/plan/"):], safe=""))
            elif path.startswith("/api/job/"):
                self._api_proxy("/jobs/" + urllib.parse.quote(path[len("/api/job/"):], safe=""))
            elif path == "/api/agent":
                self._api_list_agents()
            elif path.startswith("/api/agent/"):
                self._api_agent_status(path[len("/api/agent/"):])
            elif path.startswith("/artifacts/"):
                self._serve_service_artifact(path[len("/artifacts/"):])
            elif path.startswith("/agent-artifacts/"):
                self._serve_agent_artifact(path[len("/agent-artifacts/"):])
            else:
                self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "%s" % exc}, 500)

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/session":
                self._api_create_session()
            elif parsed.path == "/api/agent":
                self._api_start_agent()
            elif parsed.path.startswith("/api/agent/") and parsed.path.endswith("/cancel"):
                self._api_cancel(parsed.path[len("/api/agent/"):-len("/cancel")])
            else:
                self._send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "%s" % exc}, 500)

    # ---- api impl --------------------------------------------------------- #
    def _api_health(self):
        out = {"frontend": FRONTEND_ID}
        try:
            health = proxy_json("GET", "/health", None, 8.0)
            if isinstance(health, dict):
                out.update(health)
        except Exception as exc:  # noqa: BLE001
            out.update({"reachable": False, "ready": False, "error": str(exc)})
        self._send_json(out)

    def _api_proxy(self, path):
        try:
            self._send_json(proxy_json("GET", path, None, 30.0))
        except urllib.error.HTTPError as exc:
            payload = {"error": "service http %s" % exc.code}
            try:
                body = json.loads(exc.read().decode("utf-8"))
                if isinstance(body, dict):
                    payload.update(body)
            except Exception:  # noqa: BLE001
                pass
            self._send_json(payload, exc.code if exc.code in (400, 404, 409) else 502)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "service unreachable: %s" % exc}, 502)

    def _api_create_session(self):
        payload, error = self._read_body()
        if error:
            self._send_json({"error": error}, 400)
            return
        scene_id = payload.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            self._send_json({"error": "scene_id 必须是非空字符串"}, 400)
            return
        seed = payload.get("seed", 0)
        if not _valid_int(seed, 0, 2 ** 32):
            self._send_json({"error": "seed 必须是 [0, 2**32) 内的整数"}, 400)
            return
        init_state_index = payload.get("init_state_index", 0)
        if not _valid_int(init_state_index, 0):
            self._send_json({"error": "init_state_index 必须是非负整数"}, 400)
            return
        body = {"scene_id": scene_id.strip(), "seed": seed,
                "init_state_index": init_state_index}
        try:
            result = proxy_json("POST", "/sessions", body, 60.0)
        except urllib.error.HTTPError as exc:
            detail = {"error": "service http %s" % exc.code}
            try:
                parsed = json.loads(exc.read().decode("utf-8"))
                if isinstance(parsed, dict):
                    detail.update(parsed)
            except Exception:  # noqa: BLE001
                pass
            self._send_json(detail, exc.code if exc.code in (400, 409) else 502)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "service unreachable: %s" % exc}, 502)
            return
        self._send_json(result, 201)

    def _api_start_agent(self):
        payload, error = self._read_body()
        if error:
            self._send_json({"error": error}, 400)
            return
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            self._send_json({"error": "session_id 必须是非空字符串"}, 400)
            return
        session_id = session_id.strip()
        request_text = payload.get("request")
        if not isinstance(request_text, str) or not request_text.strip():
            self._send_json({"error": "request 必须是非空字符串"}, 400)
            return
        request_text = request_text.strip()
        case_id = payload.get("case_id")
        if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
            self._send_json({"error": "case_id 若提供必须是非空字符串"}, 400)
            return
        if case_id is not None:
            case_id = case_id.strip()

        # 必须有已就绪的现有 session 才允许提交自然语言请求。
        try:
            session = proxy_json("GET", "/sessions/" + urllib.parse.quote(session_id, safe=""),
                                 None, 15.0)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                self._send_json({"error": "会话不存在，请先载入场景", "session_id": session_id}, 404)
            else:
                self._send_json({"error": "读取会话失败 (http %s)" % exc.code}, 502)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "服务不可达: %s" % exc}, 502)
            return
        if not isinstance(session, dict) or session.get("state") != "ready":
            self._send_json({"error": "会话尚未就绪，请先载入或刷新场景",
                             "session_id": session_id,
                             "state": (session or {}).get("state")}, 409)
            return

        with JOBS_LOCK:
            if RUNNING.get("request_id") is not None:
                self._send_json({"error": "已有 agent 任务在运行", "busy": True,
                                 "request_id": RUNNING["request_id"]}, 409)
                return
            request_id = uuid.uuid4().hex
            run_dir = os.path.join(
                RUNS_DIR, "web_%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"), request_id))
            os.makedirs(run_dir, exist_ok=True)
            JOBS[request_id] = {
                "request_id": request_id,
                "session_id": session_id,
                "status": "queued",
                "request": request_text,
                "case_id": case_id,
                "run_dir": run_dir,
                "created": _now(),
                "started": None,
                "finished": None,
                "exit_code": None,
                "result": None,
                "log_path": None,
                "cancel_requested": False,
            }
            RUNNING["request_id"] = request_id

        thread = threading.Thread(
            target=run_job,
            args=(request_id, session_id, request_text, case_id, run_dir),
            daemon=True)
        thread.start()
        self._send_json({"request_id": request_id, "session_id": session_id,
                         "status": "queued", "run_dir": run_dir}, 202)

    def _api_cancel(self, request_id):
        """POST /api/agent/<request_id>/cancel {session_id}。

        只校验精确 request 归属的 live JOBS 项（绝非全局 latest）：
        unknown 404、wrong owner 409；completed/error/cancelled 的 live job 返回
        现有状态（noop）。queued/running/cancelling 代理服务取消（10 秒超时）；
        仅在**服务确认成功**后原子写入取消标记，并置 cancel_requested/status=cancelling。
        服务失败如实返回错误，绝不谎报取消成功；RUNNING 仍保持占用直到真实 runner 退出。
        """
        request_id = urllib.parse.unquote(request_id)
        payload, error = self._read_body()
        if error:
            self._send_json({"error": error}, 400)
            return
        session_id = payload.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            self._send_json({"error": "session_id 必须是非空字符串"}, 400)
            return
        session_id = session_id.strip()

        with JOBS_LOCK:
            job = JOBS.get(request_id)
        if job is None:
            self._send_json({"error": "unknown request_id", "request_id": request_id}, 404)
            return
        if job.get("session_id") != session_id:
            self._send_json({"error": "session_id 与该 request 不匹配",
                             "request_id": request_id, "session_id": session_id}, 409)
            return

        status = job.get("status")
        if status in ("completed", "error", "cancelled"):
            # 已完成/错误/已取消的 live job：幂等 noop，返回其现有状态，不再调用服务。
            self._send_json({"ok": True, "request_id": request_id, "session_id": session_id,
                             "cancel_requested": bool(job.get("cancel_requested")),
                             "status": status, "noop": True})
            return

        body = {"session_id": session_id}
        service_path = "/requests/" + urllib.parse.quote(request_id, safe="") + "/cancel"
        try:
            service = proxy_json("POST", service_path, body, 10.0)
        except urllib.error.HTTPError as exc:
            detail = {"error": "service http %s" % exc.code}
            try:
                parsed = json.loads(exc.read().decode("utf-8"))
                if isinstance(parsed, dict):
                    detail.update(parsed)
            except Exception:  # noqa: BLE001
                pass
            self._send_json(detail, exc.code if exc.code in (400, 404, 409) else 502)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "service unreachable: %s" % exc}, 502)
            return
        if not isinstance(service, dict) or service.get("ok") is not True:
            detail = service.get("detail") if isinstance(service, dict) else None
            self._send_json({"error": "服务未确认取消", "detail": detail}, 502)
            return

        # 仅在服务确认后落盘取消标记（原子写入），再更新 job 状态。
        try:
            write_cancel_marker(job.get("run_dir"), request_id, session_id, _now())
        except OSError as exc:
            self._send_json({"error": "无法写入取消标记: %s" % exc}, 500)
            return
        with JOBS_LOCK:
            job["cancel_requested"] = True
            if job.get("status") not in ("completed", "error", "cancelled"):
                job["status"] = "cancelling"
            # 服务确认后真实运行可能已经抢先到达终态：返回**实际当前**状态，
            # 绝不硬编码 cancelling（取消确认与完成存在竞态）。
            actual_status = job.get("status")
            actual_cancel_requested = bool(job.get("cancel_requested"))
        self._send_json({"ok": True, "request_id": request_id, "session_id": session_id,
                         "cancel_requested": actual_cancel_requested, "status": actual_status,
                         "noop": False, "service_state": service.get("state")})

    def _api_list_agents(self):
        # 在锁内构造**字典快照**列表：取消字段更新无法与 JSON 序列化竞争。
        with JOBS_LOCK:
            jobs = [public_job(job) for job in list(JOBS.values())]
        self._send_json({"jobs": jobs})

    def _api_agent_status(self, request_id):
        request_id = urllib.parse.unquote(request_id)
        # 在锁内取**字典快照**（新 dict），序列化时不再持有 live job 引用。
        with JOBS_LOCK:
            job = JOBS.get(request_id)
            snapshot = public_job(job) if job is not None else None
        if snapshot is not None:
            self._send_json(snapshot)
            return
        # 重启后的磁盘回退：仅当 run_dir 与 request_id 精确关联时才返回。
        run_dir = resolve_run_dir(request_id)
        if not run_dir:
            self._send_json({"error": "unknown request_id", "request_id": request_id}, 404)
            return
        entry = {"request_id": request_id, "status": "on_disk", "run_dir": run_dir,
                 "result": None, "request": None, "session_id": None}
        try:
            with open(os.path.join(run_dir, "agent_result.json"), "r", encoding="utf-8") as handle:
                entry["result"] = json.load(handle)
                entry["status"] = "completed"
                entry["request"] = entry["result"].get("request")
                entry["session_id"] = entry["result"].get("session_id")
        except (OSError, ValueError):
            pass
        self._send_json(entry)

    # ---- artifacts -------------------------------------------------------- #
    def _serve_service_artifact(self, rel):
        """仅代理服务 run_root 相对 artifact：PNG/MP4，安全转发 Range（206）。"""
        rel = urllib.parse.unquote(rel).lstrip("/")
        parts = [part for part in rel.split("/") if part not in ("", ".")]
        if not parts or any(part == ".." for part in parts):
            self._send_json({"error": "forbidden"}, 403)
            return
        if os.path.splitext(parts[-1])[1].lower() not in SERVICE_MEDIA_EXTS:
            self._send_json({"error": "forbidden"}, 403)
            return
        clean = "/".join(parts)
        url = SERVICE_BASE + "/artifacts/" + urllib.parse.quote(clean, safe="/")
        headers = {"Accept": "*/*"}
        range_header = self.headers.get("Range")
        if range_header:
            headers["Range"] = range_header
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with _opener().open(request, timeout=60.0) as response:
                status = response.status
                body = response.read()
                ctype = response.headers.get("Content-Type") or "application/octet-stream"
                extra = {}
                if response.headers.get("Content-Range"):
                    extra["Content-Range"] = response.headers["Content-Range"]
                if response.headers.get("Accept-Ranges"):
                    extra["Accept-Ranges"] = response.headers["Accept-Ranges"]
        except urllib.error.HTTPError as exc:
            body = b""
            try:
                body = exc.read()
            except Exception:  # noqa: BLE001
                pass
            ctype = exc.headers.get("Content-Type") if exc.headers else None
            extra = {}
            if exc.headers and exc.headers.get("Content-Range"):
                extra["Content-Range"] = exc.headers["Content-Range"]
            self._send_bytes(body, exc.code, ctype or "application/json", extra)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "service unreachable: %s" % exc}, 502)
            return
        self._send_bytes(body, status, ctype, extra)

    def _serve_agent_artifact(self, rel):
        """仅在 run_root 内、且与 request_id 精确关联时暴露允许的本机日志文件名。"""
        rel = urllib.parse.unquote(rel).lstrip("/")
        parts = [part for part in rel.split("/") if part not in ("", ".")]
        if len(parts) != 2 or any(part == ".." for part in parts):
            self._send_json({"error": "forbidden"}, 403)
            return
        request_id, basename = parts
        if basename not in ALLOWED_AGENT_FILES:
            self._send_json({"error": "forbidden"}, 403)
            return
        run_dir = resolve_run_dir(request_id)
        if not run_dir:
            self._send_json({"error": "unknown request_id", "request_id": request_id}, 404)
            return
        if not _run_dir_matches(run_dir, request_id):
            self._send_json({"error": "request mismatch"}, 403)
            return
        root = os.path.realpath(run_dir)
        target = os.path.realpath(os.path.join(root, basename))
        if target != root and not target.startswith(root + os.sep):
            self._send_json({"error": "forbidden"}, 403)
            return
        if not os.path.isfile(target):
            self._send_json({"error": "not found"}, 404)
            return
        with open(target, "rb") as handle:
            body = handle.read()
        ctype = ("application/json; charset=utf-8" if basename.endswith(".json")
                 else "text/plain; charset=utf-8")
        self._send_bytes(body, 200, ctype)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>持久场景 v2 · 真实视觉规划演示</title>
<style>
  :root { color-scheme: light; }
  body { font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
         margin: 0; background: #f4f6f9; color: #1f2933; }
  header { background: #14324f; color: #fff; padding: 14px 22px;
           display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  header h1 { font-size: 18px; margin: 0; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 13px; background: #64748b; }
  .badge.ok { background: #16a34a; }
  .badge.bad { background: #dc2626; }
  .badge.warn { background: #d97706; }
  main { max-width: 1180px; margin: 0 auto; padding: 18px; display: grid;
         grid-template-columns: 1fr 1fr; gap: 16px; }
  section { background: #fff; border-radius: 10px; padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  section.wide { grid-column: 1 / -1; }
  h2 { font-size: 15px; margin: 0 0 10px; color: #14324f; }
  textarea { width: 100%; box-sizing: border-box; min-height: 66px; padding: 8px;
             border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; resize: vertical; }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
  label { font-size: 13px; color: #475569; }
  select, input[type=number], input[type=text] { padding: 6px; border: 1px solid #cbd5e1;
             border-radius: 6px; font-size: 13px; }
  input[type=number] { width: 90px; }
  button { background: #2563eb; color: #fff; border: 0; padding: 9px 18px;
           border-radius: 6px; font-size: 14px; cursor: pointer; }
  button:disabled { background: #94a3b8; cursor: not-allowed; }
  button.stop { background: #dc2626; }
  button.stop:disabled { background: #94a3b8; }
  pre { background: #0f172a; color: #e2e8f0; padding: 10px; border-radius: 6px;
        font-size: 12px; overflow: auto; max-height: 260px; white-space: pre-wrap;
        word-break: break-all; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border-bottom: 1px solid #e2e8f0; padding: 6px 8px; text-align: left; vertical-align: top; }
  th { background: #f1f5f9; }
  .scroll { max-height: 300px; overflow: auto; }
  .kv { color: #475569; line-height: 1.9; }
  .ok-text { color: #16a34a; font-weight: 600; }
  .bad-text { color: #dc2626; font-weight: 600; }
  .warn-text { color: #d97706; font-weight: 600; }
  video { width: 100%; aspect-ratio: 1 / 1; background: #000; border-radius: 6px;
          margin-bottom: 8px; }
  img.frame { width: 100%; height: auto; aspect-ratio: 1 / 1; border-radius: 6px;
              border: 1px solid #cbd5e1; margin-bottom: 8px; }
  .hint { font-size: 12px; color: #64748b; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px;
          background: #e2e8f0; margin-right: 6px; }
  .pill.done { background: #dcfce7; color: #166534; }
  .pill.pend { background: #fef3c7; color: #92400e; }
</style>
</head>
<body>
<header>
  <h1>持久场景 v2 · 真实 Hermes 视觉规划</h1>
  <span id="svc" class="badge">服务状态: 检测中…</span>
  <span id="wf" class="badge">workflow: -</span>
  <span id="rev" class="badge">revision: -</span>
</header>

<main>
  <section class="wide">
    <h2>1. 载入场景（每个场景只创建一次，后续请求复用）</h2>
    <div class="row">
      <label>场景 <select id="scene"><option>加载中…</option></select></label>
      <label>seed <input type="number" id="seed" value="0"></label>
      <label>初始状态序号 <input type="number" id="idx" value="0"></label>
      <button id="load">载入场景</button>
      <span id="loadmsg" class="hint"></span>
    </div>
    <div id="session" class="kv">尚未载入场景。</div>
  </section>

  <section>
    <h2>2. 当前画面（agentview / wrist）</h2>
    <div id="images"><span class="hint">载入场景后显示初始画面。</span></div>
  </section>

  <section>
    <h2>3. 公开存储规则与能力</h2>
    <div id="policy" class="kv">—</div>
    <div class="scroll"><table id="caps"><thead><tr><th>能力</th><th>指令</th><th>证据</th></tr></thead>
      <tbody><tr><td colspan="3" class="hint">载入场景后显示。</td></tr></tbody></table></div>
  </section>

  <section class="wide">
    <h2>4. 自然语言请求（需先载入就绪场景）</h2>
    <textarea id="request" placeholder="例如：请帮我整理一下桌面，把碗和酒瓶收好。"></textarea>
    <div class="row">
      <label>case_id（可选，仅用于终态后的独立评测） <input type="text" id="case" placeholder="留空则不评测"></label>
      <button id="run" disabled>提交请求</button>
      <button id="stop" class="stop" disabled>停止任务</button>
      <span id="runmsg" class="hint"></span>
    </div>
    <p class="hint">停止将在当前动作或推理结束后生效，并保留当前场景。</p>
    <p class="hint">能力由真实 Hermes 依据画面与公开数据自行选择；网页与 host 都不选择能力。</p>
  </section>

  <section>
    <h2>5. 计划与子目标进度</h2>
    <div id="plan" class="kv">暂无。</div>
    <div id="subgoals"></div>
    <p class="hint">提示：该固定规则仅针对 wine 任务的放置评价，检查是否已松手并稳定支撑在架子上；标准目标命中本身并不代表已松手或稳定，也不改变 job/plan 的 success 判定。</p>
  </section>

  <section>
    <h2>6. 独立评测（无 case 时不评估）</h2>
    <div id="eval" class="kv">尚未评测。</div>
  </section>

  <section class="wide">
    <h2>7. Hermes 说明（真实 agent 输出）</h2>
    <pre id="hermes">暂无。</pre>
  </section>

  <section class="wide">
    <h2>8. 本次请求的终态视频与画面</h2>
    <div id="videos"><span class="hint">本次请求到达终态后显示其视频。</span></div>
  </section>
</main>

<script>
const RUNS_ROOT = "__RUNS_ROOT__";
const $ = (id) => document.getElementById(id);
let currentSessionId = null;
let currentSession = null;
let currentRequest = null;
let currentAgentStatus = null;   // queued | running | cancelling | completed | error | cancelled
let pollTimer = null;            // 唯一的完成驱动 setTimeout 句柄
let tickInFlight = false;        // 顶层 tick 在途保护：绝不允许两条轮询链
let pollInFlight = false;        // pollCurrent 在途保护
let sessionRefreshInFlight = false; // session 刷新在途保护
let stopInFlight = false;        // 单次在途取消，防双击
let stopRequested = false;       // 已请求停止：终态前禁用 Stop
let mediaRenderGeneration = 0;
let displayedImageKey = null;
let pendingImageKey = null;
let videoCacheKey = null;        // renderVideos 缓存键
const showCache = new Map();     // id -> 最近写入的确切源 HTML

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
// show 缓存“确切源 HTML”，写相同内容时跳过，绝不用规范化 DOM innerHTML 比较。
function show(id, html) {
  const key = String(id);
  if (showCache.get(key) === html) return;
  showCache.set(key, html);
  const el = $(id);
  if (el) el.innerHTML = html;
}
function toArtifact(p) {
  if (!p || typeof p !== "string") return null;
  if (p.startsWith(RUNS_ROOT)) return "/artifacts/" + p.slice(RUNS_ROOT.length);
  return null;
}
function fmtBool(v) {
  if (v === true) return '<span class="ok-text">true</span>';
  if (v === false) return '<span class="bad-text">false</span>';
  return '<span class="kv">未知</span>';
}
async function getJSON(url, options) {
  const r = await fetch(url, Object.assign({ cache: "no-store" }, options || {}));
  const ct = r.headers.get("content-type") || "";
  let data = null;
  if (ct.includes("json")) data = await r.json();
  if (!r.ok) throw Object.assign(new Error("HTTP " + r.status), { status: r.status, data });
  return data;
}

// queued/running/cancelling 均视为“占用中”，此时禁止新提交与载入场景。
function agentBusy(status) {
  return status === "queued" || status === "running" || status === "cancelling";
}
function agentTerminalStatus(status) {
  return status === "completed" || status === "error" || status === "cancelled";
}
function statusLabel(status) {
  if (status === "queued") return "排队中";
  if (status === "running") return "执行中";
  if (status === "cancelling") return "正在停止";
  if (status === "completed") return "已完成";
  if (status === "cancelled") return "已停止";
  if (status === "error") return "错误";
  return status || "未知";
}
// 终态判断只依据真实结果：真实 plan 终态优先；cancellation_pending 且计划仍活动时
// 绝不因 cancelled_by_user 提前判定为已停止。
function isResultCancelled(result) {
  if (!result || typeof result !== "object") return false;
  const plan = result.plan;
  const planState = (plan && typeof plan === "object") ? plan.state : null;
  if (planState === "cancelled") return true;
  if (result.cancellation_pending === true && !isTerminalPlan(planState)) return false;
  return result.cancelled_by_user === true;
}
function updateControls() {
  const busy = agentBusy(currentAgentStatus);
  const ready = !!(currentSession && currentSession.state === "ready");
  $("run").disabled = busy || !ready;
  $("load").disabled = busy;
  $("stop").disabled = !busy || stopRequested || stopInFlight;
}

async function pollHealth() {
  // 只更新服务徽标；绝不在此独立刷新 session（由统一 tick 负责，且同一 session 只刷新一次）。
  try {
    const h = await getJSON("/api/health");
    const ready = h && h.ready === true;
    const el = $("svc");
    el.textContent = "服务状态: " + (ready ? "就绪" : (h && h.reachable === false ? "不可达" : "模型加载中/未就绪"));
    el.className = "badge " + (ready ? "ok" : "warn");
    $("wf").textContent = "workflow: " + ((h && h.workflow) || "-");
    $("rev").textContent = "revision: " + (h && h.model_revision ? String(h.model_revision).slice(0, 12) : "-");
  } catch (e) {
    $("svc").textContent = "服务状态: 前端可达，服务未知";
    $("svc").className = "badge bad";
  }
}

async function loadScenes() {
  try {
    const data = await getJSON("/api/scenes");
    const scenes = (data && data.scenes) || [];
    const sel = $("scene");
    sel.innerHTML = "";
    if (!scenes.length) { sel.innerHTML = "<option>无可用场景</option>"; return; }
    scenes.forEach(s => {
      const opt = document.createElement("option");
      opt.value = s.scene_id;
      opt.textContent = (s.label || s.scene_id) + "（" + (s.variant || "-") + "）";
      opt.title = s.description || "";
      sel.appendChild(opt);
    });
  } catch (e) {
    $("scene").innerHTML = "<option>场景目录不可用</option>";
  }
}

function clearMedia() {
  mediaRenderGeneration += 1;   // 使更早场景的在途结果全部失效
  displayedImageKey = null;
  pendingImageKey = null;
  videoCacheKey = null;         // 新场景/清空使视频缓存失效
  show("images", '<span class="hint">等待场景画面…</span>');
  show("videos", '<span class="hint">本次请求到达终态后显示其视频。</span>');
}

async function loadSession() {
  const sceneId = $("scene").value;
  if (!sceneId) { $("loadmsg").textContent = "请选择场景"; return; }
  $("load").disabled = true;
  $("loadmsg").textContent = "正在创建并重置场景…";
  try {
    const s = await getJSON("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        scene_id: sceneId,
        seed: parseInt($("seed").value || "0", 10),
        init_state_index: parseInt($("idx").value || "0", 10),
      }),
    });
    currentSessionId = s.session_id;
    currentSession = s;
    currentRequest = null;
    currentAgentStatus = null;
    stopRequested = false;
    $("loadmsg").textContent = "已载入：" + s.session_id;
    renderSession(s);
    clearMedia();
    renderSessionImages(s);
    show("plan", '<span class="kv">尚未提交请求。</span>');
    show("subgoals", "");
    show("eval", '<span class="kv">尚未评测。</span>');
    $("hermes").textContent = "暂无。";
    $("runmsg").textContent = "";
  } catch (e) {
    $("loadmsg").textContent = "载入失败：" + (e.data && (e.data.error || e.data.reason) || e.message);
  } finally {
    updateControls();
  }
}

async function refreshSession() {
  // 只读刷新同一个 currentSessionId 的缓存画面/存储/能力；绝不重建/重置会话。
  if (!currentSessionId) return;
  if (sessionRefreshInFlight) return;   // 在途保护：绝不重叠刷新
  sessionRefreshInFlight = true;
  const sid = currentSessionId;
  try {
    const s = await getJSON("/api/session/" + encodeURIComponent(sid));
    if (sid !== currentSessionId) return; // 场景已切换：丢弃过期响应
    currentSession = s;
    renderSession(s);
    // 服务返回缓存画面时才更新；否则保留指令前的初始场景。
    if ((s.images || []).length) renderSessionImages(s);
  } catch (e) { /* 保持上次显示 */ }
  finally { sessionRefreshInFlight = false; }
}

function renderSession(s) {
  show("session",
    "session_id: " + esc(s.session_id) + "<br>" +
    "state: <b>" + esc(s.state) + "</b>　" +
    "scene_id: " + esc(s.scene_id) + "　" +
    "scene_version: " + esc(s.scene_version) + "<br>" +
    "env_instance_id: " + esc(s.env_instance_id) + "　" +
    "episode_resets: " + esc(s.episode_resets) + "　" +
    "total_steps: " + esc(s.total_steps) +
    (s.error ? "<br><span class='bad-text'>error: " + esc(s.error) + "</span>" : ""));

  const policy = s.storage_policy || {};
  const keys = Object.keys(policy);
  let pol = "<b>storage_policy:</b> ";
  if (!keys.length) pol += "<span class='hint'>（本场景无公开收纳规则）</span>";
  else pol += keys.map(k => esc(k) + " → " + esc(policy[k])).join("，");
  show("policy", pol);

  const caps = s.capabilities || [];
  const body = $("caps").querySelector("tbody");
  if (!caps.length) {
    body.innerHTML = '<tr><td colspan="3" class="hint">本场景没有公开能力。</td></tr>';
  } else {
    body.innerHTML = caps.map(c =>
      "<tr><td>" + esc(c.capability_id) + "</td><td>" + esc(c.instruction) +
      "</td><td>" + esc(c.evidence) + "</td></tr>").join("");
  }
}

function renderSessionImages(s) {
  // 同步入口：只在这里启动预加载，绝不直接改 DOM；全部就绪后一次性原子替换。
  const imgs = s.images || [];
  const frames = [];
  imgs.forEach(img => {
    const url = toArtifact(img.image_path);
    if (!url) return; // 无法映射为同源 artifact 的记录不参与渲染
    frames.push({
      view: img.view || img.kind || "view",
      url: url,
      revision: img.sha256 || String(s.scene_version ?? ""),
    });
  });
  if (!frames.length) { return; } // 没有可用画面：保留上一次显示

  const key = JSON.stringify([s.session_id, frames.map(frame => [frame.view, frame.url, frame.revision])]);

  if (key === displayedImageKey) {
    // 当前显示已是该键：若存在不同的在途预加载（A -> B -> A），先取消过期的 B。
    if (pendingImageKey && pendingImageKey !== key) {
      mediaRenderGeneration += 1;
      pendingImageKey = null;
    }
    return;
  }
  if (key === pendingImageKey) { return; } // 同键预加载已在途：不重复下载

  mediaRenderGeneration += 1;
  const generation = mediaRenderGeneration;
  pendingImageKey = key;

  const loads = frames.map(frame => new Promise((resolve, reject) => {
    const image = new Image();
    image.className = "frame";
    image.alt = frame.view;
    image.onload = () => resolve(image);          // 处理器必须在 src 之前赋值
    image.onerror = () => reject(new Error("image load failed"));
    image.src = frame.url + "?v=" + encodeURIComponent(frame.revision); // 修订键缓存破坏
  }));

  Promise.all(loads).then(images => {
    // 仅当本组仍是最新且会话未切换时才提交。
    if (generation !== mediaRenderGeneration) return;
    if (s.session_id !== currentSessionId) return;
    const nodes = [];
    frames.forEach((frame, index) => {
      const wrapper = document.createElement("div");
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = frame.view;
      wrapper.appendChild(pill);
      nodes.push(wrapper);       // 先标签
      nodes.push(images[index]); // 再已加载好的同一 Image 节点（不二次下载）
    });
    $("images").replaceChildren(...nodes); // 一次性替换（保留已加载好的 Image 节点）
    showCache.set("images", "__media_nodes__"); // 直改 DOM：标记 show 缓存失效以保持一致
    displayedImageKey = key;
    pendingImageKey = null;
  }).catch(() => {
    // 任一图片失败：完整保留原显示；仅当本组仍最新时可清除 pending 以便同键重试。
    if (generation === mediaRenderGeneration) pendingImageKey = null;
  });
}

async function submitRequest() {
  if (!currentSessionId) { $("runmsg").textContent = "请先载入场景"; return; }
  const request = $("request").value.trim();
  if (!request) { $("runmsg").textContent = "请输入请求"; return; }
  $("run").disabled = true;
  $("runmsg").textContent = "已提交…";
  const payload = { session_id: currentSessionId, request: request };
  const cid = $("case").value.trim();
  if (cid) payload.case_id = cid;
  try {
    const data = await getJSON("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    currentRequest = data.request_id;
    currentAgentStatus = data.status || "queued";
    stopRequested = false;      // 新请求：重新允许停止
    videoCacheKey = null;       // 新请求使视频缓存失效
    $("runmsg").textContent = "请求已受理：" + currentRequest + "（" + statusLabel(currentAgentStatus) + "）";
    // 新请求：清空旧的计划/子目标/job/视频与说明（画面保留当前场景）。
    show("plan", '<span class="kv">正在规划…（Hermes 尚未提交本次请求的计划，等待中）</span>');
    show("subgoals", "");
    show("eval", '<span class="kv">等待计划终态。</span>');
    $("hermes").textContent = "正在规划或执行，说明将在完成后显示。";
    renderVideos(currentRequest, [], false);
    updateControls();           // 受理后即在计划出现前启用 Stop
    startPolling();
  } catch (e) {
    const msg = (e.data && (e.data.error || e.data.reason)) || e.message;
    $("runmsg").textContent = "提交失败：" + msg;
    updateControls();
  }
}

function capInstruction(capId) {
  if (!currentSession) return "";
  const caps = currentSession.capabilities || [];
  const found = caps.find(c => c.capability_id === capId);
  return found ? found.instruction : "";
}

// 本地抓取阶段映射：原生取值 grasp_confirmed/attempting/approaching/unknown/
// failed_grasp/already_placed。绝不用历史确认冒充“当前持有”；其余原生值（含
// approaching/already_placed/failed_grasp）只如实转义回显，绝不编造状态。
function graspStageText(stage) {
  const s = (stage == null) ? "" : String(stage);
  if (s === "grasp_confirmed") return "已确认抓起过";
  if (s === "attempting") return "正在尝试抓取";
  if (s === "unknown") return "抓取状态待确认";
  if (!s) return "-";
  return esc(s);
}

// 计数器校验：仅有限非负整数才显示，否则显示 '-'（绝不回显注入/未校验值）。
function intOrDash(v) {
  if (typeof v === "number" && isFinite(v) && v >= 0 && Math.floor(v) === v) return String(v);
  return "-";
}

// 固定 wine 诊断单元格（供执行明细表三列）：抓取阶段/停止原因、标准目标命中、
// 语义放置评价。仅 capability_id === "wine_to_rack" 展示 wine 诊断；非 wine 一律
// '-'/'未启用'，绝不因初始字段为空而显示 false 失败。语义真值只由 semantic_spec_id
// === "semantic_wine_rack_v1" 门控，且始终独立于 job.success/plan success，绝不提升为成功。
function wineJobStatus(j) {
  const job = (j && typeof j === "object") ? j : {};

  // 列1：抓取阶段 / 停止原因（所有 job 通用，原生值一律转义）。
  let phase = graspStageText(job.grasp_stage);
  if (job.ended_reason != null && String(job.ended_reason) !== "") {
    phase += " · " + esc(String(job.ended_reason));
  }
  const graspCell = "<td>" + phase + "</td>";

  if (job.capability_id !== "wine_to_rack") {
    // 非 wine：不评估 wine 诊断，绝不显示为失败。
    return graspCell + "<td>-</td><td><span class='hint'>未启用</span></td>";
  }

  // 列2：标准目标命中（native 谓词）。null/缺失 = 尚无法确认，绝不为 false。
  const native = job.native_wine_predicate;
  let nativeCell;
  if (native === true) nativeCell = "<span class='ok-text'>已命中</span>";
  else if (native === false) nativeCell = "<span class='warn-text'>未命中</span>";
  else nativeCell = "<span class='kv'>尚无法确认</span>";

  // 列3：语义放置评价。仅合法 spec 才允许依据 semantic_success 判真；无样本仍为未知。
  const specOk = (job.semantic_spec_id === "semantic_wine_rack_v1");
  const samples = job.semantic_observation_samples;
  const samplesZero = (typeof samples === "number" && isFinite(samples) && samples === 0);
  const semanticTrue = specOk && !samplesZero && job.semantic_success === true;
  let semanticCell;
  if (!specOk || samplesZero) {
    semanticCell = "<span class='kv'>尚无法确认</span>";
  } else if (job.semantic_success === true) {
    semanticCell = "<span class='ok-text'>已松手并稳定支撑在架子上</span>";
  } else if (job.semantic_success === false) {
    semanticCell = "<span class='warn-text'>尚未满足</span>";
  } else {
    semanticCell = "<span class='kv'>尚无法确认</span>";
  }
  // native 未命中但语义取得合法真值时，独立额外提示（不改动 raw success/plan success）。
  if (native === false && semanticTrue) {
    semanticCell += "<div class='hint'>语义达成，标准区域未命中</div>";
  }
  // 计数器与状态：仅校验后的有限非负整数才显示，动态字符串一律转义。
  let meta = "<div class='hint'>样本 " + intOrDash(samples) +
             " · 连续 " + intOrDash(job.semantic_candidate_streak);
  if (job.semantic_state != null && String(job.semantic_state) !== "") {
    meta += " · 状态 " + esc(String(job.semantic_state));
  }
  meta += "</div>";
  semanticCell += meta;

  return graspCell + "<td>" + nativeCell + "</td><td>" + semanticCell + "</td>";
}

// 纯函数：只根据真实终态结果返回 HTML 字符串（不碰 DOM）。
function finalSummaryHTML(result) {
  if (!result || typeof result !== "object") return "";
  let html = "<br>run_ok: " + fmtBool(result.run_ok) +
    "　chain_ok: " + fmtBool(result.chain_ok) +
    "　plan_success: " + fmtBool(result.plan_success) +
    "　task_success: " + fmtBool(result.task_success) +
    "　execution_timeout: " + fmtBool(result.execution_timeout) +
    "　Hermes 规划/修复会话: " + esc(result.hermes_invocations) + "　wall_s: " + esc(result.wall_s);
  if (result.error) {
    html += "<br><span class='bad-text'>error: " + esc(result.error) + "</span>";
  }
  return html;
}

function renderPlan(plan, result) {
  // 计划本体 + 终态摘要一次性构建；摘要只在真实终态且带 result 时出现。
  if (!plan) {
    if (result && typeof result === "object") {
      const cancelled = isResultCancelled(result);
      const cls = cancelled ? "warn-text" : "bad-text";
      const msg = cancelled
        ? "已停止：本次请求在提交任何计划之前被取消，未执行机器人。"
        : "本次请求未提交任何计划。";
      show("plan", '<span class="' + cls + '">' + esc(msg) + "</span>" + finalSummaryHTML(result));
    } else {
      show("plan", '<span class="kv">正在规划…（Hermes 尚未提交本次请求的计划，等待中）</span>');
    }
    return;
  }
  if (plan.error && !plan.decision) {
    show("plan", '<span class="bad-text">计划错误：' + esc(plan.error) + "</span>" + finalSummaryHTML(result));
    return;
  }
  const decision = plan.decision || "-";
  let decisionText = "";
  if (decision === "clarify") decisionText = "（需要澄清，未执行机器人）";
  else if (decision === "unsupported") decisionText = "（不支持，未执行机器人）";
  else if (decision === "execute") decisionText = "（执行）";
  show("plan",
    "request_id: " + esc(plan.request_id) + "<br>" +
    "state: <b>" + esc(plan.state) + "</b>　decision: <b>" + esc(decision) + "</b> " + decisionText + "<br>" +
    "plan_success: " + fmtBool(plan.plan_success) + "　scene_version: " + esc(plan.scene_version) + "<br>" +
    "capability_ids: " + esc((plan.capability_ids || []).join(", ") || "（空）") + "<br>" +
    "rationale: " + esc(plan.rationale || "-") +
    (plan.error ? "<br><span class='bad-text'>error: " + esc(plan.error) + "</span>" : "") +
    finalSummaryHTML(result));
}

function renderSubgoals(plan, jobs) {
  // 子目标进度与「该计划自身 job」的执行明细一起、每次轮询整体重绘（不叠加）。
  const selected = (plan && plan.capability_ids) || [];
  const done = (plan && plan.completed_capability_ids) || [];
  const pend = (plan && plan.pending_capability_ids) || [];
  const decision = (plan && plan.decision) || "-";
  let html = "";
  if (!selected.length) {
    if (decision !== "execute") {
      html += '<div class="hint">本次没有选择任何能力（' + esc(decision) + '）。</div>';
    }
  } else {
    html += "<h2>子目标进度</h2>" + selected.map(capId => {
      // 最新状态取本次请求 jobs 原序中最后一个 capability_id 匹配的 job（无匹配则为 null）。
      const matches = (jobs || []).filter(j => j && j.capability_id === capId);
      const latestJob = matches.length ? matches[matches.length - 1] : null;
      const planState = (plan && plan.state) || "";
      const planTerminal = isTerminalPlan(planState);
      let cls = "pill", label = "待定";
      if (latestJob && latestJob.state === "running" && planState === "running") {
        cls += " pend"; label = "进行中";
      } else if (latestJob && latestJob.state === "queued" &&
                 (planState === "queued" || planState === "running")) {
        cls += " pend"; label = "排队中";
      } else if (done.includes(capId) || (latestJob && latestJob.success === true)) {
        cls += " done"; label = "已完成";
      } else if (latestJob && latestJob.ended_reason === "failed_grasp") {
        // 未抓起：仅 enforce 模式才代表任务被停止；shadow/观察模式只如实说明
        // “检测到未抓起”，绝不宣称已停止。completed + success false 是执行完成而非成功。
        cls += " pend";
        label = (latestJob.grasp_guard_mode === "enforce")
          ? "未抓起，任务已停止"
          : "检测到未抓起（观察模式）";
      } else if (latestJob && latestJob.success === false) {
        cls += " pend"; label = "执行未成功";
        if (pend.includes(capId) && planTerminal) label += "（后续未执行）";
      } else if (planTerminal) {
        label = "未执行（计划已停止）";
      } else if (pend.includes(capId)) {
        cls += " pend"; label = "待执行";
      }
      return "<div style='margin:4px 0'><span class='" + cls + "'>" + label + "</span>" +
             esc(capId) + " — " + esc(capInstruction(capId)) + "</div>";
    }).join("");
  }
  if (jobs && jobs.length) {
    html += "<h2>执行明细（本次请求）</h2><div class='scroll'><table><thead><tr>" +
      "<th>job_id</th><th>能力</th><th>状态</th><th>本段步数 / 场景累计步数</th><th>success</th>" +
      "<th>抓取阶段 / 停止原因</th><th>标准目标命中</th><th>语义放置评价</th></tr></thead><tbody>";
    jobs.forEach(j => {
      html += "<tr><td>" + esc(j.job_id) + "</td><td>" + esc(j.capability_id) + "</td><td>" +
              esc(j.state) + "</td><td>" +
              esc("本段 " + (j.steps != null ? j.steps : "-") + "；累计 " +
                  (j.total_steps != null ? j.total_steps : "-")) + "</td><td>" +
              fmtBool(j.success) + "</td>" + wineJobStatus(j) + "</tr>";
    });
    html += "</tbody></table></div>";
  }
  show("subgoals", html);
}

function renderVideos(reqId, jobs, terminal) {
  // 仅当本次请求到达终态（completed/blocked/cancelled/error）才显示其视频；
  // 且只使用该计划自身的 job（绝不回退到历史 job/视频）。
  // 缓存键 = [request id, terminal, 归属 job IDs, artifact URLs]；键相同则保留
  // 现有 video 节点与播放位置，绝不替换 DOM。
  const vids = (jobs || []).filter(j => j && j.rollout_path);
  const urls = vids.map(j => toArtifact(j.rollout_path)).filter(Boolean);
  const key = JSON.stringify([reqId || null, !!terminal, vids.map(j => j.job_id || null), urls]);
  if (key === videoCacheKey) return;   // 同一快照：保留节点与播放位置
  videoCacheKey = key;

  if (!terminal) {
    show("videos", '<span class="hint">等待本次请求到达终态后显示其视频。</span>');
    return;
  }
  if (!vids.length) {
    show("videos", '<span class="hint">本次请求没有产生机器人执行视频。</span>');
    return;
  }
  const nodes = [];
  vids.forEach(j => {
    const url = toArtifact(j.rollout_path);
    if (!url) return;
    const label = document.createElement("div");
    label.className = "hint";
    label.textContent = (j.capability_id || "") + " · " + (j.job_id || "");
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = url;
    nodes.push(label);
    nodes.push(video);
  });
  $("videos").replaceChildren(...nodes);
  showCache.set("videos", "__media_nodes__"); // 直改 DOM：标记 show 缓存失效以保持一致
}

function renderEvaluation(ev, decision) {
  if (ev && typeof ev === "object") {
    if (ev.error) { show("eval", '<span class="bad-text">评测失败：' + esc(ev.error) + "</span>"); return; }
    show("eval",
      "task_success: " + fmtBool(ev.task_success) + "<br>" +
      "oracle_source: " + esc(ev.oracle_source || "-") + "<br>" +
      "<span class='hint'>独立评测结果，非模型自述。</span>");
    return;
  }
  show("eval", '<span class="kv">本次请求未执行独立评测（未提供 case_id）。</span>');
}

const PLAN_TERMINAL_STATES = ["completed", "blocked", "error", "cancelled"];

function isTerminalPlan(state) {
  return PLAN_TERMINAL_STATES.includes(state);
}

function jobBelongsTo(job, requestId, sessionId) {
  // 只显示与当前请求/会话精确关联的 job：任何显式不匹配都丢弃。
  if (!job || typeof job !== "object") return false;
  if (job.request_id && job.request_id !== requestId) return false;
  if (job.session_id && job.session_id !== sessionId) return false;
  return true;
}

function renderProgress(plan, jobs, result) {
  renderPlan(plan, result);
  renderSubgoals(plan, jobs);
}

async function pollCurrent() {
  if (!currentRequest) return;
  if (pollInFlight) return;   // 在途保护：绝不重叠轮询
  pollInFlight = true;
  const reqId = currentRequest;
  const sesId = currentSessionId;
  try {
    // 后台 agent job：仅用于终态后的最终结果；进度不再由它门控。
    let agentJob = null;
    try { agentJob = await getJSON("/api/agent/" + encodeURIComponent(reqId)); }
    catch (e) { agentJob = null; }
    if (reqId !== currentRequest || sesId !== currentSessionId) return; // 丢弃过期响应

    if (agentJob && typeof agentJob.status === "string") {
      // 采纳真实归属 job 状态**之前**先记录本次“停止意图”：终态清除 stopRequested 后，
      // 仍能据此判断终态是否是一次停止竞态（completed/error 后无需停止）。
      const stopIntent = stopRequested;
      currentAgentStatus = agentJob.status;
      if (agentTerminalStatus(currentAgentStatus)) stopRequested = false;
      // 仅在真实 job 状态推进后如实刷新 runmsg（其余生命周期一律不动）：
      //  - cancelling：显示“正在停止…”或真实非空 cancel_note（绝不谎称已停止）；
      //  - cancelled：如实显示已停止且保留当前场景；
      //  - completed/error：仅当此前确有停止意图时告知任务已结束、无需停止；
      //  - queued/running：绝不覆盖 Stop API 失败原因（保留诚实失败文本）。
      if (currentAgentStatus === "cancelling") {
        const note = (typeof agentJob.cancel_note === "string" && agentJob.cancel_note)
          ? agentJob.cancel_note : "正在停止…";
        $("runmsg").textContent = note;
      } else if (currentAgentStatus === "cancelled") {
        $("runmsg").textContent = "已停止（保留当前场景）。";
      } else if (stopIntent && agentTerminalStatus(currentAgentStatus)) {
        $("runmsg").textContent = "任务已结束，无需停止。";
      }
    }
    const jobTerminal = agentTerminalStatus(currentAgentStatus);
    const result = (agentJob && agentJob.result && typeof agentJob.result === "object")
      ? agentJob.result : null;

    // 步骤1：始终按精确 request_id 拉取计划，无论 job.result 是否存在；容忍 404。
    let plan = null;
    let planMissing = false;
    let planUnreachable = false;
    try {
      plan = await getJSON("/api/plan/" + encodeURIComponent(reqId));
    } catch (e) {
      if (e.status === 404) planMissing = true;
      else planUnreachable = true;
    }
    if (reqId !== currentRequest || sesId !== currentSessionId) return; // 丢弃过期响应
    if (planUnreachable) { updateControls(); return; } // 服务暂不可用：保留上次显示

    if (planMissing || !plan || typeof plan !== "object") {
      // Hermes 尚未提交计划（404）：给出可理解的规划中/已停止提示，绝不回退到别的计划。
      show("subgoals", "");
      if (jobTerminal) {
        if (result) {
          renderPlan(null, result);   // 诚实展示“未提交计划/已停止，未执行机器人”
        } else {
          // job 已终态但没有结果对象：仍如实显示终态，绝不退回“正在规划”。
          const stopped = currentAgentStatus === "cancelled";
          show("plan", '<span class="' + (stopped ? "warn-text" : "bad-text") + '">' +
            esc(stopped ? "已停止：本次请求在提交任何计划之前被取消，未执行机器人。"
                        : "本次请求未提交任何计划（job " + statusLabel(currentAgentStatus) + "）。") +
            "</span>");
        }
        renderVideos(reqId, [], true);
        renderEvaluation(result ? result.evaluation : null, result ? result.decision : null);
        loadHermes(result);
      } else {
        renderPlan(null, null);
        renderVideos(reqId, [], false);
      }
      updateControls();
      return;
    }

    // 只接受精确属于当前请求的计划。
    if (plan.request_id && plan.request_id !== reqId) { updateControls(); return; }

    // 步骤2：仅取该计划自己的 job_ids，并校验归属后才显示。
    const jobIds = Array.isArray(plan.job_ids) ? plan.job_ids : [];
    const jobs = [];
    for (const jid of jobIds) {
      if (!jid) continue;
      try {
        const job = await getJSON("/api/job/" + encodeURIComponent(jid));
        if (reqId !== currentRequest || sesId !== currentSessionId) return; // 丢弃过期响应
        if (jobBelongsTo(job, reqId, sesId)) jobs.push(job);
      } catch (e) { /* 单个 job 尚未就绪：跳过，下次再取 */ }
    }
    if (reqId !== currentRequest || sesId !== currentSessionId) return;

    const terminal = isTerminalPlan(plan.state) || jobTerminal;

    // 运动期间即渲染初始计划与每个子目标/真实步数；终态摘要只在真实终态带 result 时出现。
    renderProgress(plan, jobs, terminal ? result : null);

    // 步骤4：视频只在本次请求终态后出现（含 blocked/cancelled/error）。
    renderVideos(reqId, jobs, terminal);

    // 步骤5：终态后继续渲染独立评测/Hermes 说明（摘要已随 renderPlan 一次构建）。
    if (terminal && jobTerminal) {
      renderEvaluation(result ? result.evaluation : null, result ? result.decision : null);
      loadHermes(result);
    }
    updateControls();
  } finally {
    pollInFlight = false;
  }
}

async function stopRequest() {
  if (stopInFlight) return;   // 单次在途取消，防双击
  if (!currentRequest || !currentSessionId) { $("runmsg").textContent = "没有可停止的请求"; return; }
  const reqId = currentRequest;
  const sesId = currentSessionId;
  stopInFlight = true;
  stopRequested = true;
  updateControls();           // 立即禁用 Stop
  $("runmsg").textContent = "正在停止…";
  try {
    const data = await getJSON("/api/agent/" + encodeURIComponent(reqId) + "/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sesId }),
    });
    if (reqId !== currentRequest || sesId !== currentSessionId) return; // 过期响应：不修改新请求
    // 如实读取取消 API 的实际状态（noop 可能表示终态竞态），绝不一律宣称已受理停止。
    // 绝不清理当前 request/session/media。
    const status = data && data.status;
    if (status === "completed" || status === "error") {
      $("runmsg").textContent = "任务已结束，无需停止。";
    } else if (status === "cancelled") {
      $("runmsg").textContent = "已停止";
    } else if (status === "cancelling") {
      // 仅真正的活动 cancelling 确认才提示正在停止/已受理。
      $("runmsg").textContent = "停止请求已受理，当前动作或推理结束后生效（保留当前场景）。";
    } else {
      // 状态异常/未知：绝不谎报已受理停止。
      $("runmsg").textContent = "停止未确认，请稍后重试。";
    }
  } catch (e) {
    if (reqId !== currentRequest || sesId !== currentSessionId) return; // 过期响应：不修改新请求
    const msg = (e.data && (e.data.error || e.data.detail || e.data.reason)) || e.message;
    $("runmsg").textContent = "停止失败：" + msg;
    stopRequested = false;    // 失败：重新启用，允许重试
  } finally {
    stopInFlight = false;
    if (reqId === currentRequest && sesId === currentSessionId) updateControls();
  }
}

async function loadHermes(result = null) {
  if (!currentRequest) return;
  const reqId = currentRequest;
  // 优先显示 Runner._finish 保存的真实合并 CLI 输出（原文 textContent，不合成解释）。
  if (result && typeof result === "object" &&
      typeof result.hermes_output === "string" && result.hermes_output) {
    const out = result.hermes_output;
    $("hermes").textContent = out.length > 8000 ? out.slice(-8000) : out;
    return;
  }
  const url = "/agent-artifacts/" + encodeURIComponent(reqId) + "/hermes_initial.log";
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (reqId !== currentRequest) return; // 请求已切换：丢弃过期响应
    if (!r.ok) { $("hermes").textContent = "说明将在完成后显示（HTTP " + r.status + "）"; return; }
    const text = await r.text();
    if (reqId !== currentRequest) return; // 请求已切换：丢弃过期响应
    $("hermes").textContent = text.length > 8000 ? text.slice(-8000) : text || "（空）";
  } catch (e) {
    $("hermes").textContent = "说明将在完成后显示。";
  }
}

async function recoverActiveJob() {
  // 启动恢复：GET /api/agent，仅当**恰好一个** active（queued/running/cancelling）
  // job 时才采纳其精确 request_id/session_id；绝不选历史终态 job，也不选全局 latest。
  // 零个保持初始视图；多个 active 一个都不选。绝不创建/重置场景：只读精确 session。
  let listing = null;
  try { listing = await getJSON("/api/agent"); } catch (e) { return; }
  const jobs = (listing && Array.isArray(listing.jobs)) ? listing.jobs : [];
  const active = jobs.filter(j => j && agentBusy(j.status));
  if (active.length !== 1) return;             // 0 -> 初始视图；>1 -> 不选
  const job = active[0];
  if (!job.request_id || !job.session_id) return;
  const reqId = job.request_id;
  let session = null;
  try { session = await getJSON("/api/session/" + encodeURIComponent(job.session_id)); }
  catch (e) { session = null; }
  if (reqId !== job.request_id) return;        // 过期保护（理论上不会变）
  currentRequest = reqId;                      // 采纳精确归属
  currentAgentStatus = job.status;
  stopRequested = false;
  currentSessionId = job.session_id;
  if (session && typeof session === "object") {
    currentSession = session;
    renderSession(session);
    if ((session.images || []).length) renderSessionImages(session);  // 只读画面
  }
  renderVideos(reqId, [], false);
  updateControls();                            // 占用中：启用 Stop，禁用新提交/载入
  startPolling();                              // 交给同一轮询调度器，绝不另起链
}

// 单一完成驱动 tick：每 2 秒一次（上一次 tick 完成后才排下一次），绝不重叠。
function startPolling() {
  // 清除任何待触发的 timeout，避免二次调用留下两条链。
  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  // 当前 tick 仍在执行：由它的 finally 负责排下一次，绝不再起第二条链。
  if (tickInFlight) return;
  tick();
}

async function tick() {
  if (tickInFlight) return;   // 忙：绝不重叠
  tickInFlight = true;        // 在所有 await 之前置位
  try {
    await pollHealth();
    await refreshSession();   // 同一 session 无论 idle 或 current request 都只刷新一次
    await pollCurrent();
  } catch (e) { /* 保持上次显示 */ }
  finally {
    tickInFlight = false;
    updateControls();
    pollTimer = setTimeout(tick, 2000);   // 恰好排一个下一次
  }
}

$("load").addEventListener("click", loadSession);
$("run").addEventListener("click", submitRequest);
$("stop").addEventListener("click", stopRequest);
loadScenes();
recoverActiveJob();   // 启动时按精确 active 归属恢复（绝不新建/重置场景）
startPolling();
</script>
</body>
</html>
"""


def render_page() -> str:
    return PAGE.replace("__RUNS_ROOT__", RUNS_DIR + "/")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="持久场景 v2 本地网页前端")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("持久场景前端已启动: http://localhost:%d" % args.port, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

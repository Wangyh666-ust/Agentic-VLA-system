#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地网页 + HTTP API：自然语言输入 -> Hermes/libero_tools -> LIBERO SmolVLA。

仅使用 Python 标准库。默认监听 127.0.0.1:8080（用 --port 修改）。

端点：
  GET  /                 中文网页
  GET  /api/health       代理服务 /health（并附前端标识）
  GET  /api/tasks        代理服务 /tasks 任务目录
  GET  /api/observe      代理服务 /observe 最近画面
  GET  /api/execution    代理服务 /status（进度；可带 ?job_id=）
  GET  /api/agent        列出独立 run_dir（内存 job + 磁盘产物）
  POST /api/agent        启动唯一后台 agent job；忙时 409
  GET  /api/agent/<id>   job 状态 queued/running/completed/error 与 result
  GET  /artifacts/<rel>  仅暴露 runs 目录下文件（支持 MP4 Range 请求）

后台不重复实现 Agent：用本文件同一个 venv 的 Python 调用 run_agent.py。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import subprocess
import sys
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

SERVICE_BASE = "http://127.0.0.1:8766"
RUNS_DIR = "/home/yhwang/fyp/libero_demo/runs"
FRONTEND_ID = "libero_demo_frontend"

MEDIA_EXTS = {".mp4", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
OTHER_ALLOWED = {"result.json", "request.json", "agent_result.json", "hermes.log"}

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
RUNNING = {"request_id": None}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def proxy_get(path: str, timeout: float = 20.0):
    req = urllib.request.Request(SERVICE_BASE + path, method="GET")
    with _opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_job(request_id: str, request: str, seed: int, init_state_index: int) -> None:
    """后台线程：调用 run_agent.py（真实 Hermes + MCP），不做任何 Agent 逻辑复刻。

    run_agent.py 会一直阻塞到本次请求的执行到达终态（或在总 deadline 到期），因此这里的
    子进程存活期间，网页 job 会保持 status=running，直到真实终态才落到 completed/error。
    """
    job = JOBS.get(request_id)
    if job is None:
        return
    run_dir = job["run_dir"]
    log_path = os.path.join(run_dir, "run_agent.log")
    cmd = [
        PYTHON, RUN_AGENT,
        "--request", request,
        "--seed", str(seed),
        "--init-state-index", str(init_state_index),
        "--run-dir", run_dir,
    ]
    with JOBS_LOCK:
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
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(stdout_text)
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
        ap = os.path.join(run_dir, "agent_result.json")
        if os.path.isfile(ap):
            try:
                with open(ap, encoding="utf-8") as fh:
                    result = json.load(fh)
            except (OSError, ValueError):
                result = None

    with JOBS_LOCK:
        job["exit_code"] = exit_code
        job["result"] = result
        job["log_path"] = log_path
        job["finished"] = _now()
        job["status"] = "completed" if exit_code == 0 else "error"
        if RUNNING.get("request_id") == request_id:
            RUNNING["request_id"] = None


def scan_runs():
    """扫描 runs 目录，列出独立 run_dir（重启后无需恢复旧进行中 job）。"""
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        path = os.path.join(RUNS_DIR, name)
        if not os.path.isdir(path):
            continue
        entry = {
            "run_dir": path,
            "run_name": name,
            "status": "on_disk",
            "request": None,
            "result": None,
        }
        ap = os.path.join(path, "agent_result.json")
        if os.path.isfile(ap):
            try:
                with open(ap, encoding="utf-8") as fh:
                    ar = json.load(fh)
                entry["result"] = ar
                entry["request"] = (ar.get("request") or {}).get("request")
                entry["status"] = "completed"
            except (OSError, ValueError):
                pass
        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "LiberoDemoUI/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---- low level -------------------------------------------------------- #
    def _send_bytes(self, body: bytes, code=200, ctype="application/octet-stream",
                    extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, code, "application/json; charset=utf-8")

    def _send_text(self, text, code=200, ctype="text/plain; charset=utf-8"):
        self._send_bytes(text.encode("utf-8"), code, ctype)

    def _send_416(self, size):
        """Range 不可满足：只回状态与头部（不写 body，HEAD 亦安全）。"""
        self.send_response(416)
        self.send_header("Content-Range", "bytes */%d" % size)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    # ---- routing ---------------------------------------------------------- #
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send_text(render_page(), 200, "text/html; charset=utf-8")
            elif path == "/api/health":
                self._api_health()
            elif path == "/api/tasks":
                self._api_proxy("/tasks")
            elif path == "/api/observe":
                self._api_proxy("/observe")
            elif path == "/api/execution":
                job_id = (query.get("job_id") or [None])[0]
                q = ("?job_id=" + urllib.parse.quote(job_id)) if job_id else ""
                self._api_proxy("/status" + q)
            elif path == "/api/agent":
                self._api_list_agents()
            elif path.startswith("/api/agent/"):
                self._api_agent_status(path[len("/api/agent/"):])
            elif path.startswith("/artifacts/"):
                self._serve_artifact(path[len("/artifacts/"):])
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
        if parsed.path != "/api/agent":
            self._send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            self._send_json({"error": "请求体必须是 JSON 对象"}, 400)
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send_json({"error": "invalid json"}, 400)
            return
        if not isinstance(payload, dict):
            self._send_json({"error": "请求体必须是 JSON 对象"}, 400)
            return
        request_text = payload.get("request")
        if not isinstance(request_text, str) or not request_text.strip():
            self._send_json({"error": "request 必须是非空字符串"}, 400)
            return
        request_text = request_text.strip()
        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed < 2 ** 32):
            self._send_json({"error": "seed 必须是 [0, 2**32) 内的整数"}, 400)
            return
        init_state_index = payload.get("init_state_index", 0)
        if (isinstance(init_state_index, bool)
                or not isinstance(init_state_index, int)
                or init_state_index < 0):
            self._send_json({"error": "init_state_index 必须是非负整数"}, 400)
            return

        with JOBS_LOCK:
            if RUNNING.get("request_id") is not None:
                self._send_json(
                    {"error": "已有 agent 任务在运行", "busy": True,
                     "request_id": RUNNING["request_id"]}, 409)
                return
            request_id = uuid.uuid4().hex[:12]
            run_dir = os.path.join(
                RUNS_DIR, "web_%s_%s" % (datetime.now().strftime("%Y%m%d_%H%M%S"),
                                         request_id))
            os.makedirs(run_dir, exist_ok=True)
            JOBS[request_id] = {
                "request_id": request_id,
                "status": "queued",
                "request": request_text,
                "seed": seed,
                "init_state_index": init_state_index,
                "run_dir": run_dir,
                "created": _now(),
                "started": None,
                "finished": None,
                "exit_code": None,
                "result": None,
                "log_path": None,
            }
            RUNNING["request_id"] = request_id

        thread = threading.Thread(
            target=run_job, args=(request_id, request_text, seed, init_state_index),
            daemon=True)
        thread.start()
        self._send_json({"request_id": request_id, "status": "queued",
                         "run_dir": run_dir}, 202)

    # ---- api impl --------------------------------------------------------- #
    def _api_health(self):
        out = {"frontend": FRONTEND_ID}
        try:
            h = proxy_get("/health")
            if isinstance(h, dict):
                out.update(h)
        except Exception as exc:  # noqa: BLE001
            out.update({"reachable": False, "ready": False, "error": str(exc)})
        self._send_json(out)

    def _api_proxy(self, path):
        try:
            self._send_json(proxy_get(path))
        except urllib.error.HTTPError as exc:
            payload = {"error": "service http %s" % exc.code}
            # 透传服务端原因（unknown_job / no_jobs 等），便于前端区分
            # “尚无执行记录”与真实错误；状态码与 error 字段保持不变。
            try:
                body = json.loads(exc.read().decode("utf-8"))
                if isinstance(body, dict):
                    for key in ("ok", "reason", "job_id"):
                        if key in body:
                            payload[key] = body[key]
            except Exception:  # noqa: BLE001
                pass
            self._send_json(payload, 502)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": "service unreachable: %s" % exc}, 502)

    def _api_list_agents(self):
        with JOBS_LOCK:
            jobs = list(JOBS.values())
        memory_dirs = {j.get("run_dir") for j in jobs}
        disk = [r for r in scan_runs() if r["run_dir"] not in memory_dirs]
        self._send_json({"jobs": jobs, "runs": disk})

    def _api_agent_status(self, request_id):
        request_id = urllib.parse.unquote(request_id)
        with JOBS_LOCK:
            job = JOBS.get(request_id)
        if job is None:
            self._send_json({"error": "unknown request_id", "request_id": request_id}, 404)
            return
        self._send_json(job)

    # ---- artifacts -------------------------------------------------------- #
    def _serve_artifact(self, rel):
        rel = urllib.parse.unquote(rel)
        root = os.path.realpath(RUNS_DIR)
        target = os.path.realpath(os.path.join(root, rel.lstrip("/")))
        if target != root and not target.startswith(root + os.sep):
            self._send_json({"error": "forbidden"}, 403)
            return
        if not os.path.isfile(target):
            self._send_json({"error": "not found"}, 404)
            return
        name = os.path.basename(target)
        ext = os.path.splitext(name)[1].lower()
        if name.startswith(".") or (ext not in MEDIA_EXTS and name not in OTHER_ALLOWED):
            self._send_json({"error": "forbidden"}, 403)
            return

        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        size = os.path.getsize(target)
        range_header = self.headers.get("Range")

        # 空文件没有可服务的字节范围。
        if size <= 0:
            self._send_416(size)
            return

        start, end, status = 0, size - 1, 200
        if range_header:
            m = re.match(r"bytes=(\d*)-(\d*)\s*$", range_header.strip())
            if not m:
                # 无法解析（或非 bytes 单位）：按不可满足处理，不整文件误发。
                self._send_416(size)
                return
            g1, g2 = m.group(1), m.group(2)
            if g1 == "" and g2 == "":
                self._send_416(size)
                return
            if g1 == "":
                n = int(g2)
                if n <= 0:
                    self._send_416(size)
                    return
                start = max(0, size - n)
                end = size - 1
            else:
                start = int(g1)
                end = min(int(g2), size - 1) if g2 else size - 1
            if start >= size or start > end:
                self._send_416(size)
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(target, "rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIBERO SmolVLA 自然语言演示</title>
<style>
  :root { color-scheme: light; }
  body { font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif;
         margin: 0; background: #f4f6f9; color: #1f2933; }
  header { background: #1e3a5f; color: #fff; padding: 14px 22px;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 18px; margin: 0; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 13px;
           background: #64748b; }
  .badge.ok { background: #16a34a; }
  .badge.bad { background: #dc2626; }
  main { max-width: 1100px; margin: 0 auto; padding: 18px; display: grid;
         grid-template-columns: 1fr 1fr; gap: 16px; }
  section { background: #fff; border-radius: 10px; padding: 16px;
            box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  section.wide { grid-column: 1 / -1; }
  h2 { font-size: 15px; margin: 0 0 10px; color: #1e3a5f; }
  textarea { width: 100%; box-sizing: border-box; min-height: 66px;
             padding: 8px; border: 1px solid #cbd5e1; border-radius: 6px;
             font-size: 14px; resize: vertical; }
  .row { display: flex; gap: 12px; align-items: center; margin-top: 10px;
         flex-wrap: wrap; }
  label { font-size: 13px; color: #475569; }
  input[type=number] { width: 90px; padding: 6px; border: 1px solid #cbd5e1;
                       border-radius: 6px; }
  button { background: #2563eb; color: #fff; border: 0; padding: 9px 18px;
           border-radius: 6px; font-size: 14px; cursor: pointer; }
  button:disabled { background: #94a3b8; cursor: not-allowed; }
  pre { background: #0f172a; color: #e2e8f0; padding: 10px; border-radius: 6px;
        font-size: 12px; overflow: auto; max-height: 240px; white-space: pre-wrap;
        word-break: break-all; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { border-bottom: 1px solid #e2e8f0; padding: 6px 8px;
           text-align: left; vertical-align: top; }
  th { background: #f1f5f9; position: sticky; top: 0; }
  .suites { max-height: 320px; overflow: auto; }
  .status-line { font-size: 14px; line-height: 1.9; }
  .kv { color: #475569; }
  .ok-text { color: #16a34a; font-weight: 600; }
  .bad-text { color: #dc2626; font-weight: 600; }
  video { width: 100%; background: #000; border-radius: 6px; }
  img.frame { width: 100%; border-radius: 6px; border: 1px solid #cbd5e1; }
  .hint { font-size: 12px; color: #64748b; }
</style>
</head>
<body>
<header>
  <h1>LIBERO SmolVLA · 自然语言任务演示</h1>
  <span id="svc" class="badge">服务状态: 检测中…</span>
  <span id="backend" class="badge">backend: -</span>
  <span id="robot" class="badge">robot: -</span>
</header>

<main>
  <section class="wide">
    <h2>1. 用自然语言描述任务</h2>
    <textarea id="request" placeholder="例如：把黑色碗放到盘子上"></textarea>
    <div class="row">
      <label>seed <input type="number" id="seed" value="0"></label>
      <label>init_state_index <input type="number" id="idx" value="0"></label>
      <button id="run">执行</button>
      <span id="runmsg" class="hint"></span>
    </div>
    <p class="hint">由 Hermes（qwen3-vl-plus）先读取任务目录，再选出最匹配的标准 LIBERO 任务并调用工具执行。
       服务会为每个 benchmark 任务自动创建对应场景；无需手动指定 task_id。</p>
  </section>

  <section>
    <h2>2. 执行状态（服务返回）</h2>
    <div id="exec" class="status-line kv">等待中…</div>
  </section>

  <section>
    <h2>3. 执行结果</h2>
    <div id="result" class="status-line kv">暂无</div>
  </section>

  <section class="wide">
    <h2>4. Hermes 说明（真实 agent 输出）</h2>
    <pre id="hermes">暂无</pre>
  </section>

  <section>
    <h2>5. 执行视频</h2>
    <video id="video" controls></video>
    <p class="hint">若浏览器支持，可拖动进度条（服务端支持 Range 请求）。</p>
  </section>

  <section>
    <h2>6. 最近画面</h2>
    <img id="frame" class="frame" alt="最近画面（若服务提供）">
    <p class="hint">来自服务 /observe 的最近一帧。</p>
  </section>

  <section class="wide">
    <h2>7. 受支持任务目录（4 套件 / 40 任务，仅供参考）</h2>
    <div class="suites"><table id="tasks">
      <thead><tr><th>suite</th><th>task_id</th><th>标准指令</th></tr></thead>
      <tbody><tr><td colspan="3" class="hint">加载中…</td></tr></tbody>
    </table></div>
  </section>
</main>

<script>
const RUNS_ROOT = "__RUNS_ROOT__";
const $ = (id) => document.getElementById(id);
let currentJob = null;    // 仅表示“正在进行的网页 job”；终态由 pollJob 清空
let pollTimer = null;
let tasksLoaded = false;
let execBaseline = null;  // 提交新请求时服务已显示的 execution job_id（旧进度基线）
let lastExec = null;      // 最近一次 /api/execution 的返回
// 本次请求的持久结果状态：与 currentJob 相互独立，currentJob 被清空后仍保留关联，
// 一直持续到用户提交下一请求。字段：
//   requestId      本次网页请求 id
//   terminal       本次请求是否已到达终态
//   executionJobId 终态且 result.execution 非 null 时为该 execution 的 job_id，否则 null
let requestOutcome = null;

function toArtifact(p) {
  if (!p || typeof p !== "string") return null;
  if (p.startsWith(RUNS_ROOT)) return "/artifacts/" + p.slice(RUNS_ROOT.length);
  return null;
}
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function show(id, html) { $(id).innerHTML = html; }

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  const ct = r.headers.get("content-type") || "";
  if (!ct.includes("json")) throw new Error("HTTP " + r.status);
  return await r.json();
}

// ---- 服务健康 ----
async function pollHealth() {
  try {
    const h = await getJSON("/api/health");
    const ready = h.ready === true;
    const el = $("svc");
    el.textContent = "服务状态: " + (ready ? "就绪" : (h.reachable === false ? "不可达" : "加载中/未就绪"));
    el.className = "badge " + (ready ? "ok" : "bad");
    $("backend").textContent = "backend: " + (h.backend || "-");
    $("robot").textContent = "robot: " + (h.robot || "-");
    // 服务就绪后若任务目录仍未取到 40 条，重试一次（服务加载期间 /tasks 可能不可用）。
    if (ready && !tasksLoaded) loadTasks();
  } catch (e) {
    const el = $("svc");
    el.textContent = "服务状态: 前端可达，服务未知";
    el.className = "badge bad";
  }
}

// ---- 任务目录 ----
function normalizeTasks(data) {
  let arr = Array.isArray(data) ? data
    : (data && (data.tasks || data.catalog || data.items)) || [];
  if (!Array.isArray(arr)) arr = [];
  return arr.map(t => ({
    suite: t.suite || t.suite_name || t.benchmark || "",
    id: t.task_id != null ? t.task_id : (t.id != null ? t.id : (t.name || "")),
    instruction: t.instruction || t.language || t.task || t.name || "",
  }));
}
async function loadTasks() {
  try {
    const data = await getJSON("/api/tasks");
    const tasks = normalizeTasks(data);
    if (tasks.length >= 40) tasksLoaded = true;
    const groups = {};
    tasks.forEach(t => { (groups[t.suite] = groups[t.suite] || []).push(t); });
    let rows = "";
    Object.keys(groups).sort().forEach(suite => {
      groups[suite].forEach(t => {
        rows += "<tr><td>" + esc(suite) + "</td><td>" + esc(t.id) +
                "</td><td>" + esc(t.instruction) + "</td></tr>";
      });
    });
    if (!rows) rows = '<tr><td colspan="3" class="hint">未获取到任务目录</td></tr>';
    $("tasks").querySelector("tbody").innerHTML = rows;
  } catch (e) {
    $("tasks").querySelector("tbody").innerHTML =
      '<tr><td colspan="3" class="hint">任务目录暂不可用：' + esc(e.message) + "</td></tr>";
  }
}

// ---- 执行进度 ----
// 服务在 job 启动时就会给出 rollout_path，但 MP4 要到 episode 结束才编码完成；
// 仅在终态（state completed/error）后才把视频挂上 src，避免访问尚未编码的文件。
function applyRolloutVideo(s) {
  if (!s) return;
  const st = s.state || s.status || "";
  if (st !== "completed" && st !== "error") return;
  const vp = toArtifact(s.rollout_path);
  if (vp && $("video").getAttribute("src") !== vp) $("video").setAttribute("src", vp);
}
function renderExec(s) {
  if (!s) {
    show("exec", '<span class="kv">尚未执行任务</span>');
    return;
  }
  if (s.error) {
    show("exec", '<span class="kv">无进度：' + esc(s.error) + "</span>");
    applyRolloutVideo(s);
    return;
  }
  const steps = s.steps != null ? s.steps
    : (s.steps_executed != null ? s.steps_executed : "-");
  const st = s.state || s.status || "-";
  show("exec",
    "state: <b>" + esc(st) + "</b><br>" +
    "job_id: " + esc(s.job_id || "-") + "<br>" +
    "真实步数: <b>" + esc(steps) + "</b><br>" +
    "suite/task_id: " + esc((s.suite || "-") + " / " + (s.task_id != null ? s.task_id : "-")) + "<br>" +
    "instruction: " + esc(s.instruction || "-") + "<br>" +
    "wall_s: " + esc(s.wall_s != null ? s.wall_s : "-"));
  applyRolloutVideo(s);
}
// 服务是否尚无任何执行记录（/status 返回 unknown_job / no_jobs，或代理转成 404）。
function isNoExecution(s) {
  if (!s) return false;
  if (s.reason === "unknown_job" || s.reason === "no_jobs") return true;
  return typeof s.error === "string" && s.error.indexOf("404") >= 0;
}
// 清空上一轮的视频/画面，并把进度区置为“正在理解请求并选择任务”。
function clearProgressDisplay() {
  const v = $("video");
  v.removeAttribute("src");
  try { v.load(); } catch (e) { /* ignore */ }
  $("frame").removeAttribute("src");
  show("exec", '<span class="kv">正在理解请求并选择任务…</span>');
}
// 只清空视频/画面元素，不改动文字状态。用于“本次请求没有触发机器人执行”时彻底移除
// 上一轮 baseline 的旧视频/画面（幂等：已清空则不重复触发 load）。
function clearMedia() {
  const v = $("video");
  if (v.getAttribute("src")) {
    v.removeAttribute("src");
    try { v.load(); } catch (e) { /* ignore */ }
  }
  if ($("frame").getAttribute("src")) $("frame").removeAttribute("src");
}
// 读取“最近画面”前先校验 /observe 返回的 job_id 与当前正在展示的 execution 一致，
// 防止轮询竞态把旧任务的画面显示成本次请求的进度。
async function refreshFrame(expectedJobId) {
  if (!expectedJobId) return;
  try {
    const o = await getJSON("/api/observe");
    if (!o || (o.job_id || null) !== expectedJobId) return;
    const ip = toArtifact(o.image_path);
    if (ip) $("frame").src = ip + (ip.indexOf("?") >= 0 ? "&" : "?") + "t=" + Date.now();
  } catch (e) { /* ignore */ }
}
async function pollExecution() {
  // (A) 本次请求已到达终态：用持久的本次请求结果持续判定，不因 currentJob 被清空而回落到
  //     上一轮 baseline。该状态一直保持到用户提交下一请求。
  if (requestOutcome && requestOutcome.terminal) {
    if (requestOutcome.executionJobId) {
      // 本次请求确实触发了机器人执行：只查询该 execution 的 job_id，持续显示其真实
      // 进度/结果，绝不把不相关的 job 当成当前请求的进度。
      let s = null;
      try {
        s = await getJSON("/api/execution?job_id=" +
                          encodeURIComponent(requestOutcome.executionJobId));
      } catch (e) { s = { error: e.message }; }
      lastExec = s;
      renderExec(s);
      await refreshFrame(requestOutcome.executionJobId);
    } else {
      // 终态且 result.execution 为 null：本次请求没有触发机器人执行。持续显示该结论，
      // 不渲染 baseline 旧任务的视频/画面，也不用上一轮服务 /status 冒充本请求。
      show("exec", '<span class="kv">本次请求没有触发机器人执行</span>');
      clearMedia();
    }
    return;
  }

  // (B) 尚无本次请求（页面首次打开，仍可显示最近一轮真实 episode）或本次请求运行中。
  let s = null;
  try { s = await getJSON("/api/execution"); }
  catch (e) { s = { error: e.message }; }
  lastExec = s;
  const running = currentJob !== null;
  const noExec = isNoExecution(s);
  const sameAsBaseline = !!(s && !s.error && (s.job_id || null) === execBaseline);
  // 当前新请求运行中，但服务 job_id 仍停留在基线（或尚无执行）时：说明 Hermes 还在理解并
  // 选择任务，绝不能把上一轮任务显示成当前请求的进度。
  const understanding = running && (noExec || sameAsBaseline);
  if (understanding) {
    show("exec", '<span class="kv">正在理解请求并选择任务…</span>');
    // 理解/选择期间不刷新“最近画面”，避免把旧任务的画面当作当前进度。
    return;
  }
  if (noExec) {
    renderExec(null);
    return;
  }
  renderExec(s);
  // 展示的 execution 就是服务当前 job：画面必须与之一致。
  await refreshFrame(s.job_id || null);
}

// ---- agent job ----
function renderJob(job) {
  const st = job.status;
  show("result",
    "请求: " + esc(job.request) + "<br>" +
    "状态: <b>" + esc(st) + "</b>　" +
    "run_ok(Hermes exit 0): " + fmtBool(job.result && job.result.run_ok) + "　" +
    "chain_ok(执行链完成): " + fmtBool(job.result && job.result.chain_ok) + "<br>" +
    "任务成功(task_success): " + fmtBool(job.result && job.result.task_success) + "<br>" +
    "run_dir: <span class='hint'>" + esc(job.run_dir) + "</span><br>" +
    "exit_code: " + esc(job.exit_code != null ? job.exit_code : "-"));
  loadHermes(job);
}
function fmtBool(v) {
  if (v === true) return '<span class="ok-text">true</span>';
  if (v === false) return '<span class="bad-text">false</span>';
  return '<span class="kv">未知</span>';
}
async function loadHermes(job) {
  const url = toArtifact((job.run_dir || "") + "/hermes.log");
  if (!url) return;
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) {
      // 文件尚未生成（Hermes 还在理解/执行）或读取失败：显示说明，不显示原始 404 JSON。
      $("hermes").textContent = (r.status === 404)
        ? "正在理解或执行，说明将在完成后显示"
        : ("hermes.log 读取失败：HTTP " + r.status);
      return;
    }
    const text = await r.text();
    $("hermes").textContent = text.length > 6000 ? text.slice(-6000) : text;
  } catch (e) {
    $("hermes").textContent = "正在理解或执行，说明将在完成后显示";
  }
}

async function pollJob() {
  if (!currentJob) return;
  try {
    const job = await getJSON("/api/agent/" + encodeURIComponent(currentJob));
    renderJob(job);
    if (job.status === "completed" || job.status === "error") {
      // 落定本次请求的持久结果状态：先于清空 currentJob 记录，之后即使 currentJob=null
      // 也能据此继续判定，不会失去与本次请求的关联。
      const execution = (job.result && typeof job.result === "object")
        ? job.result.execution : null;
      const execJobId = (execution && typeof execution === "object"
                         && execution.job_id) ? execution.job_id : null;
      requestOutcome = {
        requestId: job.request_id,
        terminal: true,
        executionJobId: execJobId,
      };
      currentJob = null;
      $("run").disabled = false;
      if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
      pollTimer = setInterval(() => { pollHealth(); pollExecution(); }, 2000);
      // 立即按持久终态刷新一次，避免等到下一个轮询周期才切换显示。
      pollExecution();
    }
  } catch (e) { /* ignore */ }
}

async function startRun() {
  const request = $("request").value.trim();
  if (!request) { $("runmsg").textContent = "请先输入任务描述"; return; }
  $("run").disabled = true;
  $("runmsg").textContent = "已提交…";
  try {
    const r = await fetch("/api/agent", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        request: request,
        seed: parseInt($("seed").value || "0", 10),
        init_state_index: parseInt($("idx").value || "0", 10),
      }),
    });
    const data = await r.json();
    if (r.status === 409) {
      $("runmsg").textContent = "服务忙：" + (data.error || "已有任务在运行");
      $("run").disabled = false;
      return;
    }
    if (!r.ok) {
      $("runmsg").textContent = "提交失败：" + (data.error || r.status);
      $("run").disabled = false;
      return;
    }
    currentJob = data.request_id;
    // 新请求开始：清空上一次的持久结果，重新建立本次请求的运行态关联。
    requestOutcome = { requestId: data.request_id, terminal: false, executionJobId: null };
    // 记录当前已显示的 execution job_id 作为基线，并清空上一轮的视频/画面，
    // 使“当前请求”与“上一轮任务”的进度显示彼此隔离。
    execBaseline = (lastExec && !lastExec.error) ? (lastExec.job_id || null) : null;
    clearProgressDisplay();
    $("runmsg").textContent = "请求已受理：" + currentJob;
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    pollTimer = setInterval(() => { pollJob(); pollHealth(); pollExecution(); }, 2000);
    pollJob();
  } catch (e) {
    $("runmsg").textContent = "提交异常：" + e.message;
    $("run").disabled = false;
  }
}

$("run").addEventListener("click", startRun);
pollHealth(); loadTasks(); pollExecution();
setInterval(() => { pollHealth(); pollExecution(); }, 2000);
</script>
</body>
</html>
"""


def render_page() -> str:
    return PAGE.replace("__RUNS_ROOT__", RUNS_DIR + "/")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="LIBERO SmolVLA 本地网页前端")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    os.makedirs(RUNS_DIR, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("LIBERO 前端已启动: http://localhost:%d" % args.port, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

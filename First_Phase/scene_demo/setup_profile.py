#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为「持久场景」演示准备一个**隔离的** Hermes profile。

只使用 Python 标准库。产出目录 /home/yhwang/fyp/scene_demo/hermes_home：

  .env          现有 /home/yhwang/.hermes/.env 的副本，权限 0600，内容从不回显，原文件不动
  config.yaml   _config_version: 12 + qwen3-vl-plus + alibaba-cn + 支持视觉，
                agent.max_turns=12，仅注册 scene_tools 一个 MCP server
  SOUL.md       英文角色说明：场景内规划器（在场景内按请求/真实画面/公开规则选物体与顺序）

只启用 scene_tools 与视觉（supports_vision）。绝不启用其它 MCP / 全局工具。

幂等：重复运行用相同内容覆盖 config.yaml / SOUL.md；
保留新 home 里已有的记忆与状态 DB，绝不复制旧会话历史。

成功时打印 PROFILE_READY。
"""

from __future__ import annotations

import os
import sys

SRC_ENV = "/home/yhwang/.hermes/.env"
HERMES_HOME = "/home/yhwang/fyp/scene_demo/hermes_home"
DST_ENV = os.path.join(HERMES_HOME, ".env")
CONFIG_PATH = os.path.join(HERMES_HOME, "config.yaml")
SOUL_PATH = os.path.join(HERMES_HOME, "SOUL.md")

MCP_SERVER = "/mnt/d/FYP/First_Phase/scene_demo/mcp_server.py"
MCP_CWD = "/mnt/d/FYP/First_Phase/scene_demo"
# 固定的系统 PATH：Hermes 会把自带 3.14 插到 stdio 子进程 PATH 最前面，
# 必须钉死，才能让 /usr/bin/python3 找到装在系统 Python 3.12 user site 的 mcp/anyio。
MCP_PATH = "/usr/bin:/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/yhwang/.local/bin"

CONFIG_YAML = f"""\
# 仅启用 scene_tools（MCP）与视觉（model.supports_vision: true）；
# 不注册任何其它 MCP server，也不启用全局 robot_tools / vla_tools / libero_tools。
_config_version: 12
model:
  default: qwen3-vl-plus
  provider: alibaba-cn
  supports_vision: true
agent:
  max_turns: 12
mcp_servers:
  scene_tools:
    command: /usr/bin/python3
    args:
      - -u
      - {MCP_SERVER}
    enabled: true
    cwd: {MCP_CWD}
    env:
      PATH: {MCP_PATH}
"""

# Shared explanatory guidance.  IDENTICAL text lives in run_agent.py and is
# embedded in BOTH phase prompts there; here it is embedded in SOUL_MD so the
# isolated profile teaches the same lazy meta-tool routing (installed Hermes may
# expose only tool_search / tool_describe / tool_call, not direct callables).
MCP_ROUTING_GUIDANCE = """\
[tool routing]
Installed Hermes may expose ONLY the meta-tools tool_search / tool_describe / tool_call; the scene_tools MCP methods may NOT be directly callable functions.
When an MCP method is not a direct callable, route to it in this order:
1. tool_search for "scene_tools" to find the available methods;
2. tool_describe to fetch the exact schema of the method you need;
3. tool_call(calls=[{"name": "mcp__scene_tools__<method>", "arguments": {...}}]) to actually invoke it.
Only call functions currently exposed as callable; never repeatedly attempt a discovered MCP name as a directly callable function.
"""

IMAGE_EFFICIENCY_GUIDANCE = """\
[image efficiency]
The native agentview image already accompanies this message. When it is sufficient, do NOT repeat vision_analyze.
Only when the attached imagery is insufficient, use observe_scene(extra_views=true) and actually inspect the extra images you need.
Checking the latest session_version and submitting a plan are still mandatory.
"""

SOUL_MD = """\
# SOUL

You are the in-scene planner for a persistent robot demonstration. You plan
inside ONE existing scene; you never create, reset or re-arrange that scene, and
you never touch the robot yourself. The host owns waiting, cancellation and any
independent evaluation.

## Your job

Turn the user's natural-language request into a plan for the CURRENT scene:

1. Read the scene's public data first (`get_scene_session`), and look at the
   ACTUAL images (the attached agentview image, plus `observe_scene` extra views
   through the native vision tool). Base object choice and ordering on the real
   pixels plus the public `storage_policy` and the public `capabilities` -- not
   on memory, not on assumptions.
2. Choose capability ids and their ORDER from the public capability list, then
   `submit_scene_plan(...)` once. The order of `capability_ids` is the execution
   order.
3. State plainly that the plan was submitted, then finish. Do not poll
   `get_scene_plan` in a loop, do not wait for the robot, do not sleep in
   45-second rounds. The host does the waiting.

## Decision rules

Fix INTENT before CAPABILITY. Always decide what the user wants FIRST, and give
a SHORT rationale that quotes the user's action word or the explicit destination
you relied on; only after intent is clear may you choose capability ids and
their order.

- Storage intents -- Chinese 整理 / 收好 / 归位 / 收纳 (and English "tidy the
  table", "put things away"): follow the PUBLIC storage rules
  (`storage_policy`) -- put each object where the public policy says it belongs.
  Do not invent destinations that are not public.
- A bare Chinese 清理 / 清理一下 (just "clean it", with NO explicit handling
  method and NO explicit destination) is AMBIGUOUS between discard, surface
  cleaning and storage: use `decision="clarify"` with an empty capability list
  and say what you need to know. NEVER use `storage_policy` to silently resolve
  this ambiguity into a storage plan.
- Discard intents -- Chinese 丢弃 / 扔掉 / 扔进垃圾桶, and throwing an object
  away generally: UNSUPPORTED whenever the current scene has no trash bin (or no
  matching capability); use `decision="unsupported"` with an empty capability
  list.
- Surface-cleaning intents -- Chinese 擦拭 / 清洗: this is cleaning, NOT storage.
  Use `decision="unsupported"` with an empty capability list when the scene has
  no cleaning capability.
- A request that DOES name its destination, e.g. 清理桌面，把碗放到盘子, may
  execute the capability it names.
- Ambiguous cleaning requests that name neither a method nor a destination
  (e.g. "clean the wine bottle") need clarification: use `decision="clarify"`
  with an empty capability list and say what you need.
- Do NOT automatically switch the stove on. Turning the stove on is only done
  when the user explicitly asks for it.
- If the scene cannot support the request at all, use `decision="unsupported"`
  with an empty capability list.

## Repair scope

修复只可重排、重试、恢复原声明目标，不得新增物体或目的地；物体身份不确定时保持blocked并说明限制；改变目标必须由用户发起新请求。

## Honesty

- Do not read, request or reveal any independent fixtures, answer keys or
  hidden evaluation data. You only ever see PUBLIC scene data.
- `candidate` capabilities MAY be used, but never claim they are reliably
  verified; an `evidence` tag of "candidate" is not a reliability guarantee.
- Do not change the scene and do not reset objects. If something looks wrong,
  report it instead of fixing it yourself.
- Report outcomes exactly as the tools return them. Never present an unverified
  or unknown result as success.

""" + MCP_ROUTING_GUIDANCE + "\n" + IMAGE_EFFICIENCY_GUIDANCE


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_home() -> None:
    os.makedirs(HERMES_HOME, mode=0o700, exist_ok=True)


def copy_env() -> None:
    """把现有 .env 复制到新 home，权限 0600，全程不回显内容，不改原文件。"""
    if not os.path.isfile(SRC_ENV):
        print(f"ERROR: 源凭据文件不存在: {SRC_ENV}", file=sys.stderr)
        raise SystemExit(1)
    src_fd = os.open(SRC_ENV, os.O_RDONLY)
    try:
        dst_fd = os.open(DST_ENV, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            while True:
                chunk = os.read(src_fd, 65536)
                if not chunk:
                    break
                os.write(dst_fd, chunk)
        finally:
            os.close(dst_fd)
    finally:
        os.close(src_fd)
    os.chmod(DST_ENV, 0o600)
    log(f"已复制凭据文件 -> {DST_ENV} (mode 0600, 内容未回显)")


def write_text(path: str, content: str, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def main() -> int:
    ensure_home()
    copy_env()
    write_text(CONFIG_PATH, CONFIG_YAML)
    write_text(SOUL_PATH, SOUL_MD)
    log(f"已写入配置: {CONFIG_PATH} (_config_version: 12, max_turns=12)")
    log(f"已写入角色说明: {SOUL_PATH}")
    log("仅启用 scene_tools 与视觉（supports_vision=true）；未启用其它 MCP / 全局工具。")
    log(f"scene_tools 使用固定解释器: /usr/bin/python3 -u {MCP_SERVER}")
    log("已保留新 home 中既有的记忆与状态 DB，未复制旧会话历史。")
    print("PROFILE_READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

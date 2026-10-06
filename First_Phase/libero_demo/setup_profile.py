#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 LIBERO SmolVLA 演示准备一个**隔离的** Hermes profile。

只使用 Python 标准库。产出目录 /home/yhwang/fyp/libero_demo/hermes_home：

  .env          现有 /home/yhwang/.hermes/.env 的副本，权限 0600，内容从不回显
  config.yaml   _config_version: 12 + qwen3-vl-plus + alibaba-cn + 支持视觉，
                agent.max_turns=24，仅注册 libero_tools 一个 MCP server
  SOUL.md       简短角色说明，用于标准 LIBERO 任务匹配（含正常等待约定）

幂等：重复运行会用相同内容覆盖 config.yaml / SOUL.md；
保留新 home 里已有的记忆与状态 DB（state.db），绝不递归复制旧会话。

成功时打印 PROFILE_READY。
"""

from __future__ import annotations

import os
import sys

SRC_ENV = "/home/yhwang/.hermes/.env"
HERMES_HOME = "/home/yhwang/fyp/libero_demo/hermes_home"
DST_ENV = os.path.join(HERMES_HOME, ".env")
CONFIG_PATH = os.path.join(HERMES_HOME, "config.yaml")
SOUL_PATH = os.path.join(HERMES_HOME, "SOUL.md")

MCP_SERVER = "/mnt/d/FYP/First_Phase/libero_demo/mcp_server.py"
MCP_CWD = "/mnt/d/FYP/First_Phase/libero_demo"
MCP_PATH = "/usr/bin:/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin:/home/yhwang/.local/bin"

CONFIG_YAML = f"""\
_config_version: 12
model:
  default: qwen3-vl-plus
  provider: alibaba-cn
  supports_vision: true
agent:
  max_turns: 24
mcp_servers:
  libero_tools:
    command: /usr/bin/python3
    args:
      - -u
      - {MCP_SERVER}
    enabled: true
    cwd: {MCP_CWD}
    env:
      PATH: {MCP_PATH}
"""

SOUL_MD = """\
# SOUL

你是接入 LIBERO SmolVLA 执行服务的任务理解与编排助手。

## 角色
把用户的自然语言目标，对照 `list_libero_tasks` 返回的真实任务目录，
在全部 40 个受支持的 LIBERO 标准 benchmark 任务中选出**最匹配**的一个。

## 原则
- 先查目录，再选任务；目录里没有的，一律报告「不支持」。
- 只做「任务选择 + 执行编排」，不臆造任务，不改写标准 instruction。
- 用户目标信息不足、无法唯一确定任务时，先要求澄清，不擅自执行。
- 执行后如实报告服务返回的 `success` 与真实 suite / task_id / instruction。
- 绝不把失败或未知当作成功。

## 等待是正常的（务必遵守）
- 任务的 `max_steps` 是**仿真动作步数**（不是秒）：一个正常任务常需 3-6 分钟才能跑完。
- 持续用同一个 `job_id` 调用 `wait_for_libero_execution(job_id, timeout_s=45)`，
  直到状态为 `completed` 或 `error`。
- 单次返回 `timed_out=true` 只表示这一轮等待窗口到期，**不是**任务失败；必须继续等待。
- 总超时由 host 统一约束，不要因为耗时较长就提前结束或宣告失败，也不要询问用户是否继续等待。
"""


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
    log(f"已写入配置: {CONFIG_PATH} (_config_version: 12)")
    log(f"已写入角色说明: {SOUL_PATH}")
    log("仅注册 libero_tools；未启用全局 robot_tools / vla_tools。")
    log("已保留新 home 中既有的记忆与状态 DB，未复制旧会话。")
    print("PROFILE_READY", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

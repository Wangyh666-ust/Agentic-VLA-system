<#
.SYNOPSIS
  一键启动 / 停止「持久场景 v2」演示（隔离 Hermes profile + 场景服务 + 本地网页）。

.DESCRIPTION
  - 默认启动：先准备隔离的 Hermes profile（setup_profile.py），再在 WSL 后台启动
    已有的 scene_demo/run_service.sh（加载 SmolVLA 模型）与 app.py（本地网页）。
  - **复用的旧共享环境依赖（必须知悉）**：本演示**不自带** Python 环境，而是复用
    `/home/yhwang/fyp/libero_demo/venv` 这个旧 demo 的 venv（其中的 torch / lerobot /
    LIBERO 依赖），以及旧的 Hermes 配置与凭据 `/home/yhwang/.hermes/.env`。因此它
    依赖旧环境仍然存在且可用；本脚本**不下载、不训练、不重建环境**。
  - 在 WSL 内使用 Python 的 subprocess.Popen(start_new_session=True, stdout=文件,
    stderr=STDOUT) 拉起进程；不使用字符串拼接的 shell 命令，不弹任何可见辅助窗口。
  - 服务端口固定 8767、网页端口固定 8081；启动前先做端口冲突检查与 /health 工作流
    （workflow=persistent_scene_v2）和模型 revision 校验，**仅完全吻合**才复用。
  - **启动新 GPU 服务前**先检查当前可用显存；显存不足会抛出可执行的提示，交由主
    agent 处理（**绝不**自动杀旧服务）。网页可在模型加载完成前显示「加载中」。
  - 启动时不下载模型、不训练。

.PARAMETER Stop
  只停止 PID 文件中记录、且 /proc/<pid>/cmdline 含本项目（scene_demo）绝对路径的进程
  （SIGTERM）。不会 wsl --shutdown，也不会动旧 libero_demo 服务或其它进程。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start_demo.ps1
  powershell -ExecutionPolicy Bypass -File .\start_demo.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [int]$ServicePort = 8767,
    [int]$AppPort = 8081,
    [string]$Distro = 'Ubuntu',
    [int]$MinFreeGpuMiB = 6000
)

$ErrorActionPreference = 'Stop'

# 与 WSL 之间以 UTF-8 交换文本，避免中文输出乱码。
try {
    $OutputEncoding = New-Object System.Text.UTF8Encoding($false)
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
} catch { }

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Write-Error '找不到 wsl.exe，请在安装 WSL2 的 Windows 上运行本脚本。'
    exit 1
}

$mode = if ($Stop) { 'stop' } else { 'start' }

# 固定的 WSL 侧启动器（Python 标准库实现）。以 base64 编码后经 python3 -c 执行：
# 命令行上只有 ASCII，既不受 PowerShell 5.1 默认 ASCII 管道编码影响，也不拼接任何
# 用户输入，避免注入。
$launcher = @'
import json, os, signal, socket, subprocess, sys, time, urllib.request

PROJECT_WSL = "/home/yhwang/fyp/scene_demo"
PROJECT_SRC = "/mnt/d/FYP/First_Phase/scene_demo"
VENV_PY = "/home/yhwang/fyp/libero_demo/venv/bin/python"
SETUP_PY = PROJECT_SRC + "/setup_profile.py"
SERVICE_SH = PROJECT_SRC + "/run_service.sh"
APP_PY = PROJECT_SRC + "/app.py"
SERVICE_PID_FILE = PROJECT_WSL + "/service.pid"
APP_PID_FILE = PROJECT_WSL + "/app.pid"
SERVICE_LOG = PROJECT_WSL + "/service.log"
APP_LOG = PROJECT_WSL + "/app.log"
SETUP_LOG = PROJECT_WSL + "/setup_profile.log"
PY3 = "/usr/bin/python3"
EXPECTED_WORKFLOW = "persistent_scene_v2"
EXPECTED_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
EXPECTED_SERVICE_PORT = 8767
EXPECTED_APP_PORT = 8081
MARKERS = (PROJECT_WSL, PROJECT_SRC)
PROXY_VARS = ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")


def log(msg):
    print(msg, flush=True)


def opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def http_json(url, timeout=5):
    req = urllib.request.Request(url, method="GET")
    with opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def port_open(port):
    sock = socket.socket()
    sock.settimeout(0.6)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS}
    env["PATH"] = ("/usr/bin:/bin:/usr/local/bin:/usr/local/sbin:"
                   "/usr/sbin:/sbin:/home/yhwang/.local/bin")
    return env


def spawn(cmd, logfile):
    handle = open(logfile, "ab")
    try:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_WSL, env=clean_env(),
            stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True)
    finally:
        handle.close()
    return proc


def save_pid(path, pid):
    with open(path, "w") as fh:
        fh.write(str(pid))


def read_pid(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def gpu_free_mib():
    """返回 (可用显存 MiB 的最大值, 错误说明)。无法探测时返回 (None, reason)。"""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
    except Exception as exc:
        return None, str(exc)
    text = proc.stdout.decode("utf-8", "replace")
    if proc.returncode != 0:
        return None, text.strip() or ("nvidia-smi rc=%d" % proc.returncode)
    values = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(float(line.split()[0])))
        except ValueError:
            continue
    if not values:
        return None, "nvidia-smi 未返回可解析的显存数值"
    return max(values), None


def run_setup():
    handle = open(SETUP_LOG, "ab")
    try:
        rc = subprocess.call([PY3, SETUP_PY], cwd=PROJECT_WSL, env=clean_env(),
                             stdout=handle, stderr=subprocess.STDOUT)
    finally:
        handle.close()
    if rc != 0:
        log("ERROR: setup_profile.py 失败 rc=%s（详见 %s）" % (rc, SETUP_LOG))
        return False
    log("隔离 Hermes profile 已就绪（日志 %s）" % SETUP_LOG)
    return True


def start(service_port, app_port, min_free_gpu):
    # 0) 固定端口
    if service_port != EXPECTED_SERVICE_PORT:
        log("ERROR: 本演示仅支持 ServicePort=%d，收到 %d。" % (EXPECTED_SERVICE_PORT, service_port))
        return 1
    if app_port != EXPECTED_APP_PORT:
        log("ERROR: 本演示仅支持 AppPort=%d，收到 %d。" % (EXPECTED_APP_PORT, app_port))
        return 1

    # 1) 复用旧共享环境：独立 venv 必须存在，缺失即报错，绝不回退 system python。
    log("提示：本演示复用旧 demo 的共享环境 %s（torch/lerobot/LIBERO）与旧凭据，"
        "不下载、不训练、不重建。" % VENV_PY)
    if not os.path.isfile(VENV_PY):
        log("ERROR: 找不到旧共享环境解释器: %s" % VENV_PY)
        log("       请先按旧 libero_demo 的 README「环境准备」准备该 venv；不回退 system python。")
        return 1
    # 运行时目录（幂等）：setup 日志与各子进程 cwd 均位于此目录下。
    os.makedirs(PROJECT_WSL, exist_ok=True)

    # 2) 先准备隔离 profile（幂等）
    if not os.path.isfile(SETUP_PY):
        log("ERROR: 找不到 %s" % SETUP_PY)
        return 1
    if not run_setup():
        return 1

    # 3) 服务（8767）
    if port_open(service_port):
        try:
            health = http_json("http://127.0.0.1:%d/health" % service_port, timeout=8)
        except Exception as exc:
            log("端口 %d 已被占用且 health 探测失败 (%s)：端口冲突，退出，不杀进程。"
                % (service_port, exc))
            return 2
        mismatches = []
        if health.get("workflow") != EXPECTED_WORKFLOW:
            mismatches.append("workflow=%r" % health.get("workflow"))
        if health.get("model_revision") != EXPECTED_REVISION:
            mismatches.append("model_revision=%r" % health.get("model_revision"))
        if mismatches:
            log("端口 %d 被非本项目服务占用 (%s)：端口冲突，退出，不杀进程。"
                % (service_port, ", ".join(mismatches)))
            return 2
        log("复用已在运行的持久场景服务 (端口 %d, workflow=%s)"
            % (service_port, EXPECTED_WORKFLOW))
    else:
        if not os.path.isfile(SERVICE_SH):
            log("ERROR: 找不到 %s" % SERVICE_SH)
            return 1
        # 启动新 GPU 服务前先看可用显存；不足则给出可执行提示，不杀旧服务。
        free_mib, reason = gpu_free_mib()
        if free_mib is None:
            log("WARNING: 无法探测 GPU 可用显存 (%s)；仍尝试启动服务，若失败请检查显存/驱动。"
                % reason)
        elif free_mib < min_free_gpu:
            log("ERROR: GPU 可用显存不足：当前约 %d MiB < 需要的 %d MiB。"
                % (free_mib, min_free_gpu))
            log("       主 agent 请先释放显存（例如结束不再使用的训练/推理进程），"
                "本脚本不会自动 kill 旧服务，也不会 wsl --shutdown。")
            log("       处理后请重跑 start_demo.ps1。")
            return 3
        else:
            log("GPU 可用显存约 %d MiB，满足启动要求 (>= %d MiB)。"
                % (free_mib, min_free_gpu))
        proc = spawn(["/bin/bash", SERVICE_SH], SERVICE_LOG)
        save_pid(SERVICE_PID_FILE, proc.pid)
        log("已启动场景服务 pid=%d（模型在后台加载），日志 %s"
            % (proc.pid, SERVICE_LOG))

    # 4) 网页前端（8081）
    if port_open(app_port):
        try:
            health = http_json("http://127.0.0.1:%d/api/health" % app_port, timeout=8)
        except Exception as exc:
            log("端口 %d 已被占用且不是本项目前端 (%s)：端口冲突，退出，不杀进程。"
                % (app_port, exc))
            return 2
        if health.get("frontend") != "scene_demo_frontend":
            log("端口 %d 被非本项目进程占用：端口冲突，退出。" % app_port)
            return 2
        log("复用已在运行的前端 (端口 %d)" % app_port)
    else:
        if not os.path.isfile(APP_PY):
            log("ERROR: 找不到 %s" % APP_PY)
            return 1
        # 前端固定用旧共享环境解释器（已在开头校验存在），不回退 system python。
        proc = spawn([VENV_PY, APP_PY, "--port", str(app_port)], APP_LOG)
        save_pid(APP_PID_FILE, proc.pid)
        log("已启动网页前端 pid=%d，日志 %s" % (proc.pid, APP_LOG))

    # 5) 等前端起来（不阻塞等待模型加载完成；网页会显示加载/就绪状态）
    for _ in range(60):
        if port_open(app_port):
            break
        time.sleep(0.5)

    log("打开浏览器访问: http://localhost:%d" % app_port)
    log("（场景模型加载完成后，网页顶部会显示「服务状态: 就绪」。）")
    return 0


def proc_state(pid):
    """返回 /proc/<pid>/stat 中的进程状态字符。

    仅当 /proc/<pid>/stat 不存在（FileNotFoundError）时返回 None，表示进程不存在；
    其余读取失败（如权限错误）或格式异常一律返回 "unknown"，表示状态无法确认，
    失败即封闭（fail closed），绝不被误判为进程已退出。
    """
    try:
        with open("/proc/%d/stat" % pid, "rb") as fh:
            data = fh.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return None
    except Exception:
        return "unknown"
    # stat 格式: pid (comm) state ...；comm 可能含空格/括号，取最后一个 ')' 之后。
    rparen = data.rfind(")")
    if rparen < 0:
        return "unknown"
    fields = data[rparen + 1:].split()
    if not fields:
        return "unknown"
    state = fields[0]
    if len(state) != 1:
        return "unknown"
    return state


def wait_exited(pid, timeout=30.0, interval=0.2):
    """轮询等待进程退出：/proc 缺失或 stat 状态为 Z 均视为已退出。

    返回 True 表示确认退出；False 表示 timeout 内仍存活（绝不 SIGKILL）。
    """
    deadline = time.time() + timeout
    while True:
        state = proc_state(pid)
        if state is None or state == "Z":
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        time.sleep(min(interval, remaining))


def wait_ports_closed(ports, timeout=5.0, interval=0.2):
    """轮询等待所有给定端口关闭；返回 True 表示全部已关闭。"""
    deadline = time.time() + timeout
    while True:
        if not any(port_open(p) for p in ports):
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        time.sleep(min(interval, remaining))


def stop():
    rc = 0
    for pidfile, label in ((APP_PID_FILE, "前端"), (SERVICE_PID_FILE, "服务")):
        pid = read_pid(pidfile)
        if pid is None:
            log("%s: 无 PID 文件，跳过。" % label)
            continue
        cmdline = ""
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as fh:
                cmdline = fh.read().decode("utf-8", "replace").replace("\x00", " ")
        except Exception as exc:
            log("%s: 读取 /proc/%d/cmdline 失败 (%s)，跳过（不杀未知进程）。"
                % (label, pid, exc))
            rc = 3
            continue
        if not any(marker in cmdline for marker in MARKERS):
            log("%s: pid=%d 的 cmdline 不含本项目 scene_demo 路径，跳过：%s"
                % (label, pid, cmdline.strip()))
            rc = 3
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log("%s: 已发送 SIGTERM 到 pid=%d，等待其退出……" % (label, pid))
        except Exception as exc:
            log("%s: SIGTERM pid=%d 失败: %s" % (label, pid, exc))
            rc = 3
            continue
        if not wait_exited(pid, timeout=30.0, interval=0.2):
            log("%s: pid=%d 在发送 SIGTERM 后 30 秒内仍未退出（state=%s）；"
                "保留 PID 文件 %s，绝不 SIGKILL。"
                % (label, pid, proc_state(pid), pidfile))
            rc = 4
            continue
        log("%s: pid=%d 已确认退出。" % (label, pid))
        try:
            os.remove(pidfile)
        except OSError:
            pass

    ports = (EXPECTED_SERVICE_PORT, EXPECTED_APP_PORT)
    if any(port_open(p) for p in ports):
        if wait_ports_closed(ports, timeout=5.0, interval=0.2):
            log("固定端口 %s 已全部释放。" % ", ".join(str(p) for p in ports))
        else:
            still_open = ", ".join(str(p) for p in ports if port_open(p))
            log("固定端口 %s 在等待 5 秒后仍未释放（端口延迟释放）。" % still_open)
            rc = 4
    return rc


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv else "start"
    service_port = int(argv[1]) if len(argv) > 1 else 8767
    app_port = int(argv[2]) if len(argv) > 2 else 8081
    min_free_gpu = int(argv[3]) if len(argv) > 3 else 6000
    if mode == "stop":
        return stop()
    if mode == "start":
        return start(service_port, app_port, min_free_gpu)
    log("usage: start|stop [service_port] [app_port] [min_free_gpu_mib]")
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
'@

$wslArgs = @()
if ($Distro) { $wslArgs += @('-d', $Distro) }
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($launcher))
$pyCode = "import base64,sys;exec(compile(base64.b64decode('$b64').decode('utf-8'),'scene_demo_launcher','exec'))"
$wslArgs += @('python3', '-c', $pyCode, $mode, $ServicePort, $AppPort, $MinFreeGpuMiB)

try {
    & wsl.exe @wslArgs
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error $_
    exit 1
}

exit $exitCode

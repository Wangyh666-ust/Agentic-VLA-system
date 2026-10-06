<#
.SYNOPSIS
  一键启动 / 停止 LIBERO SmolVLA 自然语言演示（隔离 Hermes + VLA 服务 + 本地网页）。

.DESCRIPTION
  - 默认启动：先准备隔离的 Hermes profile（setup_profile.py），再在 WSL 后台启动
    已有的 run_service.sh（加载 SmolVLA 模型）与 app.py（本地网页）。
  - 在 WSL 内使用 python 的 subprocess.Popen(start_new_session=True, stdout=文件,
    stderr=STDOUT) 拉起进程；不使用字符串拼接的 shell 命令。
  - 仅在端口空闲时启动；若端口已被占用，先做 health 检查判断是否为本项目服务，
    是本项目则复用，否则报告端口冲突并退出（绝不 kill 其它进程）。
  - 启动时不下载模型、不训练。

.PARAMETER Stop
  只停止 PID 文件中记录、且 /proc/<pid>/cmdline 含本项目绝对路径的进程（SIGTERM）。
  不会 wsl --shutdown，也不会动其它服务。

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\start_demo.ps1
  powershell -ExecutionPolicy Bypass -File .\start_demo.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [int]$ServicePort = 8766,
    [int]$AppPort = 8080,
    [string]$Distro = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'

# 与 WSL 之间以 UTF-8 交换文本，避免中文输出乱码
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

PROJECT_WSL = "/home/yhwang/fyp/libero_demo"
PROJECT_SRC = "/mnt/d/FYP/First_Phase/libero_demo"
VENV_PY = PROJECT_WSL + "/venv/bin/python"
SETUP_PY = PROJECT_SRC + "/setup_profile.py"
SERVICE_SH = PROJECT_SRC + "/run_service.sh"
APP_PY = PROJECT_SRC + "/app.py"
SERVICE_PID_FILE = PROJECT_WSL + "/service.pid"
APP_PID_FILE = PROJECT_WSL + "/app.pid"
SERVICE_LOG = PROJECT_WSL + "/service.log"
APP_LOG = PROJECT_WSL + "/app.log"
PY3 = "/usr/bin/python3"
MODEL_DIR = PROJECT_WSL + "/models/smolvla_libero"
EXPECTED_BACKEND = "smolvla"
EXPECTED_ROBOT = "Franka Panda"
EXPECTED_REVISION = "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
EXPECTED_SERVICE_PORT = 8766
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


def start(service_port, app_port):
    # 0) 本演示只支持固定的服务端口 8766（run_service.sh 与前端均按此端口编写）。
    if service_port != EXPECTED_SERVICE_PORT:
        log("ERROR: 本演示仅支持 ServicePort=%d，收到 %d；请在 8766 上运行。"
            % (EXPECTED_SERVICE_PORT, service_port))
        return 1

    # 启动前检查：独立新 venv 与已下载模型必须存在；缺失即报错，绝不回退 system python。
    if not os.path.isfile(VENV_PY):
        log("ERROR: 找不到独立 venv 解释器: %s" % VENV_PY)
        log("       请先按 README「环境准备」用 setup_env.py 准备环境；不回退 system python。")
        return 1
    if not os.path.isdir(MODEL_DIR):
        log("ERROR: 找不到模型目录: %s" % MODEL_DIR)
        log("       请先按 README「环境准备」用 setup_env.py 下载模型。")
        return 1

    # 1) 先准备隔离 profile（幂等）
    if not os.path.isfile(SETUP_PY):
        log("ERROR: 找不到 %s" % SETUP_PY)
        return 1
    rc = subprocess.call([PY3, SETUP_PY], cwd=PROJECT_WSL, env=clean_env())
    if rc != 0:
        log("ERROR: setup_profile.py 失败 rc=%s" % rc)
        return 1

    # 2) 服务（8766）
    if port_open(service_port):
        try:
            health = http_json("http://127.0.0.1:%d/health" % service_port)
        except Exception as exc:
            log("端口 %d 已被占用且 health 探测失败 (%s)：端口冲突，退出，不杀进程。"
                % (service_port, exc))
            return 2
        mismatches = []
        if health.get("backend") != EXPECTED_BACKEND:
            mismatches.append("backend=%r" % health.get("backend"))
        if health.get("robot") != EXPECTED_ROBOT:
            mismatches.append("robot=%r" % health.get("robot"))
        if health.get("model_revision") != EXPECTED_REVISION:
            mismatches.append("model_revision=%r" % health.get("model_revision"))
        if health.get("model_path") != MODEL_DIR:
            mismatches.append("model_path=%r" % health.get("model_path"))
        if mismatches:
            log("端口 %d 被非本项目服务占用 (%s)：端口冲突，退出。"
                % (service_port, ", ".join(mismatches)))
            return 2
        log("复用已在运行的 LIBERO 服务 (端口 %d)" % service_port)
    else:
        if not os.path.isfile(SERVICE_SH):
            log("ERROR: 找不到 %s" % SERVICE_SH)
            return 1
        proc = spawn(["/bin/bash", SERVICE_SH], SERVICE_LOG)
        save_pid(SERVICE_PID_FILE, proc.pid)
        log("已启动 LIBERO 服务 pid=%d（模型在后台加载），日志 %s"
            % (proc.pid, SERVICE_LOG))

    # 3) 网页前端（8080）
    if port_open(app_port):
        try:
            health = http_json("http://127.0.0.1:%d/api/health" % app_port)
        except Exception as exc:
            log("端口 %d 已被占用且不是本项目前端 (%s)：端口冲突，退出，不杀进程。"
                % (app_port, exc))
            return 2
        if health.get("frontend") != "libero_demo_frontend":
            log("端口 %d 被非本项目进程占用：端口冲突，退出。" % app_port)
            return 2
        log("复用已在运行的前端 (端口 %d)" % app_port)
    else:
        if not os.path.isfile(APP_PY):
            log("ERROR: 找不到 %s" % APP_PY)
            return 1
        # 前端固定用独立 venv 解释器（已在 start() 开头校验存在），不回退 system python。
        proc = spawn([VENV_PY, APP_PY, "--port", str(app_port)], APP_LOG)
        save_pid(APP_PID_FILE, proc.pid)
        log("已启动网页前端 pid=%d，日志 %s" % (proc.pid, APP_LOG))

    # 4) 等前端起来（不阻塞等待模型加载完成；网页会显示就绪状态）
    for _ in range(60):
        if port_open(app_port):
            break
        time.sleep(0.5)

    log("打开浏览器访问: http://localhost:%d" % app_port)
    log("（服务模型加载完成后，网页顶部会显示“服务状态: 就绪”。）")
    return 0


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
            log("%s: pid=%d 的 cmdline 不含本项目路径，跳过：%s"
                % (label, pid, cmdline.strip()))
            rc = 3
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            log("%s: 已发送 SIGTERM 到 pid=%d" % (label, pid))
        except Exception as exc:
            log("%s: SIGTERM pid=%d 失败: %s" % (label, pid, exc))
            rc = 3
            continue
        try:
            os.remove(pidfile)
        except OSError:
            pass
    return rc


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv else "start"
    service_port = int(argv[1]) if len(argv) > 1 else 8766
    app_port = int(argv[2]) if len(argv) > 2 else 8080
    if mode == "stop":
        return stop()
    if mode == "start":
        return start(service_port, app_port)
    log("usage: start|stop [service_port] [app_port]")
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
'@

$wslArgs = @()
if ($Distro) { $wslArgs += @('-d', $Distro) }
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($launcher))
$pyCode = "import base64,sys;exec(compile(base64.b64decode('$b64').decode('utf-8'),'libero_demo_launcher','exec'))"
$wslArgs += @('python3', '-c', $pyCode, $mode, $ServicePort, $AppPort)

try {
    & wsl.exe @wslArgs
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error $_
    exit 1
}

exit $exitCode

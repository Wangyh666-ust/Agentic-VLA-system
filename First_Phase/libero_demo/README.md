# LIBERO SmolVLA 自然语言演示（本地网页 + CLI）

本目录提供「自然语言输入 → Hermes 选任务 → LIBERO SmolVLA 执行 → 结果回显」的
本地测试入口：一个中文网页（`app.py`）和一个命令行入口（`run_agent.py`）。

- Hermes 通过 MCP 工具（`libero_tools`）真实地选择并执行任务；
- VLA 执行由常驻服务在 `127.0.0.1:8766` 提供；
- 网页在 `http://127.0.0.1:8080`（**推荐访问地址**；启动脚本当前实际打印的是
  `http://localhost:8080`，两者指向同一本地服务，用 `127.0.0.1` 更省事、可避免本机
  主机名解析/代理带来的干扰），CLI 与网页共用同一条执行链路。

初版范围是从任务目录选择一个标准任务并完成单次仿真执行；不是同场景多子目标规划。项目入口和当前不足见根 README：[../../README.md](../../README.md)。

---

## 一、开发分工与运行时分工（两者不同，勿混淆）

| | 分工 |
|---|---|
| **开发期** | GPT 负责规划与方案设计，DeepSeek 负责按既定步骤写代码与执行 |
| **运行时** | Hermes 主模型 = `qwen3-vl-plus`（理解自然语言、读任务目录、选任务、编排调用）；执行器 = SmolVLA **官方 LIBERO 权重** + **Franka Panda** 机械臂（LIBERO benchmark 场景） |

运行时是「Hermes 决策 + SmolVLA 执行」的固定组合，与开发期由哪个模型写代码无关。

---

## 二、目录与路径约定

| 内容 | 路径 |
|---|---|
| 脚本代码（Windows Git 目录，WSL 可见） | `/mnt/d/FYP/First_Phase/libero_demo/` |
| Python 环境（venv） | `/home/yhwang/fyp/libero_demo/venv` |
| 运行数据（每次运行的 run_dir） | `/home/yhwang/fyp/libero_demo/runs` |
| 隔离的 Hermes profile | `/home/yhwang/fyp/libero_demo/hermes_home` |
| MCP server | `/mnt/d/FYP/First_Phase/libero_demo/mcp_server.py` |
| Hermes 可执行 | `/home/yhwang/.local/bin/hermes` |

隔离 profile 只注册 `libero_tools` 一个 MCP，**不改动**用户全局 Hermes 配置，
也不启用全局 `robot_tools` / `vla_tools`。

---

## 三、环境准备

```powershell
# [Windows PowerShell] 准备 LIBERO 运行环境（幂等；用旧 venv 的 python 作启动器）
wsl -d Ubuntu -- /home/yhwang/fyp/vla/venv/bin/python /mnt/d/FYP/First_Phase/libero_demo/setup_env.py
# 期望最后打印 LIBERO_SETUP_PASS 与 runtime_manifest.json 路径
```

`setup_env.py` 的行为约定：

- **依赖旧 venv** `/home/yhwang/fyp/vla/venv` 中**已安装的 `torch` / `lerobot`**，
  不在新环境里重新安装或升级它们；
- 只用标准库启动，再在**独立的新包目录** `/home/yhwang/fyp/libero_demo/venv`
  下创建 venv，通过 `shared_vla.pth` 复用旧 venv 的 `site-packages` 与 `lerobot` 源码；
- 新环境只补齐 LIBERO 运行时依赖（`hf-libero` 等），并拉取推理所需的
  SmolVLA 权重与 LIBERO 资源；**不下载训练数据**（训练数据集不在本流程范围内）。

```powershell
# [Windows PowerShell] 准备隔离的 Hermes profile（复制凭据 0600、写 config.yaml 与 SOUL.md）
wsl -d Ubuntu -- /usr/bin/python3 /mnt/d/FYP/First_Phase/libero_demo/setup_profile.py
# 期望输出最后一行：PROFILE_READY
```

`setup_profile.py` 只做三件事：把 `/home/yhwang/.hermes/.env` 复制到新 home（权限 0600，
内容不回显，原文件不动）；写 `config.yaml`（`qwen3-vl-plus` + `alibaba-cn` +
`supports_vision: true`，`agent.max_turns: 24`，仅 `libero_tools`）；写 `SOUL.md`。
重复运行会用相同内容覆盖配置，但保留新 home 已有的记忆与状态 DB，绝不递归复制旧会话。

---

## 四、启动 / 停止

```powershell
# [Windows PowerShell] 启动（先 setup_profile，再后台起服务与网页，最后打印地址）
powershell -ExecutionPolicy Bypass -File D:\FYP\First_Phase\libero_demo\start_demo.ps1
# 输出包含：打开浏览器访问: http://localhost:8080（脚本实际打印 localhost；推荐改用 http://127.0.0.1:8080 访问）

# [Windows PowerShell] 停止（只 SIGTERM 由本脚本记录 PID 的进程）
powershell -ExecutionPolicy Bypass -File D:\FYP\First_Phase\libero_demo\start_demo.ps1 -Stop
```

启动脚本的行为：

- 用 WSL 内的 Python `subprocess.Popen(start_new_session=True, stdout=文件, stderr=STDOUT)`
  拉起 `run_service.sh` 与 `app.py`；PID 写入
  `/home/yhwang/fyp/libero_demo/service.pid` 与 `app.pid`。
- **服务端口固定为 8766**：`-ServicePort` 仅接受 8766，传入其它值会在启动前直接报错退出；
  `-AppPort`（网页端口）保持可变。
- 启动前先检查**独立 venv**（`/home/yhwang/fyp/libero_demo/venv/bin/python`）与
  **模型目录**（`/home/yhwang/fyp/libero_demo/models/smolvla_libero`）是否存在；
  任一缺失即报错退出，**不回退 system python 伪启动**。
- **只在端口空闲时启动**；端口被占用时先做 health 检查：仅当返回的
  `backend=smolvla`、`robot=Franka Panda`、
  `model_revision=6721902bc4d61e50a3bfdb11dfb4cb626f05d102`、
  `model_path=/home/yhwang/fyp/libero_demo/models/smolvla_libero` 全部吻合时，
  才复用该服务；否则报告端口冲突并退出——不会 kill 其它进程。
- 启动时**不下载模型、不训练**；服务在后台加载模型，网页顶部显示就绪状态。
- `-Stop` 只停止 PID 文件中记录、且 `/proc/<pid>/cmdline` 含本项目绝对路径的进程，
  发 SIGTERM；不做 `wsl --shutdown`，不动其它服务。

---

## 五、网页使用

打开 `http://127.0.0.1:8080`（**推荐**；启动脚本实际打印 `http://localhost:8080`，
两者等价，`127.0.0.1` 更省事）：

1. 顶部显示服务就绪状态、backend、robot；
2. 在文本框里用自然语言描述目标（例如「把黑色碗放到盘子上」），填 seed 与
   init_state_index，点「执行」；
3. 页面每 2 秒轮询：显示真实步数、执行状态、执行结果（含 `run_ok` / `chain_ok` /
   `task_success`）、Hermes 的真实输出、执行视频与最近画面；
   `chain_ok` 只在执行链按 `state=completed` 且无 `error` 结束时为真，用于区分
   「Agent 进程正常退出」与「任务执行链成功跑完」，避免 Agent 一退出就被判定通过；
   轮询与 Hermes 子进程共用同一条 deadline（`--timeout`，默认 900 秒），持续到服务返回
   终态（`completed` / `error`）或触及该 deadline 为止，不会因为 Hermes 子进程先退出就
   提前判定；
   视频仅在服务返回终态（`completed` / `error`）后才挂上 `src`，避免访问尚未编码的 MP4；
   **每次新请求都会先清空上一次请求的执行视频与最近画面**；若本次请求自始至终都没有触发执行
   （例如信息不足被判定不执行），页面会**持续显示「本次请求没有触发机器人执行」**，
   该请求的执行视频与最近画面保持为空——**不会把上一次任务（其它 job）的视频或画面关联
   到本次请求**；最近画面在执行中更新，执行视频在 episode 终态后加载；
4. 下方任务目录列出 4 个 suite / 40 个受支持任务（suite / task_id / 标准指令），
   仅作参考——**无需手动指定 task_id**，由 Hermes 依据任务目录自行选择。

---

## 六、CLI 用户测试

```powershell
# [Windows PowerShell] 单次请求（服务需已就绪）
wsl -d Ubuntu -- /home/yhwang/fyp/libero_demo/venv/bin/python `
  /mnt/d/FYP/First_Phase/libero_demo/run_agent.py `
  --request "把黑色碗放到盘子上" --seed 0 --init-state-index 0 --timeout 900
```

参数：`--request`（必填）、`--seed`（默认 0）、`--init-state-index`（默认 0）、
`--timeout`（默认 900 秒）、`--run-dir`（可选，默认在 `runs/` 下新建）。

其中 `--timeout`（默认 **900 秒**）是本次请求的 **总 deadline（默认 900 秒）**：**Hermes
子进程的执行**与**随后的终态收集（终态轮询）共用这同一条总 deadline**，并非在子进程退出后
另起一轮计时。**提前退出的是 Hermes 子进程，不是宿主（HOST）进程**：宿主进程始终保持运行，
并且**不会**替 Hermes 去选择任务、也**不会**替 Hermes 执行任务。**轮询与同 job 只读 GET**：
Hermes 子进程退出后，宿主只对 **Hermes 已经提交的同一个 job_id** 发起**只读 GET** 查询，
并**每 2 秒轮询**一次，直到该 job 状态变为 `completed` / `error`，或触及这条共享的总 deadline
为止；GET 只用于查询终态，**不新建、也不重复提交 job**。结果只在该 job 结束时关联到本次请求，
**不会重复计入**，避免把上一轮的成功误记到本轮。若 Hermes 判定信息不足、或没有受支持任务而
**选择不执行**，则不会产生新的 job，也就不会有对应的轮询与总超时判定。

CLI 会：

- 先要求服务 `/health` 的 `ready=true`，否则不运行；
- 记录运行前 `/status` 的 job_id，运行后再取新 job_id，**仅当 job_id 变化**才把该次
  执行结果关联到本次请求（避免把上一轮的成功误记到本轮）；
- 运行真实命令
  `hermes -t libero_tools -z <prompt> --usage-file <run_dir>/usage.json`，
  其中 `HERMES_HOME` 指向隔离 profile、代理变量已 unset、cwd 为新 libero_demo；
- 保存 `request.json`（原文/seed/index/timestamp）、`hermes.log`（真实 stdout/stderr）、
  `agent_result.json`（`agent_exit_code` / `request` / `execution` 或 null / `wall_s` / `log_path`）；
- **最后一行 stdout 是完整 JSON**，字段含义明确区分：
  - `run_ok`：仅表示 **Hermes 进程退出码为 0**（Agent 正常跑完、自身未报错）；
  - `chain_ok`：仅当 `execution.state == "completed"` 且 `execution.error` 为空时为真，
    表示**执行链本身无异常结束**；它不等于任务成功；
  - `task_success`：取服务返回的 `success`（环境真实判定），表示**任务是否成功**；
  - `execution_timeout`：为 `true` 时表示**总 deadline 已到期，且该 episode 既没有
    `completed` 终态、也没有 `error` 终态**（两种终态都没拿到），此时 `chain_ok=false`。
    它**不能简单等同于「未 completed」**——因为 `error` 同样是终态：episode 若以 `error`
    结束，属于拿到终态，不算 `execution_timeout`。它**不表示 episode 被强制结束**，也不影响
    对 `run_ok` 的判定——`run_ok` 只看 Hermes 进程退出码，此时仍可能为真（Hermes 进程本身
    正常退出）。
  以上字段与超时标志互不替代，均如实报告，不编造成功。

---

## 七、任务范围与场景

- 受支持任务共 **4 个 suite × 10 = 40 个**标准 LIBERO benchmark 任务，
  以服务 `GET /tasks` 返回的目录为准；Hermes 必须先用 `list_libero_tasks` 读取真实目录。
- **本次服务允许为每个 benchmark 任务自动创建对应的仿真场景**：用户只需描述目标，
  不需要手动准备或指定场景。
- Hermes 的任务选择与执行**必须发生真实工具调用**（`execute_libero_task` 等），
  不允许用直接 HTTP 调用替代 Agent。

---

## 八、结果判定与诚实性说明

- **无微调**：使用 SmolVLA 官方 LIBERO 权重，不在本流程里训练或微调任何模型。
- **成功判定**：`success` 来自仿真环境自身的 `info["is_success"]` 判定，是环境真相，
  不是模型自述、也不是本前端推断。
- **任务失败不会被伪造成通过**：CLI / 网页分别展示 `run_ok`（Hermes 进程是否 exit 0）、
  `chain_ok`（执行链是否以 `completed` 且无 `error` 结束）与 `task_success`（任务是否成功）；
  Hermes 的输出原样落盘展示。
- **服务使用标准任务 instruction 驱动 VLA**，因此不要求（也不允许）LLM 改写指令给 VLA。
- **信息不足时**：Hermes 应说明需要澄清并停止，不擅自执行；**没有受支持任务时**报告不支持。

---

## 九、产物与日志

每次运行在 `/home/yhwang/fyp/libero_demo/runs/<run_dir>/` 下产出：

| 文件 | 内容 |
|---|---|
| `request.json` | 用户原文 / seed / init_state_index / timestamp |
| `hermes.log` | Hermes 的真实 stdout/stderr |
| `usage.json` | Hermes `--usage-file` 用量 |
| `agent_result.json` | `agent_exit_code` / `request` / `execution` / `chain_ok` / `wall_s` / `log_path` |
| `run_agent.log` | 网页后台调用 `run_agent.py` 的封装输出（仅网页触发时） |

网页通过 `GET /artifacts/<relative>` 读取 `runs/` 下的文件：只允许视频/PNG 与
`result.json` / `request.json` / `agent_result.json` / `hermes.log`，其它一律拒绝；
路径解析后校验，拒绝目录越界；不暴露配置文件与 `.env`。MP4 支持 Range 请求（206 +
`Content-Range`），可在浏览器里播放与拖动。

---

## 十、后续可配置能力（本轮未实现）

- WebSearch 等外部信息工具；
- 用户偏好（偏好语言、放置位置等）的可配置化。

以上为后续可配置能力；本README只声明**已经实现**的任务理解与执行链路，
不在此列出任何未经实测的成功率或结论。指标与结论需以实际运行记录为准。

---

## 十一、版本管理与提交约定

- 本目录代码纳入 **Git 版本管理**，集成工作在分支 **`codex/libero-hermes-integration`**
  上进行，不直接在主干上堆叠功能改动。
- 提交分阶段进行：**已有项目先以 BASELINE 提交作为基线**，**后续新增能力以 FEATURES
  分阶段提交**；BASELINE 与 FEATURES 是**相互独立的阶段性提交**（先基线、后功能），
  便于审查与回退。
- 可纳入 Git 的内容为源代码、可复现所需的元数据，以及 demos 中明确精选的四个小视频。
- 以下内容**默认一律排除在 Git 之外**，不列为提交类别：
  1. **API keys / 凭据**（`.env`、密钥等；`.env` 只复制到隔离 profile 并置为 0600、
     内容不回显、原文件不动）；
  2. **模型权重**；
  3. **虚拟环境**（venv）；
  4. **日志**；
  5. **视频**（运行视频**默认排除**）。
- **例外**：运行视频默认排除，仅 demos 中的四个精选小视频明确入 Git。凭据、权重、虚拟环境和日志仍排除。
- 版本记录以 Git 历史为准；本文档**不写死具体提交 hash**，也不列出未经实测的成功率或
  结论。

---

## 十二、2026-10-06 本机实测

本次使用预训练权重，无训练或微调。完整脱敏记录见 [acceptance_results.json](acceptance_results.json)。

| 输入／方式 | 任务 | seed / 初始状态 | 结果 | 步数 | 仿真耗时 / 请求耗时 |
|---|---|---|---|---|---|
| 网页中文：把字母汤罐头放进篮子里。 | libero_object / 0 | 1 / 1 | chain_ok=true，task_success=true | 152 | 80.857 / 95.964 秒 |
| 网页英文：put the bowl on the plate | libero_goal / 8 | 0 / 0 | chain_ok=true，task_success=true | 78 | 45.770 / 58.902 秒 |
| 网页：帮我驾驶汽车去上海。 | 无 | 1 / 1 | 正确拒绝，没有新 episode | — | — / 9.865 秒 |
| 直接调用 VLA：open the middle drawer of the cabinet | libero_goal / 0 | 0 / 0 | 完成执行但任务失败，达到步数上限 | 300 | 167.504 / — 秒 |

两项网页成功案例均有真实 Hermes 工具调用、环境成功判定及可播放 H.264 视频；中文案例的工具序列和参数记录在 JSON 中。非法输入、并发提交与产物路径越界也已检查。以上是链路验收样例，不能推出全部 40 个任务的成功率。WebSearch 和可配置偏好尚未接入。

---

## 十三、初版 Demo 与当前不足

精选 Demo 的网页画面：

![web-success](demos/web-success.jpg)

| 视频 | 原始请求 | suite / task_id | 步数 | 真实仿真秒数 | 完整请求秒数 |
|---|---|---|---|---|---|
| [alphabet-soup.mp4](demos/alphabet-soup.mp4) | 把字母汤罐头放进篮子里。 | libero_object / 0 | 152 | 80.857 | 95.964 |
| [bowl-on-plate.mp4](demos/bowl-on-plate.mp4) | put the bowl on the plate | libero_goal / 8 | 78 | 45.770 | 58.902 |
| [stove-on.mp4](demos/stove-on.mp4) | 开启一下烹饪的炉灶 | libero_goal / 7 | 72 | 41.185 | 55.624 |
| [wine-on-rack.mp4](demos/wine-on-rack.mp4) | 请收拾一下酒瓶，放到架子上 | libero_goal / 9 | 173 | 90.998 | 105.763 |

四个视频仅包含动作帧的 20 fps 回放，不包含 Hermes 规划与等待时间，不是实时录屏。仿真耗时与完整请求耗时分别列出，不能用视频时长代表真实运行速度。

当前不足（均为尚未实现或尚未验证，详见根 README [../../README.md](../../README.md)）：

- 目前仅从目录选择单个任务，宽泛整理目标会先澄清；任务切换会重置到不同场景，未实现同场景多子目标执行。
- Hermes `supports_vision=true` 但观察只返回图片路径，完整看图规划尚未接入；WebSearch 与可配置偏好尚未接入。
- 40 个任务入口不代表全部成功（已有 drawer 任务在 300 步上限失败）；新场景、改顺序、空位选择与占位避让的泛化能力尚未验证。
- 公开 checkpoint 模型卡标注训练数据 unknown；本机绝对路径与共享旧环境使可移植性有限。
- 后续计划（尚未实现）：能力审计、独立测试用例、持续场景、真实视觉输入与规划反馈。

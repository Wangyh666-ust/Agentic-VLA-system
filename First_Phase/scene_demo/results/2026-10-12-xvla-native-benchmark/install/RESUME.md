# X-VLA 安装完成后的关机续接说明

日期：2026-10-12。**用户本轮只要求装好并保存进度，便于关机；本轮已停在安装和零动作验收。未执行关机命令。**

## 当前状态

- 安装和本轮验收通过：官方 benchmark、`OffScreenRenderEnv`、作者 `evaluation.libero.libero_client` 三项导入退出 0；仅查询接口签名，未实例化环境、未调用 `.step`。
- 原作者 `XVLAProcessor` / `XVLA` 从本地固定权重离线加载成功；全部模型参数为 FP32、在 CUDA 上，进程退出码 0。
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU。参数量 879,482,456；峰值 allocated 3,518,954,496 bytes，reserved 3,542,089,728 bytes。删除模型和 processor、清理缓存后，两项当前显存计数均为 0，加载子进程已正常退出。
- 推理次数 0、动作预测 0、仿真环境构造 0、仿真动作 0、机器人动作 0。
- **40 任务 × 每任务 5 次 = 200 次基准授权已确定，但尚未启动，已完成 episode 数为 0。** 本轮不进入该阶段。
- 本次评测所需 modelserver 适配实现尚未完成或启动。本 worker 未编写评测代码，未调用 HTTP `/act`。
- DeepSeek 评测实现尚未调用，新增 DeepSeek 调用 0。主 Agent 为 GPT6 Astra (`openai/gpt-6-astra`)，本地工具 worker 只执行安装/验收，不能把该工作冒称 DeepSeek 实现；也不能把名称当作真实模型调用证据。
- 只读进程检查：导入 PID 396、模型加载 PID 397 均已退出；没有命令行包含本任务 `xvla_native` 路径的残留进程。未终止其他任务或服务。

机器可读证据：[install_verification.json](install_verification.json)。它包含两套完整 `pip freeze`、固定 SHA、权重信息、各阶段返回码、接口签名、显存和进程检查结果。原始日志均保留。

## 固定源码、模型与解释器

工作区：Windows `D:/FYP`；Linux `WSL Ubuntu`。

| 项目 | 固定值 |
| --- | --- |
| 安装根目录 | `/home/yhwang/fyp/xvla_native` |
| X-VLA 官方源码 | `/home/yhwang/fyp/xvla_native/upstream` |
| X-VLA URL / SHA | `https://github.com/2toinf/X-VLA.git` / `6bc2513f5f1cbec715cc668b414392a6cae5c671` |
| LIBERO 官方源码 | `/home/yhwang/fyp/xvla_native/LIBERO` |
| LIBERO URL / SHA | `https://github.com/Lifelong-Robot-Learning/LIBERO.git` / `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| 作者客户端 | `/home/yhwang/fyp/xvla_native/upstream/evaluation/libero/libero_client.py` |
| 模型仓库 / revision | `2toINF/X-VLA-Libero` / `129e71460678b7236cee6fc9707f09d9fa0c3590` |
| 本地模型 | `/home/yhwang/fyp/xvla_native/model` |
| 权重文件 | `/home/yhwang/fyp/xvla_native/model/model.safetensors` |
| 权重 bytes | `3519068172` |
| 权重 SHA256 | `260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8` |
| 模型解释器 | `/home/yhwang/fyp/xvla_native/venv/bin/python` |
| 模型核心版本 | Python `3.12.3`，torch `2.11.0+cu128`，torchvision `0.26.0+cu128`，transformers `4.51.3`，numpy `1.26.3` |
| 客户端解释器 | `/home/yhwang/fyp/xvla_native/sim_venv/bin/python` |
| 客户端核心版本 | Python `3.10.22`，torch `1.11.0+cpu`，torchvision `0.12.0+cpu`，numpy `1.22.4`，MuJoCo `2.3.7`，robosuite `1.4.0`，termcolor `2.5.0` |
| Python 3.10 基础安装 | `/home/yhwang/fyp/xvla_native/uv_python/cpython-3.10-linux-x86_64-gnu` |
| LIBERO 配置目录 | `/home/yhwang/fyp/xvla_native/libero_config` |
| 日志与续接文件 | `D:/FYP/First_Phase/tmp/xvla_20261012_setup/` |

权重 SHA 沿用本任务早先实际计算的 `14_weight_hash.stdout.log`，本轮按授权未重算，仅在 `24_old_versions_sources_weight.stdout.log` 复查大小。模型和两套 clone 保持只读；两个 git 工作区状态为空。

模型 venv 的 `00_fyp_shared_readonly.pth` 保留两行：

```text
/home/yhwang/fyp/libero_demo/venv/lib/python3.12/site-packages
/home/yhwang/fyp/vla/venv/lib/python3.12/site-packages
```

GPU torch 来自第二个旧路径；新模型依赖覆盖在新 venv 前面。不要把这个覆盖层误认为只依靠 `--system-site-packages` 自动继承父 venv。`sim_venv` 完全独立，`include-system-site-packages = false`。

`libero` 顶层为 namespace package，所以 `libero.__file__=None`；实际 `libero.__path__` 为官方 clone 的 `LIBERO/libero`，`libero.libero.__file__` 指向 `LIBERO/libero/libero/__init__.py`。它不是旧模型环境里可见的 hf-libero。

## 恢复时的必要进程环境

仅在相应子进程设置；不要写入全局环境或修改旧 venv。解释器使用上表的绝对路径。以下是配置记录，不是基准启动命令。

模型进程（本地离线加载/未来服务器使用）：

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/home/yhwang/fyp/xvla_native/upstream
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
HF_HOME=/home/yhwang/fyp/xvla_native/cache/huggingface
XDG_CACHE_HOME=/home/yhwang/fyp/xvla_native/cache
TMPDIR=/home/yhwang/fyp/xvla_native/cache/tmp
```

客户端进程：

```text
PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=/home/yhwang/fyp/xvla_native/LIBERO:/home/yhwang/fyp/xvla_native/upstream
LIBERO_CONFIG_PATH=/home/yhwang/fyp/xvla_native/libero_config
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
XDG_CACHE_HOME=/home/yhwang/fyp/xvla_native/cache
MPLCONFIGDIR=/home/yhwang/fyp/xvla_native/cache/matplotlib
NUMBA_CACHE_DIR=/home/yhwang/fyp/xvla_native/cache/numba
TMPDIR=/home/yhwang/fyp/xvla_native/cache/tmp
```

现有配置 `libero_config/config.yaml` 指向官方 clone 的 `libero/libero`、`bddl_files`、`init_files`、`assets`；datasets 为 `/home/yhwang/fyp/xvla_native/datasets`。未修改 `~/.libero`。

历史下载使用 `HF_HUB_DISABLE_XET=1`、`HF_HUB_OFFLINE=0`、`TRANSFORMERS_OFFLINE=0` 和同一固定 revision；下载已完成，正常续接不需要再次下载。

## 已核实的真实接口与固定方案

官方加载调用已经执行成功，仅供接口记录：

```python
processor = XVLAProcessor.from_pretrained('/home/yhwang/fyp/xvla_native/model')
model = XVLA.from_pretrained(
    '/home/yhwang/fyp/xvla_native/model',
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to('cuda').eval()
```

原作者类来自 `upstream/models/modeling_xvla.py` 与 `upstream/models/processing_xvla.py`。没有调用 `generate_actions` 或任何前向推理。

实测属性：`num_actions=30`，`action_space.dim_action=20`，`processor.num_views=3`，`processor.language_max_length=50`。

作者客户端实测签名：

```text
ClientModel(host: 'str', port: 'int')
ClientModel.step(self, obs: 'Dict', goal: 'str') -> 'np.ndarray'
ClientModel._format_query(self, obs: 'Dict', goal: 'str') -> 'Dict'
LIBEROEval(task_suite_name: 'str', eval_horizon: 'int' = 600, act_type: 'str' = 'abs', num_episodes: 'int' = 10, eval_freq: 'int' = 10, init_seed: 'int' = 42) -> 'None'
LIBEROEval._init_env(self, task_suite, task_id: 'int' = 0, ep: 'int' = 0)
```

保留既定方案：**FP32、30 动作块、10 去噪步骤、domain_id=3、作者原始图像处理、absolute (`act_type='abs'`) 模式**。作者客户端通过 HTTP `/act` 和 json_numpy 编解码；`ClientModel._format_query` 将 agentview 图像上下和左右翻转，手腕图像保持原样。快照原始 CLIPImageProcessor 配置为 224×224 resize、`do_center_crop=false`、normalize/rescale 开启，mean `[0.485, 0.456, 0.406]`、std `[0.229, 0.224, 0.225]`，保留已保存的 slow processor 行为。

上述是恢复时应沿用的配置与源码事实；没有实际执行 HTTP 协议、图像处理请求或动作预测。参数块维度与客户端 action 返回格式的实际衔接仍需下一阶段首例接口验证。

## 与作者条件的差异和历史失败

用户希望**单任务条件尽量与作者一致**；当前已知差异必须保留记录：每任务样本量 5 vs 作者对照 50；客户端 Python 3.10.22 vs 作者示例 3.8；模型使用本机 Python 3.12.3 / GPU torch 2.11.0+cu128，而不是主 Agent 所述作者旧 torch 2.1；客户端 torch/torchvision 为 1.11/0.12 的 CPU 构建。不要声称完全同环境或同样本量。

**MuJoCo 2.3.7 是我们本轮明确固定的选择，不是作者锁版；作者未提供完整锁文件。** 两环境的 freeze 记录当前可见安装状态，模型 freeze 包含继承的系统/旧包，不是作者锁文件，也不是建议照单重装全部依赖。

历史原始证据保留：

1. `04_create_venv`：普通 venv 创建失败，`ensurepip is not available`；随后按明确授权以 `--without-pip` 引导并补入两行共享路径。
2. `07_install_dependencies`：单一 Python 3.12 环境安装 MuJoCo 2.3.7 走源码构建，失败 `MUJOCO_PATH environment variable is not set`；随后按明确授权拆分 Python 3.12 GPU 模型环境和 Python 3.10 原生客户端。
3. `17_sim_imports`：缺 `termcolor`；本轮仅补 `termcolor==2.5.0`，`20_sim_imports_signatures` 三项导入均通过。

保留的非阻断输出：robosuite 可选 `macros_private` 提示、Gym 维护提示、slow processor 提示、Florence2 `GenerationMixin` 提示，以及先前模型环境继承依赖冲突提示。没有创建 macros_private、改源码、换 API 或改精度。加载通过不说明所有继承依赖兼容，也未验证任何推理分支。

## 证据索引与下次恢复顺序

| 证据 | 结果 |
| --- | --- |
| `19_termcolor.*` | termcolor 2.5.0 安装，退出 0 |
| `20_sim_imports_signatures.*` | 官方三项导入/来源/签名核查，退出 0 |
| `21_fp32_cuda_load.*` | 离线 processor + FP32 CUDA model 加载、显存/属性、释放，退出 0 |
| `22_model_pip_freeze.*` | 模型环境完整 pip freeze，退出 0 |
| `23_sim_pip_freeze.*` | 客户端完整 pip freeze，退出 0 |
| `24_old_versions_sources_weight.*` | 旧环境版本不变、源码 SHA/状态、权重大小复查，退出 0 |
| `25_process_check.*` | 无本任务下载/加载/server/eval 残留，退出 0 |
| `26_artifact_validation.*` | JSON、两套 freeze（331 / 61 行）、返回码、签名、加载数据及续接引用与原日志核对通过，退出 0 |
| `27_final_process_check.*` | 归档核验后最终进程检查，无本任务残留，退出 0 |

旧环境复查仍为 numpy 2.2.6、transformers 5.5.4、MuJoCo 3.8.1、torch 2.11.0+cu128、torchvision 0.26.0+cu128。

下次先读本文件和 `install_verification.json`，从上述已有环境/权重恢复，不重复宣称或重建已经完成的安装。200 次授权继续保留，当前文件不启动基准。

进入下一阶段前，主 Agent 仍须按项目实验审查流程复用历史证据、冻结 proposal，并运行 `.agents/skills/fyp-experiment-review/scripts/review.py check --proposal <冻结方案JSON>` 门禁；该检查不代替已明确的用户范围，也不代表科学结论。实现工作按当时明确授权/模型配置实际调用，不得虚称已由 DeepSeek 实现。

完成需要的 modelserver/客户端衔接后，必须先验证首个最小样例的接口、目标分支触发和实际执行动作；若应触发未触发，应停止依赖它的后续批量验证。**模型加载通过仅为安装/加载验收，不等于推理成功、仿真任务成功或物理任务成功。** 本文件不提供或自动执行基准启动命令。

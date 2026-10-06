# 相机外参推导与验证

目的：把相机**光学帧**下的三维点变换到机器人 `base_link` 系，供 `perception/locate_object.py` 使用。
配置文件：`/home/yhwang/fyp/config/camera_extrinsics.yaml`
世界文件：`/home/yhwang/fyp/worlds/tabletop_v6.sdf`

---

## 1. 输入事实：SDF 中的相机位姿

`tabletop_v6.sdf` 里的相机模型（世界系位姿，`base_link` 与世界系重合）：

```xml
<model name="rgbd_cam">
  <static>true</static>
  <pose>0.05 -0.40 0.45 0 1.5708 0</pose>     <!-- xyz rpy, rpy 单位弧度 -->
  <link name="cam_link">
    <sensor name="rgbd" type="rgbd_camera"> ... </sensor>
  </link>
</model>
```

即相机光心位于世界坐标 `(0.05, -0.40, 0.45)` m，姿态为绕 Y 轴旋转 `1.5708 rad`（= 90°），因此**朝下俯视桌面**。
相机内参由 SDF 给出：`horizontal_fov = 1.047 rad`（≈60°）、图像 `640×480`、裁剪 `near 0.1 / far 5.0`、`update_rate 15 Hz`。

---

## 2. 第一步：`T_base_camlink`（基座 → 相机 link 帧）

`rpy = (0, 90°, 0)` 对应的旋转矩阵为 `R_y(90°) = [[0,0,1],[0,1,0],[-1,0,0]]`，平移取 SDF 的 `xyz`：

```
        ┌ 0  0  1  0.05 ┐
T_base_camlink = │ 0  1  0 -0.40 │
        │-1  0  0  0.45 │
        └ 0  0  0  1    ┘
```

**为什么 link 帧是"x 前 / y 左 / z 上"**：Gazebo 相机沿着自身 link 的 `+x` 轴看。此处 link `+x` 映射到基座的 `-z`，即朝下——与相机高挂桌面正上方的物理布置一致，可自检。

---

## 3. 第二步：光学帧 → link 帧的固定换轴

相机光学帧（OpenCV/ROS 惯例）是 **z 前 / x 右 / y 下**，而 link 帧是 **x 前 / y 左 / z 上**。把光学基向量用 link 帧表示：

| 光学轴 | 在 link 帧中的方向 |
|---|---|
| `opt_x`（右） | `(0, -1, 0)` |
| `opt_y`（下） | `(0, 0, -1)` |
| `opt_z`（前） | `(1, 0, 0)` |

因此

```
        ┌ 0  0  1 ┐
R_link_opt = │-1  0  0 │
        └ 0 -1  0 ┘
```

---

## 4. 第三步：合成 `T_base_opt`

```
T_base_opt = T_base_camlink × R_link_opt
```

旋转部分相乘得 `[[0,-1,0],[-1,0,0],[0,0,-1]]`，平移部分继承 `T_base_camlink`：

```
        ┌ 0 -1  0  0.05 ┐
T_base_opt = │-1  0  0 -0.40 │
        │ 0  0 -1  0.45 │
        └ 0  0  0  1    ┘
```

这正是 `config/camera_extrinsics.yaml` 的内容：

```yaml
matrix: [[0,-1,0,0.05],[-1,0,0,-0.40],[0,0,-1,0.45],[0,0,0,1]]
frame_id: base_link
child_frame: rgbd_optical
```

自检：`T_base_opt` 的第三列 `(0, 0, -1)` 表示光学 z（前）指向基座的 `-z`（向下），与相机俯视一致。

---

## 5. 使用方式

`perception/locate_object.py` 的流程：

1. 从 `/rgbd/image` 取 RGB，HSV 阈值分割红色（两组区间 `(0,100,80)-(10,255,255)` 与 `(170,100,80)-(180,255,255)`），`3×3` 形态学开运算，取最大连通域；
2. 用 `camera_info` 的 K 反投影掩膜像素到相机光学帧；
3. 取**中位数**表面点 `point_opt`（在相机光学帧内）；
4. 变换到基座系：

```python
surface_base = T_base_opt[:3, :3] @ point_opt + T_base_opt[:3, 3]
```

5. 物体中心由顶面下移物体半高得到：

```
center_base = surface_base - [0, 0, 0.03]      # OBJECT_HALF_HEIGHT = 0.03 m
```

这 0.03 m 是**物体模型常量**（立柱高 0.06 m，顶面比中心高半个高度），不是测量值、也不是真值。

6. 高度带门：反投影得到的顶面 z 必须落在 `[0.035, 0.10]` m（`SURFACE_Z_MIN/MAX`）。
   该门把 HOME 位姿态下**红夹爪饰件**（反投影约在 z≈0.24 m）排除掉，越界即拒绝，`reason=implausible_z`。

> 注意区分三个概念：`surface_base` 是**表面点**，`center_base` 是**几何中心**，两者都不是"抓取点"。抓取候选由 `propose_grasps` 用表面点 + 固定高度生成（见 `docs/robot_tools_api.md`）。

---

## 6. 验证门（阶段 B）

### 6.1 单点核对（`logs/phaseB_extrinsics_gate.log`）

真值：立柱 `set_pose → (0.05, -0.35, 0.03)`，顶面真值 `z = 0.06`。
模块输出（stdout 原始一行 JSON）：

```json
{"ok": true, "surface_base": [0.050703, -0.352163, 0.06],
 "center_base": [0.050703, -0.352163, 0.03],
 "confidence": {"mask_px": 1527, "valid_depth_ratio": 1.0}, "reason": ""}
```

偏差计算：

```
dx  =  0.050703 - 0.050000 = +0.70 mm
dy  = -0.352163 - (-0.350000) = -2.16 mm
dxy = 2.27 mm   < 15 mm  → PASS
z   = 0.060（真值 0.06，误差 0.00 mm）；center_base z = 0.030 = 真值中心
```

### 6.2 多点复测（`logs/phaseB_gate.log`，8 个位置）

| 位置 (x, y) | 反投影顶面 | 真值顶面 | dxy | dz | mask_px | 判定 |
|---|---|---|---|---|---|---|
| (0.050, -0.350) | (0.0507, -0.3522, 0.0600) | (0.0500, -0.3500, 0.0600) | 2.27 mm | 0.00 mm | 1527 | PASS |
| (0.050, -0.440) | (0.0507, -0.4378, 0.0620) | (0.0500, -0.4400, 0.0620) | 2.32 mm | 0.00 mm | 1463 | PASS |
| (0.050, -0.280) | (0.0507, -0.2867, 0.0600) | (0.0500, -0.2800, 0.0600) | 6.78 mm | 0.00 mm | 1957 | PASS |
| (-0.020, -0.440) | (-0.0162, -0.4387, 0.0600) | (-0.0200, -0.4400, 0.0600) | 3.99 mm | 0.00 mm | 1790 | PASS |
| (0.120, -0.440) | (0.1174, -0.4386, 0.0600) | (0.1200, -0.4400, 0.0600) | 2.93 mm | 0.00 mm | 1794 | PASS |
| (-0.020, -0.280) | — | (-0.0200, -0.2800, 0.0600) | — | — | 129 | REJECTED（`implausible_z`，被 HOME 位手臂遮挡） |
| (0.120, -0.280) | (0.1182, -0.2853, 0.0600) | (0.1200, -0.2800, 0.0600) | 5.63 mm | 0.00 mm | 2286 | PASS |
| (0.050, -0.400) | (0.0504, -0.3996, 0.0600) | (0.0500, -0.4000, 0.0600) | 0.50 mm | 0.00 mm | 1296 | PASS |

汇总：`worst dxy = 6.78 mm`、`wrong-pose failures = 0`、`rejected = 1`、`positions = 8`（阈值 15 mm）。
全部有效位置的 `dz = 0.00 mm`，说明深度—外参链路的高度方向无系统偏差。

### 6.3 闭环评估中的定位误差（`logs/phaseB_closedloop.log`）

在 10 个随机位上的感知误差（真值仅用于算误差指标）：

| 指标 | 值 |
|---|---|
| mean | 2.1 mm |
| median | 2.0 mm |
| max | 3.9 mm |

感知误差与抓放成功率比较：`carried 8/10`、`place 1/10`——说明**定位不是当前瓶颈**，瓶颈在抓取/放置环节。

---

## 7. 维护注意

- 若改了 `worlds/tabletop_v6.sdf` 里 `rgbd_cam` 的 `<pose>`，或改了机器人基座相对世界的位置，**必须**同步重算并更新 `config/camera_extrinsics.yaml`，然后重跑 §6.1 的单点核对。
- 外参只描述几何，不描述时间：帧时效由 `locate_object.py` 的 `STALE_SEC = 1.0 s` 门和工具层的 `PLAN_STALE_SEC = 30 s` 门分别把关。
- 待确认：当前外参按"`base_link` 与世界系重合"推导；若后续场景把机器人模型放到非零位姿，需要把该偏移并入 `T_base_opt`。

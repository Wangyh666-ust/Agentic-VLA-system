# 20例关节归位组合任务诊断计划（2026-10-09）

本轮冻结20条新鲜VLA序列，使用三个原生LIBERO场景，不patch场景或资产。固定顺序诊断不新增Hermes调用，不启用抓取或物体入口准备辅助器，仅在独立实验runner验证关节归位交接，不改8081／8767生产UI。

| 序列 | 场景 | suite / task | 固定能力顺序 | case |
|---:|---|---|---|---|
|1|goal_table|libero_goal / 8|wine_to_rack → bowl_to_plate|01、02|
|2|goal_table|libero_goal / 8|bowl_to_plate → wine_to_rack|03、04|
|3|goal_table|libero_goal / 8|bowl_to_plate → stove_on|05、06|
|4|goal_table|libero_goal / 8|stove_on → bowl_to_plate|07、08|
|5|basket_two|libero_10 / 0|soup_to_basket → sauce_to_basket|09、10|
|6|basket_two|libero_10 / 0|sauce_to_basket → soup_to_basket|11、12|
|7|mugs_two|libero_10 / 4|white_mug_left → yellow_mug_right|13、14|
|8|mugs_two|libero_10 / 4|yellow_mug_right → white_mug_left|15、16|
|9|goal_table|libero_goal / 8|bowl_to_plate → wine_to_rack → stove_on|17、18|
|10|goal_table|libero_goal / 8|stove_on → wine_to_rack → bowl_to_plate|19、20|

每种序列连续两例：条件A使用init_state_index=0、model_seed=2；条件B使用init_state_index=1、model_seed=3。env_seed均为0，case_id依次为case_01至case_20。每case只在初始reset一次；例内子任务连续执行，归位不重置环境。

固定策略profile为baseline_bf16：use_amp=True、num_steps=10、n_action_steps=1。每个VLA子任务最多336个实际动作；关节归位最多360个实际动作。只有子任务成功且确认释放／稳定后才归位。速度问题本轮不调。

原完成目标全程保留；归位保留所有物体位置、姿态、关节和开关状态。归位核验7个机械臂关节、2个手指及停稳条件，必须连续5个实际post-action检查通过。归位ready与case成功分开记录。

独立gold字面常量在运行前冻结，按每条序列去重，不读取Hermes计划推导：

- bowl：on / akita_black_bowl_1 / plate_1
- wine：on / wine_bottle_1 / wine_rack_1_top_region
- stove：turnon / flat_stove_1
- soup：in / alphabet_soup_1 / basket_1_contain_region
- sauce：in / tomato_sauce_1 / basket_1_contain_region
- white：on / porcelain_mug_1 / plate_1
- yellow：on / white_yellow_mug_1 / plate_2

只有独立gold连续最后5个实际post-action均真且空手／稳定，才算case成功。普通物理失败结束本case，继续下一case；基础设施错误、未知观测或取消停止campaign，剩余case显式not_run。没有自动重试、隐式调参或额外episode，最多20case，不训练权重。

统计首任务失败与有效归位后的后续任务表现，分别报告归位动作和VLA动作。初态与模型seed同时变化，不能分离归因。本轮只有固定顺序执行诊断，不评估新的语言规划能力，不将小样本结果宣称通用成功率。

准备阶段只核验冻结输入、源码、用户文件与三套原生任务初态数量；不创建环境、不加载模型、不运行物理动作或episode。用户已授权本轮20例真实实验；软件检查通过后使用本计划的冻结输入运行。

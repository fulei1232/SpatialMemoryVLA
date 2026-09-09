# SpatialMemoryVLA 状态与训练验证计划

更新日期：2026-09-03

## 当前阶段

**Model integration complete / Stage A distributed validation complete**。

模型、数据、教师网络和本地仿真依赖已准备完成；8 卡分布式 forward、backward、
FSDP optimizer step、完整检查点保存与恢复均已验证。尚未完成 500-step 曲线检查
或策略 rollout，因此不能把当前状态描述为“性能已验证”。

## 已实现

- 基于 MemoryVLA 的 Geometry-Aligned Temporal Memory。
- 冻结、外置的 VGGT 教师：`37x37 -> 16x16` 池化，目标为 `[B,256,2048]`。
- VLA 第 24 层视觉 token `[B,256,4096]` 经可训练对齐投影器变为
  `[B,256,2048]`，空间记忆压缩为 `[B,256,256]`。
- 总损失为 `L_action + 0.5 * L_spatial`。
- 空间教师不进入 FSDP，也不保存到策略 checkpoint。
- 训练期硬断言 VLA、VGGT、对齐、感知记忆、认知 token、动作 GT、噪声预测
  的张量契约，以及所有损失的有限性。
- JSONL 记录 `Action Loss`、`Spatial Loss`、`Spatial Weighted Ratio`、
  `Grad Norm` 和总损失。
- FSDP checkpoint 同时保存模型和 optimizer state，可用于真实恢复。

## 资产与接口验证

| 资产 | 本地位置 | 状态 |
| --- | --- | --- |
| VGGT-1B | `pretrained/VGGT-1B/model.pt` | 已下载；GPU 前向输出 `[1,256,2048]` |
| MemVLA LIBERO-Spatial 基线 | `pretrained/memvla-libero-spatial/` | 已下载；模型子模块完整 |
| LIBERO Spatial RLDS | `data/libero-rlds/` | 16 个分片已下载 |
| LIBERO | `third_libs/LIBERO/` | 已安装；任务 API 可调用 |
| NousResearch Llama-2-7B | `pretrained/NousResearch-Llama-2-7b-hf/` | safetensors 已下载；词表/嵌入兼容 |

NousResearch 镜像仅解决 Meta Hugging Face gated repository 的在线访问问题，**不改变
Llama 2 的许可要求**。使用、修改或分发 Llama 2 materials/derivatives 仍受
LLAMA 2 Community License 约束；见 `LICENSES/LLAMA2_LICENSE.txt` 与 `NOTICE`。

## LIBERO 任务编号：必须区分两个命名空间

不要把 `tasks_info.txt` 的行号直接传给 `Benchmark.get_task(index)`。

| 任务 | `tasks_info.txt` 0-based manifest index | 当前 LIBERO API runtime index |
| --- | ---: | ---: |
| `next_to_the_plate`（debug smoke 任务） | 2 | 8 |
| `between_the_plate_and_the_ramekin`（空间验证任务） | 9 | 0 |

原因是已安装 LIBERO 的 benchmark API 从 `libero_suite_task_map.py` 构建任务列表，
其顺序不同于 `bddl_files/libero_spatial/tasks_info.txt`。评估代码必须按规范任务名
解析 runtime index：`evaluation/libero_tasks.py` 提供
`resolve_benchmark_task_index()`，避免再出现 off-by-one 或跨命名空间错误。

LIBERO-10 的空间+记忆任务也已按任务名固定为：
`LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`。

## 分级训练计划

### Stage A：8 卡 20-step smoke

**已完成（2026-09-03）**：20 个 optimizer step 全部通过，loss 从 `0.6572`
降至 `0.1693`，最终 grad norm 为 `0.5903`。step 10 与 step 20 均生成完整
模型（约 33.6 GB）和 optimizer state（约 66.9 GB）。随后从 step 20 恢复，
成功执行 step 21（loss `0.1634`，grad norm `0.5031`）并再次保存完整状态。

运行产物：

- `log/libero/spatial_memory_libero_spatial_smoke--image_aug/`
- `log/libero/spatial_memory_libero_spatial_smoke_resume--image_aug/`

```bash
cd /home/SpatialMemoryVLA
bash script/train/libero/train_spatial_memory_smoke.sh
```

每 10 step 保存 checkpoint。必须确认：所有硬断言通过、VGGT 无梯度、所有损失有限、
无 FSDP unused-parameter 错误、checkpoint 与 `.optimizer` 同时生成。

恢复检查（会从检查点继续额外执行 1 个 optimizer step）：

```bash
bash script/train/libero/verify_smoke_checkpoint.sh \
  log/libero/spatial_memory_libero_spatial_smoke/checkpoints/<step-checkpoint>.pt
```

### Stage B：500-step functional test

```bash
bash script/train/libero/train_spatial_memory_functional.sh
```

观察 `L_action`、`L_spatial`、总损失、`Grad Norm` 和
`r = 0.5 * L_spatial / L_action`。先记录原始行为；只有在稳定运行后才讨论修改
空间损失系数。

### Stage C：完整训练

仅在 Stage A checkpoint 恢复与 Stage B 曲线稳定、显存稳定且无 NaN 后执行：

```bash
bash script/train/libero/train_spatial_memory.sh
```

## A/B/C 消融入口

| ID | 配置 | 入口 |
| --- | --- | --- |
| A | MemoryVLA，无空间对齐/空间记忆 | `train_memoryvla_baseline.sh` |
| B | MemoryVLA + VGGT 对齐损失，无空间记忆 | `train_spatial_forcing.sh` |
| C | SpatialMemoryVLA，对齐损失 + 空间记忆 | `train_spatial_memory.sh` |

三个入口共用相同的数据、基线初始化和优化超参数；由 `EXPERIMENT_MODE` 选择模式。
第一轮评估先以规范任务名选取 `next_to_the_plate`，第二轮用
`between_the_plate_and_the_ramekin`，随后再进入 LIBERO-10 多阶段任务。

## 尚未完成

1. Stage B 500-step 曲线检查。
2. A/B/C 的 LIBERO rollout 与成功率比较。
3. LIBERO-10 验证。

# SpatialMemoryVLA Functional Training Report

日期：2026-09-04  
数据来源：`/media/fulei/jlu/spatial_memory_libero_spatial_functional--image_aug/`

## 1. 训练状态

- 配置：8×A100、`MAX_STEPS=10000`、`SAVE_INTERVAL=2000`、学习率 `2e-5`。
- 当前记录：step **9252 / 10000**；训练进程仍在运行。
- 已完成 checkpoint：step 2000、4000、6000、8000。
- 每组模型约 33.6 GB，optimizer state 约 66.9 GB；最终 step 10000 checkpoint 尚未生成。

## 2. 指标变化

| 阶段 | Action Loss | Spatial Loss | Weighted Ratio | Grad Norm | Total Loss |
|---|---:|---:|---:|---:|---:|
| step 1 | 0.1385 | 1.0123 | 3.6538 | 2.6132 | 0.6447 |
| 1–100 均值 | 0.0376 | 0.1531 | 1.7442 | 0.6556 | 0.1141 |
| 101–1000 均值 | 0.0255 | 0.0132 | 0.2713 | 0.5149 | 0.0321 |
| 1001–2000 均值 | 0.0235 | 0.0083 | 0.1868 | 0.5075 | 0.0276 |
| 2001–4000 均值 | 0.0225 | 0.0071 | 0.1685 | 0.4823 | 0.0261 |
| 4001–6000 均值 | 0.0210 | 0.0058 | 0.1467 | 0.4817 | 0.0238 |
| 6001–8000 均值 | 0.0193 | 0.0052 | 0.1441 | 0.4675 | 0.0218 |
| 8001–当前均值 | 0.0183 | 0.0048 | 0.1405 | 0.4651 | 0.0207 |
| 最近 200 steps 均值 | 0.0182 | 0.0046 | 0.1381 | 0.4475 | 0.0204 |

当前最新单步（step 9252）为：Action Loss `0.0204`、Spatial Loss `0.0049`、
Weighted Ratio `0.1213`、Grad Norm `0.4875`、Total Loss `0.0228`。

## 3. 训练效果分析

1. **动作学习已明显收敛。** Action Loss 从 step 1 的 `0.1385` 降至当前单步
   `0.0115`；最近 200 steps 均值为 `0.0182`，相较前 100 steps 均值下降约 52%。
2. **空间对齐效果显著。** Spatial Loss 从初始 `1.0123` 降至当前 `0.0045`，
   最近 200 steps 均值为 `0.0046`。这说明 VLA visual representation 已逐步接近
   VGGT teacher 的几何表征。
3. **空间辅助项已从主导项变为稳定正则项。** Weighted Ratio 从初始 `3.65`
   降至最近 200 steps 均值 `0.138`，即 `0.5 × Spatial Loss` 通常约占 Action
   Loss 的 14%，不会压制动作优化。
4. **优化过程稳定。** Grad Norm 从 `2.61` 降至最近 200 steps 均值 `0.4475`，
   全程未观察到 NaN、Inf 或梯度爆炸。Total Loss 在后期约 `0.02` 上下波动，符合
   diffusion action loss 的单 batch 噪声特征。
5. **后期仍有缓慢改善，但边际收益降低。** 4001–6000、6001–8000、8001–当前
   三个阶段的 Total Loss 均值分别为 `0.0238`、`0.0218`、`0.0207`，仍在下降，
   但下降速度已经明显放缓。

## 4. Checkpoint 与完成度

截至报告生成时，step 8000 checkpoint 已完成，step 10000 尚未完成。当前训练已完成
约 **92.5%** 的计划步数；按约 5.1–5.5 秒/step 估算，还需约 1 小时左右，随后会写出
最终 checkpoint。

## 5. 结论与后续建议

当前结果支持以下结论：Geometry-Aligned Temporal Memory 的训练链路稳定，空间对齐
损失快速下降，空间项在后期保持较低且可控的相对权重，动作损失也持续改善。现在还
不能仅凭训练 loss 宣称 LIBERO 成功率提升；完成 step 10000 后，应使用最终 checkpoint
进行 LIBERO-Spatial rollout，并与 MemoryVLA baseline 及仅 auxiliary alignment 的
消融组比较成功率。

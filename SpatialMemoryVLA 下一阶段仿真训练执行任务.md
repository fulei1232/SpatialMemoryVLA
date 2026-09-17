目标：基于当前 SpatialMemoryVLA 仓库，新增并跑通 3 组针对 spatial memory 的训练实验：

1. Occlusion / Out-of-view Memory Training
2. Dynamic Object Relocation / Stale Memory Training
3. Counterfactual Spatial Relation Training

不要重新设计网络。优先复用当前：
- VLA backbone
- geometry distillation
- spatial projector
- BottleneckSE
- PerMemBank / CogMemBank
- memory retrieval
- learned gate
- DiT diffusion policy
- A / B / C 三种配置

当前方法的核心链路保持不变：

Image + Language
→ VLA
→ intermediate spatial latent
→ geometry distillation（训练期）
→ spatial compression
→ spatial memory
→ memory-conditioned DiT
→ action

VGGT 仍然只允许训练阶段使用，推理阶段不允许调用。

一、先检查仓库，不要立即改代码

先完成以下工作，并输出检查报告：

1. 找到训练入口。
2. 找到 dataset / dataloader。
3. 找到 episode_id 和 timestep 是怎么传入模型的。
4. 找到：
   - use_spatial_forcing
   - use_spatial_memory
   - memory length / capacity
   - PerMemBank
   - CogMemBank
   - _build_perception_tokens()
   - predict_action()
5. 找到 LIBERO / RoboMME 当前训练和 rollout 脚本。
6. 找到目前任务场景生成、initial state、camera 配置在哪里控制。
7. 找到 checkpoint 保存、resume、evaluation 的现有方式。

不要凭名字猜路径，以仓库实际代码为准。

最终先列出：

训练入口：
数据入口：
环境入口：
memory 实现：
evaluation 入口：
当前 A/B/C 配置方式：

确认以后再修改。

--------------------------------------------------

二、实验 1：Occlusion / Out-of-view Memory Training

目的：

构造“当前 RGB 已经不能直接看到目标，但历史看过目标”的训练 episode。

这个实验必须迫使 C 模型使用历史 spatial memory。

优先选现有 LIBERO-Spatial task，例如：

- next_to_the_plate
- between_the_plate_and_the_ramekin

如果仓库还有其他 left/right/in/on/behind/inside 类任务，也可以加入。

需要实现两种难度。

A. Occlusion

episode 前期目标物体清晰可见。

机器人开始运动后，让目标物体被以下之一遮挡：

- robot arm
- gripper
- 额外 blocker
- 大物体
- camera crop / mask

注意：
不要从 episode 一开始就遮挡。

推荐时序：

t=0~N1：
目标完全可见。

t=N1~N2：
部分遮挡。

t>N2：
目标大面积或完全遮挡。

但任务仍然要求机器人继续完成动作。

训练数据必须保留整个 episode 的顺序。

B. Out-of-view

目标开始时可见。

之后通过：

- camera viewpoint movement
或者
- robot/environment trajectory

让目标离开当前 camera FOV。

后续动作仍然依赖之前看到的目标位置。

例如：

t0：
看到 plate、bowl、ramekin。

t1：
机械臂开始抓 bowl。

t2：
camera / arm movement 后 plate 离开视野。

t3：
仍然要求把 bowl 放到 plate 与 ramekin 的目标空间关系上。

关键要求：

不能把任务修改成单帧依然容易判断。

必须确认后半段当前 RGB 本身不足以恢复目标位置。

新增配置参数，具体名字可以按仓库风格调整，例如：

training.memory_curriculum.enabled=true
training.memory_curriculum.type=occlusion

或者：

--occlusion-training
--out-of-view-training

建议支持：

occlusion_start_ratio
occlusion_duration
occlusion_probability
occlusion_strength

例如：

probability:
0.5

start ratio:
0.3~0.5 episode

strength:
partial / full

--------------------------------------------------

三、Occlusion Curriculum

不要一开始就全遮挡。

建议 curriculum：

Stage 1
历史长度需求约 1~2 timestep。
轻微遮挡。

Stage 2
历史长度需求约 4 timestep。
中等遮挡。

Stage 3
历史长度需求约 8 timestep。
重遮挡。

Stage 4
历史长度需求约 16 timestep。
完全遮挡或 out-of-view。

如果当前 memory capacity=16，就暂时不要超过 16。

如果当前训练框架不方便按 epoch 改 curriculum，也可以分别生成四个数据配置：

occ_len_2
occ_len_4
occ_len_8
occ_len_16

分别训练或者混合采样。

--------------------------------------------------

四、实验 2：Dynamic Object Relocation / Stale Memory

这是优先级最高的一组。

目的：

防止模型学成：

“memory 里的旧位置永远可信”。

要训练模型学会：

remember
→ detect scene change
→ update spatial state
→ ignore stale memory

构造 episode：

t0：
目标 A 在位置 P1。
机器人看到 A。

t1：
产生一段正常动作。

t2：
A 被移动到 P2。

t3：
机器人重新看到 A 或部分看到 A。

t4：
继续完成操作。

例如：

初始：
bowl 在 plate 左侧。

中途：
把 bowl 移动到 plate 右侧。

后续任务：
抓取 / 放置 bowl。

或者：

初始：
ramekin 在 plate 右侧。

中途：
ramekin 被移动。

后面要求：
把 bowl 放到 plate 与 ramekin 之间。

必须保证：

旧 memory 与当前场景冲突。

这样 learned gate 才需要真正学习：

什么时候信历史，
什么时候信当前视觉。

--------------------------------------------------

五、Dynamic Relocation 的实现方式

优先考虑 simulator state-level modification。

也就是直接修改目标 object pose。

不要优先通过 image augmentation 假造移动。

理想实现：

env.reset()

正常运行若干 step。

在指定 timestep：

set_object_pose(object, new_pose)

然后继续正常 rollout。

如果环境不允许中途 set pose：

则生成包含不同阶段的 scripted demonstration。

要求：

episode_id 不变，
timestep 连续。

新增配置：

relocation_probability
relocation_timestep_range
translation_range
rotation_range
target_object_selection

建议初期：

relocation_probability = 0.3

移动距离：

先做明显变化，不要只移动几个毫米。

例如桌面坐标：

5 cm
10 cm
15 cm

分别做不同 difficulty。

--------------------------------------------------

六、Stale Memory Negative Training

在 Dynamic Relocation 基础上，再做一个训练变体。

目的：

人为制造历史空间状态与当前空间状态冲突。

最推荐的数据级实现：

物体真实发生 relocation。

不要优先直接篡改 memory tensor。

原因：

这样视觉 observation、动作轨迹、空间状态之间更真实。

如果确实不好实现环境 relocation，可以额外支持一个 debugging-only 模式：

以一定概率将历史 memory 中某些 timestep 延长保留，
模拟 stale state。

但这个版本只作为 ablation/debug，不能作为主要实验。

--------------------------------------------------

七、实验 3：Counterfactual Spatial Relation Training

目的：

防止模型通过：

object identity
背景
语言模板
固定位置

解决任务。

要构造“几乎所有内容都一样，仅空间关系不同”的训练对。

例如：

Pair 1：

Scene A：
bowl left of plate

Scene B：
bowl right of plate

Pair 2：

Scene A：
bowl between plate and ramekin

Scene B：
bowl next to plate

Pair 3：

Scene A：
object inside container

Scene B：
object outside container

Pair 4：

Scene A：
object in front of target

Scene B：
object behind target

要求：

同一组 counterfactual pair 尽量保持：

- object identity 相同
- texture 相同
- lighting 相同
- background 相同
- camera 尽量相同
- instruction template 尽量相同

只改变：

空间位置
或者
空间关系词。

例如：

“put the black bowl to the left of the plate”

和

“put the black bowl to the right of the plate”

不要同时改变 bowl、plate、背景和 camera。

否则无法证明模型学习的是 geometry。

--------------------------------------------------

八、Counterfactual 数据最好成对生成

推荐 dataset metadata 增加：

pair_id
relation_type
relation_label
difficulty

例如：

pair_id = 001
relation_type = left_right
relation_label = left

pair_id = 001
relation_type = left_right
relation_label = right

后续 evaluation 时可以计算：

counterfactual consistency

即：

模型是否会随着关系改变正确改变动作。

--------------------------------------------------

九、不要只训练 C

三组新数据至少跑：

A：MemoryVLA baseline
B：Geometry-supervised VLA
C：SpatialMemoryVLA

保持训练：

dataset
steps
batch size
optimizer
learning rate
seed
action horizon

完全一致。

A/B/C 唯一改变模型配置。

当前定义应该保持：

A：
普通 MemoryVLA。

B：
geometry loss 训练 backbone，
但 action perception branch 不消费 geometry-aligned spatial feature。

C：
geometry-aligned spatial latent
→ spatial compression
→ spatial memory
→ DiT。

不要误改 B。

--------------------------------------------------

十、增加 Memory Length Ablation

C 模型至少额外跑：

memory length = 1
memory length = 2
memory length = 4
memory length = 8
memory length = 16

重点比较：

length=1

vs

length=16

因为 length=1 基本等价于没有长期 history。

如果：

C length=1 ≈ B

而：

C length=16 > C length=1

则可以证明收益来自 temporal memory，而不是简单 feature enhancement。

如果 GPU 时间有限，优先：

1
4
16

--------------------------------------------------

十一、必须记录 gate 行为

当前 memory 有：

current state X_t

retrieved history R_t

以及 learned gate alpha。

请增加日志。

至少记录每个 episode：

mean alpha
std alpha
alpha over timestep

如果当前公式是：

X_fused = alpha * X_current
        + (1-alpha) * X_history

那么预期：

正常无遮挡：
alpha 相对更高。

当前帧遮挡：
alpha 应下降，更依赖 history。

object relocation 后重新看到目标：
alpha 应升高，更依赖 current observation。

请确认代码实际定义方向，不要根据这里文字硬套。

如果 gate 定义相反，则相应解释反过来。

输出：

gate_curve.npy / csv

最好同时保存：

episode_id
timestep
occlusion flag
relocation flag
alpha

方便后面画图。

--------------------------------------------------

十二、增加 memory retrieval diagnostic

如果方便，在 evaluation 阶段记录：

当前 query 与历史各 timestep 的 attention weight。

例如：

t=12 query

history:
t0
t1
...
t11

保存 retrieval attention。

希望能够检查：

遮挡发生后，
模型是否检索遮挡前最后一个有效空间状态。

这是论文 figure 很有价值的分析。

如果改动成本很高，这项可以作为 optional。

--------------------------------------------------

十三、训练数据比例

第一版不要完全替换正常数据。

建议 mixture：

Normal:
50%

Occlusion / Out-of-view:
20%

Dynamic relocation:
15%

Counterfactual spatial:
15%

或者近似比例。

先验证稳定性。

如果 C 的正常任务性能明显掉，再调低 hard-case 占比。

第二阶段再试：

Normal:
30%

Memory-critical:
70%

--------------------------------------------------

十四、训练顺序

不要直接启动大规模 full training。

Phase 1：Smoke Test

每种新数据先生成少量：

10~50 episodes。

检查：

episode_id 是否正确。

timestep 是否连续。

memory 是否跨 timestep 工作。

teacher feature 是否正常。

spatial projector 是否正常反传。

action loss 是否正常。

geo loss 是否正常。

无 NaN。

checkpoint 能保存。

rollout 能加载。

--------------------------------------------------

Phase 2：Small Overfit

每种数据取：

50~200 episodes。

训练到明显 overfit。

目标不是泛化，而是确认：

Occlusion：
模型在遮挡后还能成功。

Relocation：
模型能够更新目标位置。

Counterfactual：
左右等关系不会混淆。

如果 small overfit 都做不到，先不要 full training。

--------------------------------------------------

Phase 3：Matched Training

正式跑：

A
B
C

所有配置 matched。

至少 3 random seeds，如果计算资源不足：

先 1 seed 找趋势，
再给核心实验补 3 seeds。

--------------------------------------------------

十五、Evaluation 必须拆场景

不能只报告 overall success。

至少分别报告：

Normal

Occlusion

Out-of-view

Dynamic relocation

Counterfactual relation

最好再按：

short memory dependency
medium memory dependency
long memory dependency

分类。

例如：

short:
需要记 1~2 step

medium:
4~8 step

long:
8~16 step

--------------------------------------------------

十六、建议最终结果表

Table 1：

Model | Normal | Occlusion | Out-of-view | Relocation | Counterfactual | Avg

A
B
C

--------------------------------------------------

Table 2：

Memory Length | Occlusion | Out-of-view | Relocation

1
2
4
8
16

--------------------------------------------------

Table 3：

Model | Inference VGGT | Success | FPS / latency

A | No
B | No
C | No

证明 teacher-free deployment。

--------------------------------------------------

十七、建议额外做两个关键测试

Test A：Memory Reset

同一个 C checkpoint。

正常模式：
使用历史 memory。

测试模式：
每一步都 clear/reset memory。

比较 success rate。

这个实验非常重要。

如果 reset memory 后：

Occlusion
Out-of-view

性能显著下降，

可以直接证明 C 确实用了历史。

--------------------------------------------------

Test B：Wrong History

只在 evaluation 做。

把当前 episode 的 history 替换成另一个 episode 的 history。

不要用于主训练。

比较：

correct history

vs

wrong history

如果 wrong history 明显降低性能，
说明策略不是忽略 memory。

注意：
必须保证这种测试不会污染正式指标，单独记录。

--------------------------------------------------

十八、Camera / domain randomization 第二阶段再做

前三组跑通以后，再增加：

camera yaw
camera pitch
camera translation
FOV
lighting
texture
background
object scale
RGB noise
motion blur

不要一开始同时开很多 domain randomization。

否则无法区分提升来自 spatial memory 还是增强数据。

建议独立一个：

sim2real_randomization=true

的实验。

--------------------------------------------------

十九、禁止事项

1. 不允许 inference 调 VGGT。
2. 不允许为了让 C 赢而给 C 更多数据。
3. A/B/C 不允许使用不同训练步数。
4. 不要修改 action horizon 来制造性能差异。
5. 不要只看 training loss。
6. 不要仅跑 C。
7. 不要只在普通 LIBERO task 上报告结果。
8. 不要让 occlusion 从第一帧就开始，否则历史没有信息可记。
9. relocation 后不要让任务答案仍然由旧位置决定。
10. counterfactual pair 不要同时修改 object identity、camera、背景等大量因素。

--------------------------------------------------

二十、每次实验需要输出什么

每次训练结束后统一输出目录：

experiment_name/
    config.yaml
    train.log
    checkpoint/
    eval/
        normal.json
        occlusion.json
        out_of_view.json
        relocation.json
        counterfactual.json
    diagnostics/
        gate.csv
        memory_attention.*
    videos/
    summary.json

summary.json 至少包含：

model_type
seed
training_steps
dataset_size
memory_length
use_spatial_forcing
use_spatial_memory
geo_loss_weight
overall_success
normal_success
occlusion_success
out_of_view_success
relocation_success
counterfactual_success

--------------------------------------------------

二十一、优先级

如果资源有限，严格按照下面顺序：

P0：

C 模型
Normal vs Occlusion
memory length 1 vs 16

先证明 history 有用。

P1：

Dynamic object relocation

证明 memory 能更新，而不是只会记忆。

P2：

A/B/C matched comparison

证明：

B-A =
geometry representation benefit

C-B =
temporal spatial memory benefit

P3：

Counterfactual relation training。

P4：

camera/domain randomization。

--------------------------------------------------

二十二、第一轮最小可执行版本

如果希望最快出结果，只需要先做：

任务：

next_to_the_plate
between_the_plate_and_the_ramekin

模型：

B
C

C memory：

1
16

训练数据：

normal
occlusion

Evaluation：

每种配置至少 100 rollout，或按现有评测规范使用相同 episode 数。

得到：

B
C-memory1
C-memory16

在：

normal
occlusion

上的 success rate。

如果结果表现为：

normal：
三者接近。

occlusion：
C-16 > C-1 ≈ B。

那么这就是非常好的第一批结果。

--------------------------------------------------

二十三、第二轮

加入 relocation。

重点观察：

C-memory16 是否：

在旧位置已经失效后，
能够根据当前视觉覆盖历史信息。

同步分析 gate：

遮挡时偏 history。

relocation 后重新观测时偏 current。

如果成功，这一组可以直接成为论文中的 qualitative analysis。

--------------------------------------------------

二十四、最终你需要给我返回

不要只告诉我“代码改好了”。

返回：

1. 修改了哪些文件。
2. 每个文件改了什么。
3. 新增了哪些 CLI / config 参数。
4. 数据是如何生成的。
5. 一个具体 episode 的时序例子。
6. smoke test 结果。
7. small-overfit 结果。
8. 正式训练命令。
9. evaluation 命令。
10. A/B/C 对应的完整配置。
11. checkpoint 路径。
12. evaluation JSON。
13. success rate 汇总。
14. gate diagnostic。
15. 失败案例。
16. 是否发现任何 data leakage / shortcut。
17. 是否确认 inference 完全没有调用 VGGT。

如果仓库实际结构与上述描述不同，以代码为准，但不要擅自改变实验科学问题。

核心科学问题始终是：

Q1：
geometry supervision 是否提升 spatial representation？

Q2：
history 中保存 spatial state 是否比单帧 spatial feature 更有效？

Q3：
memory 是否能够在遮挡时保留信息？

Q4：
环境变化后 memory 是否能够纠正旧状态？

Q5：
模型是否真的区分空间关系，而不是利用 dataset shortcut？
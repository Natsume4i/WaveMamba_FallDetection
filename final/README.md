
# DualMCN 论文写作说明文档

## 1. 研究目标

本项目主要研究 **跨环境 CSI 跌倒检测**。

核心问题是：CSI 信号对环境变化非常敏感，当训练环境和测试环境不一致时，模型性能容易下降。因此，本工作的重点不是单一环境下的高准确率，而是提升模型在不同环境、不同布置、不同遮挡条件下的跨环境泛化能力。

论文中建议围绕以下主线展开：

> 面向跨环境 CSI 跌倒检测，提出一种结合长程依赖建模和局部时频特征提取的轻量级双分支网络 DualMCN，并在多个公开数据集和自建数据集上验证其跨环境泛化能力。

---

## 2. 模型名称说明

最终模型名称为：

> **DualMCN**

全称：

> **Dual Mamba-Conv Network**

含义如下：

| 缩写 | 含义 |
|---|---|
| Dual | 双分支结构 |
| M | Mamba / BiMamba 分支 |
| C | Convolution / Multi-kernel Conv 分支 |
| N | Network |

需要注意：模型之前曾暂时命名为 **WaveMamba**，但最终论文中建议统一使用 **DualMCN**。

### 改名原因

原名称 WaveMamba 更强调“小波 + Mamba”，但从最终模型设计和实验结果来看，模型的核心特征更准确地说是：

1. 通过双分支结构同时建模不同类型特征；
2. BiMamba 分支负责长程依赖建模；
3. Multi-kernel Conv 分支负责局部时频模式提取；
4. 两个分支的互补性比“小波”本身更能代表模型贡献。

因此，最终改名为 **DualMCN**，更能体现模型的主要设计思想：  
**Dual Mamba-Conv Network**。

论文中请不要再使用 WaveMamba 作为最终模型名。

---

## 3. 模型结构说明

DualMCN 的整体结构如下：

```text
Input CSI amplitude map
    ↓
Stem Encoder
    ↓
Stage 1: DualMC Block
    ↓
Transition
    ↓
Stage 2: DualMC Block
    ↓
Global Average Pooling
    ↓
Linear Classifier
    ↓
Fall / Nonfall
~~~

其中，核心模块是 **DualMC Block**。

DualMC Block 内部主要流程如下：

```text
Input Feature
    ↓
Block-level SE
    ↓
Temporal Haar Analysis
    ↓
Low-frequency Stream      High-frequency Stream
    ↓                     ↓
BiMamba Branch            Multi-kernel Conv Branch
    ↓                     ↓
Long-range Modeling       Local Time-frequency Pattern Extraction
    ↓                     ↓
Fusion
    ↓
Reconstruction
    ↓
Residual Add
    ↓
Output Feature
```

各模块在论文中的解释建议如下：

| 模块                     | 作用                             |
| ------------------------ | -------------------------------- |
| Temporal Haar Analysis   | 将特征分解为低频和高频信息       |
| BiMamba Branch           | 建模长程时序 / 时频依赖          |
| Multi-kernel Conv Branch | 提取局部时频模式                 |
| Fusion & Reconstruction  | 融合双分支特征并恢复到残差空间   |
| Residual Add             | 保持特征传递稳定，提升训练稳定性 |

------

## 4. 模型代码说明

目前主要有两个模型代码文件：

| 文件               | 用途                                               |
| ------------------ | -------------------------------------------------- |
| `dualmcn.py`       | 实验版模型代码，用于训练、对比实验和消融实验       |
| `dualmcn_clean.py` | 纯净版模型代码，用于理解模型结构、辅助写论文和画图 |

说明：

- `dualmcn.py` 保留了消融实验所需的配置项；
- `dualmcn_clean.py` 固定为最终模型结构，去掉了大量实验开关；
- 写论文方法部分和画模型图时，建议优先参考 `dualmcn_clean.py`；
- 实验复现时，应参考 `dualmcn.py` 和统一 run 代码。

------

## 5. 数据集与跨环境设置

本项目涉及三个数据集：

1. CSI-Bench
2. ENetFall
3. ourdata，自建数据集

### 5.1 CSI-Bench

| 项目          | 设置             |
| ------------- | ---------------- |
| 任务          | FallDetection    |
| 输入尺寸      | `500 × 232`      |
| 环境          | E21–E26          |
| 跨环境测试    | E24              |
| 训练 / 验证   | 不包含 E24       |
| 主要测试指标  | `test_cross_env` |
| Batch size    | 32               |
| Learning rate | 1e-3             |
| Epochs        | 120              |

说明：

CSI-Bench 使用 E24 作为跨环境测试环境，训练和验证阶段不包含 E24。

------

### 5.2 ENetFall

| 项目          | 设置                                      |
| ------------- | ----------------------------------------- |
| 输入尺寸      | `625 × 90`                                |
| 训练环境      | meeting room、home lab(L)、home lab(R)    |
| 测试环境      | living room、lecture room                 |
| 主要测试指标  | `test_living_room` 和 `test_lecture_room` |
| Batch size    | 32                                        |
| Learning rate | 1e-3                                      |
| Epochs        | 100                                       |

说明：

ENetFall 有两个跨环境测试集：LivingRoom 和 LectureRoom。
验证集从训练环境中划分，不作为源域测试集。

------

### 5.3 ourdata

| 项目          | 设置                            |
| ------------- | ------------------------------- |
| 数据来源      | 自建数据集                      |
| 样本数        | 667 个 CSV 样本                 |
| 输入尺寸      | 约 `1200 × 1026`                |
| 环境设置      | 同一会议室下三种布置 / 干扰情况 |
| 训练 / 验证   | e1、e2                          |
| 跨环境测试    | e3                              |
| Batch size    | 16                              |
| Learning rate | 2e-4                            |
| Epochs        | 100                             |

说明：

ourdata 是自建数据集，主要用于验证模型在真实自采场景下的跨环境鲁棒性。
e3 作为跨环境测试场景。
该数据集的输入时间长度不完全统一，训练时进行了统一长度处理。

------

## 6. 模型保存与评估策略

需要在论文实验设置中说明：三个数据集使用的模型保存策略不同。

| 数据集    | 模型保存 / 评估策略                                          |
| --------- | ------------------------------------------------------------ |
| CSI-Bench | 使用验证集最优 EMA 模型，即 `valbest_ema`                    |
| ENetFall  | 使用验证集最优原始模型，即 `valbest_raw`                     |
| ourdata   | 使用验证集 Top-5 最优模型平均，并进行 BN recalibration，即 `top5avg_bnrecal` |

建议论文中写成：

> 为保证不同数据集实验设置与原始训练逻辑一致，本文在三个数据集上采用了对应的模型选择策略：CSI-Bench 使用验证集最优的 EMA 模型，ENetFall 使用验证集最优的原始模型，自建数据集使用验证集 Top-5 最优模型平均并进行 BN 统计重校准。

英文可写为：

> For each dataset, we followed its corresponding training protocol and checkpoint selection strategy. CSI-Bench uses the validation-best EMA checkpoint, ENetFall uses the validation-best raw checkpoint, and our self-collected dataset uses top-5 validation-best checkpoint averaging with BN recalibration.

------

## 7. 数据增强实验设置

数据增强是本文的重要实验内容之一。

实验分为三类：

| 实验组        | 是否使用数据增强 | 作用             |
| ------------- | ---------------- | ---------------- |
| compare_aug   | 使用             | 主对比实验       |
| compare_noaug | 不使用           | 验证数据增强影响 |
| ablation_aug  | 使用             | 消融实验默认设置 |

说明：

- 对比实验同时整理了有增强和无增强两组；
- 消融实验统一使用增强版本；
- 数据增强由各数据集对应的 dataset 代码实现；
- 数据增强不一定在所有模型和所有数据集上都绝对提升，但整体用于增强跨环境鲁棒性。

------

## 8. 实验结果表格组织方式

每个数据集最终整理三张表：

1. 消融实验表；
2. 对比实验表 with augmentation；
3. 对比实验表 without augmentation。

### 8.1 消融实验表

每个数据集消融实验保留四个核心模型：

| 模型                          | 说明                               |
| ----------------------------- | ---------------------------------- |
| `dualmcn_default`             | 最终 DualMCN                       |
| `dualmcn_stem_se_no_block_se` | 对比 SE 放置策略                   |
| `dualmcn_no_wavelet`          | 验证 Temporal Haar Analysis 的影响 |
| `dualmcn_main_only`           | 验证 Multi-kernel Conv 分支贡献    |

说明：

为了降低论文解释成本，消融实验只保留与模型核心设计最相关的四个变体。

------

### 8.2 对比实验表

对比模型包括：

| 模型     |
| -------- |
| DualMCN  |
| ResNet18 |
| LSTM     |
| GRU      |
| TCN      |

说明：

- Vim 虽然在配置中保留过，但由于环境依赖和复现稳定性问题，当前最终表格中暂不放入；
- 对比表分为 with augmentation 和 without augmentation 两组；
- 如果某个数据集的 noaug 下 DualMCN 尚未补跑，可以先不放 DualMCN，后续补齐。

------

## 9. 指标说明

### 9.1 CSI-Bench 和 ourdata

主要报告：

| 指标           | 含义                |
| -------------- | ------------------- |
| Test Acc       | 跨环境测试准确率    |
| Test Macro-F1  | 跨环境测试宏平均 F1 |
| Fall Recall    | 跌倒类召回率        |
| Nonfall Recall | 非跌倒类召回率      |

### 9.2 ENetFall

ENetFall 有两个测试环境，需要分别报告：

| 指标                       |
| -------------------------- |
| LivingRoom Acc             |
| LivingRoom Macro-F1        |
| LivingRoom Fall Recall     |
| LivingRoom Nonfall Recall  |
| LectureRoom Acc            |
| LectureRoom Macro-F1       |
| LectureRoom Fall Recall    |
| LectureRoom Nonfall Recall |

说明：

跌倒检测任务中，Macro-F1 和类别召回率非常重要。
尤其是 Fall Recall，关系到跌倒样本是否被漏检，不能只看 Accuracy。

------

## 10. 结果解读建议

### 10.1 不要只看验证集结果

多个模型在验证集上表现很高，但跨环境测试性能不一定好。

论文中应重点分析：

- `test_cross_env`
- `test_living_room`
- `test_lecture_room`

而不是只看 `val_macro_f1`。

------

### 10.2 不要写 DualMCN 在所有指标上都第一

更稳妥的写法是：

> DualMCN 在多个数据集和跨环境测试设置下表现出较好的综合泛化能力，尤其在 Macro-F1 和类别召回率平衡方面具有优势。

不要写成：

> DualMCN 在所有指标上均取得最优结果。

因为部分数据集、部分环境或部分指标上，某些 baseline 可能局部更高。

------

### 10.3 ourdata 结果偏低是正常现象

ourdata 的跨环境结果整体低于 ENetFall，说明自建数据集更难。

可能原因包括：

- 样本量较小；
- 环境扰动更复杂；
- 存在遮挡物影响；
- 家具移动导致 CSI 分布变化；
- 更接近真实部署场景。

这可以作为论文讨论部分的内容。

------

### 10.4 ENetFall 需要同时看两个测试环境

ENetFall 有 LivingRoom 和 LectureRoom 两个测试环境。
论文中不要只挑一个环境说明，需要分别报告两个环境的结果。

------

## 11. 模型图说明

模型图建议画成两部分：

### Figure (a): Overall Architecture

展示整体流程：

```text
Input CSI → Stem Encoder → Stage 1 → Transition → Stage 2 → GAP → Classifier
```

### Figure (b): DualMC Block

展示核心模块：

```text
Block SE
→ Temporal Haar Analysis
→ BiMamba Branch / Multi-kernel Conv Branch
→ Fusion
→ Reconstruction
→ Residual Add
```

图中不建议画：

- 所有 BN / ReLU；
- fallback linear；
- `F.interpolate`；
- 复杂消融开关；
- 过多 tensor shape。

另外需要注意：
目前已有的两张模型图是由 AI 生成的，只能作为结构和风格参考，不建议直接放入最终论文。正式论文中的图需要重新绘制.

------

## 12. 写作注意事项总结

论文负责人需要特别注意以下几点：

1. 最终模型名为 **DualMCN**，不是 WaveMamba。
3. 论文方法部分重点解释 Mamba-Conv 双分支互补建模。
4. 三个数据集的 checkpoint 策略不同，需要在实验设置中写清楚。
5. 对比实验要区分 with augmentation 和 without augmentation。
6. 消融实验只展示四个核心变体，降低解释成本。
7. 结果分析重点看跨环境测试，不要只看验证集。
8. 不要声称 DualMCN 所有指标都第一，应强调综合泛化能力和稳定性。
9. ENetFall 需要分别报告 LivingRoom 和 LectureRoom。
10. AI 生成的模型图仅供参考，最终论文图需要重新绘制。


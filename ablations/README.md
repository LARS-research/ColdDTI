# ColdDTI 消融实验

本目录统一提供修订稿中 Figure 3、Table 2、Figure 5 对应的实验入口。
默认数据集为 DrugBank；实验划分使用现成的 train/val/test CSV，本目录不重新划分数据。
默认训练 seed=57，边界随机 seed=57。

当前数据已按统一结构过滤和编号更新。先阅读 [完整复现流程](../README.md)，
并通过 `get_emb.py` 重算特征；`feature_status.json` 为 false 时不允许混用旧数组。

## 论文与代码对应

| 论文实验 | 参数 `--variant` | 实现 |
| --- | --- | --- |
| 完整模型 / Origin | `Full` | 复用原模型的投影、token 后处理、交互分数尺度及融合公式 |
| Figure 3(a), w/o Secondary | `wo_secondary` | 去掉二级结构的两个交互模块及对应融合分支 |
| Figure 3(a), w/o Tertiary | `wo_tertiary` | 去掉三级结构的两个交互模块及对应融合分支 |
| Figure 3(a), w/o Quaternary | `wo_quaternary` | 去掉全局蛋白表示的两个交互模块及对应融合分支 |
| Figure 3(b), w/o Local | `wo_local` | 去掉四个药物局部交互模块及局部药物融合分支 |
| Figure 3(b), w/o Global | `wo_global` | 去掉四个药物全局交互模块及全局药物融合分支 |
| Table 2, + Self-Attn | `self_attention` | 在每个二级、三级结构单元内部，对 token 表示做额外注意力后再池化 |
| Figure 5, Annotated | `annotated_secondary` | 使用原注释的二级结构残基分组作为边界实验基线 |
| Figure 5, Length-matched Random | `random_secondary` | 每条链内随机放置片段，保持片段数量及各片段残基长度 |
| Figure 5, Shuffled | `shuffle_secondary` | 每条链内打乱片段长度顺序及原有间隔长度顺序，改变片段成员 |

`--suite protein / drug / attention / boundary / paper` 可批量运行对应组；
`paper` 包含上表的 10 个变体，完整模型只列一次。

## 实现约定

主入口转发到本目录的统一训练器；旧参数 `Secondary/Tertiary/Quaternary/Local/Global`
分别对应真正的分支删除。被删除分支不参与交互和融合。`Full` 的交互分数缩放为
`ceil(dim)`（默认 704）；self-attention 在二级、三级单元内部均使用 `batch_first=True`。
每个蛋白只生成一次固定边界对照，训练、验证、测试共用。

所有变体从训练开始应用，仅按验证 AUC 选 checkpoint 和早停，最后在测试集评估一次。
共同基础来自 `model/colddti.py`。基础 token 注意力使用 `batch_first=True`。
环境、输入准备、批量实验和预测导出命令见仓库根目录的
[README](../README.md)。

## 文件

```text
ablations/
  run.py                # 单次训练与批量实验入口
  model.py              # 完整模型、五种分支删除、结构内注意力
  boundaries.py         # 两种固定、长度匹配的二级边界对照
  data.py               # 读取现有 CSV 和冻结特征，不重新采样
  summarize.py          # 收集各次结果，计算均值和样本标准差
  __init__.py
  README.md
```

## 输入数据

使用仓库原有环境，无须安装额外依赖，也无须在消融训练中加载 Transformers 编码器权重。
输入为原 `get_emb.py` 生成的固定特征：

```text
<data-root>/<dataset>/
  smiles.csv
  protein.csv
  preprocessing.json
  feature_status.json
  smilestokenized.npy
  smilesembeddings.npy
  proteinstokenized.npy
  proteinsembeddings.npy
  <split>/
    train.csv
    val.csv
    test.csv
```

划分 CSV 必须含 `smiles_nid, protein_nid, protein_id, Y`；编号须与同目录实体表和特征数组一致。
缺失的实体特征会报错，不会通过删除样本来改变测试集。
默认特征维度为药物 384、蛋白 1024、交互 704，与原模型一致。
单个样本按变长序列前向，`--batch-size` 控制平均梯度累积的样本数量。

默认 `<data-root>` 为仓库自身的 `data/`。
可通过 `--train-samples 2048 --epochs 5` 做真实数据的短程训练：只从训练集按固定
seed=57 抽样，验证和测试集保持完整，训练 seed 改变时也不更换抽出的训练子集。
先用 `get_emb.py` 生成当前数据对应的全部特征。特征和训练权重仅保留在本地，
由 `.gitignore` 排除，不上传到 Git 源码仓库。

## 运行示例

以下命令在 ColdDTI 仓库根目录执行。

先预览全部运行组合，不加载特征，也不开始训练：

```bash
python ablations/run.py --suite paper \
  --splits cold_drug cold_protein cold_pair --seeds 57 58 59 --dry-run
```

单独训练 w/o Secondary：

```bash
python ablations/run.py --dataset drugbank --split cold_drug \
  --variant wo_secondary --seed 57 --cuda 0 \
  --data-root data
```

运行 Figure 3 的蛋白层级消融，三种划分、三个训练 seed：

```bash
python ablations/run.py --dataset drugbank --suite protein \
  --splits cold_drug cold_protein cold_pair --seeds 57 58 59 --cuda 0
```

其他实验使用同一组参数，将 `--suite protein` 替换为：

```text
--suite drug        # Figure 3(b)
--suite attention  # Table 2
--suite boundary   # Figure 5
--suite paper      # 以上全部，共 90 次训练（3 划分 × 3 seeds × 10 变体）
```

这些批量示例使用默认 `data/`；使用其他目录时追加 `--data-root`。
论文报告三次运行的平均值，但所附正文没有给出三个 seed 的具体值；
57/58/59 是本实现明确选择的运行配置，不声称是原论文的训练 seed。
数据划分仍固定为此前生成的 seed=57，不随训练 seed 改变。

## 边界对照的具体定义

为避免二级结构标记和类型标记影响长度匹配，Figure 5 组的三个变体均只对残基表示池化。
`Full` 为兼容原发布模型保留其含边界 token 的池化，因此边界实验应比较
`annotated_secondary / random_secondary / shuffle_secondary` 三者。

- 使用 ProtBERT 原词表的残基 token ID 5–29，排除特殊 token 和后加的结构 token。
- 结构标记 ID 从经过校验的特征元数据读取；显式参数须与实际 tokenizer ID 一致。
- 片段不重叠，不跨越配置的三级作用域；domain 以外的间隙独立处理以保留其二级单元；保持每个作用域内的片段数量和残基长度多重集合。
- Random 保留片段长度顺序，将剩余残基数均匀随机分配到片段间隔。
- Shuffled 分别打乱原片段长度与间隔长度的顺序；不是仅调整池化后向量的排列。
- 随机发生器使用 `boundary_seed + protein_nid`，不依赖 batch 顺序、训练 seed 或 epoch；默认 boundary_seed=57。
- 没有有效二级片段的蛋白沿用统一均值回退；仅一个覆盖全链片段等情形可能无法改变配置。
  `boundary_counts.json` 记录访问过的蛋白数、无片段数及实际改变的蛋白数。

这两个对照作用于固定缓存表示的**二级结构池化边界**，不会重新排列蛋白序列、
重提取 ProtBERT 特征或删除编码器输入中的原结构标记。因此它们检验的是给定缓存表示下的
片段分组作用，不能解读为彻底移除编码器所见的结构信息。
论文所附正文没有说明随机放置分布、缓存重编码等全部细节；以上是补充后的明确实现，
不能据此保证与已发表图表数值完全一致，也不附带未实际运行的论文结果。

## 输出与汇总

每次训练输出到：

```text
output/ablations/<dataset>/<split>/<variant>/seed_<seed>/
  config.json            # 本次参数
  initial_validation.json # 随机初始化时的验证集指标
  validation.csv         # 每个 epoch 的训练 loss、验证集指标
  best.pt                # 由验证集 AUC 选择的权重，含运行与数据版本约定
  metrics.json           # 最佳模型的最终 test 指标
  test_predictions.csv   # 原测试行、概率和分类结果
  boundary_counts.json
```

汇总文件 `results.csv` 保存逐次结果，`summary.csv` 按数据集、划分、变体、边界 seed
给出 AUC、AUPR、F1、Precision、Recall、Accuracy 的均值和样本标准差，另列运行次数。
只有一个 seed 时标准差留空，不伪装成三次运行的结果。
再次运行同一组合会覆盖该组合的输出；比较不同边界 seed 时应指定不同 `--output-dir`。

```bash
python ablations/summarize.py --output-dir output/ablations
```

默认 t 是链范围聚合，q 是全输入均值。可选 domain 模式读取实际范围；
这些模块名不表示默认已使用完整三级/四级几何输入。旧结果文件属于其当时输入版本。

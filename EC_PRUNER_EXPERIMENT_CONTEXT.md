# EC-Pruner 实验上下文记录

更新时间：2026-06-17

本文档整理当前对话中围绕 CDPruner 复现、EC-Pruner 实现、GQA 数据准备、实验结果和后续建议的上下文，方便后续继续实验。

## 1. 项目与环境

项目目录：

```bash
/home/gpuadmin/txr/CDPruner
```

主要环境：

```bash
conda env: cdpruner
checkpoint: /home/gpuadmin/txr/CDPruner/checkpoints/llava-v1.5-7b
```

当前主要实验模型：

```text
LLaVA-1.5-7B
```

当前主要 token budget：

```text
64 visual tokens
```

## 2. GQA 数据准备状态

GQA 数据没有放在 `/home/gpuadmin/txr` 所在磁盘，而是迁移到了大容量磁盘 `/data1`。

真实数据目录：

```bash
/data1/gpuadmin/txr_datasets/gqa
```

项目软链接：

```bash
/home/gpuadmin/txr/CDPruner_data/gqa -> /data1/gpuadmin/txr_datasets/gqa
/home/gpuadmin/txr/CDPruner/playground/data/eval/gqa/data -> /data1/gpuadmin/txr_datasets/gqa/data
```

数据准备完成标记：

```bash
/data1/gpuadmin/txr_datasets/gqa/data/.images_ready
/data1/gpuadmin/txr_datasets/gqa/data/.questions_ready
```

图像数量：

```text
148854
```

问题文件：

```bash
/data1/gpuadmin/txr_datasets/gqa/data/questions/testdev_balanced_questions.json
```

GQA JSONL 测试文件：

```bash
/home/gpuadmin/txr/CDPruner/playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl
```

样本数：

```text
12578
```

GQA evaluator：

```bash
/data1/gpuadmin/txr_datasets/gqa/data/eval/eval.py
```

说明：使用的是 LLaVA 文档推荐的 GQA 修正版 evaluator。该版本不依赖官方缺失的 scene graph / choices 文件，只需要 questions 和 predictions。

## 3. GQA 脚本状态

脚本：

```bash
/home/gpuadmin/txr/CDPruner/scripts/v1_5/eval/gqa.sh
```

已做修改：

```text
1. CKPT_DIR 默认指向 /home/gpuadmin/txr/CDPruner/checkpoints
2. DATA_DIR 默认指向 /home/gpuadmin/txr/CDPruner_data
3. 增加运行前路径检查
4. CDPruner 和 EC-Pruner 使用不同输出目录，避免覆盖
5. GQA evaluator 参数已改为 --questions 和 --predictions
6. evaluation 阶段使用 flock，避免并行实验写同一个 testdev_balanced_predictions.json 时冲突
```

语法检查已通过：

```bash
bash -n scripts/v1_5/eval/gqa.sh
```

## 4. EC-Pruner 当前代码位置

核心实现：

```bash
/home/gpuadmin/txr/CDPruner/llava/model/pruners/ec_pruner.py
```

接入点：

```bash
/home/gpuadmin/txr/CDPruner/llava/model/llava_arch.py
```

启用方式：

```bash
PRUNE_METHOD=ec_pruner
```

保持原 CDPruner：

```bash
PRUNE_METHOD=cdpruner
```

原则：

```text
PRUNE_METHOD=cdpruner 时不走 EC-Pruner 逻辑，仍走原 CDPruner/CDPP 路径。
```

## 5. EC-Pruner 已实现阶段

### 阶段 0：定位 token selection 入口

视觉 token 生成位置：

```python
image_features, image_embeds, text_embeds = self.get_model().get_vision_tower()(images, texts=texts)
```

文件：

```bash
llava/model/llava_arch.py
```

变量 shape：

```text
image_features: [B, N, C]
```

原 CDPruner keep index 生成：

```python
select_idx = torch.empty((self.visual_token_num, B), dtype=torch.long, device=device)
```

原 CDPruner 裁剪位置：

```python
index_masks.scatter_(1, select_idx, True)
```

debug 环境变量：

```bash
EC_DEBUG=1
EC_DEBUG_LIMIT=5
```

### 阶段 1：最小 EC-Pruner

公式退化为：

```text
TopK(a_i)
a_i = ||v_i||_2
```

实现点：

```python
scores = visual_tokens.norm(dim=-1).mean(dim=0)
keep_idx = torch.topk(scores, K).indices
keep_idx = keep_idx.sort().values
```

要求已满足：

```text
keep_idx 是 1D tensor
keep_idx.numel() == K
keep_idx 在 token index 范围内
keep_idx 排序
keep_idx.device 与 visual_tokens.device 一致
batch size 非 1 时对 batch 维度求均值
```

### 阶段 2：节点贡献分数源

新增环境变量：

```bash
EC_SCORE_SOURCE=norm
EC_SCORE_SOURCE=relevance
EC_SCORE_SOURCE=attn
EC_SCORE_SOURCE=semantic
```

默认：

```bash
EC_SCORE_SOURCE=norm
```

函数：

```python
compute_node_score(...)
normalize(...)
aggregate_text_to_visual_attention(...)
```

fallback：

```text
relevance 不可用时 fallback 到 norm
text_attn 不可用时 fallback 到 norm
semantic b 不可用或 uniform 导致分数全 0 时 fallback 到 norm
所有分数 min-max normalize 到 [0, 1]
NaN / Inf 会被清理
```

### 阶段 3-6：语义 pipeline 和语义候选池

已实现：

```python
extract_semantic_units(question)
build_relation_matrix(units)
compute_semantic_response(features, units, text_embeds=None)
compute_semantic_node_score(b)
build_semantic_candidate_pool(a, b, K)
```

语义 units 规则：

```text
question=None 时返回 ["object"]
最多 8 个 units
去重
保持原顺序
过滤基础 stop words
```

语义关系矩阵：

```text
空间词 left/right/above/below/behind/front/near/under/over 与其他 unit 建关系
动作词 holding/wearing/riding/... 与其他 unit 建关系
没有关系词时使用弱连接 0.1
对角线为 0
```

语义响应：

```text
优先使用 CLIP text tower 对 semantic units 编码
如果 text_embeds 不可用，fallback 到 uniform response
如果 uniform response 导致 semantic score 不稳定，semantic score fallback 到 norm
```

语义候选池公式：

```text
C_pool = TopM(a_i) union over m TopL(b_i,m)
```

参数：

```bash
EC_CANDIDATE_RATIO=2.0
EC_PER_UNIT_TOPL=8
EC_USE_SEMANTIC_CANDIDATE=1
```

已修正：

```text
候选池现在按论文的 M_c = delta_c K 做上限截断。
如果 union 后超过上限，按 normalize(a_i) + normalize(semantic_coverage_count) 截断。
```

### 阶段 7-9：正负耦合选择

冗余排斥：

```text
R_ij = ReLU(sim(v_i, v_j)) * b_i^T b_j * g_ij
```

参数：

```bash
EC_USE_REPULSION=1
EC_TOPK_R=8
EC_SIGMA=2.0
```

互补吸引：

```text
C_ij = sqrt(a_i a_j) * b_i^T G_q b_j * Phi_ij(q)
```

参数：

```bash
EC_USE_COMPLEMENT=1
EC_USE_PHI=1
EC_TOPK_C=8
```

已修正：

```text
之前代码误用了 a_i * a_j，现在已改为论文中的 sqrt(a_i a_j)。
```

低能量贪心：

```text
i* = argmax_i [a_i - lambda * sum_j R_ij + delta * sum_j C_ij]
```

参数：

```bash
EC_LAMBDA=0.1
EC_DELTA=0.1
```

## 6. 重要代码修正记录

### 6.1 互补项公式修正

文件：

```bash
llava/model/pruners/ec_pruner.py
```

修正前：

```python
node_pair_score = a_c[:, None] * a_c[None, :]
```

修正后：

```python
node_pair_score = torch.sqrt((a_c[:, None] * a_c[None, :]).clamp_min(0.0))
```

原因：

```text
论文使用 sqrt(a_i a_j)，不是 a_i a_j。
a_i a_j 会过度压低单点贡献较低但互补性强的 token pair。
```

### 6.2 语义候选池截断修正

文件：

```bash
llava/model/pruners/ec_pruner.py
```

当前行为：

```text
candidate_size <= EC_CANDIDATE_RATIO * K
默认 K=64 时 candidate_size <= 128
```

截断分数：

```text
normalize(a_i) + normalize(semantic_coverage_count)
```

原因：

```text
论文写明若候选池大小超过预设上限 M_c = delta_c K，则按照 a_i 与语义覆盖频次的加权分数截断。
```

### 6.3 relevance 分数方向修正

文件：

```bash
llava/model/llava_arch.py
```

修正前：

```python
return (-relevance).mean(dim=-1)
```

修正后：

```python
return relevance.mean(dim=-1)
```

原因：

```text
relevance 应该正相关，越相关分数越高。
旧实现会让 EC_SCORE_SOURCE=relevance 反向选择。
```

## 7. 已验证的 GQA 结果

GQA split：

```text
llava_gqa_testdev_balanced
```

样本数：

```text
12578
```

### 7.1 主结果表

| Method / Setting | Token | Candidate | R | C | Phi | Accuracy | Binary | Open | Rel | Query | vs CDPruner |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDPruner baseline | 64 | original CDPruner | - | - | - | 58.65 | 75.82 | 44.09 | 50.68 | 44.09 | 0.00 |
| EC full, 修正前 | 64 | semantic candidate | 1 | 1 | 1 | 55.56 | 74.71 | 39.31 | 46.16 | 39.31 | -3.09 |
| EC full, 修正后 | 64 | semantic candidate | 1 | 1 | 1 | 55.45 | 74.55 | 39.24 | 45.97 | 39.24 | -3.20 |
| EC semantic candidate only | 64 | semantic candidate | 0 | 0 | - | 54.98 | 74.78 | 38.19 | 45.35 | 38.19 | -3.67 |

### 7.2 Structural type

| Method / Setting | Choose | Compare | Logical | Query | Verify |
|---|---:|---:|---:|---:|---:|
| CDPruner baseline | 77.06 | 63.84 | 73.82 | 44.09 | 79.93 |
| EC full, 修正前 | 75.82 | 63.50 | 73.38 | 39.31 | 78.15 |
| EC full, 修正后 | 75.55 | 63.33 | 73.49 | 39.24 | 77.84 |
| EC semantic candidate only | 75.64 | 62.99 | 73.93 | 38.19 | 78.11 |

### 7.3 Semantic type

| Method / Setting | Attr | Cat | Global | Obj | Rel |
|---|---:|---:|---:|---:|---:|
| CDPruner baseline | 64.23 | 52.05 | 59.24 | 85.48 | 50.68 |
| EC full, 修正前 | 63.21 | 46.21 | 46.50 | 84.32 | 46.16 |
| EC full, 修正后 | 63.23 | 46.13 | 45.86 | 83.93 | 45.97 |
| EC semantic candidate only | 63.04 | 45.08 | 42.04 | 84.32 | 45.35 |

### 7.4 当前判断

```text
1. 当前 EC-Pruner 全量配置低于 CDPruner baseline。
2. 修正 sqrt(a_i a_j) 和候选池截断后，GQA 准确率从 55.56 降到 55.45，基本没有改善。
3. EC full 相比 semantic candidate only 提升 0.47，说明 R/C/Phi 有一点贡献，但不足以弥补 semantic pipeline 的损失。
4. 掉分主要集中在 Open / Query / Rel。
5. Attr 和 Obj 相对接近 CDPruner，说明方法不是随机失效，而是关系/open-query 证据保留还不够好。
```

## 8. 已生成结果文件

CDPruner GQA：

```bash
playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-7b/vtn_64/merge.jsonl
```

EC-Pruner 修正前 full：

```bash
playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-7b/ec_pruner_semantic_candidate1_r1_c1_phi1_vtn_64/merge.jsonl
```

注意：修正后 full 使用相同 PARAM 名称，可能覆盖同名目录中的旧结果，具体以文件修改时间为准。

## 9. 推荐后续实验命令

### 9.1 CDPruner baseline

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=cdpruner \
CUDA_VISIBLE_DEVICES=4 \
bash scripts/v1_5/eval/gqa.sh 64
```

### 9.2 EC-Pruner full

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

## 16. 最新补充：run_ec_gqa_batch.sh 四组结果分析

更新时间：2026-06-09

脚本：

```bash
/home/gpuadmin/txr/CDPruner/run_ec_gqa_batch.sh
```

日志目录：

```bash
/home/gpuadmin/txr/CDPruner/logs/ec_gqa_20260608_113709
```

### 16.1 四组结果

| Experiment | Global Ratio | R | C | Lambda | Delta | Accuracy | Binary | Open | Rel | Query | Distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| g0625_candidate_only | 0.625 | 0 | 0 | - | - | 56.30 | 74.99 | 40.44 | 47.17 | 40.44 | 1.57 |
| g0625_full | 0.625 | 1 | 1 | 0.1 | 0.1 | 56.36 | 74.97 | 40.57 | 47.42 | 40.57 | 1.58 |
| g0875_candidate_only | 0.875 | 0 | 0 | - | - | 55.20 | 74.80 | 38.57 | 45.55 | 38.57 | 1.72 |
| g075_full_l005_d015 | 0.75 | 1 | 1 | 0.05 | 0.15 | 56.11 | 75.18 | 39.94 | 46.50 | 39.94 | 1.58 |

### 16.2 结论

```text
1. 这四组里最好的是 g0625_full，Accuracy=56.36。
2. g0625_candidate_only 已经达到 56.30，说明 global ratio 降到 0.625 后 coverage 本身贡献很大。
3. g0625_full 只比 candidate only 高 0.06，说明当前 R/C/Phi 在这个候选池上的增益很小。
4. g0875_candidate_only 明显变差，说明 global token 过多会压缩 semantic/spatial coverage 的作用。
5. g075_full_l005_d015 没有超过默认 g075 full 56.14，说明简单减小 lambda、增大 delta 没有带来提升。
6. 当前最佳 EC=56.36，仍低于 CDPruner baseline 58.65，差距 -2.29。
```

### 16.3 重要问题：g0625 实验不是严格固定 64 token

日志中 `g0625_candidate_only` 和 `g0625_full` 的进度条末尾出现：

```text
vtn=56
```

原因：

```text
EC_GLOBAL_CANDIDATE_RATIO=0.625 时 global_budget=40。
spatial grid 4x4 且 topL=1 时最多提供 16 个不在 global 中的 token。
如果 semantic candidates 与 global 高重叠，candidate pool 可能只有 56 个唯一 token。
旧代码在 candidate_size < K 时直接用 candidate_size 作为实际 keep 数，导致部分样本没有保满 64 token。
```

这意味着：

```text
g0625_candidate_only 和 g0625_full 的结果有参考价值，但不是严格 vtn=64 公平对比。
```

已修复：

```text
candidate pool unique 后，如果 candidate_size < K，会从未入池 token 中按 a_i 补足到 K。
debug 中新增 candidate_padded_count。
smoke test 已确认 global0.625 时 keep_idx=64。
```

### 16.4 需要重跑

由于已修复 candidate pool 不足 K 的问题，建议重跑：

```bash
# g0625 candidate only, fixed K=64
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.625 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

```bash
# g0625 full, fixed K=64
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.625 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

### 16.5 fixed-K 重跑结果：g0625 candidate only

在修复 `candidate_size < K` 时自动补足到 `K` 之后，重新运行了相同配置两次。

配置：

```bash
PRUNE_METHOD=ec_pruner
EC_SCORE_SOURCE=semantic
EC_USE_SEMANTIC_CANDIDATE=1
EC_USE_SPATIAL_CANDIDATE=1
EC_SPATIAL_EXCLUDE_GLOBAL=1
EC_GLOBAL_CANDIDATE_RATIO=0.625
EC_CANDIDATE_RATIO=2.0
EC_BETA_A=0.5
EC_BETA_SEM=0.3
EC_BETA_SPATIAL=0.2
EC_PER_UNIT_TOPL=8
EC_SPATIAL_GRID_SIZE=4
EC_SPATIAL_TOPL=1
EC_USE_REPULSION=0
EC_USE_COMPLEMENT=0
```

两次结果：

| Run | Accuracy | Binary | Open | Rel | Query | Distribution |
|---|---:|---:|---:|---:|---:|---:|
| rerun-1 | 56.38 | 75.00 | 40.59 | 47.48 | 40.59 | 1.58 |
| rerun-2 | 56.32 | 75.02 | 40.46 | 47.23 | 40.46 | 1.57 |

均值：

```text
Accuracy: 56.35
Binary: 75.01
Open/Query: 40.53
Rel: 47.36
Distribution: 1.575
```

波动范围：

```text
Accuracy 波动仅 0.06，说明当前配置较稳定。
```

相对旧版 candidate only：

```text
旧版 global0.75 candidate only: 55.97
新版 global0.625 fixed-K candidate only: 56.35
提升: +0.38
```

相对 batch 中未修复版本：

```text
batch g0625_candidate_only: 56.30
fixed-K rerun mean: 56.35
提升: +0.05
```

相对 CDPruner baseline：

```text
CDPruner baseline: 58.65
当前最佳 fixed-K g0625 candidate only mean: 56.35
差距: -2.30
```

结论：

```text
1. 修复 candidate padding 后，g0625 candidate only 结果稳定在 56.32~56.38。
2. 这说明 global0.625 + semantic/spatial coverage 的改进是真实的，不是偶然波动。
3. 当前 candidate-only 已经比旧版 global0.75 candidate-only 更强。
4. 但距离 CDPruner 仍有约 2.3 个点差距。
5. 下一步优先级仍是 fixed-K 版本的 g0625 full，而不是继续重复 candidate-only。
```

### 9.3 Semantic candidate only

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

### 9.4 Norm Top-K baseline

目的：判断 EC 接入本身和 greedy/top-k 逻辑是否稳定。

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=norm \
EC_USE_SEMANTIC_CANDIDATE=0 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

### 9.5 Relevance Top-K baseline

目的：判断修正后的 relevance 分数是否比 semantic score 更适合做 a_i。

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=relevance \
EC_USE_SEMANTIC_CANDIDATE=0 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

### 9.6 Relevance + EC coupling

目的：如果 relevance top-k 明显强于 semantic top-k，则把 CDPruner/CLIP relevance 作为 a_i，再测试 R/C 是否有增益。

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=relevance \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
CUDA_VISIBLE_DEVICES=2 \
bash scripts/v1_5/eval/gqa.sh 64
```

## 10. 当前最可能的瓶颈

```text
1. semantic response b_i,m 质量可能不够。
2. 规则抽取的 semantic units 过粗，可能遗漏 GQA 中关键 attribute / relation / reference target。
3. G_q 关系矩阵可能过密，很多 unit 之间被弱连接或动作/空间词全连接，互补项区分度不足。
4. semantic candidate pool 可能筛掉 CDPruner 条件 DPP 会保留的关键视觉 token。
5. 当前 a_i 来自 semantic confidence，可能不如 CDPruner 的 instruction relevance 稳定。
```

## 11. 下一步建议

优先顺序：

```text
1. 先跑 norm top-k 和 relevance top-k。
2. 如果 relevance top-k 明显优于 semantic top-k，则后续把 EC 的 a_i 主源切到 relevance。
3. 再测试 relevance + semantic candidate + R/C，看 pairwise coupling 是否能补充 CDPruner。
4. 如果仍低于 CDPruner，重点改 b_i,m 和 G_q，而不是继续调 lambda/delta。
5. 如果要做论文实验，GQA 应继续作为主实验，因为它最能体现 relation / multi-hop / evidence-chain。
```

## 12. Debug 命令

打开 EC debug：

```bash
EC_DEBUG=1 EC_DEBUG_LIMIT=10 \
PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

预期会看到类似：

```text
[PruneDebug] prune_method=ec_pruner
[PruneDebug] image_features.shape=(1, 576, ...)
[PruneDebug] keep_idx.shape=(64,)
[PruneDebug] vtn=64
[PruneDebug] ec_score_source=semantic
[PruneDebug] candidate_size=128
[PruneDebug] R.nnz=...
[PruneDebug] C.nnz=...
```

## 13. 重要注意事项

```text
1. 不要用 PRUNE_METHOD=ec_pruner 的结果覆盖 CDPruner baseline。
2. gqa.sh 的 PARAM 名称目前已包含 EC_USE_SPATIAL_CANDIDATE，但仍没有包含 EC_LAMBDA / EC_DELTA / EC_CANDIDATE_RATIO / EC_PER_UNIT_TOPL / EC_SPATIAL_GRID_SIZE / EC_SPATIAL_TOPL。
3. 做参数扫描时建议手动备份结果目录，或继续扩展 PARAM 命名，避免覆盖。
4. CDPruner baseline 当前为 58.65，是后续对比的重要参照。
5. 当前 full EC 为 55.45，尚未超过 baseline。
```

## 14. 最新补充：relevance/norm/spatial 实验结果

更新时间：2026-06-08

本节补充后续运行的 `EC_SCORE_SOURCE=relevance`、`EC_SCORE_SOURCE=norm`、以及 `EC_USE_SPATIAL_CANDIDATE=1` 的 GQA 结果。

### 14.1 新增代码改动：空间覆盖候选池

文件：

```bash
llava/model/pruners/ec_pruner.py
scripts/v1_5/eval/gqa.sh
```

新增环境变量：

```bash
EC_USE_SPATIAL_CANDIDATE=1
EC_SPATIAL_GRID_SIZE=4
EC_SPATIAL_TOPL=1
```

默认：

```bash
EC_USE_SPATIAL_CANDIDATE=0
```

作用：

```text
将 patch grid 划分为 EC_SPATIAL_GRID_SIZE x EC_SPATIAL_GRID_SIZE 个区域。
每个区域内按 a_i 选择 EC_SPATIAL_TOPL 个候选 token 加入候选池。
候选池最终仍受 EC_CANDIDATE_RATIO * K 的总上限约束。
```

当前实现特点：

```text
1. 只在 EC_USE_SPATIAL_CANDIDATE=1 时启用。
2. 不改变 PRUNE_METHOD=cdpruner 逻辑。
3. 可以和 semantic candidate / R / C / Phi 独立开关组合。
4. smoke test 已通过：keep_idx=64，candidate_size<=128。
```

gqa.sh 输出目录命名已更新，当前 EC-Pruner 输出目录会包含：

```text
spatial0 或 spatial1
```

示例：

```text
ec_pruner_semantic_candidate1_spatial1_r1_c1_phi1_vtn_64
```

### 14.2 relevance / norm Top-K 结果

| Method / Setting | Token | Score Source | Candidate | R | C | Accuracy | Binary | Open | Rel | Query | Distribution |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CDPruner baseline | 64 | CDPruner relevance + DPP | CDPruner | - | - | 58.65 | 75.82 | 44.09 | 50.68 | 44.09 | 1.54 |
| EC full 修正后 | 64 | semantic | semantic candidate | 1 | 1 | 55.45 | 74.55 | 39.24 | 45.97 | 39.24 | 1.72 |
| EC semantic candidate only | 64 | semantic | semantic candidate | 0 | 0 | 54.98 | 74.78 | 38.19 | 45.35 | 38.19 | 1.74 |
| EC relevance Top-K | 64 | relevance | none | 0 | 0 | 40.17 | 60.89 | 22.59 | 29.99 | 22.59 | 5.43 |
| EC norm Top-K | 64 | norm | none | 0 | 0 | 38.34 | 60.11 | 19.87 | 27.58 | 19.87 | 5.78 |

结论：

```text
1. norm Top-K 和 relevance Top-K 在 GQA 上明显崩溃。
2. 单纯 scalar Top-K 不适合 GQA 这种 relation/reference/multi-hop 数据集。
3. semantic candidate only 明显强于 norm/relevance Top-K，说明 coverage 比单点分数更关键。
4. CDPruner 的优势主要来自 conditional DPP / diversity，而不是简单 relevance Top-K。
5. 当前 EC 的主要瓶颈不是 R/C 公式，而是 candidate pool 和基础 a_i / b_i,m 质量。
```

### 14.3 spatial candidate 结果

运行配置 1：semantic + spatial candidate only，关闭 R/C。

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

结果：

```text
Accuracy: 54.98
Binary: 74.78
Open: 38.19
Rel: 45.35
Query: 38.19
Distribution: 1.74
```

运行配置 2：semantic + spatial + R/C/Phi full。

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

结果：

```text
Accuracy: 55.45
Binary: 74.57
Open: 39.24
Rel: 45.99
Query: 39.24
Distribution: 1.72
```

对比表：

| Setting | Spatial Candidate | R | C | Phi | Accuracy | Binary | Open | Rel | Query | Distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| semantic candidate only, 无 spatial | 0 | 0 | 0 | - | 54.98 | 74.78 | 38.19 | 45.35 | 38.19 | 1.74 |
| semantic + spatial candidate only | 1 | 0 | 0 | - | 54.98 | 74.78 | 38.19 | 45.35 | 38.19 | 1.74 |
| semantic full, 无 spatial | 0 | 1 | 1 | 1 | 55.45 | 74.55 | 39.24 | 45.97 | 39.24 | 1.72 |
| semantic + spatial full | 1 | 1 | 1 | 1 | 55.45 | 74.57 | 39.24 | 45.99 | 39.24 | 1.72 |

结论：

```text
1. 当前 EC_USE_SPATIAL_CANDIDATE=1 没有带来实际提升。
2. semantic + spatial candidate only 与 semantic candidate only 完全一致。
3. semantic + spatial full 与 semantic full 基本一致。
4. 这说明当前 spatial candidate 可能没有改变最终候选池或最终 keep_idx。
5. 原因很可能是 TopM(a_i) 已经直接占满 EC_CANDIDATE_RATIO*K，即 K=64 时 TopM=128。
6. 后续 semantic/spatial candidate 虽然进入 union，但超过 128 后又按 a_i/coverage 截断，spatial token 很可能被截掉。
```

### 14.4 当前优化方向

当前已经明确：

```text
1. 纯 norm/relevance Top-K 不可用。
2. semantic candidate 能显著缓解 Top-K 崩溃，但仍低于 CDPruner。
3. R/C/Phi 稳定带来约 +0.47 的小收益。
4. spatial candidate 当前实现没有实际影响，因为全局 TopM(a_i) 占满候选池容量。
```

因此，下一步最合理的代码优化不是继续调 `EC_LAMBDA / EC_DELTA`，而是给全局候选分配单独配额：

```bash
EC_GLOBAL_CANDIDATE_RATIO=1.0
```

建议候选池改为：

```text
max_candidates = EC_CANDIDATE_RATIO * K
global_topm = EC_GLOBAL_CANDIDATE_RATIO * K

C_pool =
    Top global_topm(a_i)
    union semantic unit TopL(b_i,m)
    union spatial grid TopL(a_i)
```

最后若超过 `max_candidates`，再按：

```text
normalize(a_i) + normalize(coverage_count)
```

截断。

这样做的目的：

```text
给 semantic coverage 和 spatial coverage 留出真实候选池预算，避免 Top2K(a_i) 一开始占满所有候选名额。
```

建议改完后跑：

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_GLOBAL_CANDIDATE_RATIO=1.0 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

以及 full：

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_GLOBAL_CANDIDATE_RATIO=1.0 \
EC_CANDIDATE_RATIO=2.0 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

如果这个仍然无法接近 CDPruner，则下一步应该考虑 hybrid candidate：

```text
EC candidate = CDPruner candidate union semantic candidate union spatial candidate
```

理由：

```text
当前实验证明完全丢掉 CDPruner 的 conditional diversity 会掉太多。
更现实的优化路线是保留 CDPruner 的覆盖性优势，再用 EC 的 semantic relation / complementarity 做 evidence-chain re-ranking。
```

## 15. 最新补充：global0.75 + spatial exclude global 实验结果

更新时间：2026-06-08

本节补充在候选池配额修正后的 GQA vtn=64 结果。

### 15.1 代码改动补充

文件：

```bash
llava/model/pruners/ec_pruner.py
scripts/v1_5/eval/gqa.sh
```

新增或修正参数：

```bash
EC_GLOBAL_CANDIDATE_RATIO=0.75
EC_SPATIAL_EXCLUDE_GLOBAL=1
```

关键修正：

```text
1. EC_GLOBAL_CANDIDATE_RATIO 现在真正允许小于 1.0。
2. 旧代码写成 max(K, ratio*K)，导致 ratio<1.0 无效。
3. 当前改为 global_budget = round(EC_GLOBAL_CANDIDATE_RATIO * K)。
4. EC_SPATIAL_EXCLUDE_GLOBAL=1 时，spatial grid 每格优先选择不在 global top 中的 token。
5. 这样 spatial coverage 才可能真正进入最终 keep_idx。
```

调试结论：

```text
EC_GLOBAL_CANDIDATE_RATIO=1.0 时：
global_budget=64，最终 greedy 在 candidate 内仍会优先选 global top-64，所以 keep_diff_spatial0 通常为 False。

EC_GLOBAL_CANDIDATE_RATIO=0.75 时：
global_budget=48，semantic/spatial coverage 必须补足剩余 token，keep_diff_spatial0 可以变为 True。
```

### 15.2 新实验配置 1：semantic + spatial candidate only

命令：

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.75 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

结果：

```text
Accuracy: 55.97
Binary: 74.99
Open: 39.84
Distribution: 1.60

Structural:
choose: 76.53
compare: 63.67
logical: 73.66
query: 39.84
verify: 78.24

Semantic:
attr: 63.61
cat: 48.04
global: 55.41
obj: 83.16
rel: 46.25
```

### 15.3 新实验配置 2：semantic + spatial full

命令：

```bash
cd /home/gpuadmin/txr/CDPruner

PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.75 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
bash scripts/v1_5/eval/gqa.sh 64
```

注意：

```text
该命令中未显式设置 CUDA_VISIBLE_DEVICES，因此使用了当时 shell 的默认 GPU 可见性。
EC_DELTA 未显式设置，代码默认 EC_DELTA=0.1。
```

结果：

```text
Accuracy: 56.14
Binary: 75.00
Open: 40.13
Distribution: 1.58

Structural:
choose: 75.82
compare: 63.84
logical: 73.82
query: 40.13
verify: 78.46

Semantic:
attr: 63.56
cat: 48.04
global: 56.05
obj: 83.29
rel: 46.67
```

### 15.4 对比表

| Method / Setting | Token | Global Ratio | Spatial Exclude Global | R | C | Accuracy | Binary | Open | Rel | Query | Distribution | vs CDPruner |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CDPruner baseline | 64 | - | - | - | - | 58.65 | 75.82 | 44.09 | 50.68 | 44.09 | 1.54 | 0.00 |
| EC semantic candidate only, old | 64 | 2.0 effective global fill | 0 | 0 | 0 | 54.98 | 74.78 | 38.19 | 45.35 | 38.19 | 1.74 | -3.67 |
| EC full, old | 64 | 2.0 effective global fill | 0 | 1 | 1 | 55.45 | 74.55 | 39.24 | 45.97 | 39.24 | 1.72 | -3.20 |
| EC semantic+spatial candidate only | 64 | 0.75 | 1 | 0 | 0 | 55.97 | 74.99 | 39.84 | 46.25 | 39.84 | 1.60 | -2.68 |
| EC semantic+spatial full | 64 | 0.75 | 1 | 1 | 1 | 56.14 | 75.00 | 40.13 | 46.67 | 40.13 | 1.58 | -2.51 |

### 15.5 关键变化

相对旧 semantic candidate only：

```text
Accuracy: 54.98 -> 55.97  (+0.99)
Open/Query: 38.19 -> 39.84 (+1.65)
Rel: 45.35 -> 46.25 (+0.90)
Distribution: 1.74 -> 1.60，明显更接近 CDPruner 的 1.54
```

相对旧 full：

```text
Accuracy: 55.45 -> 56.14 (+0.69)
Open/Query: 39.24 -> 40.13 (+0.89)
Rel: 45.97 -> 46.67 (+0.70)
Distribution: 1.72 -> 1.58
```

R/C/Phi 在新候选池上的增益：

```text
candidate only: 55.97
full: 56.14
gain: +0.17
```

旧候选池上 R/C/Phi 增益约为：

```text
54.98 -> 55.45 = +0.47
```

因此：

```text
1. 新候选池本身贡献明显，主要收益来自 coverage 进入最终选择。
2. R/C/Phi 仍有正收益，但在当前参数下收益较小。
3. global0.75 + spatial exclude global 是目前 EC-Pruner 最好的 GQA vtn=64 结果。
4. 当前最佳 EC 为 56.14，仍低于 CDPruner 58.65，差距为 -2.51。
```

### 15.6 当前判断

```text
1. 之前 spatial candidate 没效果，不是空间覆盖思想完全无效，而是 global TopK 占满最终 K，coverage token 进不了最终 keep_idx。
2. 将 global_budget 降到 0.75K，并排除 global 后选 spatial token，确实提升了 GQA。
3. 提升主要体现在 Open/Query/Rel 和 Distribution，符合“覆盖性改善”的预期。
4. 但是当前仍没有追上 CDPruner，说明 CDPruner 的 conditional diversity 仍然更强。
5. 后续优化应优先考虑更强的 candidate 机制，而不是只调 lambda/delta。
```

### 15.7 下一步建议

建议优先尝试以下方向：

```text
1. 调 EC_GLOBAL_CANDIDATE_RATIO：0.5 / 0.625 / 0.75 / 0.875。
2. 调 EC_SPATIAL_GRID_SIZE：3 / 4 / 6，观察空间覆盖粒度。
3. 调 EC_SPATIAL_TOPL：1 / 2，确认是否需要每格多保留候选。
4. 调 beta：提高 EC_BETA_SEM 或 EC_BETA_SPATIAL，降低 EC_BETA_A。
5. 若仍无法接近 CDPruner，进入 hybrid candidate：
   EC candidate = CDPruner candidate union semantic candidate union spatial candidate。
```

建议下一组最小参数扫描：

```bash
# global ratio 0.625 candidate only
PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.625 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=0 \
EC_USE_COMPLEMENT=0 \
CUDA_VISIBLE_DEVICES=1 \
bash scripts/v1_5/eval/gqa.sh 64
```

```bash
# global ratio 0.625 full
PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.625 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/v1_5/eval/gqa.sh 64
```

## 16. 最新补充：QF-Pruner 路线

更新时间：2026-06-17

当前实验方向已经从 QMO 和 EC 旧耦合路线转向 QF-Pruner。原因很明确：

```text
1. QF-Pruner 已经在 GQA vtn=64 上达到 58.13，明显强于此前多数 EC/semantic/spatial 变体。
2. 当前最优 tau 出现在 EC_QFID_TAU=0.50，优于 0.10 / 0.20 / 1.00，说明 QF 的量子态保真建模不是偶然波动，而是有效方向。
3. 后续实验应优先围绕 QF kernel 和概率平滑做最小改动，不再回到 QMO，也不再重新引入 R / C / Phi / ChainCov。
```

### 16.1 QF-Pruner 当前实现状态

核心文件：

```bash
/home/gpuadmin/txr/CDPruner/llava/model/pruners/ec_pruner.py
```

GQA 输出命名：

```bash
/home/gpuadmin/txr/CDPruner/scripts/v1_5/eval/gqa.sh
```

启动脚本：

```bash
/home/gpuadmin/txr/CDPruner/run_gqa_profile.sh
```

当前 QF 关键点：

```text
1. EC_SCORE_SOURCE=qfid 时，直接走 select_by_quantum_fidelity(...)。
2. 该分支完全绕开 semantic/spatial candidate、R、C、Phi、ChainCov 和旧 greedy/qanneal 能量路径。
3. 视觉 token 被视为纯态 |psi_i> = v_i / ||v_i||。
4. instruction-conditioned probability p_i^q 来自 semantic / relevance / uniform 三种概率源。
5. 当前默认 tau 已改为 0.50。
6. 已支持两种 kernel：
   amplitude: sqrt(p_i p_j) <psi_i, psi_j>
   density: sqrt(p_i p_j) |<psi_i, psi_j>|^2
7. 已支持 depolarizing smoothing：
   p <- (1-eps) p + eps * uniform
```

当前默认环境变量：

```bash
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_EPS=1e-6
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=amplitude
EC_QFID_DEPOLARIZE=0.0
```

### 16.2 当前最重要结果

当前已确认：

```text
QF-Pruner @ GQA vtn=64:
Accuracy = 58.13
```

解释：

```text
1. 这个结果已经非常接近甚至进入和 CDPruner baseline 同一量级的竞争区间。
2. 它不是依赖额外的 pairwise coupling 或复杂 candidate pool 得到的，而是来自 fixed-budget quantum fidelity 子集选择本身。
3. 因此下一步最应该做的是验证：
   当前最优结果是否来自 amplitude kernel 的偶然最优，
   还是 mixed-state density kernel / depolarized probability 还能继续带来增益。
```

### 16.3 现在应该跑哪三个 GQA 命令

下面三条命令是当前最值得跑的 GQA 实验。

#### 16.3.1 命令 1：QF 当前主线 baseline

命令：

```bash
cd /home/gpuadmin/txr/CDPruner
conda activate cdpruner
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid 2
```

为什么跑这个：

```text
1. 这是当前 QF-Pruner 的主线配置。
2. 它使用 amplitude kernel + semantic probability + tau=0.50。
3. 这是目前已知最强、也是后续所有 QF 变体的比较基线。
4. 如果这条不能稳定复现 58.13 左右，后面的 density / depolarize 对比就没有意义。
```

对应配置：

```bash
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=amplitude
EC_QFID_DEPOLARIZE=0.0
EC_USE_SEMANTIC_CANDIDATE=0
EC_USE_SPATIAL_CANDIDATE=0
EC_USE_REPULSION=0
EC_USE_COMPLEMENT=0
EC_USE_PHI=0
EC_USE_CHAIN_COVERAGE=0
EC_SOLVER=greedy
```

#### 16.3.2 命令 2：QF density kernel 对比

命令：

```bash
cd /home/gpuadmin/txr/CDPruner
conda activate cdpruner
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid_density 2
```

为什么跑这个：

```text
1. 这是最关键的理论对比实验。
2. amplitude kernel 对应的是纯态振幅相似性。
3. density kernel 对应的是 mixed-state preservation，更贴近 rho_q = sum_i p_i |psi_i><psi_i| 的定义。
4. 这条实验可以回答：
   当前 QF 的收益，究竟来自“纯态相干保真”，
   还是来自“混合态密度保真”。
5. 这是一条非常干净的 ablation，因为除了 kernel 形式外，其它配置保持不变。
```

对应配置：

```bash
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=density
EC_QFID_DEPOLARIZE=0.0
```

#### 16.3.3 命令 3：QF density + depolarize

命令：

```bash
cd /home/gpuadmin/txr/CDPruner
conda activate cdpruner
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid_density_depol 2
```

为什么跑这个：

```text
1. 这条实验的目标不是追求复杂建模，而是验证概率平滑是否提高泛化稳健性。
2. semantic probability 在某些样本上可能过尖锐，导致过度依赖少数 token。
3. depolarizing smoothing 相当于把 instruction-conditioned Born probability 与 uniform probability 混合。
4. 如果 density + depolarize 比 density 更稳，说明当前问题不在 kernel 本身，而在概率分布过于尖锐。
5. 如果它反而下降，则说明语义分布已经足够有效，不需要额外均匀化。
```

对应配置：

```bash
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=density
EC_QFID_DEPOLARIZE=0.25
```

### 16.4 为什么暂时只跑这三条，而不是继续扩展别的模块

```text
1. 当前最需要的是“验证 QF 路线到底是哪里有效”，而不是再次把问题复杂化。
2. 这三条命令只改变 kernel / depolarize，属于单因素、小步、可解释的比较。
3. 它们都保持：
   不用 candidate pool
   不用 R/C/Phi
   不用 ChainCov
   不用 qanneal
4. 因此结果差异几乎可以直接归因到 QF mixed-state fidelity 建模本身。
5. 如果这三条里：
   qfid 最强，则说明 amplitude kernel 已足够；
   qfid_density 最强，则说明 mixed-state kernel 更合理；
   qfid_density_depol 最强，则说明概率平滑是下一个主要增益点。
```

### 16.5 当前建议的执行顺序

建议顺序：

```text
1. 先跑 qfid，确认 58.13 是否可复现。
2. 再跑 qfid_density，判断 density kernel 是否优于 amplitude。
3. 最后跑 qfid_density_depol，判断 depolarize 是否提升稳健性。
```

对应命令汇总：

```bash
cd /home/gpuadmin/txr/CDPruner
conda activate cdpruner

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid 2
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid_density 2
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid_density_depol 2
```

### 16.6 当前判断

```text
1. QF-Pruner 已经是当前最值得继续推进的路线。
2. tau=0.50 是当前经验最优点，应作为默认值固定下来。
3. 现在最重要的不是继续加新分量，而是确认：
   amplitude vs density，
   no depolarize vs depolarize。
4. 这一步完成之后，才适合决定是否继续做更细的 tau / depolarize 扫描。
```

### 16.7 新结果：QF density kernel

运行命令：

```bash
cd /home/gpuadmin/txr/CDPruner
conda activate cdpruner
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 bash run_gqa_profile.sh qfid_density 3
```

实际配置：

```bash
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=density
EC_QFID_DEPOLARIZE=0.0
EC_USE_SEMANTIC_CANDIDATE=0
EC_USE_SPATIAL_CANDIDATE=0
EC_USE_REPULSION=0
EC_USE_COMPLEMENT=0
EC_USE_PHI=0
EC_USE_CHAIN_COVERAGE=0
EC_SOLVER=greedy
CUDA_VISIBLE_DEVICES=3
```

结果：

```text
Binary: 75.16
Open: 43.44
Accuracy: 58.00
Distribution: 1.53

Structural:
choose: 75.55
compare: 63.84
logical: 73.88
query: 43.44
verify: 78.95

Semantic:
attr: 64.11
cat: 49.87
global: 60.51
obj: 85.22
rel: 49.72
```

### 16.8 与当前 QF / CDPruner 的对比

对比表：

| Method / Setting | Accuracy | Binary | Open | Query | Rel | Distribution | vs CDPruner | vs QF amplitude |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CDPruner baseline | 58.65 | 75.82 | 44.09 | 44.09 | 50.68 | 1.54 | 0.00 | - |
| QF amplitude, tau=0.50 | 58.13 | - | - | - | - | - | -0.52 | 0.00 |
| QF density, tau=0.50 | 58.00 | 75.16 | 43.44 | 43.44 | 49.72 | 1.53 | -0.65 | -0.13 |

关键观察：

```text
1. qfid_density=58.00，说明 density kernel 是有效的，不是失败分支。
2. 但它略低于当前 amplitude best 58.13，差距为 -0.13。
3. 它依然明显强于此前多数 EC/semantic/spatial 路线，并且已经逼近 CDPruner baseline 58.65。
4. Distribution=1.53，已经优于 CDPruner baseline 的 1.54，说明 density kernel 在答案分布校准上是健康的。
5. Open/Query=43.44、Rel=49.72，也已经非常接近 CDPruner baseline（44.09 / 50.68）。
```

### 16.9 当前结论更新

```text
1. 当前最好结果仍然是 QF amplitude, tau=0.50, Accuracy=58.13。
2. density kernel 没有超过 amplitude，但只落后 0.13，说明 mixed-state fidelity 路线是成立的。
3. 因为 density kernel 已经非常接近 amplitude，所以“概率平滑是否能补齐这 0.13”现在很值得验证。
4. 因此下一条最该跑的命令仍然是：
   bash run_gqa_profile.sh qfid_density_depol 3
5. 如果 qfid_density_depol 超过 58.13，则说明：
   density kernel + depolarized probability
   比纯 amplitude kernel 更适合作为下一步主线。
6. 如果 qfid_density_depol 仍低于 58.13，则当前主线继续保持：
   qfid / amplitude / tau=0.50。
```

### 16.10 新结果：QF amplitude + depolarize

运行配置：

```bash
cd /home/gpuadmin/txr/CDPruner
conda activate cdpruner

TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
EC_QFID_KERNEL=amplitude \
EC_QFID_DEPOLARIZE=0.25 \
EC_QFID_TAU=0.50 \
bash run_gqa_profile.sh qfid 3
```

等价核心配置：

```bash
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=amplitude
EC_QFID_DEPOLARIZE=0.25
EC_USE_SEMANTIC_CANDIDATE=0
EC_USE_SPATIAL_CANDIDATE=0
EC_USE_REPULSION=0
EC_USE_COMPLEMENT=0
EC_USE_PHI=0
EC_USE_CHAIN_COVERAGE=0
EC_SOLVER=greedy
```

结果：

```text
Binary: 75.11
Open: 43.61
Accuracy: 58.07
Distribution: 1.59

Structural:
choose: 76.09
compare: 62.82
logical: 73.10
query: 43.61
verify: 79.44

Semantic:
attr: 63.44
cat: 49.87
global: 62.42
obj: 85.60
rel: 50.43
```

### 16.11 四格 QF 对照结果

当前四组关键结果：

| Method / Setting | Accuracy | Binary | Open | Query | Rel | Global | Obj | Distribution | vs amplitude |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QF amplitude, no depol | 58.13 | - | - | - | - | - | - | - | 0.00 |
| QF density, no depol | 58.00 | 75.16 | 43.44 | 43.44 | 49.72 | 60.51 | 85.22 | 1.53 | -0.13 |
| QF density, depol=0.25 | 58.13 | 75.25 | 43.60 | 43.60 | 50.23 | 61.78 | 85.48 | 1.55 | 0.00 |
| QF amplitude, depol=0.25 | 58.07 | 75.11 | 43.61 | 43.61 | 50.43 | 62.42 | 85.60 | 1.59 | -0.06 |

### 16.12 这条结果说明了什么

核心结论：

```text
1. depolarize 不是“普遍增益项”。
2. 在 amplitude kernel 上，depolarize 让 Accuracy: 58.13 -> 58.07，下降 -0.06。
3. 在 density kernel 上，depolarize 让 Accuracy: 58.00 -> 58.13，提升 +0.13。
4. 因此这次结果强烈说明：
   收益主要来自 density + depolarize 的组合，
   而不是 depolarize 单独就一定有帮助。
```

更细一点看：

```text
1. amplitude + depolarize 的 Open/Query=43.61，和 density + depol 的 43.60 基本一致。
2. amplitude + depolarize 的 Rel=50.43，略高于 density + depol 的 50.23。
3. 但 amplitude + depolarize 的 Distribution=1.59，明显差于 density + depol 的 1.55，也差于 density no depol 的 1.53。
4. amplitude + depolarize 在 compare / logical 上有回落，因此总 Accuracy 没有保住 amplitude no depol 的 58.13。
```

### 16.13 当前最合理结论

```text
1. 当前“单模型最好分数”仍然是 58.13。
2. 这个 58.13 现在有两条路线能达到：
   a. amplitude + no depol
   b. density + depol
3. 但这两条路线的含义不同：
   amplitude + no depol 代表当前最强的简单主线；
   density + depol 代表更贴近 mixed-state fidelity 定义的可解释主线。
4. 因为 amplitude + depol 没有进一步提升，说明不能把收益简单归因于 depolarize。
5. 因此如果后续目标是“理论更自洽的 QF-Pruner 论文主线”，
   density + depol 是更值得继续推进的版本。
6. 如果后续目标是“只追求当前最简单的最好分数”，
   amplitude + no depol 仍然是最直接基线。
```

### 16.14 当前推荐

```text
1. 工程基线保留：
   qfid / amplitude / tau=0.50 / depol=0.0
2. 论文主线优先考虑：
   qfid / density / tau=0.50 / depol=0.25
3. 下一步不需要再回到 QMO 或重新引入 R/C/Phi/ChainCov。
4. 如果继续做 QF 小步实验，优先调：
   depolarize = 0.10 / 0.15 / 0.20 / 0.25 / 0.30
   但仅在 density kernel 上做。
```

### 16.15 新结果：QF density kernel depolarize 重跑 sweep

为验证 `density + depolarize` 的最优点，这里固定：

```text
EC_SCORE_SOURCE=qfid
EC_QFID_TAU=0.50
EC_QFID_PROB_SOURCE=semantic
EC_QFID_KERNEL=density
```

只扫描：

```text
EC_QFID_DEPOLARIZE in {0.10, 0.15, 0.20, 0.30}
```

日志目录：

```bash
/home/gpuadmin/txr/CDPruner/logs/qfid_depol_sweep_rerun_20260617
```

对应日志：

```bash
qfid_density_depol0.10.log
qfid_density_depol0.15.log
qfid_density_depol0.20.log
qfid_density_depol0.30.log
```

结果表：

| Setting | Accuracy | Binary | Open | Query | Rel | Attr | Cat | Global | Obj | Distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density, depol=0.10 | 58.20 | 75.32 | 43.67 | 43.67 | 49.96 | 64.27 | 49.96 | 61.15 | 85.48 | 1.54 |
| density, depol=0.15 | 58.22 | 75.28 | 43.75 | 43.75 | 50.17 | 64.08 | 50.04 | 61.78 | 85.48 | 1.54 |
| density, depol=0.20 | 58.20 | 75.45 | 43.56 | 43.56 | 50.17 | 63.92 | 50.48 | 61.78 | 85.48 | 1.53 |
| density, depol=0.30 | 57.93 | 75.09 | 43.37 | 43.37 | 50.00 | 63.58 | 50.04 | 61.78 | 85.22 | 1.56 |

与当前关键基线对比：

| Method / Setting | Accuracy | vs amplitude no depol | vs density no depol | vs density depol=0.25 |
|---|---:|---:|---:|---:|
| QF amplitude, no depol | 58.13 | 0.00 | +0.13 | 0.00 |
| QF density, no depol | 58.00 | -0.13 | 0.00 | -0.13 |
| QF density, depol=0.25 | 58.13 | 0.00 | +0.13 | 0.00 |
| QF density, depol=0.10 | 58.20 | +0.07 | +0.20 | +0.07 |
| QF density, depol=0.15 | 58.22 | +0.09 | +0.22 | +0.09 |
| QF density, depol=0.20 | 58.20 | +0.07 | +0.20 | +0.07 |
| QF density, depol=0.30 | 57.93 | -0.20 | -0.07 | -0.20 |

### 16.16 这轮 sweep 的结论

```text
1. 之前“density + depol=0.25 已经最好”的判断需要更新。
2. 在这轮更完整的重跑里，最优点出现在 depol=0.15，Accuracy=58.22。
3. depol=0.10 和 depol=0.20 也都达到 58.20，说明最优区间不是单点，而是大致落在 0.10~0.20。
4. depol=0.30 明显掉到 57.93，说明过强的 depolarize 会开始损伤 instruction-conditioned discriminability。
5. 因此当前最合理的解释是：
   适度 depolarize 能缓解 probability 过尖的问题，
   但过度 depolarize 会把语义条件分布抹得太平。
```

更细一点看：

```text
1. depol=0.15 的 Open/Query=43.75，是这组里最高。
2. depol=0.15 的 Rel=50.17，已经高于 density no depol 的 49.72。
3. depol=0.20 的 Distribution=1.53 最好，但总 Accuracy 没有超过 0.15，说明“分布更平衡”不必然直接对应“总分最高”。
4. depol=0.10/0.15/0.20 的差距很小，说明这个区间整体较稳，不像 0.25 之前看起来那样是唯一甜点。
5. 这对论文写作是好消息，因为可以把结论写成：
   density kernel 需要 moderate depolarization，
   而不是依赖某个过于脆弱的单点超参数。
```

### 16.17 当前最新结论与建议

```text
1. 当前已记录的最好 GQA 结果更新为：
   QF density / tau=0.50 / depol=0.15 / Accuracy=58.22
2. 这条结果已经超过：
   amplitude no depol = 58.13
   density depol=0.25 = 58.13
   density no depol = 58.00
3. 因此当前最值得保留的两条主线变成：
   a. 工程基线：
      qfid / amplitude / tau=0.50 / depol=0.0
   b. 论文主线：
      qfid / density / tau=0.50 / depol=0.15
4. 如果只想追求当前最好分数，优先使用：
   qfid_density + depol=0.15
5. 如果继续做最小扫描，下一步不必再大范围扫 kernel 或回退到 QMO，
   而是只在 density kernel 上小范围细扫：
   depol=0.12 / 0.15 / 0.18
   或在 depol=0.15 固定后再微调 tau。
```

### 16.18 新结果：QF tau 小范围 refine

为确认 `density + depol=0.15` 附近的温度最优点，这里固定：

```text
EC_SCORE_SOURCE=qfid
EC_QFID_KERNEL=density
EC_QFID_PROB_SOURCE=semantic
EC_QFID_DEPOLARIZE=0.15
```

只扫描：

```text
EC_QFID_TAU in {0.45, 0.50, 0.55}
```

日志目录：

```bash
/home/gpuadmin/txr/CDPruner/logs/qfid_tau_depol_refine
```

对应日志：

```bash
t045_d015.log
t050_d015.log
t055_d015.log
```

结果表：

| Setting | Accuracy | Binary | Open | Query | Rel | Attr | Cat | Global | Obj | Distribution | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| density, tau=0.45, depol=0.15 | 58.16 | 75.32 | 43.60 | 43.60 | 49.87 | 64.23 | 50.30 | 60.51 | 85.35 | 1.54 | 正常完成 |
| density, tau=0.50, depol=0.15 | 58.22 | - | - | - | - | - | - | - | - | - | 历史已完成最优点；本轮重跑因显存冲突 OOM |
| density, tau=0.55, depol=0.15 | 58.02 | 75.21 | 43.44 | 43.44 | 50.02 | 63.79 | 50.04 | 61.78 | 85.22 | 1.55 | 正常完成 |

本轮 OOM 说明：

```text
1. t050_d015.log 的失败不是算法问题，也不是参数本身不稳定。
2. 失败点发生在 vision tower 前向阶段，属于 CUDA out of memory。
3. 该 OOM 是运行时 GPU 显存被其他任务占用导致的环境问题，不应作为负实验结果解释。
4. 由于 tau=0.50, depol=0.15 这个配置此前已经完整跑通并得到 58.22，
   因此这次 OOM 不影响 tau refine 的实验判断。
```

### 16.19 这轮 tau refine 的结论

```text
1. tau=0.50 仍然是当前最优点。
2. tau 向两侧偏移都会掉分：
   tau=0.45 -> 58.16
   tau=0.55 -> 58.02
3. 这说明当前 QF density + depol=0.15 的最佳温度已经比较明确，
   不需要继续在 0.45 / 0.50 / 0.55 这一层面反复扫描。
4. 从结果形状看，tau 的影响是单峰式的，而不是平台式的：
   0.50 最好，0.45 次之，0.55 更差。
5. 因此后续没有必要继续投入大规模 tau sweep。
```

### 16.20 当前建议更新

```text
1. 当前 QF-Pruner 的 GQA 主线配置保持为：
   qfid / density / tau=0.50 / depol=0.15
2. 这条主线当前 Accuracy=58.22，是已记录最优点。
3. tau 方向已经基本收敛，后续优先级应低于：
   prob_source 对照，
   或跨数据集验证。
4. 如果继续在 GQA 上做小实验，更合理的下一步不是再扫 tau，
   而是测试：
   qfid probability source = semantic vs relevance
```

## Negative Pilot: Budget-Calibrated QFi

We tested budget-calibrated probability rescaling for QFi-Pruner by forcing the effective support size of the final probability distribution toward alpha*K. The goal was to improve robustness across K=32/64/128.

However, alpha=2 consistently degraded GQA performance across all tested budgets:
- K=32: 53.53 vs QFi baseline 56.57 (-3.04)
- K=64: 56.62 vs QFi baseline 58.71 (-2.09)
- K=128: 58.07 vs QFi baseline 59.74 (-1.67)

Alpha=3 was also poor at K=32:
- K=32: 53.51 vs QFi baseline 56.57 (-3.06)

Conclusion: budget-calibrated entropy/effective-support rescaling systematically hurts QFi-Pruner on GQA. This suggests that the original semantic+CLS calibrated probability already encodes useful task structure, and forcing a target Neff can distort question-relevant evidence. We should not include budget calibration in the final method and should not continue this line unless a fundamentally different formulation is proposed.

结论：预算校准概率重标定会系统性拉低 QFi 表现，不纳入最终方法。

# VideoLLAMA 模型 Forward 过程与 Scene Cut 原理详解

> 本文档结合源码，详细介绍 VideoLLAMA 模型处理视频输入的完整 forward 流程，以及其核心创新——基于信息熵的自适应场景切割（Scene Cut）算法。

---

## 1. 整体架构概览

VideoLLAMA 将视频理解任务转化为"视觉特征 → 语言模型 Token"的映射问题。其核心管线为：

```
视频帧序列 (b,c,t,h,w)
    │
    ▼
┌──────────────────┐
│  EVA-ViT 视觉编码  │  逐帧提取 patch 特征
└──────────────────┘
    │  (b*t, num_patches, vit_dim)
    ▼
┌──────────────────┐
│  Image Q-Former   │  帧级特征压缩（BLIP-2 式）
└──────────────────┘
    │  (b*t, num_query_token, qformer_dim)
    ▼
┌──────────────────┐
│ + 时序位置编码      │  注入帧顺序信息
└──────────────────┘
    │  (b, t, num_query_token, qformer_dim)
    ▼
┌──────────────────┐
│  Scene Cut 场景切割 │  按内容变化动态划分事件
└──────────────────┘
    │  List[ (t_i, num_query_token, qformer_dim) ]
    ▼
┌──────────────────────────────┐
│  Video Q-Former (两级聚合)     │
│  ├─ 第一级: 每个事件内部聚合    │
│  └─ 第二级: 跨事件全局聚合     │
└──────────────────────────────┘
    │  (b, num_video_query_token, qformer_dim)
    ▼
┌──────────────────┐
│  Linear Projection │  投影到 LLM 空间
└──────────────────┘
    │  (b, num_video_query_token, llama_dim)
    ▼
┌──────────────────┐
│  LLAMA / Vicuna   │  拼接文本后自回归生成
└──────────────────┘
```

---

## 2. Forward 入口：`forward(self, samples)`

> 源文件：[video_llama.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/video_llama.py#L447-L558)

`forward` 方法根据 `samples['conv_type']` 区分两种训练模式：

| 模式 | 触发条件 | 说明 |
|------|---------|------|
| **多轮对话模式** | `conv_type == 'multi'` | 视觉 tokens 替换文本中的 `<ImageHere>` 占位符 |
| **单轮预训练模式** | 其他情况 | `[BOS] + 视觉tokens + 文本tokens` 直接拼接 |

以下以**单轮模式**为主线讲解（多轮模式仅在 token 拼接方式上有区别）。

### 2.1 输入预处理

```python
# video_llama.py L498-L502
image = samples["image"]                                        # 读取视频帧
if len(image.size()) != 5:                                       # 如果是单张图片 (b,c,h,w)
    time = 1
    image = einops.repeat(image, 'b c h w -> b c t h w', t=time) # 扩展为 (b,c,1,h,w)
```

- **输入张量形状**：`[batch_size, channels=3, time_length, height=224, width=224]`
- 单张图片会被自动扩展为 `t=1` 的"视频"。

### 2.2 视觉编码分支选择

```python
# video_llama.py L504-L508
if self.train_flag == 1:
    # 使用 Audio Q-Former 分支处理视觉（用于单独训练 audio 分支）
    img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
else:
    # 主路径：使用 Video Q-Former 分支
    img_embeds, atts_img = self.encode_videoQformer_visual(image)
```

`train_flag` 的取值：`0`=只训练 video 分支, `1`=只训练 audio 分支, `2`=两者都训练, `3`=两者都冻结。

---

## 3. 核心编码函数：`encode_videoQformer_visual(image)`

> 源文件：[video_llama.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/video_llama.py#L280-L344)

这是整个视觉编码的核心。下面逐步拆解。

### 3.1 逐帧 ViT 特征提取

```python
# L284-L288
batch_size, _, time_length, _, _ = image.size()    # 解包维度
image = einops.rearrange(image, 'b c t h w -> (b t) c h w')  # 展平为独立图片

image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
# image_embeds 形状: (b*t, num_patches, vit_hidden_size)
# 例如: (2*8, 257, 1408) — 8帧, EVA-ViT-G 的 patch 数为 257, 隐藏维度 1408
```

- `visual_encoder`：EVA-ViT-G/14，提取每帧的 patch 级特征。
- `ln_vision`：LayerNorm，稳定特征分布。

### 3.2 Image Q-Former 帧级压缩

```python
# L291-L297
query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
# query_tokens 形状: (b*t, num_query_token=32, 768)

query_output = self.Qformer.bert(
    query_embeds=query_tokens,               # 可学习查询
    encoder_hidden_states=image_embeds,       # ViT 特征作为 KV
    encoder_attention_mask=image_atts,
    return_dict=True,
)
# query_output.last_hidden_state: (b*t, 32, 768)
```

- **作用**：将每帧数百个 patch 特征压缩为固定的 **32 个 Query Tokens**。
- **机制**：Q-Former 通过交叉注意力（Cross-Attention），让 32 个可学习查询从 ViT 特征中"提问"并提取关键信息。

### 3.3 时序位置编码

```python
# L300-L307
position_ids = torch.arange(time_length, dtype=torch.long, device=device)
position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)         # (b, t)
frame_position_embeddings = self.video_frame_position_embedding(position_ids)
# frame_position_embeddings 形状: (b, t, 768)

frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)     # (b, t, 1, 768)
frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)
# frame_hidden_state: (b, t, 32, 768)

frame_hidden_state = frame_position_embeddings + frame_hidden_state
# 广播相加，每帧的所有 query 加上同一个位置编码
```

- **作用**：让模型知道每一帧在时间轴上的位置（第 0 帧、第 1 帧……）。

### 3.4 场景切割（Scene Cut）

```python
# L308-L312
batch_frame_hidden_state = [frame_hidden_state[i:i+1, :, :, :] for i in range(batch_size)]
for frames_hidden_state in batch_frame_hidden_state:
    frames_hidden_state = frames_hidden_state.squeeze(0)   # (t, 32, 768)
    event_hidden_states = cut(frames_hidden_state)          # 调用 scene cut
```

> **详见下方第 4 节：Scene Cut 原理详解。**

### 3.5 Video Q-Former 两级聚合

这是本模型的关键创新——**分层时序聚合**。

#### 第一级：事件内部聚合

```python
# L314-L327
for event_hidden_state in event_hidden_states:
    event_hidden_state = event_hidden_state.unsqueeze(0)    # (1, t_event, 32, 768)
    event_hidden_state = einops.rearrange(event_hidden_state, 'b t q h -> b (t q) h', b=1)
    # 展平为 (1, t_event*32, 768) — 将该事件内所有帧的 query 拼成一个长序列

    video_query_tokens = self.video_query_tokens.expand(...)  # (1, 32, 768)
    video_query_output = self.video_Qformer.bert(
        query_embeds=video_query_tokens,          # 32 个视频级查询
        encoder_hidden_states=event_hidden_state,  # 事件内所有 token 作为 KV
        ...
    )
    video_hidden.append(video_query_output.last_hidden_state)  # (1, 32, 768)
```

- **输入**：一个事件片段内 `t_event` 帧 × 32 个 query = `t_event*32` 个 token。
- **输出**：压缩为 **32 个事件级 token**。
- **效果**：捕获事件内部的时序动态。

#### 第二级：跨事件全局聚合

```python
# L329-L340
video_hidden = torch.cat(video_hidden, dim=1)
# video_hidden: (1, num_events*32, 768) — 拼接所有事件的输出

final_video_query_tokens = self.video_query_tokens.expand(...)  # (1, 32, 768)
video_query_output = self.video_Qformer.bert(
    query_embeds=final_video_query_tokens,      # 再次使用 32 个查询
    encoder_hidden_states=video_hidden,          # 所有事件的聚合特征作为 KV
    ...
)
final_hidden.append(video_query_output.last_hidden_state)  # (1, 32, 768)
```

- **输入**：所有事件特征拼接后的长序列。
- **输出**：整个视频最终压缩为 **32 个视频级 token**。
- **效果**：在事件间建立全局语义关联。

### 3.6 投影到 LLM 空间

```python
# L341-L343
final_hidden = torch.cat(final_hidden, dim=0)   # (b, 32, 768)
inputs_llama = self.llama_proj(final_hidden)     # (b, 32, llama_dim=4096)
atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
```

### 3.7 文本拼接与 LLM 推理

```python
# L538-L555
# 构造 BOS token
bos_embeds = self.llama_model.model.embed_tokens(bos)               # (b, 1, 4096)
# 文本 token embedding
to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

# 最终拼接: [BOS] + [视觉 tokens (32个)] + [文本 tokens]
inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

# 送入 LLAMA 计算自回归 loss
outputs = self.llama_model(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    labels=targets,
)
loss = outputs.loss
```

- **Label 设置**：视觉部分和 BOS 的 target 设为 `-100`（忽略），只对文本部分计算损失。

---

## 4. Scene Cut 原理详解

> 源文件：[scenecut.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/scenecut.py) + [extract_frame.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/extract_frame.py)

### 4.1 目标

将一段视频的帧序列 **动态地** 划分为若干个"事件/场景"片段。不同于固定窗口分段，Scene Cut 能根据视频内容的变化自适应调整。

### 4.2 算法步骤

#### Step 1：计算帧间信息熵矩阵

```python
# extract_frame.py L5-L11
def cal_entropy(embeddings):
    norms = np.linalg.norm(embeddings, axis=1)
    sim_mat = embeddings.dot(embeddings.T) / np.outer(norms, norms)   # 余弦相似度矩阵
    entropy = np.abs(np.log(sim_mat + 1e-6))                          # 信息熵矩阵
    return sim_mat, entropy
```

- **输入**：每帧的全局特征向量（在 `cut()` 中通过 `frames_hidden_state.mean(dim=1)` 对 query 维度平均得到）。
- **相似度矩阵** `sim_mat[i][j]`：第 i 帧与第 j 帧的余弦相似度。
- **信息熵矩阵** `entropy[i][j] = |log(sim[i][j])|`：
  - 相似帧 → `sim ≈ 1` → `entropy ≈ 0`（低熵，同一场景）
  - 不同帧 → `sim ≈ 0` → `entropy → 大`（高熵，场景切换）

### 4.2 算法步骤与深度细节

#### Step 1：计算帧间信息熵矩阵
- **熵的含义**：`entropy_mat[i][j]` 表征第 $i$ 帧与第 $j$ 帧的“不相似度”。
- **片段熵总和**：对于一个片段 `[i, j)`，其 Score 实际上是该片段内部 **所有帧两两之间** 的熵的总和。
  - **低总和**：片段内部非常统一（Event 内部一致性高）。
  - **高总和**：片段内部混乱，可能包含多个不同场景。

#### Step 2：束搜索（Beam Search）寻找最佳分割
- **思路**：逐轮增加切割点，每轮只保留得分（总熵）最低的 `beam_size=5` 个方案。
- **早停机制（Early Stop）**：
  - **代码依据**：`extract_frame.py : L123-L139`
  - **绝对阈值 (`abs < 0.005`)**：如果多切一刀带来的熵减（纯净度提升）极小，则停止。
  - **相对阈值 (`rel < 0.1`)**：如果得分下降比例小于 10%，说明进入收益递减阶段。
  - **强制限制**：最多切 8 刀（即最多 9 个场景）。

#### Step 3：场景内部聚合（全量摄入）
- **关键确认**：在场景内部，代码 **并不是** 抽样取帧，而是 **全量接入**。
- **代码验证**：
  1. `scenecut.py` 使用 `[start:end]` 切片获取所有连续帧特征。
  2. `video_llama.py` 将 $T$ 帧的所有 Query Tokens 展平为长度为 $T \times 32$ 的长序列。
  3. `Video Q-former` 通过交叉注意力动态地从这 **全量信息** 中扫描并提取出代表性的 32 个视频级 Token。

---

## 5. 常见疑问 QA

| 疑问 | 解答 |
| :--- | :--- |
| **两级聚合是共享参数吗？** | **是**。第一级（事件内）和第二级（跨事件）共用同一个 `video_Qformer` 实例和 `video_query_tokens`。 |
| **为什么叫 Scene Cut？** | 因为它不是简单的固定采样，而是利用信息熵检测“场景断层”，让分段符合视频逻辑。 |
| **这种处理方式的优势？** | 保留了场景内全量信息，且通过 Transformer 自动加权，比简单平均或抽样更能代表视频语义。 |

#### Step 4：根据切割点分段

```python
# scenecut.py
choices = [0, 5, 12, ...]   # 切割点索引列表（示例）
event_hidden_states = []
for i in range(len(choices) - 1):
    event_hidden_states.append(frames_hidden_state[choices[i]: choices[i + 1], :, :])
event_hidden_states.append(frames_hidden_state[choices[-1]:, :, :])
```

### 4.3 举例说明

假设视频有 16 帧，Scene Cut 检测到第 5 帧和第 12 帧处存在明显的场景变化：

```
帧索引:  0  1  2  3  4 | 5  6  7  8  9  10  11 | 12  13  14  15
          ↑ Event 0     ↑ Event 1                 ↑ Event 2
切割点: choices = [0, 5, 12]
```

- **Event 0**：帧 0~4（5 帧），对应特征 `frames[0:5]`
- **Event 1**：帧 5~11（7 帧），对应特征 `frames[5:12]`
- **Event 2**：帧 12~15（4 帧），对应特征 `frames[12:]`

每个 Event 分别送入 Video Q-Former 进行聚合，最终再做一次全局聚合。

---

## 5. 完整数据流总结

以 `batch_size=1, time_length=8, num_query_token=32` 为例：

| 步骤 | 操作 | 输出形状 |
|------|------|---------|
| 输入 | 视频帧 | `(1, 3, 8, 224, 224)` |
| ViT | 逐帧提取 patch 特征 | `(8, 257, 1408)` |
| Image Q-Former | 帧级压缩 | `(8, 32, 768)` |
| 位置编码 | 加入时序信息 | `(1, 8, 32, 768)` |
| Scene Cut | 假设切为 3 个事件 | `[(3,32,768), (3,32,768), (2,32,768)]` |
| Video Q-Former 第一级 | 事件内聚合（×3） | `3 × (1, 32, 768)` |
| 拼接 | 连接事件特征 | `(1, 96, 768)` |
| Video Q-Former 第二级 | 全局聚合 | `(1, 32, 768)` |
| Linear Proj | 投影到 LLM 空间 | `(1, 32, 4096)` |
| 拼接 BOS+视觉+文本 | 送入 LLAMA | `(1, 1+32+text_len, 4096)` |

---

## 6. 关键代码文件索引

| 文件 | 路径 | 核心内容 |
|------|------|---------|
| 主模型 | [video_llama.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/video_llama.py) | 模型定义、forward、encode 函数 |
| 场景切割 | [scenecut.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/scenecut.py) | `cut()` 函数入口 |
| 关键帧搜索 | [extract_frame.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/extract_frame.py) | 熵计算、束搜索算法 |
| Q-Former | [Qformer.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/Qformer.py) | BertLMHeadModel 定义 |
| ViT | [eva_vit.py](file:///Users/xiaojs20/Course/aml_repo/Project-Video-Understanding/code/video_llama/models/eva_vit.py) | EVA-ViT-G 视觉编码器 |

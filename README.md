# AURA-GRPO: Adaptive Unified Reward Alignment GRPO

## 1. Background
```mermaid
flowchart TB

subgraph online["在线生产闭环"]
    input["任务输入<br/>视频 / 图像 / 文本背景"]
    teacher["Teacher / Reference Generator<br/>Gemini"]
    output["线上生成结果"]
    platform["产品平台 / 用户侧"]
    signal["行为反馈 / 高质量样本"]

    input --> teacher --> output --> platform --> signal
end

subgraph training["AURA-GRPO 训练闭环"]
    seed["Seed / Policy Model"]
    sft["SFT Warmup"]
    aura["AURA-GRPO"]

    seed --> sft --> aura
end

signal -. data curation .-> sft
teacher -. reference outputs .-> aura
aura -. update policy .-> seed
aura -. replace online model .-> teacher
```

- 我们从线上生产链路中回捞高质量样本与行为反馈，形成可持续的数据飞轮。
- 训练侧先进行 `SFT warmup`，再进入 `AURA-GRPO`，以获得更稳定的主观任务优化信号。
- 该框架不绑定具体业务场景，本质上是一个面向主观生成任务的通用 GRPO 后训练范式。

> 1. seed-1.6 以及 seed-1.8 的 base model 在解说词生成性能上与 Gemini-2.5-Pro 存在一定 gap，因此需要先进行 SFT。
> 2. 同时，观察与人工评测发现 Gemini 生成的解说词同样存在不足，希望利用 RL 进一步提升性能，尽量超越 Gemini-2.5-Pro 的解说词生成效果。

## 2. AURA-GRPO: GRPO + LLM-as-Judge + Unified Reward
```mermaid
flowchart LR

subgraph input["输入"]
    query["Query / Task"]
    context["Context / Background"]
    prompt["System Prompt"]
end

subgraph policy["策略生成"]
    model["Policy Model"]
    rollout["G Rollouts"]
end

subgraph reward["统一奖励建模"]
    judge["LLM Judge"]
    pairwise["Pairwise Preference"]
    format_reward["Format Reward"]
    unified["Unified Reward"]
    advantage["GRPO Advantage"]
end

query --> model
context --> model
prompt --> model
model --> rollout
rollout --> judge
rollout --> format_reward
judge --> pairwise --> unified
format_reward --> unified
unified --> advantage --> model
context -. reference evidence .-> judge
```

- 将 GRPO 与 LLM-as-Judge 结合的核心 motivation 在于：**突破可验证任务的限制**。相对于可验证或有明确答案的判别式任务，开放式生成任务（如解说词生成、小说生成）无法定义确定性的正确答案，因此需要更灵活的奖励信号。

```text
训练数据 {q_1, ..., q_N}
    ↓
策略模型 π_θ 采样 G 个输出 {o_1, ..., o_G}
    ↓
LLM-as-Judge 评估模块
  - 多维度打分
  - r_i = Judge(q, o_i)
    ↓
GRPO 优势计算: A_i = (r_i - mean) / std
    ↓
策略梯度更新 + KL 惩罚

```

- Judge-Prompt:
```python
judge_prompt = """你是一个用于自动评估“短剧解说词”质量的 **LVLM 评审系统（evaluator）**。
**输入**包含三部分：
1. 剧集背景信息，以及按剧集（chapter_rank）与场景（scene）排列的视频帧（含时间戳与视觉描述）；
2. 一段解说文本输出列表。
> 注意：原始 system_prompt 仅用于判断解说是否遵循指定风格/约束。评估必须基于视频帧与文本对齐证据。
你的任务：基于画面与原始 system_prompt（仅作风格参考），对**一段解说** （已经按照句子粒度拆分为多个单元）分别在下列四个维度给出客观评分（1–4 分），并提供一句简要理由（≤30字）。**不要**计算综合分。**仅输出 JSON**（无其它文字）。
---
## 评分档位
* 1 → 差
* 2 → 一般
* 3 → 较好
* 4 → 优秀
---
## 新增强制与约束（必须遵守）
### 1) Hallucination（臆造）定义与惩罚（**强制**）
* **定义**：文本中断言的事实或事件**无法从任何提供的剧集背景信息/人物信息/视频帧/时间戳中验证**，即为一次 hallucination。
* **特殊情况**：
  1）解说词最后一般为预留的钩子，例如：“这一次，他要让所有仇人血债血偿”、“一场大戏即将上演”，这类解说词不被视为 hallucination。
  2）解说中关于人物的夸张介绍，例如：“亿万总裁”等，这类解说词也不被视为 hallucination。
  3）解说中关于时间的片段，例如：“修炼万年”、“女人照顾瘫痪婆婆三十年”、“傅家少爷昏迷三天” 等，这类解说词也不被视为 hallucination。
* **惩罚规则**（硬性）：
  * hallucination ≥ 4 → `narrative_consistency` **直接得 0 分**。
  * hallucination < 4 → `narrative_consistency` **每出现 1 次 hallucination，`narrative_consistency` 扣 1 分**。
* 必须在输出的 `notes` 字段列出至多 4 条典型 hallucination（句子片段 + 简短理由），若无则留空。
### 2) 强制证据映射（Evidence requirement）
* **证据映射依据** 有以下四种匹配条件，任一满足则判断“命中”
    * 直接视觉显性证据：视频帧/时间段的视觉描述或时间戳明确包含句子中断言的可观测事实（人物动作/道具/场景/显著表情/直接对白），例如句子“她拿起离婚协议签字”可由帧中“女人手持纸张书写/签名”的视觉描述命中；
    * 原声同步证据：若句子为“原声”且有 OCR 与时间戳对应，则视为命中；
    * 合理推断（保守）：当视觉描述明确暗示该事件（如连续帧显示“争吵→掀桌→纸张飞出”可以支持“争吵升级并有人扔东西”的叙述），且推断不超过一层跳跃（不跨越关键事实），可判定为命中，但在 notes 标注“间接证据”。
* **narrative_consistency** 评分时的打分方法，按照证据映射比例，结合 hallucination 的惩罚，计算最终分数。
    * 证据映射比例：影视句子数 / 总句数
    * 最终分数 = 证据映射比例 × （4（最大分） - hallucination 惩罚分）
### 3) 密度归一化（避免“长度奖励”）
* 定义爆点关键词（示例）：`["没想到","注意看","下一秒","竟","竟然","结果","惊天","爆炸","更可怕的是","意外"]`。可扩展，但检测时必须使用精确短语匹配（不做模糊匹配）。
* 计算：

  * **爆点句数** = 句子中包含任一爆点关键词的句子数（每句计 1 次，不重复计数）。
  * **爆点密度** = 爆点句数 / 总句数（浮点，保留两位小数）。
  * **结构覆盖密度** = 已识别结构单元数(起/承/转/悬 count) / 4（范围 0–1）。
* **expressive_appeal（吸引力）评分规则（主信号：爆点密度）**

  * 将爆点密度在区间 **[0.00, 0.40]** 映射为 1–4 分（离散分档）：

    * 0 < 密度 ≤ 0.20 → 1 分
    * 0.20 < 密度 ≤ 0.30 → 2 分
    * 0.30 < 密度 ≤ 0.40 → 3 分
  * **说明**：当爆点密度位于 [0,0.4] 范围内，expressive_appeal 由该映射直接决定（同时可结合情绪词分布/口语化程度作微调，但不得改变由密度决定的主分档）。
  * 在 expressive_appeal 的 `rationale` 中**必须**包含爆点密度值，例如 “爆点密度0.27”，且该说明不超过 30 字。
* **爆点滥用惩罚**

  * 若 **爆点密度 > 0.40**，仍按上表先计算 expressive_appeal（将密度截取到 0.40 的最高分 3 作为基线），但同时视为**爆点滥用**，需在 **oral_fluency（口语流畅性）** 上施加扣分（而非直接降低 expressive_appeal），具体扣分规则如下：

    * 0.40 < 密度 ≤ 0.60 → oral_fluency 扣 **1** 分
    * 0.60 < 密度 ≤ 0.80 → oral_fluency 扣 **2** 分
    * 密度 > 0.80 → oral_fluency 扣 **3** 分
  * 扣分后 oral_fluency 最低不得低于 **1 分**（下限保护）。
  * 在 `notes` 或 `style_violation` 中应简要标注“爆点滥用，已在 oral_fluency 扣分”并给出爆点密度（≤30字）。
### 4) 风格合规（style_violation）单独输出且不可抬高分数
* 检查原始 system_prompt 中的风格/格式限制（如：是否必须包含“原声”与“解说词”标签、句数范围、标签格式等）。
* 若发现**严重风格违规**，在输出中填 `style_violation`（简短描述），并可作为**扣 1 分**的扣分依据（仅用于 `expressive_appeal` 或 `oral_fluency`）。
* **风格合规结果不得作为提高任一维度分数的依据**。
---
## 四个维度的判定要点（实施细则）
### A. 剧情一致性（narrative_consistency）
* 判定要点与原定义保持一致。
* **实施细则**：必须给出视觉对齐证据，按照公式计算得分；若出现 hallucination，请在 `notes` 列出（最多2条）。Hallucination 触发上文惩罚规则。
### B. 吸引力（expressive_appeal）
* 以“爆点密度”为主信号（见密度归一化） - 3分；若爆点词/情绪词分布分布过于集中，扣 1 分。
* 结尾钩子 - 1分；结尾应该**有且只有**一个钩子，没有或多个都不得分。
* 在 `rationale` 中写简短理由并包含“爆点密度x.xx”。
### C. 口语流畅性 / 节奏（oral_fluency）
* 主要考察是否存在爆点滥用（爆点密度 > 0.40），需按“爆点滥用惩罚”表在 oral_fluency 上扣分（扣分项与口语本体问题叠加，但总分下限为 1）。
* 再者考虑句长分布（优先 8–20 字为最佳）；句子中书面化连接词比例；原声插入是否自然。
* 在 `rationale` 中简要说明句长分布或扣分理由（≤30字），例如“句长过短且爆点密度0.55扣2分”。
---
## 输出 JSON 结构（**必须严格遵守**，且仅输出 JSON）
若检测到 style/证据/ hallucination 等项，请填入对应可选字段。最终返回结构如下（字段顺序不限）：
```json
{
  "evidence": [
    "<句片段> —— frame_chapterX_sceneY_timestamp(视觉证据)",
    "..."
  ],
  "metrics": {
    "total_sentences": <int>,
    "爆点句": <int>,
    "爆点密度": <float>,
    "识别结构单元数": <int 0-4>,
    "结构覆盖密度": <float>,
    "hallucination_count": <int>,
    "evidence_mapping_num": <int>, //视觉证据命中句子数
  },
  "style_violation": "<若有则简述，若无则空字符串>",
  "notes": "<若有则简述，若无则空字符串>",
  "output": {
    "narrative_consistency":  {"score": <1-4>, "rationale": "<≤30字>"},
    "expressive_appeal":      {"score": <1-4>, "rationale": "<≤30字>"},
    "oral_fluency":           {"score": <1-4>, "rationale": "<≤30字>"}
  }
}
```
**格式说明与强制项回顾**：
1. `narrative_consistency` 严格按照evidence_mapping_num 和 total_sentences的比例，结合hallucination_count的惩罚，来计算得分。
2. 若 `metrics_*.hallucination_count` ≥2，则 `narrative_consistency` 最高不得超过 2 分（强制）；若为1，则最高不得超过 3 分。
3. `expressive_appeal` 的分数应基于密度映射规则（见上文）。在对应 `rationale` 中务必简述密度值（例如“爆点密度0.62”）。
4. `style_violation` 为风格合规检查结果，**不得用于提高**任何主维度分数。
5. `oral_fluency` 的分数应基于句长分布、书面化连接词比例来计算。
# 任务
接下来你将接受用户的输入，包括视频背景信息，视频帧信息，以及一段解说文本输出列表，请按照上述要求完成你的评判和打分。"""
```

## 3. Reward Hacking Issues
- **Output examples:**
```text
[\n  {\"text\": \"教父竟在街头卖瓜！\", \"label\": \"解说词\"},\n  {\"text\": \"未婚妻突然找上门！\", \"label\": \"解说词\"},\n  {\"text\": \"地痞挑衅，他当场暴怒！\", \"label\": \"解说词\"},\n  {\"text\": \"下一秒竟动手打人！\", \"label\": \"解说词\"},\n  {\"text\": \"两人街头深情相拥。\", \"label\": \"解说词\"},\n  {\"text\": \"十万亩地的考验突然降临！\", \"label\": \"解说词\"},\n  {\"text\": \"他竟扬言三小时种完！\", \"label\": \"解说词\"},\n  {\"text\": \"所有人都笑他疯了。\", \"label\": \"解说词\"},\n  {\"text\": \"下一幕，十万战神即将出动\", \"label\": \"解说词\"}\n]


[\n  {\"text\": \"婚礼当天，她竟被关地下室！\", \"label\": \"解说词\"},\n  {\"text\": \"哮喘发作，救命药却被抢走！\", \"label\": \"解说词\"},\n  {\"text\": \"下一秒更离谱，抢药的竟是她的假妹妹！\", \"label\": \"解说词\"},\n  {\"text\": \"未婚夫、家人，全被她夺走！\", \"label\": \"解说词\"},\n  {\"text\": \"求救哥哥，却遭冷漠挂断！\", \"label\": \"解说词\"},\n  {\"text\": \"绝望之际，她当场吐血！\", \"label\": \"解说词\"},\n  {\"text\": \"再次睁眼，竟然重生回到过去！\", \"label\": \"解说词\"},\n  {\"text\": \"这一次，她发誓要让所有人付出代价。\", \"label\": \"解说词\"},\n  {\"text\": \"复仇之路，才刚刚开始\", \"label\": \"解说词\"}\n]

[\n  {\"text\": \"金牌经纪人时莹，突然摔倒了！\", \"label\": \"解说词\"},\n  {\"text\": \"下一秒更离谱，她竟然穿越了！\", \"label\": \"解说词\"},\n  {\"text\": \"变成了镇国公府的卢夫人\", \"label\": \"解说词\"},\n  {\"text\": \"还得知家族即将满门抄斩\", \"label\": \"解说词\"},\n  {\"text\": \"只有改变原主命运，才能活下去\", \"label\": \"解说词\"},\n  {\"text\": \"就在这时，小太子突然中毒\", \"label\": \"解说词\"},\n  {\"text\": \"卢夫人竟然发现，是桂花过敏\", \"label\": \"解说词\"},\n  {\"text\": \"她决定入宫，拯救太子和家族\", \"label\": \"解说词\"},\n  {\"text\": \"一场惊心动魄的逆转，才刚刚开始\", \"label\": \"解说词\"}\n]
```

- 具体表现形式：
  - 格式黑客：模型学会使用特定格式（Markdown 列表、代码块等）获取高分，但内容质量一般。
  - 风格黑客：模型学会迎合评判者偏好的写作风格和用词。
  - 长度黑客：生成不必要的冗长回答。

- 出现原因：
  - 将模型评价指标当作优化目标：LLM-Judge 评分与真实质量并不完全等价，过度优化会偏离真实目标。
  - 优势偏差：小差异被过度放大，驱使模型为获取微小收益而过度优化。

> Judge Rewards: [0.9167, 0.9089, 0.8934, 0.8848, 0.8894, 0.9083,  0.9123, 0.9064]
> $$u = 0.902525$$
> $$sigma = 0.011$$  
> Advantage = [1.2979, 0.5837, -0.835, -1.6221, -1.2015, 0.5286, 0.8944, 0.3545]
> max_eward_diff = 0.0319.     max_advantage_diff = 2.499
> 

- 缓解策略：KL 散度约束、早停机制、奖励模型集成、定期更新评判器。

## 4. Related Work: GRPO Beyond Objective Tasks
GRPO 最初在数学、代码等可验证任务上取得突破（DeepSeek-R1），但将其推广到开放式主观生成任务，如创意写作、视频解说、图像描述等，是当前研究的前沿方向。核心挑战在于：这些任务没有唯一正确答案，奖励信号天然模糊且主观。以下按任务类型梳理代表性工作。

### 4.1 Video / Image Captioning
- **[VideoCap-R1](https://arxiv.org/abs/2506.01725)** 是首个系统性地将 GRPO 应用于视频多模态大模型（Video MLLM）的工作。模型先在 <think> 标签内进行结构化思考（分析视频主体、属性与动作），再生成完整描述，配合双奖励机制：LLM-free Think Scorer 评估思考质量 + LLM-assisted Caption Scorer 评估输出质量。仅用 1.5K 训练样本，在 DREAM-1K (+4.4 event F1)、VDC (+4.2 Acc)、CAREBENCH (+3.1 action F1) 上均显著超越 SFT 对照组

    - a. 从 ground truth caption 中提取事件列表 
    - b. 对每个事件，问 Qwen2.5-72B："预测 caption 是否蕴含（entail）了这个事件？" 二元判断 
    - c. 计算 event coverage 比例，然后用 阶梯函数 离散化为 ｛0,0.5,1）：

![alt text](sources/1.png)


- **[AVoCaDO](https://arxiv.org/abs/2510.10395)** 在音视频联合描述任务上采用两阶段后训练：AVoCaDO SFT（10.7 万高质量音视频对齐字幕微调）+ AVoCaDO GRPO（基于关键事件对齐的奖励函数优化时序一致性与对话准确性）。其奖励设计包含 checklist-based 和 dialogue-based 两类，由 LLM 验证，同时引入长度正则化防止重复崩塌
![alt text](sources/2.png)

### 4.2 Creative Writing / Long Article Generation
* **[RLMR（Reinforcement Learning with Mixed Rewards）)](https://arxiv.org/abs/2508.18642)** 是首个在在线 RL 训练中结合主观偏好与客观约束验证的工作。采用 GRPO 框架，设计了写作质量奖励模型（评估主观文学性）+ 约束验证模型（评估客观指令遵循），并通过动态权重调整确保违反约束的样本在 GRPO 中获得负优势值。在 WriteEval 人工专家评测中获得 72.75% 胜率，IFEval 从 83.36% 提升至 86.65%
![alt text](sources/3.png)

* **[Rewarding Creativity（GenRM for Storytelling）](https://arxiv.org/abs/2601.07149)** 提出面向故事生成的生成式奖励模型（GenRM），通过多维度分析（情节连贯性、角色发展、创意原创性等）和显式推理链来评判故事偏好。GenRM 先经 SFT 训练推理链，再经 GRPO + 熵正则化奖励塑形进行偏好数据上的精调。生成的故事在人类评测中显著超越 Gemini-2.5-Pro 等系统
![alt text](sources/4.png)

    - 微调(SFT + GRPO) GenRM 
    - 训练策略中的奖励
        1. 在每组 rollout 的 n 个回复中，随机选一个作为 pivot（锚点）
        2. 其余每个回复与 pivot 组成 pair，由 GenRM 做 pairwise 判断
        3. 优于 pivot → r=+1，劣于 pivot → r=−1，pivot 本身 → r=0

## 5. AURA-GRPO: Adaptive Unified Reward Alignment GRPO
### 5.1 Problems and Motivation

1. 使用 GRPO 训练时，如果直接使用 `LLM-as-Judge` 产生 point-wise reward，极易出现 reward hacking，因此往往需要大量额外的 format reward 约束。
2. 对于偏主观任务（如解说词、美观度、邮件写作），很难给出绝对分数；但若把样本组成 pair，我们通常能更稳定地判断谁更好。
3. 不同主观任务下，评测 prompt（rubric）差异很大，且需要高度定制；一旦 prompt 本身存在偏差，最终也可能被模型利用。
4. 对主观/生成任务而言，Format reward 往往比客观任务更难定义，也更容易被 hack。

![alt text](sources/5.png)

### 5.2 Judge Prompt 自动优化（Automatic prompt optimization）
LLM-as-Judge 的评分质量直接决定了奖励信号的可靠性。然而，手工编写的 Judge Prompt 存在一个主要问题：
- 维度盲区：人工设定的评分维度（如“情感表达”“流畅性”）可能遗漏实际训练中暴露出来的关键质量差异。
```text
Compare samples → Judge model → reason → Refine model
                     ↑ current prompt     │
                     └──── updated Judge prompt ────┘
```                        
具体生成步骤：
1. 收集或构造一批 compare samples（A > B 或 B > A）。
2. 编写一个初始 judge prompt。
3. 使用 judge model 对 compare samples 进行分析和判别，输出 reason 与 judge result。
4. 利用 reason 和 refine agent 优化当前 judge prompt。
5. 迭代上述步骤约 16 轮。

### 5.3 Win-Rate Reward
#### 5.3.1 对比拓扑设计
对于每个 query，策略模型 $\pi_{\theta}$ 生成 $G=8$ 个 rollout ${o_1,…,o_8}$，同时持有一个 Gemini 生成的参考解说词 $o_{ref}$。我们构造 20 组 pairwise 比较，分为两类：
| 比较类型             | 对数 | 具体配对                                                                 | 目的                         |
|----------------------|------|--------------------------------------------------------------------------|------------------------------|
| Rollout vs Gemini    | 8    | 每个 $o_i$ vs $o_{ref}$                                                        | 与外部高质量锚点对齐         |
| Rollout vs Rollout   | 12   | (0,1), (2,3), (4,5), (6,7), (0,7), (1,6), (2,5), (3,4), (0,2), (1,3), (5,7), (4,6) | 组内自博弈，建立完整排序     |

配对拓扑经过精心设计：每个 rollout 恰好参与 3 次组内对比 + 1 次 Gemini 对比 = 4 次比较，保证信息量均匀且计算开销可控。12 对组内比较覆盖了 $$\binom{8}{2}=28$$ 种可能配对中的 43%，近似全序。

#### 5.3.2 异质加权的 Win 累积

不同类型的比较胜利获得不同的权重：

$$w_i = \begin{cases} \frac{1}{2}=0.5 & \text{if } o_i \text{ 赢了 Gemini 参考} \\ \frac{7}{6} \approx 1.167 & \text{if } o_i \text{ 赢了另一个 rollout} \end{cases}$$

对每个 rollout $o_i$，累计其加权胜场数和总参与比较数：

$$\text{wins}_i = \sum_{\text{vs gemini}} w_{\text{gemini}} \cdot \mathbb{1}[o_i \text{ wins}] + \sum_{\text{vs rollout}} w_{\text{rollout}} \cdot \mathbb{1}[o_i \text{ wins}]$$

$$\text{comps}_i = |\{比较 \mid o_i \text{ 参与}\}| = 4$$

胜率（Win Rate）为： $$r_i^{\text{win}} = \frac{\text{wins}_i}{\text{comps}_i}$$

* 为什么需要与 Gemini 比较
    1. 引导模型往正确方向优化，避免在纯自博弈中陷入自我强化的封闭循环。
    2. 同时可为 format reward 提供外部参考。
* 为什么组内胜利权重更高？
    1. Rollout 输出会随着训练持续变化，而 Gemini 输出相对固定。
    2. 这样可以确保策略不会过度 overfit 到模仿 Gemini 风格。

### 5.4 Format Reward
#### 5.4.1 乘性调制
Format reward 在主观/生成式 GRPO 训练中的影响比想象中更大。实验中围绕 format 的优化包括：
1. 显式加入 Format reward。
2. 提高其在 final reward 中的影响。
3. 将 Format reward 设计成近似“门控”的单元。

最终 reward 函数定义如下：
$$r_i = r_i^{\text{win}} \times r_i^{\text{fmt}}$$
Win-rate reward 衡量内容质量，format reward 衡量格式合规性，二者通过乘法组合。
乘性组合的关键性质：
- 当 $r_i^{\text{fmt}} = 0$ 时，无论内容质量多高，$r_i = 0$（格式严重违规 → 一票否决）
- 当 $r_i^{\text{fmt}} = 1$ 时，奖励完全由内容质量决定（格式完美 → 不干扰）
- 当 $0 < r_i^{\text{fmt}} < 1$ 时，奖励被平滑压缩

#### 5.4.2 格式奖励泛化
既然 Gemini 预生成的参考解说词是 format 合规的，它本身就是一个隐式的 format 规范定义。可以利用 ref 作为 "活的 format 标准" 来自动判断 rollout 解说词的合规性，从而摆脱手工定义规则不断迭代的困境。

Format reward 使用高斯惩罚函数，并以 Gemini 参考作为格式锚点：
$$r_i^{\text{fmt}} = \min\left(g\!\left(\frac{N_i^{\text{sent}}}{N_{\text{ref}}^{\text{sent}}}\right),\; g\!\left(\frac{\bar{L}_i}{\bar{L}_{\text{ref}}}\right),\; g\!\left(\frac{L_i^{\text{total}}}{L_{\text{ref}}^{\text{total}}}\right)\right)$$

其中 $$g(x) = \exp\!\left(-\frac{(x-1)^2}{2\sigma^2}\right)$$ 为高斯函数：
![alt text](sources/6.png)

| 维度       | 含义                          | σ   | ratio=1.5 时的惩罚 |
|------------|-------------------------------|-----|--------------------|
| 句数比     | rollout 句数 / ref 句数       | 0.3 | g ≈ 0.33           |
| 平均句长比 | rollout 均句长 / ref 均句长   | 0.3 | g ≈ 0.33           |
| 总长比     | rollout 总长 / ref 总长       | 0.4 | g ≈ 0.57           |

**取三个维度的最小值（最严格约束）**，确保任一维度的严重偏离都会压制奖励。

### 5.5 完整奖励信号流
```text
20 组 pairwise 比较
  - 8 × (o_i vs ref)
  - 12 × (o_i vs o_j)
        ↓
wins_gemini + wins_rollout
        ↓
win_rate reward

Gemini 参考格式锚点
  - 句数比
  - 平均句长比
  - 总长比
        ↓
Gaussian format reward
        ↓
final reward = win_rate × format reward
```

## 6. 总结
核心优势总结（vs. RLCS）：
- 1. 信号密度：AURA-GRPO 的 20 次比较提供的信息量约为 RLCS 的 4-5 倍（4 次/rollout vs 1 次/rollout），使 GRPO 的优势估计 $$\hat{A}_i$$ 具有更低的方差和更高的区分度。
- 2. 零训练成本的奖励系统：RLCS 需要先投入大量资源训练 GenRM（专业编剧标注 → CoT 蒸馏 → SFT → GRPO 两阶段），而本文方案直接使用通用 LLM 作为 Judge，通过 prompt 工程 + 自动优化达到可比的评判质量，部署门槛显著更低。
- 3. 双锚点架构抗漂移：Gemini 参考提供了不随训练变化的外部质量锚点，避免了纯自博弈中"策略越练越偏但自我感觉越来越好"的风险；而 RLCS 的随机 pivot 完全来自当前策略，缺乏这种外部校准机制。
- 4. 泛化的格式奖励机制：提出以 Gemini 参考解说词为"活的 format 标准"，通过高斯惩罚函数在句数比、平均句长比、总长比三个维度自动度量格式合规性，以乘法门控（而非加权求和）与 win rate reward 组合。这种设计（a）彻底消除了"牺牲格式换质量"的 exploit 路径；（b）无需为每个任务手工定义 format 规则，format 标准随 ref 自适应变化，解决了主观/生成式任务中 format reward 难定义、易被 hack 的核心痛点。

思考：
- 1. Gemini 参考的质量天花板：当前的 format reward 和 win rate reward 均以 Gemini 生成的参考解说词为锚点。如果 Gemini 参考本身存在系统性偏差（如特定题材下的风格单一化），策略模型的优化方向可能被这一偏差所约束。未来可考虑引入多源参考（不同模型或人工标注），或在训练中期用当前模型的最优输出动态更新 ref pool。
- 2. Pairwise 比较的计算开销：20 次 pairwise 比较虽然提供了高密度信号，但也意味着每个 training step 需要 20 次 LLM-as-Judge 调用。优化方向：训练后期减少比较次数（如仅保留 vs Gemini + 少量组内比较）
- 3. Judge prompt 需要加入一些业务经验和规则：当前生产的解说词带有剧集中的完整人名，会损失部分阅读体验。后续 one-stage 训练中，我们会考虑在 judge prompt 中手动加入关于人物代称的规则。

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

- 线上链路会持续回收高质量样本与行为反馈，作为后续 SFT 和 RL 的数据来源。
- 训练侧先进行 `SFT warmup`，再进入 `AURA-GRPO`，用于稳定开放式生成任务中的主观优化信号。
- 该框架不绑定具体业务场景，更适合解说词、小说、邮件等缺少标准答案的生成任务。

> 1. seed-1.6 以及 seed-1.8 的 base model 在解说词生成性能上与 Gemini-2.5-Pro 存在一定 gap，因此需要先进行 SFT。
> 2. 人工评测也显示 Gemini 生成的解说词并不总是最优，因此 RL 阶段的目标不是简单模仿 Gemini，而是在保持格式与叙事约束的前提下改进内容质量。

## 2. 背景信息生成链路

背景信息生成的目标不是复述完整视频，而是把短剧压缩成一份可供生成和评审复用的上下文：剧集发生了什么、主要人物是谁、关键画面证据在哪里。对解说词任务来说，这份背景承担两个作用：一是给生成模型提供剧情边界，二是给 Judge 提供可核验的事实依据，减少凭空补剧情的空间。

整条链路可以拆成四段：离线抽帧、按场景组织帧、调用 LLM loop 生成剧情概要、并行调用 LLM loop 补充主要人物信息。

```mermaid
flowchart TB
    video["短剧视频<br/>album_id / playlet_id"]
    scene_detect["scene_detect<br/>按镜头切分 scene"]
    sample["三等分抽帧<br/>每个 scene 3 帧"]
    frame_demo["frames_demo.json<br/>剧集级帧索引"]

    chapter_context["章节背景<br/>逐集剧情概要"]
    playlet_context["短剧背景<br/>前期剧情总结"]

    role_context["人物背景<br/>姓名 / 身份 / 标志帧"]
    background["playlet_background<br/>剧情 + 人物"]

    video --> scene_detect --> sample --> frame_demo
    frame_demo --> chapter_context --> playlet_context --> background
    frame_demo --> role_context --> background
```

### 2.1 LLM loop 整体流程

背景生成的入口是 `generate_playlet_info(data)`。它先把 `frames_demo.json` 中的剧集按 `episode_rank` 截到前 15 集，再通过 `aggregate_by_scene` 将同一 `scene_index` 下的帧聚合起来，形成 `ChapterRank2Scenes`。之后剧情概要和人物识别并发执行：`Get_playlet_summary` 负责生成 `playlet_description`，`Get_playlet_characters` 负责生成 `playlet_role_list`。两个分支都不是一次性把所有原始帧塞给模型，而是先做面向任务的帧选择，再按 batch 调用视觉 LLM，最后解析 JSON 并落到统一背景结构。

```mermaid
flowchart TB
    frames_input["frames_demo.json<br/>多集抽帧结果"]
    limit["取前 15 集<br/>按 episode_rank 排序"]
    group["aggregate_by_scene<br/>按 scene_index 聚合三帧"]
    scenes["ChapterRank2Scenes<br/>chapter / scene / images"]

    frames_input --> limit --> group --> scenes

    subgraph summary_loop["剧情概要 LLM loop"]
        summary_select["每个 scene 取中间帧<br/>降低输入量，保留事件主状态"]
        summary_batch["累计到 batch<br/>total_frames &gt;= 500<br/>或 5 集<br/>或到 max_chapters"]
        prev_context["附带最近 3 条历史章节概要<br/>维持跨集人物与因果连续性"]
        chapter_llm["视觉 LLM<br/>输出章节 summary JSON 数组"]
        parse_chapter["清理 markdown fence<br/>json.loads 校验"]
        retry_chapter["失败重试<br/>最多 MAX_RETRIES"]
        chapters_info["chapters_info<br/>逐集概要列表"]
        full_llm["文本 LLM<br/>合并为 full_summary"]
    end

    subgraph role_loop["人物识别 LLM loop"]
        role_select["按 scene 时长选帧<br/>&gt; 3s 取首尾帧<br/>&lt;= 3s 取中间帧"]
        role_batch["累计到 batch<br/>total_frames &gt;= 500<br/>或 max_chapters"]
        frame_token["为每帧注入<br/>chapter_rank + frame_index"]
        role_llm["视觉 LLM<br/>识别最多 3 位主要人物"]
        parse_role["解析 characters JSON<br/>回填 iconic frame 图片路径"]
        retry_role["失败重试<br/>最多 MAX_RETRIES"]
        role_info["characters<br/>name / role_title / iconic_frames"]
    end

    scenes --> summary_select --> summary_batch --> prev_context --> chapter_llm --> parse_chapter --> chapters_info --> full_llm
    chapter_llm -. error .-> retry_chapter -. retry .-> chapter_llm
    scenes --> role_select --> role_batch --> frame_token --> role_llm --> parse_role --> role_info
    role_llm -. error .-> retry_role -. retry .-> role_llm

    full_llm --> final["construct_playlet_info<br/>playlet_description + playlet_role_list"]
    role_info --> final
```

这个 loop 里有三层约束。第一层是输入约束：每张图在进入 LLM 前会转成 base64，并通过 `smart_resize` 控制像素上限，避免单次请求被少量大图挤爆。第二层是 batch 约束：剧情概要分支最多积累 5 集或约 500 帧就发起一次请求，人物分支在达到 `max_chapters` 或约 500 帧时发起请求。第三层是输出约束：prompt 要求只返回 JSON；代码会去掉 Markdown code fence 这类包裹，再解析 JSON，失败则丢弃当前 batch 或进入重试。这种设计的核心不是让 LLM 写得更漂亮，而是把背景生成控制在「可批处理、可解析、可回溯」的工程边界内。

剧情概要分支有一个小的记忆窗口：每次生成新 batch 的章节概要时，会把最近 `max_prev_summarie = 3` 条历史概要附加给 LLM。这样做是为了让第 4、5 集之后的总结知道前面人物关系和冲突因果，但又不把所有历史概要都塞回上下文，避免早期错误被过度放大。所有章节概要生成后，再调用一次文本 LLM，把 `chapters_info` 合并为不超过 800 字的 `full_summary`，重点保留主要人物关系、关键事件、冲突转折和结局走向。

人物识别分支和剧情概要分支并行，但它的目标更窄：识别最多 3 位主要人物，并为每个人返回不超过 2 个标志帧。这里每张输入图前都会插入 `chapter_rank` 和 `frame_index` 文本标记，LLM 返回 iconic frame 后，代码再用 `Chapter2Idx2Path` 把这个临时索引映射回真实图片路径。也就是说，人物分支输出的不是抽象的「这个人很重要」，而是可以定位到具体集数和具体帧的证据。

### 2.2 背景生成的帧选择策略

抽帧发生在背景生成之前。每部短剧以 `album_id` 或 `playlet_id` 作为主键，最多取前 15 集进入背景构建。这个上限主要是成本和信息密度的折中：短剧的核心人物关系、世界观设定和主要矛盾通常在前几集集中出现，继续扩大集数会显著增加视觉输入量，但对「前期剧情背景」的边际收益不稳定。

单集视频先使用 `scene_detect` 这类开源镜头切分工具划分 scene。每个 scene 再按时间三等分抽取三帧，分别覆盖 scene 的前段、中段和后段。以 `frames_demo.json` 第一集为例，`scene_index_1` 的时间区间是 `[0.0, 2.27]` 秒，对应抽到 `0.57`、`1.13`、`1.7` 秒三帧；`scene_index_2` 的时间区间是 `[2.27, 6.37]` 秒，对应抽到 `3.29`、`4.32`、`5.34` 秒三帧。这个采样方式比只取关键帧更稳，尤其适合短剧里「人物进场、冲突升级、表情反应」连续发生的场景。

离线抽帧得到的是候选证据池，真正进入背景生成 LLM 时还会二次选择。剧情概要分支采用「每个 scene 取中间帧」策略，即 `images[len(images) // 2]`。原因是逐集概要更关心 scene 的主状态：人物是否同处一室、冲突对象是谁、道具或环境是否已经出现。中间帧通常比首帧更少受到转场影响，也比尾帧更少只留下反应结果，因此适合作为低成本的剧情代表帧。这个策略会牺牲一部分动作连续性，但能显著降低输入帧数，使多集 batch 和历史概要上下文可以一起进入模型。

人物识别分支采用另一套策略：如果 `scene_duration > 3.0` 秒，就取该 scene 的首帧和尾帧；如果 `scene_duration <= 3.0` 秒，就只取中间帧。长 scene 往往包含人物进出、镜头反打或姿态变化，首尾帧更容易覆盖不同人物和正脸机会；短 scene 的信息窗口很窄，首尾差异不稳定，只取中间帧可以避免把转场、模糊表情或不完整构图送进模型。人物分支还会让 LLM 优先挑「人物清晰可见、正脸、无遮挡、带字幕或场景提示」的标志帧，所以帧选择不只是为了识别人物，也是在为后续可视证据绑定做准备。

因此，背景生成里的帧选择是两级策略：第一级离线抽帧保证每个 scene 有前中后三个候选证据；第二级按任务目标过滤候选帧。剧情概要更偏向事件代表性，所以用中间帧；人物识别更偏向身份可辨识度，所以长 scene 用首尾帧增加覆盖面，短 scene 用中间帧保稳定。抽帧结果不直接等同于剧情理解，它只是把视频压成一组有时间戳和场景边界的视觉证据，后续概要生成和人物识别都基于这组证据展开。

### 2.3 `frame_demo.json` 字段格式

这里的 `frame_demo.json` 指抽帧后的剧集级索引文件；仓库中的示例文件名是 `frames_demo.json`，当前样例包含 13 集。顶层是剧集数组，每个元素对应一集。单集下的 `frames` 是该集所有 scene 的抽帧结果，通常同一个 `scene_index` 会出现 3 条记录。

| 层级 | 字段                     | 含义                                                  |
| ---- | ------------------------ | ----------------------------------------------------- |
| 剧集 | `playlet_id`           | 短剧 ID。业务上也可理解为`album_id` 对齐后的剧 ID。 |
| 剧集 | `episode_gid`          | 单集视频 ID。                                         |
| 剧集 | `episode_rank`         | 当前集序号，示例中为字符串形式。                      |
| 剧集 | `frames`               | 当前集抽到的全部帧列表。                              |
| 帧   | `video_duration`       | 单集视频总时长，单位为秒。                            |
| 帧   | `total_frames`         | 单集视频总帧数。                                      |
| 帧   | `scene_index`          | scene 编号，由镜头切分结果产生。                      |
| 帧   | `scene_frame_interval` | 当前 scene 在原视频中的帧号区间。                     |
| 帧   | `scene_time_interval`  | 当前 scene 在原视频中的时间区间。                     |
| 帧   | `scene_duration`       | 当前 scene 的持续时间。                               |
| 帧   | `frame_idx`            | 当前抽样帧在原视频中的帧号。                          |
| 帧   | `cut_time`             | 当前抽样帧对应的视频时间点，单位为秒。                |
| 帧   | `image`                | 抽样帧图片路径。                                      |

这个字段设计保留了两类信息：一类是剧集身份信息，用于把多集内容重新按顺序组织；另一类是可追溯的视觉证据，用于回到原视频定位某个场景、某个时间点和某张图片。后续 Judge 判断 hallucination 时，真正依赖的就是这类可回溯信息。

### 2.4 剧情背景生成

剧情背景生成先按 `episode_rank` 恢复剧集顺序，再按 `scene_index` 聚合同一场景下的三帧。每个 scene 的三帧提供了一个很小的时间窗口：前一帧看场景状态，中间帧看冲突或动作，后一帧看结果或反应。概要生成实际送入 LLM 的是每个 scene 的中间帧，相当于用一个稳定代表帧去近似该 scene 的主要事件状态。对短剧解说而言，这比随机抽帧更容易保住事件链，也比全量送三帧更省上下文。

生成时先得到逐集概要，再把逐集概要合并成短剧前期背景。逐集概要解决「每一集发生了什么」，合并概要解决「这部剧到当前阶段讲到了哪里」。这里不追求文学化表达，而是保留人物关系、关键事件、冲突转折和结局走向。后续生成解说词时，模型可以基于这段背景判断哪些事件已经发生，哪些内容只能作为悬念处理。

这个环节的风险在于摘要会传播错误。若某一集把人物身份或事件因果概括错，后面的全局背景会继续吸收这个错误。因此背景信息更适合作为可核验上下文，而不是不可质疑的事实源；高价值样本仍然需要抽样回看原始帧。

### 2.5 人物背景生成

人物背景和剧情概要使用同一批抽帧证据，但关注点不同。剧情概要关心事件推进，人物背景关心「谁是主要人物、身份是什么、有没有清晰的标志帧」。因此人物分支不会固定每个 scene 只取一个代表帧，而是根据 scene 时长决定覆盖范围：长 scene 取首尾帧，短 scene 取中间帧。标志帧的作用不是展示素材，而是让后续流程能把人物描述和视觉证据绑定起来。

最终人物信息会整理为人物名、人物身份或关系、代表帧列表。短剧任务里这一步很有价值，因为人物身份经常承担叙事功能：总裁、少主、假千金、重生女主、赘婿、家族长辈等标签会直接影响解说词的措辞和冲突解释。没有人物背景时，模型更容易把「画面里的人」写成泛化角色，甚至把不同人物混在一起。

### 2.6 最终背景结构

背景生成完成后，会得到一份以短剧为粒度的结构化信息：

| 字段                                   | 含义                                     |
| -------------------------------------- | ---------------------------------------- |
| `playlet_id`                         | 短剧 ID。                                |
| `playlet_description`                | 基于前若干集生成的剧情背景概要。         |
| `playlet_role_list`                  | 主要人物列表。                           |
| `playlet_role_list.role`             | 人物名或可识别的人物称谓。               |
| `playlet_role_list.role_description` | 人物身份、关系或叙事功能。               |
| `playlet_role_list.role_img`         | 人物标志帧，保留集数、帧索引和图片路径。 |

这份背景不是最终对用户展示的文案，更像训练样本的上下文层。它把视频压缩成「剧情概要 + 人物表 + 标志帧证据」三类信息。后续无论是生成解说词，还是用 LLM-as-Judge 做剧情一致性评估，都可以沿着这份结构回到原始视觉证据，判断一句解说是否有依据。

## 3. AURA-GRPO: GRPO + LLM-as-Judge + Unified Reward

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

- 将 GRPO 与 LLM-as-Judge 结合，主要是为了解决可验证奖励覆盖不到的任务。解说词生成、小说生成等开放式任务没有唯一正确答案，奖励信号只能来自偏好、约束和人工可解释的 rubric。

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

- Judge prompt:

````python
judge_prompt = """你是一个用于自动评估「短剧解说词」质量的 LVLM evaluator。

输入包含三部分：
1. 剧集背景信息；
2. 按剧集（chapter_rank）与场景（scene）排列的视频帧，包括时间戳与视觉描述；
3. 已按句子粒度拆分的解说文本列表。

原始 system_prompt 只用于检查解说是否遵循指定风格和格式约束。评分必须基于视频帧、时间戳、人物信息和剧情背景中的可验证证据。

你的任务是对解说文本的三个主维度给出客观评分，并提供一句不超过 30 字的理由。不要计算综合分。仅输出 JSON，不输出任何解释性文字。

---
## 评分档位

* 1: 差
* 2: 一般
* 3: 较好
* 4: 优秀

除 narrative_consistency 触发强制归零外，其余维度均在 1-4 分范围内评分。

---
## 强制约束

### 1. Hallucination 定义与惩罚

若解说文本断言的事实或事件无法从剧集背景、人物信息、视频帧或时间戳中验证，则记为一次 hallucination。

以下情况不计为 hallucination：
1. 结尾预留钩子，例如「这一次，他要让所有仇人血债血偿」「一场大戏即将上演」；
2. 人物夸张称谓，例如「亿万总裁」；
3. 时间跨度类短剧设定，例如「修炼万年」「女人照顾瘫痪婆婆三十年」「傅家少爷昏迷三天」。

惩罚规则：
* hallucination_count >= 4 时，`narrative_consistency` 直接记为 0 分。
* hallucination_count < 4 时，每出现 1 次 hallucination，`narrative_consistency` 扣 1 分。
* 必须在 `notes` 字段列出至多 4 条典型 hallucination，包括句子片段和简短理由；没有则留空字符串。

### 2. 证据映射要求

任一条件满足即可判定该句证据命中：
* 直接视觉证据：视频帧、时间段描述或时间戳明确包含句子中的可观测事实，例如人物动作、道具、场景、显著表情或直接对白。
* 原声同步证据：若句子标注为「原声」，且 OCR 或对白与时间戳对应，则视为命中。
* 保守推断证据：视觉描述明确暗示该事件，且推断不超过一层跳跃，不跨越关键事实。此类命中需在 `notes` 中标注「间接证据」。

`narrative_consistency` 的基础分按证据映射比例计算：

* 证据映射比例 = evidence_mapping_num / total_sentences
* 基础分 = 证据映射比例 * 4
* 最终分 = 基础分 - hallucination_count
* 最终分低于 0 时按 0 计，高于 4 时按 4 计。

### 3. 密度归一化

爆点关键词示例：`["没想到", "注意看", "下一秒", "竟", "竟然", "结果", "惊天", "爆炸", "更可怕的是", "意外"]`。可扩展，但检测时必须使用精确短语匹配，不做模糊匹配。

计算方式：

* 爆点句数 = 包含任一爆点关键词的句子数，每句只计 1 次。
* 爆点密度 = 爆点句数 / 总句数，保留两位小数。
* 结构覆盖密度 = 已识别结构单元数（起/承/转/悬 count）/ 4，范围为 0-1。

`expressive_appeal` 以爆点密度为主信号：

* 0 <= 密度 <= 0.20: 1 分
* 0.20 < 密度 <= 0.30: 2 分
* 0.30 < 密度 <= 0.40: 3 分
* 结尾有且只有一个有效钩子时，可在上述基础上加 1 分，最高不超过 4 分。

`expressive_appeal.rationale` 必须包含爆点密度值，例如「爆点密度0.27」，且不超过 30 字。

### 4. 爆点滥用惩罚

若爆点密度 > 0.40，`expressive_appeal` 仍以 3 分作为密度基线；爆点滥用只扣 `oral_fluency`：

* 0.40 < 密度 <= 0.60: `oral_fluency` 扣 1 分
* 0.60 < 密度 <= 0.80: `oral_fluency` 扣 2 分
* 密度 > 0.80: `oral_fluency` 扣 3 分

扣分后 `oral_fluency` 最低为 1 分。若触发该规则，需在 `notes` 或 `style_violation` 中简要说明「爆点滥用，已在 oral_fluency 扣分」，并给出爆点密度。

### 5. 风格合规单独输出

检查原始 system_prompt 中的风格和格式限制，例如是否必须包含「原声」「解说词」标签、句数范围、标签格式等。

若发现严重风格违规，在 `style_violation` 中简短描述，并可作为扣 1 分依据，仅用于 `expressive_appeal` 或 `oral_fluency`。风格合规结果不得用于提高任何维度分数。

---
## 三个主维度的判定要点

### A. 剧情一致性（narrative_consistency）

必须给出视觉对齐证据，并按证据映射比例与 hallucination 惩罚计算得分。若出现 hallucination，请在 `notes` 中列出至多 4 条典型样例。

### B. 吸引力（expressive_appeal）

以爆点密度为主信号，结尾钩子作为加分项。若爆点词或情绪词集中堆叠，可扣 1 分。`rationale` 必须包含「爆点密度x.xx」。

### C. 口语流畅性 / 节奏（oral_fluency）

主要检查爆点滥用、句长分布、书面化连接词比例，以及原声插入是否自然。句长以 8-20 字为较优区间。`rationale` 需说明句长分布或扣分原因，例如「句长过短且爆点密度0.55扣2分」。

---
## 输出 JSON 结构

必须严格遵守以下结构，且仅输出 JSON。字段顺序不限。

```json
{
  "evidence": [
    "句片段 —— frame_chapterX_sceneY_timestamp（视觉证据）"
  ],
  "metrics": {
    "total_sentences": 0,
    "爆点句": 0,
    "爆点密度": 0.0,
    "识别结构单元数": 0,
    "结构覆盖密度": 0.0,
    "hallucination_count": 0,
    "evidence_mapping_num": 0
  },
  "style_violation": "",
  "notes": "",
  "output": {
    "narrative_consistency": {"score": 0, "rationale": "不超过30字"},
    "expressive_appeal": {"score": 1, "rationale": "爆点密度0.00"},
    "oral_fluency": {"score": 1, "rationale": "不超过30字"}
  }
}
```

格式说明：

1. `narrative_consistency` 按 `evidence_mapping_num / total_sentences` 与 `hallucination_count` 计算。
2. `hallucination_count = 1` 时，`narrative_consistency` 最高 3 分；`hallucination_count` 为 2-3 时最高 2 分；`hallucination_count >= 4` 时直接 0 分。
3. `expressive_appeal` 依据爆点密度与结尾钩子计算，`rationale` 必须包含密度值，例如「爆点密度0.62」。
4. `style_violation` 只记录风格合规问题，不得用于提高任何主维度分数。
5. `oral_fluency` 依据句长分布、书面化连接词比例、原声插入自然度和爆点滥用扣分计算。

# 任务

接下来你将收到视频背景信息、视频帧信息和一段解说文本输出列表。请按上述规则完成评判和打分。"""

````

## 4. Reward Hacking Issues
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
>
> $$
> u = 0.902525
> $$
>
> $$
> sigma = 0.011
> $$
>
> Advantage = [1.2979, 0.5837, -0.835, -1.6221, -1.2015, 0.5286, 0.8944, 0.3545]
> max_reward_diff = 0.0319.     max_advantage_diff = 2.499

- 缓解策略：KL 散度约束、早停机制、奖励模型集成、定期更新评判器。

## 5. Related Work: GRPO Beyond Objective Tasks

GRPO 最初主要用于数学、代码等可验证任务（DeepSeek-R1）。开放式生成任务的问题更麻烦：创意写作、视频解说、图像描述都没有唯一正确答案，奖励信号通常来自主观偏好、结构约束或外部 judge。以下按任务类型整理几项相关工作。

### 5.1 Video / Image Captioning

- **[VideoCap-R1](https://arxiv.org/abs/2506.01725)** 系统性地将 GRPO 用到视频多模态大模型（Video MLLM）。模型先在 `<think>` 标签内进行结构化思考（分析视频主体、属性与动作），再生成完整描述。奖励由两部分组成：LLM-free Think Scorer 评估思考质量，LLM-assisted Caption Scorer 评估输出质量。论文报告仅用 1.5K 训练样本，在 DREAM-1K (+4.4 event F1)、VDC (+4.2 Acc)、CAREBENCH (+3.1 action F1) 上超过 SFT 对照组。

  1. 从 ground truth caption 中提取事件列表。
  2. 对每个事件，问 Qwen2.5-72B："预测 caption 是否蕴含（entail）了这个事件？" 二元判断。
  3. 计算 event coverage 比例，然后用阶梯函数离散化为 {0, 0.5, 1}：

![alt text](assets/1.png)

- **[AVoCaDO](https://arxiv.org/abs/2510.10395)** 在音视频联合描述任务上采用两阶段后训练：AVoCaDO SFT（10.7 万高质量音视频对齐字幕微调）+ AVoCaDO GRPO（基于关键事件对齐的奖励函数优化时序一致性与对话准确性）。奖励设计包含 checklist-based 和 dialogue-based 两类，由 LLM 验证，同时引入长度正则化，避免模型通过重复内容获得偏高奖励。
  ![alt text](assets/2.png)

### 5.2 Creative Writing / Long Article Generation

* **[RLMR（Reinforcement Learning with Mixed Rewards）](https://arxiv.org/abs/2508.18642)** 在在线 RL 训练中同时使用主观偏好和客观约束验证。方法采用 GRPO 框架，设计写作质量奖励模型（评估主观文学性）和约束验证模型（评估客观指令遵循），再通过动态权重调整，让违反约束的样本在 GRPO 中获得负优势值。论文报告在 WriteEval 人工专家评测中获得 72.75% 胜率，IFEval 从 83.36% 提升至 86.65%。
  ![alt text](assets/3.png)
* **[Rewarding Creativity（GenRM for Storytelling）](https://arxiv.org/abs/2601.07149)** 提出面向故事生成的生成式奖励模型（GenRM），用情节连贯性、角色发展、创意原创性等维度和显式推理链评判故事偏好。GenRM 先通过 SFT 学习推理链，再用 GRPO + 熵正则化奖励塑形在偏好数据上精调。论文的人类评测结果显示，其故事生成质量高于 Gemini-2.5-Pro 等系统。
  ![alt text](assets/4.png)

  - 微调(SFT + GRPO) GenRM
  - 训练策略中的奖励
    1. 在每组 rollout 的 n 个回复中，随机选一个作为 pivot（锚点）
    2. 其余每个回复与 pivot 组成 pair，由 GenRM 做 pairwise 判断
    3. 优于 pivot → r=+1，劣于 pivot → r=−1，pivot 本身 → r=0

## 6. AURA-GRPO: Adaptive Unified Reward Alignment GRPO

### 6.1 Problems and Motivation

1. 使用 GRPO 训练时，如果直接使用 `LLM-as-Judge` 产生 point-wise reward，极易出现 reward hacking，因此往往需要大量额外的 format reward 约束。
2. 对于偏主观任务（如解说词、美观度、邮件写作），很难给出绝对分数；但若把样本组成 pair，我们通常能更稳定地判断谁更好。
3. 不同主观任务下，评测 prompt（rubric）差异很大，且需要高度定制；一旦 prompt 本身存在偏差，最终也可能被模型利用。
4. 对主观/生成任务而言，Format reward 往往比客观任务更难定义，也更容易被 hack。

![alt text](assets/5.png)

### 6.2 Judge Prompt 自动优化（Automatic prompt optimization）

LLM-as-Judge 的评分质量直接影响奖励信号。手工编写 Judge Prompt 时，最常见的问题是维度盲区：人工设定的评分维度（如“情感表达”“流畅性”）可能漏掉训练中真正区分样本质量的因素。

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

### 6.3 Win-Rate Reward

#### 6.3.1 对比拓扑设计

对于每个 query，策略模型 $\pi_{\theta}$ 生成 $G=8$ 个 rollout ${o_1,…,o_8}$，同时持有一个 Gemini 生成的参考解说词 $o_{ref}$。我们构造 20 组 pairwise 比较，分为两类：

| 比较类型           | 对数 | 具体配对                                                                           | 目的                     |
| ------------------ | ---- | ---------------------------------------------------------------------------------- | ------------------------ |
| Rollout vs Gemini  | 8    | 每个$o_i$ vs $o_{ref}$                                                         | 与外部高质量锚点对齐     |
| Rollout vs Rollout | 12   | (0,1), (2,3), (4,5), (6,7), (0,7), (1,6), (2,5), (3,4), (0,2), (1,3), (5,7), (4,6) | 组内自博弈，建立完整排序 |

该拓扑让每个 rollout 恰好参与 3 次组内对比 + 1 次 Gemini 对比 = 4 次比较，使信息量分布更均匀，也控制了 judge 调用成本。12 对组内比较覆盖

$$
\binom{8}{2}=28
$$

种可能配对中的 43%，可以近似建立组内排序。

#### 6.3.2 异质加权的 Win 累积

不同类型的比较胜利获得不同的权重：

$$
w_i = \begin{cases} \frac{1}{2}=0.5 & \text{if } o_i \text{ 赢了 Gemini 参考} \\ \frac{7}{6} \approx 1.167 & \text{if } o_i \text{ 赢了另一个 rollout} \end{cases}
$$

对每个 rollout $o_i$，累计其加权胜场数和总参与比较数：

$$
\text{wins}_i = \sum_{\text{vs gemini}} w_{\text{gemini}} \cdot \mathbb{1}[o_i \text{ wins}] + \sum_{\text{vs rollout}} w_{\text{rollout}} \cdot \mathbb{1}[o_i \text{ wins}]
$$

$$
\text{comps}_i = |\{比较 \mid o_i \text{ 参与}\}| = 4
$$

胜率（Win Rate）为：

$$
r_i^{\text{win}} = \frac{\text{wins}_i}{\text{comps}_i}
$$


* 为什么需要与 Gemini 比较
  1. 引导模型往正确方向优化，避免在纯自博弈中陷入自我强化的封闭循环。
  2. 同时可为 format reward 提供外部参考。
* 为什么组内胜利权重更高？
  1. Rollout 输出会随着训练持续变化，而 Gemini 输出相对固定。
  2. 组内胜利权重更高，可以减少策略过度贴近 Gemini 风格的倾向。

### 6.4 Format Reward

#### 6.4.1 乘性调制

Format reward 在主观/生成式 GRPO 训练中很容易主导优化方向。实验中围绕 format 做了三类处理：

1. 显式加入 Format reward。
2. 提高其在 final reward 中的影响。
3. 将 Format reward 设计成近似“门控”的单元。

最终 reward 函数定义如下：

$$
r_i = r_i^{\text{win}} \times r_i^{\text{fmt}}
$$

Win-rate reward 衡量内容质量，format reward 衡量格式合规性，二者通过乘法组合。
乘性组合有三个性质：

- 当 $r_i^{\text{fmt}} = 0$ 时，无论内容质量多高，$r_i = 0$（格式严重违规 → 一票否决）
- 当 $r_i^{\text{fmt}} = 1$ 时，奖励完全由内容质量决定（格式完美 → 不干扰）
- 当 $0 < r_i^{\text{fmt}} < 1$ 时，奖励被平滑压缩

#### 6.4.2 格式奖励泛化

Gemini 预生成的参考解说词已经满足当前任务的 format 要求，因此可以把 ref 视为隐式格式锚点，用它判断 rollout 解说词的句数、句长和总长度是否偏离任务规范。这样不需要为每个业务反复手写 format 规则。

Format reward 使用高斯惩罚函数，并以 Gemini 参考作为格式锚点：

$$
r_i^{\text{fmt}} = \min\left(g\!\left(\frac{N_i^{\text{sent}}}{N_{\text{ref}}^{\text{sent}}}\right),\; g\!\left(\frac{\bar{L}_i}{\bar{L}_{\text{ref}}}\right),\; g\!\left(\frac{L_i^{\text{total}}}{L_{\text{ref}}^{\text{total}}}\right)\right)
$$

其中

$$
g(x) = \exp\!\left(-\frac{(x-1)^2}{2\sigma^2}\right)
$$

为高斯函数：
![alt text](assets/6.png)

| 维度       | 含义                        | σ  | ratio=1.5 时的惩罚 |
| ---------- | --------------------------- | --- | ------------------ |
| 句数比     | rollout 句数 / ref 句数     | 0.3 | g ≈ 0.33          |
| 平均句长比 | rollout 均句长 / ref 均句长 | 0.3 | g ≈ 0.33          |
| 总长比     | rollout 总长 / ref 总长     | 0.4 | g ≈ 0.57          |

最终取三个维度的最小值。只要句数、平均句长或总长度中任一项偏离过大，format reward 都会压低最终奖励。

### 6.5 完整奖励信号流

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

## 7. 总结

与 RLCS 相比，AURA-GRPO 的差异主要在四点：

1. 信号密度更高。20 次比较约为 RLCS 的 4-5 倍（4 次/rollout vs 1 次/rollout），使 GRPO 的优势估计 $\hat{A}_i$ 方差更低、区分度更高。
2. 奖励系统不需要额外训练。RLCS 需要专业编剧标注、CoT 蒸馏、SFT 和 GRPO 两阶段训练 GenRM；本文方案直接使用通用 LLM 作为 judge，通过 prompt 工程和自动优化获得可用的评判质量。
3. 双锚点可以降低漂移风险。Gemini 参考提供不随训练变化的外部质量锚点，避免纯自博弈中"策略越练越偏但自我感觉越来越好"的问题；RLCS 的随机 pivot 来自当前策略，缺少这类外部校准。
4. Format reward 更容易迁移。以 Gemini 参考解说词作为 format 锚点后，句数比、平均句长比、总长比可以通过高斯惩罚函数自动度量，并以乘法门控方式接入 win rate reward。这会压缩"牺牲格式换质量"的 exploit 空间，也减少了为每个任务手写 format 规则的成本。

仍需验证的部分：

1. Gemini 参考的质量天花板。当前 format reward 和 win rate reward 均以 Gemini 生成的参考解说词为锚点。如果 Gemini 参考存在系统性偏差（如特定题材下风格单一），策略模型的优化方向也会被约束。后续可以引入多源参考（不同模型或人工标注），或在训练中期用当前模型的最优输出动态更新 ref pool。
2. Pairwise 比较的计算开销。20 次 pairwise 比较虽然提供了高密度信号，但每个 training step 都需要 20 次 LLM-as-Judge 调用。训练后期可以减少比较次数，例如仅保留 vs Gemini 和少量组内比较。
3. Judge prompt 还需要业务规则。当前生产的解说词会保留剧集中的完整人名，阅读体验会受影响。后续 one-stage 训练中，可以在 judge prompt 中手动加入人物代称相关规则。

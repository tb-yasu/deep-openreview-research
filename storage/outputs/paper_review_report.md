# 論文レビューレポート

**生成日時**: 2025年11月04日 11:33

## 検索条件

- **学会**: NeurIPS 2025
- **キーワード**: large language model
- **検索論文数**: 149件
- **評価論文数**: 149件
- **ランク対象論文数**: 55件

## 評価基準

- **研究興味**: large language models, efficiency, fine-tuning, reasoning, multimodal, agents
- **最小関連性スコア**: 0.3
- **最小レビュー評価**: 4.5/10
- **新規性重視**: はい
- **インパクト重視**: はい

## 統計情報

- **平均総合スコア**: 0.458
- **最高スコア**: 0.560
- **最低スコア**: 0.377
- **平均レビュー評価**: 4.68/10

## トップ論文

### 1. VidEmo: Affective-Tree Reasoning for Emotion-Centric Video Foundation Models

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.560** |
| 関連性 | 0.450 |
| 新規性 | 0.630 |
| インパクト | 0.585 |
| OpenReview評価 | 4.50/10 |

**著者**: Zhicheng Zhang, Weicheng Wang, Yongjie Zhu, Wenyu Qin, Pengfei Wan 他2名

**キーワード**: Video, Multimodal Large Language Model, Video MLLM, Facial Analysis, Video Emotion Analysis

**概要**: Understanding and predicting emotions from videos has gathered significant attention in recent studies, driven by advancements in video large language models (VideoLLMs). While advanced methods have made progress in video emotion analysis, the intrinsic nature of emotions—characterized by their open-set, dynamic, and context-dependent properties—poses challenge in understanding complex and evolving emotional states with reasonable rationale. To tackle these challenges, we propose a novel affective cues-guided reasoning framework that unifies fundamental attribute perception, expression analysis, and high-level emotional understanding in a stage-wise manner. At the core of our approach is a family of video emotion foundation models (VidEmo), specifically designed for emotion reasoning and instruction-following. These models undergo a two-stage tuning process: first, curriculum emotion learning for injecting emotion knowledge, followed by affective-tree reinforcement learning for emotion reasoning. Moreover, we establish a foundational data infrastructure and introduce a emotion-centric fine-grained dataset (Emo-CFG) consisting of 2.1M diverse instruction-based samples. Emo-CFG includes explainable emotional question-answering, fine-grained captions, and associated rationales, providing essential resources for advancing emotion understanding tasks. Experimental results demonstrate that our approach achieves competitive performance, setting a new milestone across 15 face perception tasks.

**評価**: 平均レビュースコア: 4.50/10 | レビュアー信頼度: 4.50/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.56 | 関連性: 0.45 | 新規性: 0.63 | インパクト: 0.58

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=x8lg9aihwl)
- [PDF](https://openreview.net/pdf?id=x8lg9aihwl)

---

### 2. KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.545** |
| 関連性 | 0.550 |
| 新規性 | 0.445 |
| インパクト | 0.640 |
| OpenReview評価 | 5.50/10 |

**著者**: Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen 他24名

**キーワード**: LLM; Evaluation; RL; Game

**概要**: Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM’s general reasoning potential. To address this limitation, we introduce the **Knowledge Orthogonal Reasoning Gymnasium (KORGym)**, a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments.

**評価**: 平均レビュースコア: 5.50/10 | レビュアー信頼度: 4.25/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.54 | 関連性: 0.55 | 新規性: 0.45 | インパクト: 0.64

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=uAeqQePu4c)
- [PDF](https://openreview.net/pdf?id=uAeqQePu4c)

---

### 3. LogicTree: Improving Complex Reasoning of LLMs via Instantiated Multi-step Synthetic Logical Data

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.542** |
| 関連性 | 0.500 |
| 新規性 | 0.575 |
| インパクト | 0.545 |
| OpenReview評価 | 5.00/10 |

**著者**: Zehao Wang, Lin Yang, Jie Wang, Kehan Wang, Hanzhu Chen 他5名

**キーワード**: large language model, logical reasoning, data synthesis

**概要**: Despite their remarkable performance on various tasks, Large Language Models (LLMs) still struggle with logical reasoning, particularly in complex and multi-step reasoning processes. 
Among various efforts to enhance LLMs' reasoning capabilities, synthesizing large-scale, high-quality logical reasoning datasets has emerged as a promising direction. 
However, existing methods often rely on predefined templates for logical reasoning data generation, limiting their adaptability to real-world scenarios. 
To address the limitation, we propose **LogicTree**, a novel framework for efficiently synthesizing multi-step logical reasoning dataset that excels in both complexity and instantiation.
By iteratively searching for applicable logic rules based on structural pattern matching to perform backward deduction, **LogicTree** constructs multi-step logic trees that capture complex reasoning patterns. 
Furthermore, we employ a two-stage LLM-based approach to instantiate various real-world scenarios for each logic tree, generating consistent real-world reasoning processes that carry contextual significance.   This helps LLMs develop generalizable logical reasoning abilities across diverse scenarios rather than merely memorizing templates.
Experiments on multiple benchmarks demonstrate that our approach achieves an average improvement of 9.4\% in accuracy on complex logical reasoning tasks.

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 3.25/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.54 | 関連性: 0.50 | 新規性: 0.57 | インパクト: 0.54

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=z4AMrCOetn)
- [PDF](https://openreview.net/pdf?id=z4AMrCOetn)

---

### 4. SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.537** |
| 関連性 | 0.500 |
| 新規性 | 0.575 |
| インパクト | 0.530 |
| OpenReview評価 | 5.00/10 |

**著者**: Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng 他2名

**キーワード**: Audio-Visual, 3D Spatial Reasoning, Multi-modal LLMs, Audio-Visual QA

**概要**: 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of carefully curated question–answer pairs probing both directional and distance relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 3.00/5 | レビュー数: 4 | 採択判定: Accept (oral) | 
総合スコア: 0.54 | 関連性: 0.50 | 新規性: 0.57 | インパクト: 0.53

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=zwCb9cKHpd)
- [PDF](https://openreview.net/pdf?id=zwCb9cKHpd)

---

### 5. SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.537** |
| 関連性 | 0.500 |
| 新規性 | 0.500 |
| インパクト | 0.605 |
| OpenReview評価 | 5.00/10 |

**著者**: Jinyang Li, Xiaolong Li, Ge Qu, Per Jacobsson, Bowen Qin 他15名

**キーワード**: text-to-SQL, LLM, SQL issues

**概要**: Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging on SQL issues. In order to address this gap, we introduce **BIRD-CRITIC**, a new SQL issue debugging benchmark comprising 530 carefully curated PostgreSQL tasks (**BIRD-CRITIC-PG**) and 570 multi-dialect tasks (**BIRD-CRITIC-Multi**), which are distilled from authentic user issues and replayed within new environments to facilitate rigorous and contamination-free evaluation. Baseline evaluations on BIRD-CRITIC underscore the task's complexity, with the leading reasoning model **O3-Mini** achieving only 38.87% success rate on **BIRD-CRITIC-PG** and 33.33% on **BIRD-CRITIC-Multi**. Meanwhile, realizing open-source models for database tasks is crucial which can empower local development while safeguarding data privacy. Therefore, we present **Six-Gym** (**S**ql-f**IX**-Gym), a training environment for elevating the capabilities of open-source models specifically for SQL issue debugging. This environment leverages **SQL-Rewind** strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose *f*-Plan Boosting, which extracts high-level debugging plans automatically from SQL solutions, enabling the teacher LLMs to harvest and produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, **BIRD-Fixer**. Based on Qwen-2.5-Coder-14B, **BIRD-Fixer** raises its success rate to 38.11% on **BIRD-CRITIC-PG** and 29.65% on **BIRD-CRITIC-Multi**, surpassing many leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities for both research and industry.

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 4.25/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.54 | 関連性: 0.50 | 新規性: 0.50 | インパクト: 0.60

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=yRxXTdElLv)
- [PDF](https://openreview.net/pdf?id=yRxXTdElLv)

---

### 6. SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.521** |
| 関連性 | 0.500 |
| 新規性 | 0.500 |
| インパクト | 0.560 |
| OpenReview評価 | 5.00/10 |

**著者**: Gabriele Oliaro, Zhihao Jia, Daniel F Campos, Aurick Qiao

**キーワード**: speculative decoding, LLM agents, model-free speculation, SWE-Bench, LLM inference

**概要**: Speculative decoding is widely adopted to reduce latency in large language model (LLM) inference by leveraging smaller draft models capable of handling diverse user tasks. However, emerging AI applications, such as LLM-based agents, present unique workload characteristics: instead of diverse independent requests, agentic frameworks typically submit repetitive inference requests, such as multi-agent pipelines performing similar subtasks or self-refinement loops iteratively enhancing outputs. These workloads result in long and highly predictable sequences, which current speculative decoding methods do not effectively exploit. To address this gap, we introduce \emph{SuffixDecoding}, a novel method that utilizes efficient suffix trees to cache long token sequences from prompts and previous outputs. By adaptively speculating more tokens when acceptance likelihood is high and fewer when it is low, SuffixDecoding effectively exploits opportunities for longer speculations while conserving computation when those opportunities are limited. Evaluations on agentic benchmarks, including SWE-Bench and Text-to-SQL, demonstrate that SuffixDecoding achieves speedups of up to 3.9$\times$, outperforming state-of-the-art methods -- 2.2$\times$ faster than model-based approaches like EAGLE-2/3 and 1.6$\times$ faster than model-free approaches such as Token Recycling. SuffixDecoding is open-sourced.

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 3.50/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.52 | 関連性: 0.50 | 新規性: 0.50 | インパクト: 0.56

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=uwL0vbeEVn)
- [PDF](https://openreview.net/pdf?id=uwL0vbeEVn)

---

### 7. Semantic Representation Attack against Aligned Large Language Models

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.516** |
| 関連性 | 0.500 |
| 新規性 | 0.500 |
| インパクト | 0.545 |
| OpenReview評価 | 5.00/10 |

**著者**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei 他1名

**キーワード**: Semantic Representation, Attack, Aligned Large Language Models

**概要**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content. Current methods typically target exact affirmative responses, suffering from limited convergence, unnatural prompts, and high computational costs. We introduce semantic representation attacks, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs. Rather than targeting exact textual patterns, our approach exploits the semantic representation space that can elicit diverse responses that share equivalent harmful meanings. This innovation resolves the inherent trade-off between attack effectiveness and prompt naturalness that plagues existing methods. Our Semantic Representation Heuristic Search (SRHS) algorithm efficiently generates semantically coherent adversarial prompts by maintaining interpretability during incremental search. We establish rigorous theoretical guarantees for semantic convergence and demonstrate that SRHS achieves unprecedented attack success rates (89.4% averaged across 18 LLMs, including 100% on 11 models) while significantly reducing computational requirements. Extensive experiments show that our method consistently outperforms existing approaches.

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 3.25/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.52 | 関連性: 0.50 | 新規性: 0.50 | インパクト: 0.54

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=yzl5tL0Z2M)
- [PDF](https://openreview.net/pdf?id=yzl5tL0Z2M)

---

### 8. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.514** |
| 関連性 | 0.475 |
| 新規性 | 0.490 |
| インパクト | 0.573 |
| OpenReview評価 | 4.75/10 |

**著者**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang 他1名

**キーワード**: AI-generated Text Detection

**概要**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce **DNA-DetectLLM**, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of **5.55\%** in AUROC and **2.08\%** in F1 score across multiple public benchmark datasets. Code and data are available at https://github.com/Xiaoweizhu57/DNA-DetectLLM.

**評価**: 平均レビュースコア: 4.75/10 | レビュアー信頼度: 4.00/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.51 | 関連性: 0.47 | 新規性: 0.49 | インパクト: 0.57

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=yQoHUijSHx)
- [PDF](https://openreview.net/pdf?id=yQoHUijSHx)

---

### 9. 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.510** |
| 関連性 | 0.500 |
| 新規性 | 0.425 |
| インパクト | 0.605 |
| OpenReview評価 | 5.00/10 |

**著者**: Tianyi Zhang, Mohsen Hariri, Shaochen Zhong, Vipin Chaudhary, Yang Sui 他2名

**キーワード**: Compression, Lossless Compression, LLM, Efficiency, Diffusion Models

**概要**: Large-scale AI models, such as Large Language Models (LLMs) and Diffusion Models (DMs), have grown rapidly in size, creating significant challenges for efficient deployment on resource-constrained hardware. In this paper, we introduce Dynamic-Length Float (DFloat11), a lossless compression framework that reduces LLM and DM size by 30\% while preserving outputs that are bit-for-bit identical to the original model. DFloat11 is motivated by the low entropy in the BFloat16 weight representation of LLMs, which reveals significant inefficiency in the existing storage format. By applying entropy coding, DFloat11 assigns dynamic-length encodings to weights based on frequency, achieving near information-optimal compression without any loss of precision. To facilitate efficient inference with dynamic-length encodings, we develop a custom GPU kernel for fast online decompression. Our design incorporates the following: (i) compact, hierarchical lookup tables (LUTs) that fit within GPU SRAM for efficient decoding, (ii) a two-phase GPU kernel for coordinating thread read/write positions using lightweight auxiliary variables, and (iii) transformer-block-level decompression to minimize latency. Experiments on Llama 3.3, Qwen 3, Mistral 3, FLUX.1, and others validate our hypothesis that DFloat11 achieves around 30\% model size reduction while preserving bit-for-bit identical outputs. Compared to a potential alternative of offloading parts of an uncompressed model to the CPU to meet memory constraints, DFloat11 achieves 2.3--46.2$\times$ higher throughput in token generation. With a fixed GPU memory budget, DFloat11 enables 5.7--14.9$\times$ longer generation lengths than uncompressed models. Notably, our method enables lossless inference of Llama 3.1 405B, an 810GB model, on a single node equipped with 8$\times$80GB GPUs.

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 4.25/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.51 | 関連性: 0.50 | 新規性: 0.42 | インパクト: 0.60

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=xdNAVP7TGy)
- [PDF](https://openreview.net/pdf?id=xdNAVP7TGy)

---

### 10. SpaceServe: Spatial Multiplexing of Complementary Encoders and Decoders for Multimodal LLMs

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.503** |
| 関連性 | 0.500 |
| 新規性 | 0.500 |
| インパクト | 0.510 |
| OpenReview評価 | 5.00/10 |

**著者**: zhicheng li, Shuoming Zhang, Jiacheng Zhao, Siqi Li, Xiyu Shi 他6名

**キーワード**: Multimodal large language models; Inference optimizations; Infrastructure

**概要**: Recent multimodal large language models (MLLMs) marry modality-specific
vision or audio encoders with a shared text decoder. While the encoder is compute-
intensive but memory-light, the decoder is the opposite, yet state-of-the-art serving
stacks still time-multiplex these complementary kernels, idling SMs or HBM in
turn. We introduce SpaceServe, a serving system that space-multiplexes MLLMs:
it decouples all modality encoders from the decoder, and co-locates them on the
same GPU using fine-grained SM partitioning available in modern runtimes. A
cost-model-guided Space-Inference Scheduler (SIS) dynamically assigns SM slices,
while a Time-Windowed Shortest-Remaining-First (TWSRFT) policy batches en-
coder requests to minimise completion latency and smooth decoder arrivals. 
Evaluation shows that SpaceServe reduces time-per-output-token by 4.81×
on average and up to 28.9× on Nvidia A100 GPUs. SpaceServe is available at
https://github.com/gofreelee/SpaceServe

**評価**: 平均レビュースコア: 5.00/10 | レビュアー信頼度: 2.67/5 | レビュー数: 3 | 採択判定: Accept (poster) | 
総合スコア: 0.50 | 関連性: 0.50 | 新規性: 0.50 | インパクト: 0.51

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=w4qJ056WhI)
- [PDF](https://openreview.net/pdf?id=w4qJ056WhI)

---

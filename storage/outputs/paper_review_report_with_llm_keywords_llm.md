# 論文レビューレポート

**生成日時**: 2025年11月05日 10:42

## 検索条件

- **学会**: NeurIPS 2025
- **キーワード**: large language model
- **検索論文数**: 1127件
- **評価論文数**: 1127件
- **ランク対象論文数**: 286件

## 評価基準

- **研究興味**: large language models, efficiency, fine-tuning, reasoning, multimodal, agents
- **最小関連性スコア**: 0.3
- **最小レビュー評価**: 4.5/10
- **新規性重視**: はい
- **インパクト重視**: はい

## 統計情報

- **平均総合スコア**: 0.555
- **最高スコア**: 0.781
- **最低スコア**: 0.385
- **平均レビュー評価**: 4.66/10

## トップ論文

### 1. $\texttt{G1}$: Teaching LLMs to Reason on Graphs with Reinforcement Learning

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.781** |
| 関連性 | 1.000 |
| 新規性 | 0.637 |
| インパクト | 0.632 |
| OpenReview評価 | 4.75/10 |

**著者**: Xiaojun Guo, Ang Li, Yifei Wang, Stefanie Jegelka, Yisen Wang

**キーワード**: Graph, Large Language Models, Reasoning, Reinforcement Learning

**概要**: Although Large Language Models (LLMs) have demonstrated remarkable progress, their proficiency in graph-related tasks remains notably limited, hindering the development of truly general-purpose models. Previous attempts, including pretraining graph foundation models or employing supervised fine-tuning, often face challenges such as the scarcity of large-scale, universally represented graph data. We introduce $\texttt{G1}$, a simple yet effective approach demonstrating that Reinforcement Learning (RL) on synthetic graph-theoretic tasks can significantly scale LLMs' graph reasoning abilities. To enable RL training, we curate \erdos, the largest graph reasoning dataset to date comprising 50 diverse graph-theoretic tasks of varying difficulty levels, 100k training data and 5k test data, all drived from real-world graphs. With RL on \erdos, $\texttt{G1}$ obtains substantial improvements in graph reasoning, where our finetuned 3B model even outperforms Qwen2.5-72B-Instruct (24x size). RL-trained models also show strong zero-shot generalization to unseen tasks, domains, and graph encoding schemes, including other graph-theoretic benchmarks as well as real-world node classification and link prediction tasks, without compromising general reasoning abilities. Our findings offer an efficient, scalable path for building strong graph reasoners by finetuning LLMs with RL on graph-theoretic tasks, which combines the strengths of pretrained LLM capabilities with abundant, automatically generated synthetic data, suggesting that LLMs possess graph understanding abilities that RL can elicit successfully. Our implementation is open-sourced at https://github.com/PKU-ML/G1, with models and datasets hosted on Hugging Face collections https://huggingface.co/collections/PKU-ML/g1-683d659e992794fc99618cf2 for broader accessibility.

**評価**: 平均レビュースコア: 4.75/10 | レビュアー信頼度: 3.50/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.78 | 関連性: 1.00 | 新規性: 0.64 | インパクト: 0.63

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=Lq4nneD2xX)
- [PDF](https://openreview.net/pdf?id=Lq4nneD2xX)

---

### 2. d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.770** |
| 関連性 | 0.935 |
| 新規性 | 0.525 |
| インパクト | 0.795 |
| OpenReview評価 | 4.50/10 |

**著者**: Siyan Zhao, Devaansh Gupta, Qinqing Zheng, Aditya Grover

**キーワード**: diffusion language models, post-training, reinforcement learning, reasoning, large language models

**概要**: Recent large language models (LLMs) have demonstrated strong reasoning capabilities that benefits from online reinforcement learning (RL).
These capabilities have primarily been demonstrated within the left-to-right autoregressive (AR) generation paradigm. 
In contrast, non-autoregressive paradigms based on diffusion generate text in a coarse-to-fine manner. Although recent diffusion-based large language models (dLLMs) have achieved competitive language modeling performance compared to their AR counterparts, it remains unclear if dLLMs can also leverage recent advances in LLM reasoning.
To this end, we propose, a framework to adapt pre-trained masked dLLMs into reasoning models via a combination of supervised finetuning (SFT) and RL.
Specifically, we develop and extend techniques to improve reasoning in pretrained dLLMs: (a) we utilize a masked SFT technique to distill knowledge and instill self-improvement behavior directly from existing datasets, and (b) we introduce a novel critic-free, policy-gradient based RL algorithm called diffu-GRPO, the first integration of policy gradient methods to masked dLLMs. Through empirical studies, we investigate the performance of different post-training recipes on multiple mathematical and planning benchmarks. We find that d1 yields the best performance and significantly improves performance of a state-of-the-art dLLM.

**評価**: 平均レビュースコア: 4.50/10 | レビュアー信頼度: 4.00/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.77 | 関連性: 0.94 | 新規性: 0.53 | インパクト: 0.80

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=7ZVRlBFuEv)
- [PDF](https://openreview.net/pdf?id=7ZVRlBFuEv)

---

### 3. Accelerating RL for LLM Reasoning with Optimal Advantage Regression

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.764** |
| 関連性 | 1.000 |
| 新規性 | 0.558 |
| インパクト | 0.655 |
| OpenReview評価 | 4.50/10 |

**著者**: Kianté Brantley, Mingyu Chen, Zhaolin Gao, Jason D. Lee, Wen Sun 他2名

**キーワード**: reinforcement learning, large language models, reasoning

**概要**: Reinforcement learning (RL) has emerged as a powerful tool for fine-tuning large language models (LLMs) to improve complex reasoning abilities. However, state-of-the-art policy optimization methods often suffer from high computational overhead and memory consumption, primarily due to the need for multiple generations per prompt and the reliance on critic networks or advantage estimates of the current policy. In this paper, we propose $A^\star$-PO, a novel two-stage policy optimization framework that directly approximates the optimal advantage function and enables efficient training of LLMs for reasoning tasks. In the first stage, we leverage offline sampling from a reference policy to estimate the optimal value function $V^\star$, eliminating the need for costly online value estimation. In the second stage, we perform on-policy updates using a simple least-squares regression loss with only a single generation per prompt. Theoretically, we establish performance guarantees and prove that the KL-regularized RL objective can be optimized without requiring complex exploration strategies. Empirically, $A^\star$-PO achieves competitive performance across a wide range of mathematical reasoning benchmarks, while reducing training time by up to 2$\times$ and peak memory usage by over 30\% compared to PPO, GRPO, and REBEL. Implementation of $A^\star$-PO can be found at https://github.com/ZhaolinGao/A-PO.

**評価**: 平均レビュースコア: 4.50/10 | レビュアー信頼度: 4.25/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.76 | 関連性: 1.00 | 新規性: 0.56 | インパクト: 0.66

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=T1V8BJO0iG)
- [PDF](https://openreview.net/pdf?id=T1V8BJO0iG)

---

### 4. When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.760** |
| 関連性 | 0.943 |
| 新規性 | 0.475 |
| インパクト | 0.802 |
| OpenReview評価 | 4.75/10 |

**著者**: Xiaomin Li, Zhou Yu, Zhiwei Zhang, Xupeng Chen, Ziji Zhang 他3名

**キーワード**: Large language models, Reasoning, Instruction Following, Chain of Thoughts

**概要**: Reasoning-enhanced large language models (RLLMs), whether explicitly trained for reasoning or prompted via chain-of-thought (CoT), have achieved state-of-the-art performance on many complex reasoning tasks. However, we uncover a surprising and previously overlooked phenomenon: explicit CoT reasoning can significantly degrade instruction-following accuracy. Evaluating 20+ models on two benchmarks: IFEval (with simple, rule-verifiable constraints) and ComplexBench (with complex, compositional constraints), we consistently observe performance drops when CoT prompting is applied. Through large-scale case studies and an attention-based analysis, we identify common patterns where reasoning either helps (e.g., with formatting or lexical precision) or hurts (e.g., by neglecting simple constraints or introducing unnecessary content). We propose a metric, constraint attention, to quantify model focus during generation and show that CoT reasoning often diverts attention away from instruction-relevant tokens. To mitigate these effects, we introduce and evaluate four strategies: in-context learning, self-reflection, self-selective reasoning, and classifier-selective reasoning. Our results demonstrate that selective reasoning strategies, particularly classifier-selective reasoning, can substantially recover lost performance. To our knowledge, this is the first work to systematically expose reasoning-induced failures in instruction-following and offer practical mitigation strategies.

**評価**: 平均レビュースコア: 4.75/10 | レビュアー信頼度: 4.00/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.76 | 関連性: 0.94 | 新規性: 0.47 | インパクト: 0.80

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=w5uUvxp81b)
- [PDF](https://openreview.net/pdf?id=w5uUvxp81b)

---

### 5. RF-Agent: Automated Reward Function Design via Language Agent Tree Search

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.743** |
| 関連性 | 0.842 |
| 新規性 | 0.571 |
| インパクト | 0.782 |
| OpenReview評価 | 4.75/10 |

**著者**: Ning Gao, Xiuhui Zhang, Xingyu Jiang, Mukang You, Mohan Zhang 他1名

**キーワード**: Reward Design, Reinforcement Learning, Large Language Models, Language Agent, Monte Carlo Tree Search

**概要**: Designing efficient reward functions for low-level control tasks is a challenging problem. Recent research aims to reduce reliance on expert experience by using Large Language Models (LLMs) with task information to generate dense reward functions. These methods typically rely on training results as feedback, iteratively generating new reward functions with greedy or evolutionary algorithms. However, they suffer from poor utilization of historical feedback and inefficient search, resulting in limited improvements in complex control tasks. To address this challenge, we propose RF-Agent, a framework that treats LLMs as language agents and frames reward function design as a sequential decision-making process, enhancing optimization through better contextual reasoning. RF-Agent integrates Monte Carlo Tree Search (MCTS) to manage the reward design and optimization process, leveraging the multi-stage contextual reasoning ability of LLM. This approach better utilizes historical information and improves search efficiency to identify promising reward functions. Outstanding experimental results in 17 diverse low-level control tasks demonstrate the effectiveness of our method.

**評価**: 平均レビュースコア: 4.75/10 | レビュアー信頼度: 3.50/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.74 | 関連性: 0.84 | 新規性: 0.57 | インパクト: 0.78

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=dZ94ZS410X)
- [PDF](https://openreview.net/pdf?id=dZ94ZS410X)

---

### 6. Ravan: Multi-Head Low-Rank Adaptation for Federated Fine-Tuning

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.736** |
| 関連性 | 1.000 |
| 新規性 | 0.483 |
| インパクト | 0.637 |
| OpenReview評価 | 4.67/10 |

**著者**: Arian Raje, Baris Askin, Divyansh Jhunjhunwala, Gauri Joshi

**キーワード**: Federated Learning, LoRA, Fine-Tuning, Efficiency

**概要**: Large Language Models (LLMs) have yet to effectively leverage the vast amounts of edge-device data, and Federated Learning (FL) offers a promising paradigm to collaboratively fine-tune LLMs without transferring private edge data to the cloud. To operate within the computational and communication constraints of edge devices, recent literature on federated fine-tuning of LLMs proposes the use of low-rank adaptation (LoRA) and similar parameter-efficient methods. However, LoRA-based methods suffer from accuracy degradation in FL settings, primarily because of data and computational heterogeneity across clients. We propose Ravan, an adaptive multi-head LoRA method that balances parameter efficiency and model expressivity by reparameterizing the weight updates as the sum of multiple LoRA heads, $s_i\textbf{B}_i\textbf{H}_i\textbf{A}_i$, in which only the $\textbf{H}_i$ parameters and their lightweight scaling factors $s_i$ are trained. These trainable scaling factors let the optimization focus on the most useful heads, recovering a higher-rank approximation of the full update without increasing the number of communicated parameters since clients upload $s_i\textbf{H}_i$ directly. Experiments on vision and language benchmarks show that Ravan improves test accuracy by 2–8\% over prior parameter-efficient baselines, making it a robust and scalable solution for federated fine-tuning of LLMs.

**評価**: 平均レビュースコア: 4.67/10 | レビュアー信頼度: 3.67/5 | レビュー数: 3 | 採択判定: Accept (poster) | 
総合スコア: 0.74 | 関連性: 1.00 | 新規性: 0.48 | インパクト: 0.64

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=gyn4n8oC9B)
- [PDF](https://openreview.net/pdf?id=gyn4n8oC9B)

---

### 7. Language Models (Mostly) Know When to Stop Reading

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.732** |
| 関連性 | 0.935 |
| 新規性 | 0.558 |
| インパクト | 0.635 |
| OpenReview評価 | 4.50/10 |

**著者**: Roy Xie, Junlin Wang, Paul Rosu, Chunyuan Deng, Bolun Sun 他2名

**キーワード**: Large language models, efficiency, context processing, context compression

**概要**: Large language models (LLMs) process entire input contexts indiscriminately, which is inefficient when the information required to answer a query is localized within the context. We present dynamic context cutoff, a novel method enabling LLMs to self-terminate processing upon acquiring sufficient task-relevant information. Through analysis of model internals, we discover that specific attention heads inherently encode "sufficiency signals" -- detectable through lightweight classifiers -- that predict when critical information has been processed. This reveals a new efficiency paradigm: models' internal understanding naturally dictates processing needs rather than external compression heuristics. Comprehensive experiments across six QA datasets (up to 40K tokens) with three model families (LLaMA/Qwen/Mistral, 1B-70B) demonstrate 3.4% accuracy improvement while achieving 1.33x token reduction on average. Furthermore, our method demonstrates superior performance compared to other context efficiency methods at equivalent token reduction rates. Additionally, we observe an emergent scaling phenomenon: while smaller models require probing for sufficiency detection, larger models exhibit intrinsic self-assessment capabilities through prompting.

**評価**: 平均レビュースコア: 4.50/10 | レビュアー信頼度: 3.75/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.73 | 関連性: 0.94 | 新規性: 0.56 | インパクト: 0.64

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=naAUSeyoZ7)
- [PDF](https://openreview.net/pdf?id=naAUSeyoZ7)

---

### 8. LLM Meeting Decision Trees on Tabular Data

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.726** |
| 関連性 | 0.743 |
| 新規性 | 0.637 |
| インパクト | 0.792 |
| OpenReview評価 | 4.75/10 |

**著者**: Hangting Ye, Jinmeng Li, He Zhao, Dandan Guo, Yi Chang

**キーワード**: Tabular data, Large language models, Classification and regression

**概要**: Tabular data have been playing a vital role in diverse real-world fields, including healthcare, finance, etc. 
With the recent success of Large Language Models (LLMs), early explorations of extending LLMs to the domain of tabular data have been developed. Most of these LLM-based methods typically first serialize tabular data into natural language descriptions, and then tune LLMs or directly infer on these serialized data. However, these methods suffer from two key inherent issues: (i) data perspective: existing data serialization methods lack universal applicability for structured tabular data, and may pose privacy risks through direct textual exposure, and (ii) model perspective: LLM fine-tuning methods struggle with tabular data, and in-context learning scalability is bottle-necked by input length constraints (suitable for few-shot learning). This work explores a novel direction of integrating LLMs into tabular data through logical decision tree rules as intermediaries, proposing a decision tree enhancer with LLM-derived rule for tabular prediction, DeLTa. The proposed DeLTa avoids tabular data serialization, and can be applied to full data learning setting without LLM fine-tuning. 
Specifically, we leverage the reasoning ability of LLMs to redesign an improved rule given a set of decision tree rules. Furthermore, we provide a calibration method for original decision trees via new generated rule by LLM, which approximates the error correction vector to steer the original decision tree predictions in the direction of ``errors'' reducing.
Finally, extensive experiments on diverse tabular benchmarks show that our method achieves state-of-the-art performance.

**評価**: 平均レビュースコア: 4.75/10 | レビュアー信頼度: 3.75/5 | レビュー数: 4 | 採択判定: Accept (spotlight) | 
総合スコア: 0.73 | 関連性: 0.74 | 新規性: 0.64 | インパクト: 0.79

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=SRDF3RV0KP)
- [PDF](https://openreview.net/pdf?id=SRDF3RV0KP)

---

### 9. Hyperbolic Fine-Tuning for Large Language Models

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.721** |
| 関連性 | 0.744 |
| 新規性 | 0.640 |
| インパクト | 0.772 |
| OpenReview評価 | 4.80/10 |

**著者**: Menglin Yang, Ram Samarth B B, Aosong Feng, Bo Xiong, Jiahong Liu 他2名

**キーワード**: Large Language Models, Hyperbolic Space, Low-Rank Adaptation, Embedding Space

**概要**: Large language models (LLMs) have demonstrated remarkable performance on various tasks. However, it remains an open question whether the default Euclidean space is the most suitable choice for embedding tokens in LLMs.
   In this study, we investigate the non-Euclidean characteristics of LLMs. 
   Our findings reveal that token frequency follows a power-law distribution, with high-frequency tokens clustering near the origin and low-frequency tokens positioned farther away. Additionally, token embeddings exhibit a high degree of hyperbolicity, indicating a latent tree-like structure in the embedding space. 
   Motivated by these observations, we propose to efficiently fine-tune LLMs in hyperbolic space to better exploit the underlying complex structures. 
   However, we find that this hyperbolic fine-tuning cannot be achieved through the naive application of exponential and logarithmic maps when the embedding and weight matrices both reside in Euclidean space.
   To address this technical issue, we introduce hyperbolic low-rank efficient fine-tuning, HypLoRA, which performs low-rank adaptation directly on the hyperbolic manifold, preventing the cancellation effect produced by consecutive exponential and logarithmic maps and thereby preserving hyperbolic modeling capabilities.
   Extensive experiments across various base models and two different reasoning benchmarks, specifically arithmetic and commonsense reasoning tasks, demonstrate that HypLoRA substantially improves LLM performance.

**評価**: 平均レビュースコア: 4.80/10 | レビュアー信頼度: 3.20/5 | レビュー数: 5 | 採択判定: Accept (spotlight) | 
総合スコア: 0.72 | 関連性: 0.74 | 新規性: 0.64 | インパクト: 0.77

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=TkEdQv0bXB)
- [PDF](https://openreview.net/pdf?id=TkEdQv0bXB)

---

### 10. Think-RM: Enabling Long-Horizon Reasoning in Generative Reward Models

#### スコア

| 項目 | スコア |
|------|--------|
| 総合 | **0.717** |
| 関連性 | 1.000 |
| 新規性 | 0.450 |
| インパクト | 0.605 |
| OpenReview評価 | 4.50/10 |

**著者**: Ilgee Hong, Changlong Yu, Liang Qiu, Weixiang Yan, Zhenghao Xu 他6名

**キーワード**: Generative Reward Models, Large Language Models, Reasoning, RLHF

**概要**: Reinforcement learning from human feedback (RLHF) has become a powerful post-training paradigm for aligning large language models with human preferences. A core challenge in RLHF is constructing accurate reward signals, where the conventional Bradley-Terry reward models (BT RMs) often suffer from sensitivity to data size and coverage, as well as vulnerability to reward hacking. Generative reward models (GenRMs) offer a more robust alternative by generating chain-of-thought (CoT) rationales followed by a final verdict. However, existing GenRMs rely on shallow, vertically scaled reasoning, limiting their capacity to handle nuanced or complex tasks. Moreover, their pairwise preference outputs are incompatible with standard RLHF algorithms that require pointwise reward signals. In this work, we introduce Think-RM, a training framework that enables long-horizon reasoning in GenRMs by modeling an internal thinking process. Rather than producing structured, externally provided rationales, Think-RM generates flexible, self-guided reasoning traces that support advanced capabilities such as self-reflection, hypothetical reasoning, and divergent reasoning. To elicit these reasoning abilities, we first warm-up the models by supervised fine-tuning (SFT) over long CoT data. We then further improve the model's long-horizon abilities by rule-based reinforcement learning (RL). In addition, we propose a novel pairwise RLHF pipeline that directly optimizes policies from pairwise comparisons, eliminating the need for pointwise reward conversion. Experiments show that Think-RM outperforms baselines on both in-distribution and out-of-distribution tasks, with particularly strong gains on reasoning-heavy benchmarks: more than 10\% and 5\% on RewardBench's Chat Hard and Reasoning, and 12\% on RM-Bench's Math domain. When combined with our pairwise RLHF pipeline, it demonstrates superior end-policy performance compared to traditional approaches. This depth-oriented approach not only broadens the GenRM design space but also establishes a new paradigm for preference-based policy optimization in RLHF.

**評価**: 平均レビュースコア: 4.50/10 | レビュアー信頼度: 3.00/5 | レビュー数: 4 | 採択判定: Accept (poster) | 
総合スコア: 0.72 | 関連性: 1.00 | 新規性: 0.45 | インパクト: 0.60

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=UfQAFbP6xq)
- [PDF](https://openreview.net/pdf?id=UfQAFbP6xq)

---

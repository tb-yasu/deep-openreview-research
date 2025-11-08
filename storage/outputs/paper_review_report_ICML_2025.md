# 論文レビューレポート

**生成日時**: 2025年11月08日 15:37

## 検索条件

- **学会**: ICML 2025
- **キーワード**: 指定なし
- **検索論文数**: 3422件
- **評価論文数**: 3422件
- **ランク対象論文数**: 28件

## 評価基準

- **研究興味**: graph generation, drug discovery, computational biology, molecular modeling, data-driven approaches, machine learning, network analysis
- **最小関連性スコア**: 0.2
- **新規性重視**: はい
- **インパクト重視**: はい

## キーワードと同義語

各キーワードに対してLLMが生成した同義語を使用して論文を検索しました。

### graph generation

**同義語**:
- graph synthesis
- graph construction
- graph modeling
- network generation
- graph creation

### drug discovery

**同義語**:
- pharmaceutical development
- medicinal chemistry
- compound screening
- drug design
- dd

### computational biology

**同義語**:
- bioinformatics
- systems biology
- computational genomics
- cb
- biological data analysis

### molecular modeling

**同義語**:
- molecular simulation
- computational chemistry
- molecular dynamics
- molecular visualization
- mm

### data-driven approaches

**同義語**:
- data-centric methods
- analytics-based strategies
- evidence-based approaches
- data-informed techniques
- ddm

### machine learning

**同義語**:
- artificial intelligence
- ai
- deep learning
- ml
- predictive analytics

### network analysis

**同義語**:
- graph analysis
- social network analysis
- network modeling
- network theory
- nwa

## 統計情報

- **平均総合スコア**: 0.388
- **最高スコア**: 0.456
- **最低スコア**: 0.306
- **平均レビュー評価**: 3.33/10

## トップ論文

### 1. DeFoG: Discrete Flow Matching for Graph Generation

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.808** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.850 |
| OpenReview評価 | 4.00/10 |

**採択判定**: Accept (oral)
  - 🎤 **発表形式**: Oral Presentation（口頭発表）

**著者**: Yiming QIN, Manuel Madeira, Dorina Thanou, Pascal Frossard

**キーワード**: Graph Generation, Flow Matching

#### 概要

Graph generative models are essential across diverse scientific domains by capturing complex distributions over relational data. Among them, graph diffusion models achieve superior performance but face inefficient sampling and limited flexibility due to the tight coupling between training and sampling stages. We introduce DeFoG, a novel graph generative framework that disentangles sampling from training, enabling a broader design space for more effective and efficient model optimization. DeFoG employs a discrete flow-matching formulation that respects the inherent symmetries of graphs. We theoretically ground this disentangled formulation by explicitly relating the training loss to the sampling algorithm and showing that DeFoG faithfully replicates the ground truth graph distribution. Building on these foundations, we thoroughly investigate DeFoG's design space and propose novel sampling methods that significantly enhance performance and reduce the required number of refinement steps. Extensive experiments demonstrate state-of-the-art performance across synthetic, molecular, and digital pathology datasets, covering both unconditional and conditional generation settings. It also outperforms most diffusion-based models with just 5–10\% of their sampling steps.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しており、実用性も高いが、さらなるスケーラビリティの検証が求められる。

#### 📊 レビュー要約

レビューワーはDeFoGの理論的基盤と実験結果を高く評価し、特に効率性とサンプリングの質の向上を指摘。一方で、既存のアイデアの組み合わせに留まる点が新規性の限界として挙げられた。Program Chairsは全体的な評価から採択を推奨。

#### 🔍 評価データソース

ICMLのoverall_recommendationフィールド(平均4.0)とclaims_and_evidence、experimental_designs_or_analysesを主に参照しました。

#### 📝 採択理由

All reviewer suggested to accept the paper and the rebuttal could clarify a few remaining points, leading to reviewers increasing their score. The reviewers particularly liked the systematic empirical evaluation and hope for an easy to use code-base.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 4.00 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=KPRIwWhqAZ)
- [PDF](https://openreview.net/pdf?id=KPRIwWhqAZ)

---

### 2. Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.803** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.800 |
| OpenReview評価 | 3.67/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Mohit Pandey, Gopeshh Subbaraj, Artem Cherkasov, Martin Ester, Emmanuel Bengio

**キーワード**: Generative Flow Networks, Foundation model, Drug Discovery

#### 概要

Generative Flow Networks (GFlowNets) have recently emerged as a suitable framework for generating diverse and high-quality molecular structures by learning from rewards treated as unnormalized distributions. Previous works in this framework often restrict exploration by using predefined molecular fragments as building blocks, limiting the chemical space that can be accessed. In this work, we introduce Atomic GFlowNets (A-GFNs), a foundational generative model leveraging individual atoms as building blocks to explore drug-like chemical space more comprehensively. We propose an unsupervised pre-training approach using drug-like molecule datasets, which teaches A-GFNs about inexpensive yet informative molecular descriptors such as drug-likeliness, topological polar surface area, and synthetic accessibility scores. These properties serve as proxy rewards, guiding A-GFNs towards regions of chemical space that exhibit desirable pharmacological properties. We further implement a goal-conditioned finetuning process, which adapts A-GFNs to optimize for specific target properties. In this work, we pretrain A-GFN on a subset of ZINC dataset, and by employing robust evaluation metrics we show the effectiveness of our approach when compared to other relevant baseline methods for a wide range of drug design tasks.  The code is accessible at https://github.com/diamondspark/AGFN.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、特定の実験条件に対する詳細な検討が求められる。

#### 📊 レビュー要約

レビューワーは新しい原子ベースの手法の有効性を評価し、実験結果が主張を支持していると述べている。一方で、実験の範囲や特定の質問に対する明確な回答が求められている。Program Chairsは新規性と実験の質を考慮し、採択を決定した。

#### 🔍 評価データソース

ICMLのoverall_recommendationフィールド(平均4.0)とclaims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper proppose  Atomic GFlowNets (A-GFNs), a foundational generative model leveraging individual atoms as building blocks to explore drug-like chemical space.  The authors pre-train A-GFN on inexpensive molecular properties as rewards, and then further fine-tune on  downstream tasks. The authors show the effectiveness of the proposed method on a variety of drug design tasks.

Strengths of the paper: 
- The proposed method is well-motivated by insights into molecular generation. 
- The rewards are computationally efficient. 
- The paper provides a thorough evaluation of A-GFN across variou...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.67 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=zy15E0X3Dq)
- [PDF](https://openreview.net/pdf?id=zy15E0X3Dq)

---

### 3. Graph Generative Pre-trained Transformer

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.803** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.800 |
| OpenReview評価 | 3.33/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Xiaohui Chen, Yinkai Wang, Jiaxing He, Yuanqi Du, Soha Hassoun 他2名

**キーワード**: Graph generation, transformer, graph representation learning

#### 概要

Graph generation is a critical task in numerous domains, including molecular design and social network analysis, due to its ability to model complex relationships and structured data. While most modern graph generative models utilize adjacency matrix representations, this work revisits an alternative approach that represents graphs as sequences of node set and edge set. We advocate for this approach due to its efficient encoding of graphs and propose a novel representation. Based on this representation, we introduce the Graph Generative Pre-trained Transformer (G2PT), an auto-regressive model that learns graph structures via next-token prediction. To further exploit G2PT's capabilities as a general-purpose foundation model, we explore fine-tuning strategies for two downstream applications: goal-oriented generation and graph property prediction. We conduct extensive experiments across multiple datasets. Results indicate that G2PT achieves superior generative performance on both generic graph and molecule datasets. Furthermore, G2PT exhibits strong adaptability and versatility in downstream tasks from molecular design to property prediction.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、関連文献との比較が不足している点が気になる。

#### 📊 レビュー要約

レビューワーは手法のシンプルさと効果を高く評価し、実験結果も良好であると認めた。一方で、関連文献との比較が不足しているとの指摘もあった。Program Chairsは新しい表現方法と実験の多様性を評価し、採択を推奨した。

#### 🔍 評価データソース

OpenReviewのreviewフィールド(特にclaims_and_evidence、experimental_designs_or_analyses)を主に参照しました。

#### 📝 採択理由

This paper focuses on graph generation.  The paper proposes a straight-forward yet effective method for graph generation.  The approaches centers on a new way to represent a graph as sequence of tokens, that contains both node definitions and edge definitions.  Generation is achieved via an auto-regressive framework on this representation.

The paper had divergent reviews after the rebuttal phase.  The reviewers appreciated the simplicity and effectiveness of the framework.  The results on standard benchmarks (Planar, Tree, Lobster, and SBM) were solid.  Concerns were about the results on mole...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.33 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=FmHnhDLlOX)
- [PDF](https://openreview.net/pdf?id=FmHnhDLlOX)

---

### 4. PyTDC: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.803** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.800 |
| OpenReview評価 | 3.75/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Alejandro Velez-Arce, Marinka Zitnik

**キーワード**: biological chemistry, biomedical AI, computational biology, biomedical informatics, machine learning platform, machine learning framework, AI for drug discovery

#### 概要

Existing biomedical benchmarks do not provide end-to-end infrastructure for training, evaluation, and inference of models that integrate multimodal biological data and a broad range of machine learning tasks in therapeutics. We present PyTDC, an open-source machine-learning platform providing streamlined training, evaluation, and inference software for multimodal biological AI models. PyTDC unifies distributed, heterogeneous, continuously updated data sources and model weights and standardizes benchmarking and inference endpoints. This paper discusses the components of PyTDC's architecture and, to our knowledge, the first-of-its-kind case study on the introduced single-cell drug-target nomination ML task. We find state-of-the-art methods in graph representation learning and domain-specific methods from graph theory perform poorly on this task. Though we find a context-aware geometric deep learning method that outperforms the evaluated SoTA and domain-specific baseline methods, the model is unable to generalize to unseen cell types or incorporate additional modalities, highlighting PyTDC's capacity to facilitate an exciting avenue of research developing multimodal, context-aware, foundation models for open problems in biomedical AI.

#### 🤖 AI評価

この論文はユーザーの興味に関連するグラフ生成と創薬への応用に焦点を当てており、実用的なプラットフォームを提供している。新規性はあるが、実験の範囲が限られているため、さらなる検証が期待される。

#### 📊 レビュー要約

レビューワーはPyTDCのシステム的貢献を高く評価し、特に多様な生物データの統合における重要性を指摘。一方で、実験の限界や新規性の観点からのさらなる検証が求められている。Program Chairsは、バイオメディカルAIコミュニティにおけるニーズに応える点を評価し、採択を推奨した。

#### 🔍 評価データソース

レビューのclaims_and_evidence、experimental_designs_or_analyses、methods_and_evaluation_criteriaフィールドを主に参照しました。

#### 📝 採択理由

The paper introduces PyTDC, an open-source platform for training, evaluating, and deploying multimodal biomedical machine learning models. It integrates diverse biological data sources, supports a wide range of therapeutic tasks, and provides infrastructure for benchmarking and inference. Reviewers agree that the work addresses a timely and important need in the biomedical AI community. While the contributions are primarily system-oriented, the paper introduces several useful components, including task definitions, evaluation metrics, and a case study that illustrates practical challenges. Som...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.75 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=HV8vZDDoYc)
- [PDF](https://openreview.net/pdf?id=HV8vZDDoYc)

---

### 5. Learning-Order Autoregressive Models with Application to Molecular Graph Generation

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Zhe Wang, Jiaxin Shi, Nicolas Heess, Arthur Gretton, Michalis Titsias

**キーワード**: Generative Modeling, Variational Inference, Autoregressive Model, Graph Generation, Molecule Generation

#### 概要

Autoregressive models (ARMs) have become the workhorse for sequence generation tasks, since many problems can be modeled as next-token prediction. While there appears to be a natural ordering for text (i.e., left-to-right), for many data types, such as graphs, the canonical ordering is less obvious. To address this problem, we introduce a variant of ARM that generates high-dimensional data using a probabilistic ordering that is sequentially inferred from data. This model incorporates a trainable probability distribution, referred to as an order-policy, that dynamically decides the autoregressive order in a state-dependent manner. To train the model, we introduce a variational lower bound on the exact log-likelihood, which we optimize with stochastic gradient estimation. We demonstrate experimentally that our method can learn meaningful autoregressive orderings in image and graph generation. On the challenging domain of molecular graph generation, we achieve state-of-the-art results on the QM9 and ZINC250k benchmarks, evaluated using the Fréchet ChemNet Distance (FCD), Synthetic Accessibility Score (SAS), Quantitative Estimate of Drug-likeness (QED).

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、さらなる多様なデータセットでの検証が求められる。

#### 📊 レビュー要約

レビューワーは新しい生成順序を学習する手法の独自性と実験結果を高く評価。一方で、実験の多様性や評価基準の追加を求める声もあり。Program Chairsは新規性と実験の質を考慮し、採択を決定した。

#### 🔍 評価データソース

ICMLのoverall_recommendationフィールド(平均3.25)とexperimental_designs_or_analysesを主に参照しました。

#### 📝 採択理由

The paper introduces an auto-regressive model (ARM) that generates images and graphs using a probabilistic ordering inferred from data. This is achieved by using a trainable probability distribution that dynamically decides the sampling order of the data dimensions, which is optimized using stochastic gradient estimation the variation lower bound of the exact log-likelihood. Experimentally, the paper shows good results on molecular generation (QM9 and ZINC250k benchmarks).

This paper was well-received by the reviewers, in particular the clarity of exposition and the motivations behind the pap...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.25 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=EY6pXIDi3G)
- [PDF](https://openreview.net/pdf?id=EY6pXIDi3G)

---

### 6. A Non-Asymptotic Convergent Analysis for Scored-Based Graph Generative Model via a System of Stochastic Differential Equations

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.67/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Junwei Su, Chuan Wu

**キーワード**: graph generation, score-based generative model, convergence analysis

#### 概要

This paper investigates the convergence behavior of score-based graph generative models (SGGMs). Unlike common score-based generative models (SGMs) that are governed by a single stochastic differential equation (SDE), SGGMs utilize a system of dependent SDEs, where the graph structure and node features are modeled separately, while accounting for their inherent dependencies. This distinction makes existing convergence analyses from SGMs inapplicable for SGGMs. In this work, we present the first convergence analysis for SGGMs, focusing on the convergence bound (the risk of generative error) across three key graph generation paradigms: (1) feature generation with a fixed graph structure, (2) graph structure generation with fixed node features, and (3) joint generation of both graph structure and node features. Our analysis reveals several unique factors specific to SGGMs (e.g., the topological properties of the graph structure) which significantly affect the convergence bound. Additionally, we offer theoretical insights into the selection of hyperparameters (e.g., sampling steps and diffusion length) and advocate for techniques like normalization to improve convergence. To validate our theoretical findings, we conduct a controlled empirical study using a synthetic graph model. The results in this paper contribute to a deeper theoretical understanding of SGGMs and offer practical guidance for designing more efficient and effective SGGMs.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で理論的な分析が行われているが、実験の深さや規模に関してはさらなる検証が求められる。

#### 📊 レビュー要約

レビューワーは理論的貢献を高く評価し、特にグラフ生成モデルの収束分析における新規性を指摘。一方で、実験の規模や深さに関しては改善の余地があると述べている。Program Chairsは理論的な洞察と実用的な指針を提供する点を評価し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendationフィールド(平均3.67)とstrengths_and_weaknessesを主に参照しました。

#### 📝 採択理由

The authors present a strong theoretical contribution on the consistency of graph-based diffusion generative models. Their analysis is heavily buidling on prior work on consistency of score based diffusion models, albeit by explicitly analyzing the convergence behavior of the graph component and the node value component they shed light on practical aspects of graph generative modeling.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.67 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=XGZeCEzeRT)
- [PDF](https://openreview.net/pdf?id=XGZeCEzeRT)

---

### 7. AnalogGenie-Lite: Enhancing Scalability and Precision in Circuit Topology Discovery through Lightweight Graph Modeling

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.783** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.800 |
| OpenReview評価 | 3.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Jian Gao, Weidong Cao, Xuan Zhang

**キーワード**: Graph modeling, Application of Generative Models, Electronic Design Automation

#### 概要

The sustainable performance improvements of integrated circuits (ICs) drive the continuous advancement of nearly all transformative technologies. Since its invention, IC performance enhancements have been dominated by scaling the semiconductor technology. Yet, as Moore's law tapers off, a crucial question arises: ***How can we sustain IC performance in the post-Moore era?*** Creating new circuit topologies has emerged as a promising pathway to address this fundamental need. This work proposes AnalogGenie-Lite, a decoder-only transformer that discovers novel analog IC topologies with significantly enhanced scalability and precision via lightweight graph modeling.
AnalogGenie-Lite makes several unique contributions, including concise device-pin representations (i.e., advancing the best prior art from $O\left(n^2\right)$ to $O\left(n\right)$), frequent sub-graph mining, and optimal sequence modeling. Compared to state-of-the-art circuit topology discovery methods, it achieves $5.15\times$ to $71.11\times$ gains in scalability and 23.5\% to 33.6\% improvements in validity. Case studies on other domains' graphs are also provided to show the broader applicability of the proposed graph modeling approach. Source code: https://github.com/xz-group/AnalogGenie-Lite.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、特定の条件下での性能評価が限られている。

#### 📊 レビュー要約

レビューワーは手法の新規性と実験結果の有望さを評価しつつ、実験設定の明確さや応用の幅についての疑問を指摘。Program Chairsは新規性と実用性のバランスから採択を推奨。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均3.0)とstrengths_and_weaknessesフィールドを主に参照しました。

#### 📝 採択理由

This paper proposes AnalogGenie-Lite, a decoder-only generative framework for discovering novel analog circuit topologies. By leveraging lightweight graph modeling, frequent subgraph mining, and sequence optimization via the Chinese Postman Problem, the method improves both efficiency and generation quality. Experiments on real-world analog circuits demonstrate its effectiveness in generating diverse and high-quality designs.

All reviewers recommended accepting the paper.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.25 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=KRk0WTII0I)
- [PDF](https://openreview.net/pdf?id=KRk0WTII0I)

---

### 8. FDGen: A Fairness-Aware Graph Generation Model

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Zichong Wang, Wenbin Zhang

**キーワード**: Fairness, Graph Learning, Graph Generation

#### 概要

Graph generation models have shown significant potential across various domains. However, despite their success, these models often inherit societal biases, limiting their adoption in real-world applications. Existing research on fairness in graph generation primarily addresses structural bias, overlooking the critical issue of feature bias. To address this gap, we propose FDGen, a novel approach that defines and mitigates both feature and structural biases in graph generation models. Furthermore, we provide a theoretical analysis of how bias sources in graph data contribute to disparities in graph generation tasks. Experimental results on four real-world datasets demonstrate that FDGen outperforms state-of-the-art methods, achieving notable improvements in fairness while maintaining competitive generation performance.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、特定のデータセットに依存している点が懸念材料。

#### 📊 レビュー要約

レビューワーは手法の新規性と理論的分析を高く評価しているが、実験の適用範囲に関して懸念を示している。Program Chairsは、特に公平性の観点からの貢献を重視し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均2.75)、claims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper introduces FDGen, a fairness-aware graph generation model that mitigates both structural and feature biases through the integration of a fairness regularizer and a diffusion-based generation framework. A key contribution lies in the theoretical analysis that formalises how biases emerge and propagated in the graph generation process and provides analytical tools to disentangle legitimate group differences from unfair biases. This enables the model to preserve graph quality while improving fairness across demographic groups.

The theoretical contribution is well recognised and apprec...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.00 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=6YUdCt7rUR)
- [PDF](https://openreview.net/pdf?id=6YUdCt7rUR)

---

### 9. Compositional Flows for 3D Molecule and Synthesis Pathway Co-design

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Tony Shen, Seonghwan Seo, Ross Irwin, Kieran Didi, Simon Olsson 他2名

**キーワード**: drug discovery, synthesizable molecular design, GFlowNets, flow matching

#### 概要

Many generative applications, such as synthesis-based 3D molecular design, involve constructing compositional objects with continuous features.
Here, we introduce Compositional Generative Flows (CGFlow), a novel framework that extends flow matching to generate objects in compositional steps while modeling continuous states. 
Our key insight is that modeling compositional state transitions can be formulated as a straightforward extension of the flow matching interpolation process.
We further build upon the theoretical foundations of generative flow networks (GFlowNets), enabling reward-guided sampling of compositional structures. 
We apply CGFlow to synthesizable drug design by jointly designing the molecule's synthetic pathway with its 3D binding pose.
Our approach achieves state-of-the-art binding affinity and synthesizability on all 15 targets from the LIT-PCBA benchmark, and 4.2x improvement in sampling efficiency compared to 2D synthesis-based baseline.
To our best knowledge, our method is also the first to achieve state of-art-performance in both Vina Dock (-9.42) and AiZynth success rate (36.1\%) on the CrossDocked2020 benchmark.

#### 🤖 AI評価

この論文はグラフ生成に関連し、ユーザーの興味に直接結びつく。新しい手法であり、実験も充実しているが、一般化の限界が懸念される。

#### 📊 レビュー要約

レビューワーは手法の理論的堅牢性と実験結果を高く評価する一方で、一般化の限界や追加評価の必要性を指摘。Program Chairsは新規性と実用性の観点から採択を推奨。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均2.75)、claims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

The paper on Compositional Generative Flows presents a novel and methodologically sound approach for jointly modeling 3D structure and synthesizability in molecular design, addressing a significant real-world challenge in drug discovery. Reviewers identified various strengths, such as well-supported claims, a logical extension of flow matching, and appropriate benchmarks, but also raised concerns like limited generalizability, theoretical issues, lack of certain evaluations, and performance compared to baselines. The authors made substantial efforts in their rebuttals to address these concerns...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.00 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=4aXfSLfM0Z)
- [PDF](https://openreview.net/pdf?id=4aXfSLfM0Z)

---

### 10. MF-LAL: Drug Compound Generation Using Multi-Fidelity Latent Space Active Learning

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Peter Eckmann, Dongxia Wu, Germano Heinzelmann, Michael K Gilson, Rose Yu

**キーワード**: drug discovery, multi-fidelity learning, active learning

#### 概要

Current generative models for drug discovery primarily use molecular docking as an oracle to guide the generation of active compounds. However, such models are often not useful in practice because even compounds with high docking scores do not consistently show real-world experimental activity. More accurate methods for activity prediction exist, such as molecular dynamics based binding free energy calculations, but they are too computationally expensive to use in a generative model. To address this challenge, we propose Multi-Fidelity Latent space Active Learning (MF-LAL), a generative modeling framework that integrates a set of oracles with varying cost-accuracy tradeoffs. Using active learning, we train a surrogate model for each oracle and use these surrogates to guide generation of compounds with high predicted activity. Unlike previous approaches that separately learn the surrogate model and generative model, MF-LAL combines the generative and multi-fidelity surrogate models into a single framework, allowing for more accurate activity prediction and higher quality samples. Our experiments on two disease-relevant proteins show that MF-LAL produces compounds with significantly better binding free energy scores than other single and multi-fidelity approaches (~50% improvement in mean binding free energy score). The code is available at https://github.com/Rose-STL-Lab/MF-LAL.

#### 🤖 AI評価

この論文はグラフ生成に関連し、ユーザーの研究興味に合致。新しいアプローチを提案しており、実験結果も示されているが、評価基準やデータセットの多様性に関する懸念が残る。

#### 📊 レビュー要約

レビューワーは手法の新規性と理論的な堅牢性を評価しつつ、実験の限界や評価基準の不足を指摘。Program Chairsは全体的な新規性と実用性を考慮し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均3.0)とexperimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This is a nice paper, and all reviewers generally had positive comments about the novelty, appropriateness, and applicability of the proposed multi-fidelity approach. All reviewers agreed the problem setting is important and the methodology is sound. All reviewers had various more minor concerns, mostly regarding evaluation or baselines. The authors provided detailed rebuttals for most issues; the consensus across all reviewers was a "weak accept", and I think based on its merits the paper would be a good contribution to ICML.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.00 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=3xdFvqsVnM)
- [PDF](https://openreview.net/pdf?id=3xdFvqsVnM)

---

### 11. Efficient and Scalable Density Functional Theory Hamiltonian Prediction through Adaptive Sparsity

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.50/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Erpai Luo, Xinran Wei, Lin Huang, Yunyang Li, Han Yang 他5名

**キーワード**: Equivariant network, Hamiltonian Matrix, Computational Chemistry, Efficiency

#### 概要

Hamiltonian matrix prediction is pivotal in computational chemistry, serving as the foundation for determining a wide range of molecular properties. While SE(3) equivariant graph neural networks have achieved remarkable success in this domain, their substantial computational cost—driven by high-order tensor product (TP) operations—restricts their scalability to large molecular systems with extensive basis sets. To address this challenge, we introduce **SPH**Net, an efficient and scalable equivariant network, that incorporates adaptive **SP**arsity into **H**amiltonian prediction. SPHNet employs two innovative sparse gates to selectively constrain non-critical interaction combinations, significantly reducing tensor product computations while maintaining accuracy. To optimize the sparse representation, we develop a Three-phase Sparsity Scheduler, ensuring stable convergence and achieving high performance at sparsity rates of up to 70\%. Extensive evaluations on QH9 and PubchemQH datasets demonstrate that SPHNet achieves state-of-the-art accuracy while providing up to a 7x speedup over existing models. Beyond Hamiltonian prediction, the proposed sparsification techniques also hold significant potential for improving the efficiency and scalability of other SE(3) equivariant networks, further broadening their applicability and impact.

#### 🤖 AI評価

この論文はグラフ生成に関連し、ユーザーの研究興味に合致。新しい手法を提案し、実験も充実しているが、特定のデータセットに依存している点が懸念材料。

#### 📊 レビュー要約

レビューワーはSPHNetの効率性と新規性を高く評価し、特に計算コストの削減に寄与する点を強調。一方で、実験結果の一部に疑問が呈され、さらなる検証が求められている。Program Chairsは新規性と実験の質を考慮し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均3.5)とclaims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper proposes SPHNet, an efficient SE(3) equivariant graph neural network for Hamiltonian matrix prediction. By introducing sparse gating mechanisms and an adaptive sparsity scheduler, SPHNet effectively reduces computational costs while maintaining high accuracy. Experiments on two common datasets demonstrate that SPHNet achieves state-of-the-art performance with significant speed and memory improvements.

The reviewers were all recommending acceptance of the paper.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.50 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=K3lykWhXON)
- [PDF](https://openreview.net/pdf?id=K3lykWhXON)

---

### 12. Boosting Protein Graph Representations through Static-Dynamic Fusion

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.33/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Pengkang Guo, Bruno Correia, Pierre Vandergheynst, Daniel Probst

**キーワード**: Graph Neural Networks, Protein Modeling, Molecular Dynamics, Heterogeneous Graph

#### 概要

Machine learning for protein modeling faces significant challenges due to proteins' inherently dynamic nature, yet most graph-based machine learning methods rely solely on static structural information. Recently, the growing availability of molecular dynamics trajectories provides new opportunities for understanding the dynamic behavior of proteins; however, computational methods for utilizing this dynamic information remain limited. We propose a novel graph representation that integrates both static structural information and dynamic correlations from molecular dynamics trajectories, enabling more comprehensive modeling of proteins. By applying relational graph neural networks (RGNNs) to process this heterogeneous representation, we demonstrate significant improvements over structure-based approaches across three distinct tasks: atomic adaptability prediction, binding site detection, and binding affinity prediction. Our results validate that combining static and dynamic information provides complementary signals for understanding protein-ligand interactions, offering new possibilities for drug design and structural biology applications.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、基準モデルの選定に関する指摘があり、さらなる実験の必要性が示唆された。

#### 📊 レビュー要約

レビューワーは新しいグラフ表現手法の有効性を評価し、実験結果が競合手法に対して優れていることを確認。一方で、基準モデルの選定に関する指摘があり、さらなる実験の必要性が示唆された。Program Chairsは新規性と実験の質を評価し、採択を推奨した。

#### 🔍 評価データソース

OpenReviewのreviewフィールド(特にclaims_and_evidence、experimental_designs_or_analyses)を主に参照しました。

#### 📝 採択理由

This paper proposes to study protein representation learning by leveraging both their static structure and dynamic correlation information form MD trajectories. Experimental results on three different tasks including atomic adaptability prediction, binding site detection, and binding affinity prediction prove the effectiveness of the proposed approach over competitive baselines. 

All reviewers agree the novelty of leveraging the dynamic information from MD trajectories for protein understanding. The evaluations are comprehensive and compelling. Therefore, the AC votes for acceptance.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.33 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=QbtBIE36Fd)
- [PDF](https://openreview.net/pdf?id=QbtBIE36Fd)

---

### 13. Aligning Protein Conformation Ensemble Generation with Physical Feedback

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Jiarui Lu, Xiaoyin Chen, Stephen Zhewen Lu, Aurelie Lozano, Vijil Chenthamarakshan 他2名

**キーワード**: Protein, generative models, molecular dynamics, conformation generation, alignments

#### 概要

Protein dynamics play a crucial role in protein biological functions and properties, and their traditional study typically relies on time-consuming molecular dynamics (MD) simulations conducted in silico. Recent advances in generative modeling, particularly denoising diffusion models, have enabled efficient accurate protein structure prediction and conformation sampling by learning distributions over crystallographic structures. However, effectively integrating physical supervision into these data-driven approaches remains challenging, as standard energy-based objectives often lead to intractable optimization. In this paper, we introduce Energy-based Alignment (EBA), a method that aligns generative models with feedback from physical models, efficiently calibrating them to appropriately balance conformational states based on their energy differences. Experimental results on the MD ensemble benchmark demonstrate that EBA achieves state-of-the-art performance in generating high-quality protein ensembles. By improving the physical plausibility of generated structures, our approach enhances model predictions and holds promise for applications in structural biology and drug discovery.

#### 🤖 AI評価

この論文は生成モデルと物理モデルの統合に関する新しいアプローチを提案しており、ユーザーの研究興味に関連。実験結果は良好だが、さらなる検証が求められる。

#### 📊 レビュー要約

レビューワーは手法の新規性と実験結果の質を評価しつつ、比較対象の不足や実験の限定性を指摘。Program Chairsは物理モデルとの統合による新しいアプローチを評価し、採択を推奨。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均2.75)、claims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper introduces Energy-Based Alignment (EBA), a method for fine-tuning diffusion models to generate protein conformational ensembles. By incorporating energy-based feedback from molecular simulations into the generative training process, the authors propose a way to bridge data-driven learning and classical physical modeling. Empirical evaluation shows improved performance over prior methods, such as AlphaFlow and MDGen.

The paper addresses an important and timely problem in the intersection of structural biology and generative modeling. The proposed method is well-motivated and represe...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.00 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=Asr955jcuZ)
- [PDF](https://openreview.net/pdf?id=Asr955jcuZ)

---

### 14. CellFlux: Simulating Cellular Morphology Changes via Flow Matching

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.67/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Yuhui Zhang, Yuchang Su, Chenyu Wang, Tianhong Li, Zoe Wefers 他6名

**キーワード**: flow matching, cell image, drug discovery, generative models

#### 概要

Building a virtual cell capable of accurately simulating cellular behaviors in silico has long been a dream in computational biology. We introduce CellFlux, an image-generative model that simulates cellular morphology changes induced by chemical and genetic perturbations using flow matching. Unlike prior methods, CellFlux models distribution-wise transformations from unperturbed to perturbed cell states, effectively distinguishing actual perturbation effects from experimental artifacts such as batch effects—a major challenge in biological data. Evaluated on chemical (BBBC021), genetic (RxRx1), and combined perturbation (JUMP) datasets, CellFlux generates biologically meaningful cell images that faithfully capture perturbation-specific morphological changes, achieving a 35% improvement in FID scores and a 12% increase in mode-of-action prediction accuracy over existing methods. Additionally, CellFlux enables continuous interpolation between cellular states, providing a potential tool for studying perturbation dynamics. These capabilities mark a significant step toward realizing virtual cell modeling for biomedical research. Project page: https://yuhui-zh15.github.io/CellFlux/.

#### 🤖 AI評価

この論文は創薬への応用に関連し、ユーザーの研究興味に合致。新しい手法を提案しており、実験も多様だが、他の研究との関連性や実験の限界が指摘されている。

#### 📊 レビュー要約

レビューワーはCellFluxの新規性と実験結果を高く評価し、特に生物学的な応用における重要性を指摘。一方で、実験の限界や他の関連研究との重複についての懸念も示された。Program Chairsは全体的な貢献を考慮し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendationフィールド(平均3.67)とstrengths_and_weaknessesを主に参照しました。

#### 📝 採択理由

The paper utilizes state of the art generative modelling method for in-silico biological experiments that may have an important impact for future research in this direction. The reviewers are in agreement that even though the work has some limitations, it is overall well executed and is a valuable contribution to the field.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.67 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=3NLNmdheIi)
- [PDF](https://openreview.net/pdf?id=3NLNmdheIi)

---

### 15. Symmetry-Aware GFlowNets

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.770** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.720 |
| 　├ インパクト | 0.680 |
| 　└ 実用性 | 0.800 |
| OpenReview評価 | 3.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Hohyun Kim, Seunggeun Lee, Min-hwan Oh

**キーワード**: GFlowNet, graph generation, molecule optimization

#### 概要

Generative Flow Networks (GFlowNets) offer a powerful framework for sampling graphs in proportion to their rewards. However, existing approaches suffer from systematic biases due to inaccuracies in state transition probability computations. These biases, rooted in the inherent symmetries of graphs, impact both atom-based and fragment-based generation schemes. To address this challenge, we introduce Symmetry-Aware GFlowNets (SA-GFN), a method that incorporates symmetry corrections into the learning process through reward scaling. By integrating bias correction directly into the reward structure, SA-GFN eliminates the need for explicit state transition computations. Empirical results show that SA-GFN enables unbiased sampling while enhancing diversity and consistently generating high-reward graphs that closely match the target distribution.

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、大規模データセットでの検証が限定的。

#### 📊 レビュー要約

レビューワーは手法の理論的堅牢性を高く評価。一方で実験の限定性を指摘。Program Chairsは新規性と実験品質のバランスから採択を推奨。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均2.75)、theoretical_claims、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper addresses the symmetry-induced bias in GFlowNets by proposing a reward scaling method based on graph automorphisms. While this paper is heavily inspired by a prior work (Ma et al., 2024), the authors clearly demonstrate improved scalability and methodological contributions on being able to correct biases without requiring transition-level equivalence checks. Empirical results on synthetic and molecular tasks support the method’s effectiveness. Overall, I recommend acceptance.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.25 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=JD4eHocSPi)
- [PDF](https://openreview.net/pdf?id=JD4eHocSPi)

---

### 16. Wyckoff Transformer: Generation of Symmetric Crystals

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.761** |
| 　├ 関連性 | 0.800 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Nikita Kazeev, Wei Nong, Ignat Romanov, Ruiming Zhu, Andrey E Ustyuzhanin 他2名

**キーワード**: material design, machine learning, crystal generation, space group symmetry, Transformer, Wyckoff position, generative model, autoregressive model

#### 概要

Crystal symmetry plays a fundamental role in determining its physical, chemical, and electronic properties such as electrical and thermal conductivity, optical and polarization behavior, and mechanical strength. Almost all known crystalline materials have internal symmetry. However, this is often inadequately addressed by existing generative models, making the consistent generation of stable and symmetrically valid crystal structures a significant challenge. We introduce WyFormer, a generative model that directly tackles this by formally conditioning on space group symmetry. It achieves this by using Wyckoff positions as the basis for an elegant, compressed, and discrete structure representation. To model the distribution, we develop a permutation-invariant autoregressive model based on the Transformer encoder and an absence of positional encoding. Extensive experimentation demonstrates WyFormer's compelling combination of attributes: it achieves best-in-class symmetry-conditioned generation, incorporates a physics-motivated inductive bias, produces structures with competitive stability, predicts material properties with competitive accuracy even without atomic coordinates, and exhibits unparalleled inference speed.

#### 🤖 AI評価

この論文は材料設計における生成モデルに関するもので、ユーザーの興味に関連するグラフ生成の応用に寄与する可能性がある。新しい手法であり、実験も充実しているが、サンプル数の制限が影響している。

#### 📊 レビュー要約

レビューワーはWyFormerの理論的なアプローチと実験結果を評価しつつ、実験のサンプル数の少なさや論文の構成に対する改善点を指摘。Program Chairsは新規性と実験の質を考慮し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均2.75)、claims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper presents a method named WyFormer, which is designed to generate crystal structures autoregressively, by generating the space group of the unit cell, the elements and the Wyckoff positions. There is no consensus among reviewers about the overall assessment and recommendation of the paper. On the one hand, the method seems solid and the results are convincing though not groundbreaking. On the other hand, the structure, clarity and writing of the paper could be largely improved. This has been noted by several reviewers and is also my own assessment. For these reasons, I am recommending...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.00 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=eFHfRQRjJo)
- [PDF](https://openreview.net/pdf?id=eFHfRQRjJo)

---

### 17. Action-Minimization Meets Generative Modeling: Efficient Transition Path Sampling with the Onsager-Machlup Functional

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.761** |
| 　├ 関連性 | 0.800 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.780 |
| OpenReview評価 | 3.75/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Sanjeev Raja, Martin Sipka, Michael Psenka, Tobias Kreiman, Michal Pavelka 他1名

**キーワード**: Transition path sampling, molecular dynamics, generative models, diffusion models, flow matching models

#### 概要

Transition path sampling (TPS), which involves finding probable paths connecting two points on an energy landscape, remains a challenge due to the complexity of real-world atomistic systems. Current machine learning approaches rely on expensive training procedures and under-utilize growing quantities of atomistic data, limiting scalability and generalization. Generative models of atomistic conformational ensembles sample temporally independent states from energy landscapes, but their application to TPS remains mostly unexplored. In this work, we address TPS by interpreting candidate paths as trajectories sampled from stochastic dynamics induced by the learned score function of generative models, namely denoising diffusion and flow matching. Under these dynamics, finding high-likelihood transition paths becomes equivalent to minimizing the Onsager-Machlup (OM) action functional, enabling us to repurpose pre-trained generative models for TPS in a zero-shot fashion. We demonstrate our approach on a Müller-Brown potential and several fast-folding proteins, where we obtain diverse, physically realistic transition pathways, as well as tetrapeptides, where we demonstrate successful TPS on systems not seen by the generative model during training. Our method can be easily incorporated into new generative models, making it practically relevant as models continue to scale and improve.

#### 🤖 AI評価

この論文はグラフ生成に関連する新しい手法を提案しており、ユーザーの研究興味に合致。実験は充実しているが、他の手法との比較が不足しているため、影響力には限界がある。

#### 📊 レビュー要約

レビューワーは手法の新規性と理論的な基盤を高く評価しているが、実験の比較が不足している点を指摘。Program Chairsは新しいアプローチの実用性とスケーラビリティを考慮し、採択を推奨した。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均4.0)とstrengths_and_weaknessesフィールドを主に参照しました。

#### 📝 採択理由

The author present an application of the Onsager-Machlup action minimization functional to sample transition paths in pretrained molecular ensemble generation models, e.g. pre-trained Boltzmann Generators or emulators. While the approach is conceptually simple and well-known in the molecular simulation community (work by Vanden-Eijden and others is discussed in the paper), however, their practicality have been limited by computational demands in its implementation. The application to pre-trained generative models is as far the reviewers and I am concerned, new. However, when applied this setti...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.75 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=QwoGfQzuMa)
- [PDF](https://openreview.net/pdf?id=QwoGfQzuMa)

---

### 18. SBGD: Improving Graph Diffusion Generative Model via Stochastic Block Diffusion

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.752** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.700 |
| 　├ インパクト | 0.650 |
| 　└ 実用性 | 0.750 |
| OpenReview評価 | 3.33/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Junwei Su, shan Wu

**キーワード**: graph generation, scalable, size generalization

#### 概要

Graph diffusion generative models (GDGMs) have emerged as powerful tools for generating high-quality graphs. However, their broader adoption faces challenges in \emph{scalability and size generalization}. GDGMs struggle to scale to large graphs due to their high memory requirements, as they typically operate in the full graph space, requiring the entire graph to be stored in memory during training and inference. This constraint limits their feasibility for large-scale real-world graphs. GDGMs also exhibit poor size generalization, with limited ability to generate graphs of sizes different from those in the training data, restricting their adaptability across diverse applications. To address these challenges, we propose the stochastic block graph diffusion (SBGD) model, which refines graph representations into a block graph space. This space incorporates structural priors based on real-world graph patterns, significantly reducing memory complexity and enabling scalability to large graphs. The block representation also improves size generalization by capturing fundamental graph structures.   Empirical results show that SBGD achieves significant memory improvements (up to 6$\times$) while maintaining comparable or even superior graph generation performance relative to state-of-the-art methods. Furthermore, experiments demonstrate that SBGD better generalizes to unseen graph sizes. The significance of SBGD extends beyond being a scalable and effective GDGM; \emph{it also exemplifies the principle of modularization in generative modelling, offering a new avenue for exploring generative models by decomposing complex tasks into more manageable components.}

#### 🤖 AI評価

この論文はグラフ生成に特化しており、ユーザーの興味に直接関連。新しい手法で実験も充実しているが、特に大規模データセットでの検証が不足している点が懸念される。

#### 📊 レビュー要約

レビューワーはSBGDの理論的な新規性と実験結果を評価する一方で、実験の限界や他の手法との比較不足を指摘。Program Chairsは新規性と実用性の観点から採択を推奨。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均3.33)とclaims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

This paper presents SBGD, a diffusion-based graph generative model leveraging stochastic block structures to address key challenges in scalability and size generalization. The reviewers broadly appreciate the paper’s core idea of decomposing graphs into block representations to reduce memory overhead, enabling training on large-scale graphs. Empirical results support claims of competitive performance with improved efficiency.

Following the rebuttal, all reviewers either maintained or increased their scores, indicating that key concerns were satisfactorily addressed.

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.33 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=FMg9mEKH17)
- [PDF](https://openreview.net/pdf?id=FMg9mEKH17)

---

### 19. Linear Mode Connectivity between Multiple Models modulo Permutation Symmetries

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.750** |
| 　├ 関連性 | 0.750 |
| 　├ 新規性 | 0.800 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.750 |
| OpenReview評価 | 3.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Akira Ito, Masanori Yamada, Atsutoshi Kumagai

**キーワード**: Linear mode connectivity, deep learning, permutation symmetry

#### 概要

Ainsworth et al. empirically demonstrated that linear mode connectivity (LMC) can be achieved between two independently trained neural networks (NNs) by applying an appropriate parameter permutation. LMC is satisfied if a linear path with non-increasing test loss exists between the models, suggesting that NNs trained with stochastic gradient descent (SGD) converge to a single approximately convex low-loss basin under permutation symmetries. However, Ainsworth et al. verified LMC for two models and provided only limited discussion on its extension to multiple models. In this paper, we conduct a more detailed empirical analysis. First, we show that existing permutation search methods designed for two models can fail to transfer multiple models into the same convex low-loss basin. Next, we propose a permutation search method using a straight-through estimator for multiple models (STE-MM). We then experimentally demonstrate that even when multiple models are given, the test loss of the merged model remains nearly the same as the losses of the original models when using STE-MM, and the loss barriers between all permuted model pairs are also small. Additionally, from the perspective of the trace of the Hessian matrix, we show that the loss sharpness around the merged model decreases as the number of models increases with STE-MM, indicating that LMC for multiple models is more likely to hold. The source code implementing our method is available at https://github.com/e5-a/STE-MM.

#### 🤖 AI評価

この論文はグラフ生成に直接関連する内容ではないが、深層学習の新しい手法を提案しており、実用的な応用の可能性も示唆されている。

#### 📊 レビュー要約

レビューワーは新しい手法の有効性と理論的な貢献を高く評価しているが、実験の範囲や明確さに関していくつかの指摘がある。Program Chairsは新規性と実験の質を考慮し、採択を決定した。

#### 🔍 評価データソース

ICMLのoverall_recommendation(平均3.0)とclaims_and_evidence、experimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

The paper addresses the challenge of achieving linear mode connectivity (LMC) among multiple neural networks (NNs) trained using SGD. The prior works experience degradation in performance when merging > 2 models. The current work introduces a novel method, Straight-Through Estimator for Multiple Models (STE-MM), that directly optimizes for loss barrier. The authors also provide a method to accelerate the permutation search. The proposed method outperforms existing approaches in improving LMC, demonstrated through experiments on multiple datasets and network architectures. 

Overall, the review...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.25 | 4 |

*4件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=qaJuLzY6iL)
- [PDF](https://openreview.net/pdf?id=qaJuLzY6iL)

---

### 20. Context is Key: A Benchmark for Forecasting with Essential Textual Information

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.750** |
| 　├ 関連性 | 0.750 |
| 　├ 新規性 | 0.800 |
| 　├ インパクト | 0.700 |
| 　└ 実用性 | 0.750 |
| OpenReview評価 | 3.33/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Andrew Robert Williams, Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Jithendaraa Subramanian 他6名

**キーワード**: time series, forecasting, multimodality, foundation models, contextual forecasting, deep learning, machine learning, natural language processing

#### 概要

Forecasting is a critical task in decision-making across numerous domains. While historical numerical data provide a start, they fail to convey the complete context for reliable and accurate predictions. Human forecasters frequently rely on additional information, such as background knowledge and constraints, which can efficiently be communicated through natural language. However, in spite of recent progress with LLM-based forecasters, their ability to effectively integrate this textual information remains an open question. To address this, we introduce "Context is Key" (CiK), a time-series forecasting benchmark that pairs numerical data with diverse types of carefully crafted textual context, requiring models to integrate both modalities; crucially, every task in CiK requires understanding textual context to be solved successfully. We evaluate a range of approaches, including statistical models, time series foundation models, and LLM-based forecasters, and propose a simple yet effective LLM prompting method that outperforms all other tested methods on our benchmark. Our experiments highlight the importance of incorporating contextual information, demonstrate surprising performance when using LLM-based forecasting models, and also reveal some of their critical shortcomings. This benchmark aims to advance multimodal forecasting by promoting models that are both accurate and accessible to decision-makers with varied technical expertise.
The benchmark can be visualized at https://servicenow.github.io/context-is-key-forecasting/v0.

#### 🤖 AI評価

この論文は、時系列予測における文脈情報の統合に関する新しいベンチマークを提案しており、ユーザーの研究興味に関連性がある。実験が多岐にわたるが、特定のモデルの限界についての分析が不足している。

#### 📊 レビュー要約

レビューワーは、文脈情報を統合した時系列予測の重要性を強調し、実験デザインの質を高く評価。一方で、特定のモデルのパフォーマンスに関する懸念も示された。Program Chairsは、実験の包括性と新規性を考慮し、採択を推奨した。

#### 🔍 評価データソース

OpenReviewのreview_summaryとexperimental_designs_or_analysesフィールドを主に参照しました。

#### 📝 採択理由

The paper received three reviews, all of which were positive (two weak accepts and one accept). Reviewers commended the paper’s comprehensive experimental design, which spans 71 tasks across seven domains, where textual context is essential for accurate time series forecasting. The paper proposes a region-of-interest variant of the CRPS evaluation metric (RCRPS), and provides a structured evaluation of classical statistical models, time series foundation models, and  LLM-based predictors. Overall, the reviewers recognized this work as one of the early contributions to the emerging area of inte...

#### 📊 レビュースコアの平均

| 項目 | 平均値 | レビュー数 |
|------|--------|-----------|
| 総合評価 (Overall Recommendation) | 3.33 | 3 |

*3件のレビューから集計*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=ih2WuBT1Fn)
- [PDF](https://openreview.net/pdf?id=ih2WuBT1Fn)

---

# 論文レビューレポート

**生成日時**: 2025年11月06日 20:24

## 検索条件

- **学会**: NeurIPS 2025
- **キーワード**: 指定なし
- **検索論文数**: 5539件
- **評価論文数**: 5539件
- **ランク対象論文数**: 82件

## 評価基準

- **研究興味**: graph generation, graph algorithms, network modeling, data structures, machine learning, graph theory, computational complexity, random graphs
- **最小関連性スコア**: 0.2
- **新規性重視**: はい
- **インパクト重視**: はい

## キーワードと同義語

各キーワードに対してLLMが生成した同義語を使用して論文を検索しました。

### graph generation

**同義語**:
- graph synthesis
- graph creation
- graph modeling
- network generation
- graph construction

### graph algorithms

**同義語**:
- graph theory
- graph traversal
- network algorithms
- grafos
- graph data structures

### network modeling

**同義語**:
- network simulation
- graph modeling
- topology analysis
- network architecture
- nm

### data structures

**同義語**:
- data organization
- data models
- data formats
- ds
- data representation

### machine learning

**同義語**:
- artificial intelligence
- ai
- deep learning
- ml
- predictive analytics

### graph theory

**同義語**:
- graph mathematics
- network theory
- graph algorithms
- gt
- graph structures

### computational complexity

**同義語**:
- algorithmic complexity
- complexity theory
- np-completeness
- computational hardness
- cc

### random graphs

**同義語**:
- stochastic graphs
- probabilistic graphs
- random networks
- rg
- graph theory

## 統計情報

- **平均総合スコア**: 0.431
- **最高スコア**: 0.518
- **最低スコア**: 0.296
- **平均レビュー評価**: 4.27/10

## トップ論文

### 1. Flatten Graphs as Sequences: Transformers are Scalable Graph Generators

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.702** |
| OpenReview総合 | 0.405 |
| 　├ 関連性 | 0.250 |
| 　├ 新規性 | 0.392 |
| 　└ インパクト | 0.625 |
| AI評価（関連性） | 1.000 |
| AI評価（新規性） | 0.800 |
| AI評価（実用性） | 0.900 |
| OpenReview評価 | 4.50/10 |

**著者**: Dexiong Chen, Markus Krimmel, Karsten Borgwardt

**キーワード**: graph generation, transformers, autoregressive modeling, language models, LLMs

#### 概要

We introduce AutoGraph, a scalable autoregressive model for attributed graph generation using decoder-only transformers. By flattening graphs into random sequences of tokens through a reversible process, AutoGraph enables modeling graphs as sequences without relying on additional node features that are expensive to compute, in contrast to diffusion-based approaches. This results in sampling complexity and sequence lengths that scale optimally linearly with the number of edges, making it scalable and efficient for large, sparse graphs. A key success factor of AutoGraph is that its sequence prefixes represent induced subgraphs, creating a direct link to sub-sentences in language modeling. Empirically, AutoGraph achieves state-of-the-art performance on synthetic and molecular benchmarks, with up to 100x faster generation and 3x faster training than leading diffusion models. It also supports substructure-conditioned generation without fine-tuning and shows promising transferability, bridging language modeling and graph generation to lay the groundwork for graph foundation models. Our code is available at https://github.com/BorgwardtLab/AutoGraph.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.50/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.405 （内訳：関連性 0.250、 新規性 0.392、 インパクト 0.625） 
レビュアーの信頼度は3.50/5（高い）です。

#### AI評価（内容分析）

この論文はGraph Generationに特化しており、ユーザーの研究興味に直接関連しています。新しいアプローチであるAutoGraphは、従来の手法と比較して効率的であり、特に大規模なグラフに対して優れた性能を示しています。実用性も高く、生成速度やトレーニング速度の向上が実証されているため、実際の応用においても有用です。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=eszmES7j1F)
- [PDF](https://openreview.net/pdf?id=eszmES7j1F)

---

### 2. A Unified Framework for Fair Graph Generation: Theoretical Guarantees and Empirical Advances

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.651** |
| OpenReview総合 | 0.427 |
| 　├ 関連性 | 0.213 |
| 　├ 新規性 | 0.525 |
| 　└ インパクト | 0.615 |
| AI評価（関連性） | 0.900 |
| AI評価（新規性） | 0.800 |
| AI評価（実用性） | 0.700 |
| OpenReview評価 | 4.50/10 |

**著者**: Zichong Wang, Zhipeng Yin, Wenbin Zhang

**キーワード**: Fairness, Graph Generation, GNN

#### 概要

Graph generation models play pivotal roles in many real-world applications, from data augmentation to privacy-preserving. Despite their deployment successes, existing approaches often exhibit fairness issues, limiting their adoption in high-risk decision-making applications. Most existing fair graph generation works are based on autoregressive models that suffer from ordering sensitivity, while primarily addressing structural bias and overlooking the critical issue of feature bias. To this end, we propose FairGEM, a novel one-shot graph generation framework designed to mitigate both graph structural bias and node feature bias simultaneously. Furthermore, our theoretical analysis establishes that FairGEM delivers substantially stronger fairness guarantees than existing models while preserving generation quality. Extensive experiments across multiple real-world datasets demonstrate that FairGEM achieves superior performance in both generation quality and fairness.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.50/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.427 （内訳：関連性 0.213、 新規性 0.525、 インパクト 0.615） 
レビュアーの信頼度は3.25/5（高い）です。

#### AI評価（内容分析）

この論文はGraph Generationに関するものであり、ユーザーの研究興味に非常に関連しています。新しいフレームワークFairGEMを提案しており、既存のモデルの限界を克服する点で新規性がありますが、実用性については実験結果が示されているものの、具体的な応用例が不足しているため、やや低めの評価となります。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=T85ADT8a2y)
- [PDF](https://openreview.net/pdf?id=T85ADT8a2y)

---

### 3. Hierarchical Semantic-Augmented Navigation: Optimal Transport and Graph-Driven Reasoning for Vision-Language Navigation

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.649** |
| OpenReview総合 | 0.422 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.513 |
| 　└ インパクト | 0.594 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.900 |
| AI評価（実用性） | 0.700 |
| OpenReview評価 | 3.60/10 |

**著者**: Xiang Fang, Wanlong Fang, Changshuo Wang

**キーワード**: Hierarchical Semantic-Augmented Navigation

#### 概要

Vision-Language Navigation in Continuous Environments (VLN-CE) poses a formidable challenge for autonomous agents, requiring seamless integration of natural language instructions and visual observations to navigate complex 3D indoor spaces. Existing approaches often falter in long-horizon tasks due to limited scene understanding, inefficient planning, and lack of robust decision-making frameworks. We introduce the \textbf{Hierarchical Semantic-Augmented Navigation (HSAN)} framework, a groundbreaking approach that redefines VLN-CE through three synergistic innovations. First, HSAN constructs a dynamic hierarchical semantic scene graph, leveraging vision-language models to capture multi-level environmental representations—from objects to regions to zones—enabling nuanced spatial reasoning. Second, it employs an optimal transport-based topological planner, grounded in Kantorovich's duality, to select long-term goals by balancing semantic relevance and spatial accessibility with theoretical guarantees of optimality. Third, a graph-aware reinforcement learning policy ensures precise low-level control, navigating subgoals while robustly avoiding obstacles. By integrating spectral graph theory, optimal transport, and advanced multi-modal learning, HSAN addresses the shortcomings of static maps and heuristic planners prevalent in prior work. Extensive experiments on multiple challenging VLN-CE datasets demonstrate that HSAN achieves state-of-the-art performance, with significant improvements in navigation success and generalization to unseen environments.

#### OpenReview評価

この論文は5件のレビューを受け、 平均3.60/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.422 （内訳：関連性 0.225、 新規性 0.513、 インパクト 0.594） 
レビュアーの信頼度は3.40/5（高い）です。

#### AI評価（内容分析）

この論文は、視覚と言語の統合に基づくナビゲーションに関するものであり、特にグラフ生成に関連する階層的なセマンティックシーングラフを構築しているため、関連性が高いです。新規性に関しては、最適輸送とグラフ駆動の推論を組み合わせたアプローチは独創的であり、従来の手法の限界を克服する可能性があります。実用性は高いものの、理論的な枠組みが実際のアプリケーションにどの程度適用できるかは、さらなる検証が必要です。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=ypVW5jvguX)
- [PDF](https://openreview.net/pdf?id=ypVW5jvguX)

---

### 4. Topology-aware Graph Diffusion Model with Persistent Homology

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.628** |
| OpenReview総合 | 0.471 |
| 　├ 関連性 | 0.250 |
| 　├ 新規性 | 0.600 |
| 　└ インパクト | 0.635 |
| AI評価（関連性） | 0.900 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.50/10 |

**著者**: Joonhyuk Park, Donghyun Lee, Yujee Song, Guorong Wu, Won Hwa Kim

**キーワード**: Graph Generation, Diffusion, Topology, Brain Network

#### 概要

Generating realistic graphs faces challenges in estimating accurate distribution of graphs in an embedding space while preserving structural characteristics. However, existing graph generation methods primarily focus on approximating the joint distribution of nodes and edges, often overlooking topological properties such as connected components and loops, hindering accurate representation of global structures. To address this issue, we propose a Topology-Aware diffusion-based Graph Generation (TAGG), which aims to sample synthetic graphs that closely resemble the structural characteristics of the original graph based on persistent homology. Specifically, we suggest two core components: 1) Persistence Diagram Matching (PDM) loss which ensures high topological fidelity of generated graphs, and 2) topology-aware attention module (TAM) which induces the denoising network to capture the homological characteristics of the original graphs. Extensive experiments on conventional graph benchmarks demonstrate the effectiveness of our approach demonstrating high generation performance across various metrics, while achieving closer alignment with the distribution of topological features observed in the original graphs. Furthermore, application to real brain network data showcases its potential for complex and real graph applications.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.50/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.471 （内訳：関連性 0.250、 新規性 0.600、 インパクト 0.635） 
レビュアーの信頼度は3.75/5（高い）です。

#### AI評価（内容分析）

この論文はGraph Generationに関するものであり、ユーザーの研究興味に直接関連しています。新規性については、トポロジーに基づくアプローチを提案している点が評価できますが、既存の手法との比較が不十分なため、スコアはやや控えめです。実用性は、実際の脳ネットワークデータへの応用が示されているものの、具体的な応用例や実装の詳細が不足しているため、スコアは中程度に留まります。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=sye27MizdM)
- [PDF](https://openreview.net/pdf?id=sye27MizdM)

---

### 5. Scaling Epidemic Inference on Contact Networks: Theory and Algorithms

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.626** |
| OpenReview総合 | 0.514 |
| 　├ 関連性 | 0.412 |
| 　├ 新規性 | 0.546 |
| 　└ インパクト | 0.617 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.25/10 |

**著者**: Guanghui Min, Yinhan He, Chen Chen

**キーワード**: Computational Epidemiology, Graph Theory, Algorithm Acceleration

#### 概要

Computational epidemiology is crucial in understanding and controlling infectious diseases, as highlighted by large-scale outbreaks such as COVID-19. Given the inherent uncertainty and variability of disease spread, Monte Carlo (MC) simulations are widely used to predict infection peaks, estimate reproduction numbers, and evaluate the impact of non-pharmaceutical interventions (NPIs). While effective, MC-based methods require numerous runs to achieve statistically reliable estimates and variance, which suffer from high computational costs. In this work, we present a unified theoretical framework for analyzing disease spread dynamics on both directed and undirected contact networks, and propose an algorithm, **RAPID**, that significantly improves computational efficiency. Our contributions are threefold. First, we derive an asymptotic variance lower bound for MC estimates and identify the key factors influencing estimation variance. Second, we provide a theoretical analysis of the probabilistic disease spread process using linear approximations and derive the convergence conditions under non-reinfection epidemic models. Finally, we conduct extensive experiments on six real-world datasets, demonstrating our method's effectiveness and robustness in estimating the nodes' final state distribution. Specifically, our proposed method consistently produces accurate estimates aligned with results from a large number of MC simulations, while maintaining a runtime comparable to a single MC simulation. Our code and datasets are available at https://github.com/GuanghuiMin/RAPID.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.25/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.514 （内訳：関連性 0.412、 新規性 0.546、 インパクト 0.617） 
レビュアーの信頼度は3.50/5（高い）です。

#### AI評価（内容分析）

この論文は感染症の拡散をグラフ理論に基づいて分析しており、Graph Generationに関連する要素が含まれています。新しいアルゴリズムRAPIDの提案は新規性がありますが、既存のMCシミュレーションに依存しているため、完全に独自とは言えません。実用性は高いものの、特定の応用に限られる可能性があるため、スコアはやや低めです。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=qF5IrJfJDS)
- [PDF](https://openreview.net/pdf?id=qF5IrJfJDS)

---

### 6. A Generalized Binary Tree Mechanism for Private Approximation of All-Pair Shortest Distances

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.612** |
| OpenReview総合 | 0.479 |
| 　├ 関連性 | 0.375 |
| 　├ 新規性 | 0.475 |
| 　└ インパクト | 0.622 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.75/10 |

**著者**: Zongrui Zou, Chenglin Fan, Michael Dinitz, Jingcheng Liu, Jalaj Upadhyay

**キーワード**: differential privacy, graph theory

#### 概要

We study the problem of approximating all-pair distances in a weighted undirected graph with differential privacy, introduced by Sealfon [Sea16]. Given a publicly known undirected graph, we treat the weights of edges as sensitive information, and two graphs are neighbors if their edge weights differ in one edge by at most one. We obtain efficient algorithms with significantly improved bounds on a broad class of graphs which we refer to as *recursively separable*. In particular, for any $n$-vertex $K_h$-minor-free graph, our algorithm achieve an additive error of $ \widetilde{O}(h(nW)^{1/3} ) $, where $ W $ represents the maximum edge weight; For grid graphs, the same algorithmic scheme achieve additive error of $ \widetilde{O}(n^{1/4}\sqrt{W}) $.

Our approach can be seen as a generalization of the celebrated binary tree mechanism for range queries, as releasing range queries is equivalent to computing all-pair distances on a path graph. In essence, our approach is based on generalizing the binary tree mechanism to graphs that are *recursively separable*.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.75/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.479 （内訳：関連性 0.375、 新規性 0.475、 インパクト 0.622） 
レビュアーの信頼度は3.25/5（高い）です。

#### AI評価（内容分析）

この論文は、グラフ理論における全対最短距離の近似に関するものであり、Graph Generationに関連する研究者にとって興味深い内容です。新しいアルゴリズムの提案は新規性がありますが、特定のグラフクラスに限定されているため、一般的な応用には限界があります。実用性はあるものの、実際の応用においてはさらなる検証が必要です。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=vpJDCWOnPj)
- [PDF](https://openreview.net/pdf?id=vpJDCWOnPj)

---

### 7. Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.606** |
| OpenReview総合 | 0.464 |
| 　├ 関連性 | 0.237 |
| 　├ 新規性 | 0.617 |
| 　└ インパクト | 0.613 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.33/10 |

**著者**: Chenglin Fan, Dahoon Lee, Euiwoong Lee

**キーワード**: Correlation Clustering, Chromatic Clustering, Approximation Algorithms, Graph Algorithms

#### 概要

Correlation Clustering (CC) is a foundational problem in unsupervised learning that models binary similarity relations using labeled graphs. While classical CC has been well studied, many real-world applications involve more nuanced relationships—either multi-class categorical interactions or varying confidence levels in edge labels. To address these, two natural generalizations have been proposed: Chromatic Correlation Clustering (CCC), which assigns semantic colors to edge labels, and pseudometric-weighted CC, which allows edge weights satisfying the triangle inequality. In this paper, we develop improved approximation algorithms for both settings. Our approach leverages LP-based pivoting techniques combined with problem-specific rounding functions. For the pseudometric-weighted correlation clustering problem, we present a tight $\frac{10}{3}$-approximation algorithm, matching the best possible bound achievable within the framework of standard LP relaxation combined with specialized rounding. For the Chromatic Correlation Clustering (CCC) problem, we improve the approximation ratio from the previous best of $2.5$ to  $2.15$, and we establish a lower bound of $2.11$ within the same analytical framework, highlighting the near-optimality of our result.

#### OpenReview評価

この論文は3件のレビューを受け、 平均4.33/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.464 （内訳：関連性 0.237、 新規性 0.617、 インパクト 0.613） 
レビュアーの信頼度は3.33/5（高い）です。

#### AI評価（内容分析）

この論文はグラフアルゴリズムに関連しており、特に相関クラスタリングの新しいアプローチを提案しているため、Graph Generationに興味がある研究者にとって関連性が高いです。新しい近似アルゴリズムの開発は新規性を持っていますが、既存の研究に基づいているため、完全に革新的とは言えません。実用性については、理論的な結果が実際のアプリケーションにどの程度適用できるかは不明であり、やや低めの評価となっています。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=0JSolJVzjd)
- [PDF](https://openreview.net/pdf?id=0JSolJVzjd)

---

### 8. PointTruss: K-Truss for Point Cloud Registration

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.601** |
| OpenReview総合 | 0.452 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.575 |
| 　└ インパクト | 0.630 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.00/10 |

**著者**: Yue Wu, Jun Jiang, Yongzhe Yuan, Maoguo Gong, Qiguang Miao 他3名

**キーワード**: Point cloud registration; compatibility graph; outlier removal; k-truss; correspondence selection

#### 概要

Point cloud registration is a fundamental task in 3D computer vision. Recent advances have shown that graph-based methods are effective for outlier rejection in this context. However, existing clique-based methods impose overly strict constraints and are NP-hard, making it difficult to achieve both robustness and efficiency. While the k-core reduces computational complexity, which only considers node degree and ignores higher-order topological structures such as triangles, limiting its effectiveness in complex scenarios. To overcome these limitations, we introduce the $k$-truss from graph theory into point cloud registration, leveraging triangle support as a constraint for inlier selection. We further propose a consensus voting-based low-scale sampling strategy to efficiently extract the structural skeleton of the point cloud prior to $k$-truss decomposition. Additionally, we design a spatial distribution score that balances coverage and uniformity of inliers, preventing selections that concentrate on sparse local clusters. Extensive experiments on KITTI, 3DMatch, and 3DLoMatch demonstrate that our method consistently outperforms both traditional and learning-based approaches in various indoor and outdoor scenarios, achieving state-of-the-art results.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.00/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.452 （内訳：関連性 0.225、 新規性 0.575、 インパクト 0.630） 
レビュアーの信頼度は4.00/5（非常に高い）です。

#### AI評価（内容分析）

この論文は、グラフ理論に基づく新しい手法を用いて点群登録の問題に取り組んでおり、特にk-trussを導入する点が関連性を高めています。新規性はあるものの、既存の手法に対する明確な優位性が示されているため、実用性はやや低めと評価しました。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=MuxBO5f8mL)
- [PDF](https://openreview.net/pdf?id=MuxBO5f8mL)

---

### 9. Doodle to Detect: A Goofy but Powerful Approach to Skeleton-based Hand Gesture Recognition

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.599** |
| OpenReview総合 | 0.448 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.558 |
| 　└ インパクト | 0.635 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.50/10 |

**著者**: Sang Hoon Han, Seonho Lee, Hyeok Nam, Jae Hyeon Park, Min Hee Cha 他5名

**キーワード**: Hand Gesture Recognition, Skeleton based Action Recognition, Online Recognition, Modality Transform, Vision Transformer

#### 概要

Skeleton-based hand gesture recognition plays a crucial role in enabling intuitive human–computer interaction. Traditional methods have primarily relied on hand-crafted features—such as distances between joints or positional changes across frames—to alleviate issues from viewpoint variation or body proportion differences. However, these hand-crafted features often fail to capture the full spatio-temporal information in raw skeleton data, exhibit poor interpretability, and depend heavily on dataset-specific preprocessing, limiting generalization. In addition, normalization strategies in traditional methods, which rely on training data, can introduce domain gaps between training and testing environments, further hindering robustness in diverse real-world settings. To overcome these challenges, we exclude traditional hand-crafted features and propose Skeleton Kinematics Extraction Through Coordinated grapH (SKETCH), a novel framework that directly utilizes raw four-dimensional (time, x, y, and z) skeleton sequences and transforms them into intuitive visual graph representations. The proposed framework incorporates a novel learnable Dynamic Range Embedding (DRE) to preserve axis-wise motion magnitudes lost during normalization and visual graph representations, enabling richer and more discriminative feature learning. This approach produces a graph image that richly captures the raw data’s inherent information and provides interpretable visual attention cues. Furthermore, SKETCH applies independent min–max normalization on fixed-length temporal windows in real time, mitigating degradation from absolute coordinate fluctuations caused by varying sensor viewpoints or differences in individual body proportions. Through these designs, our approach becomes inherently topology-agnostic, avoiding fragile dependencies on dataset- or sensor-specific skeleton definitions. By leveraging pre-trained vision backbones, SKETCH achieves efficient convergence and superior recognition accuracy. Experimental results on SHREC’19 and SHREC’22 benchmarks show that it outperforms state-of-the-art methods in both robustness and generalization, establishing a new paradigm for skeleton-based hand gesture recognition. The code is available at https://github.com/capableofanything/SKETCH.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.50/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.448 （内訳：関連性 0.225、 新規性 0.558、 インパクト 0.635） 
レビュアーの信頼度は3.75/5（高い）です。

#### AI評価（内容分析）

この論文は、Skeleton-based hand gesture recognitionにおける新しいアプローチを提案しており、特にGraph Generationに関連する視覚的グラフ表現を用いているため、関連性が高いです。新規性はあるものの、従来の手法との比較が不十分であり、実用性は実験結果に基づくものの、実際の応用における課題が残る可能性があります。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=u8SXX5ITE6)
- [PDF](https://openreview.net/pdf?id=u8SXX5ITE6)

---

### 10. Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with Large Language Models

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.598** |
| OpenReview総合 | 0.445 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.546 |
| 　└ インパクト | 0.637 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.25/10 |

**著者**: Siwei Zhang, Yun Xiong, Yateng Tang, Jiarong Xu, Xi Chen 他4名

**キーワード**: Temporal Text-attributed Graph, Large Language Models, Data Mining

#### 概要

Temporal graph neural networks (TGNNs) have shown remarkable performance in temporal graph modeling. However, real-world temporal graphs often possess rich textual information, giving rise to temporal text-attributed graphs (TTAGs). Such combination of dynamic text semantics and evolving graph structures introduces heightened complexity. Existing TGNNs embed texts statically and rely heavily on encoding mechanisms that biasedly prioritize structural information, overlooking the temporal evolution of text semantics and the essential interplay between semantics and structures for synergistic reinforcement.
To tackle these issues, we present $\textbf{CROSS}$, a flexible framework that seamlessly extends existing TGNNs for TTAG modeling. CROSS is designed by decomposing the TTAG modeling process into two phases: (i) temporal semantics extraction; and (ii) semantic-structural information unification. The key idea is to advance the large language models (LLMs) to $\textit{dynamically}$ extract the temporal semantics in text space and then generate $\textit{cohesive}$ representations unifying both semantics and structures.
Specifically, we propose a Temporal Semantics Extractor in the CROSS framework, which empowers LLMs to offer the temporal semantic understanding of node's evolving contexts of textual neighborhoods, facilitating semantic dynamics.
Subsequently, we introduce the Semantic-structural Co-encoder, which collaborates with the above Extractor for synthesizing illuminating representations by jointly considering both semantic and structural information while encouraging their mutual reinforcement. Extensive experiments show that CROSS achieves state-of-the-art results on four public datasets and one industrial dataset, with 24.7\% absolute MRR gain on average in temporal link prediction and 3.7\% AUC gain in node classification of industrial application.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.25/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.445 （内訳：関連性 0.225、 新規性 0.546、 インパクト 0.637） 
レビュアーの信頼度は4.00/5（非常に高い）です。

#### AI評価（内容分析）

この論文は、テキスト属性を持つ時系列グラフのモデリングに関するものであり、Graph Generationに関連する要素が含まれています。新しいフレームワークCROSSは、既存のTGNNを拡張する点で新規性がありますが、特に実用性に関しては、実験結果が示す通りの効果がどの程度の範囲で適用可能かは不明です。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=9env0BdcDV)
- [PDF](https://openreview.net/pdf?id=9env0BdcDV)

---

### 11. Learning to Plan Like the Human Brain via Visuospatial Perception and Semantic-Episodic Synergistic Decision-Making

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.598** |
| OpenReview総合 | 0.444 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.563 |
| 　└ インパクト | 0.616 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.60/10 |

**著者**: Tianyuan Jia, Ziyu Li, Qing Li, Xiuxing Li, Xiang Li 他3名

**キーワード**: Brain-inspired learning; Motion planning; Graph neural networks;

#### 概要

Motion planning in high-dimensional continuous spaces remains challenging due to complex environments and computational constraints. Although learning-based planners, especially graph neural network (GNN)-based, have significantly improved planning performance, they still struggle with inaccurate graph construction and limited structural reasoning, constraining search efficiency and path quality. The human brain exhibits efficient planning through a two-stage Perception-Decision model. First, egocentric spatial representations from visual and proprioceptive input are constructed, and then semantic–episodic synergy is leveraged to support decision-making in uncertainty scenarios. Inspired by this process, we propose NeuroMP, a brain-inspired planning framework that learns to plan like the human brain. NeuroMP integrates a Perceptive Segment Selector inspired by visuospatial perception to construct safer graphs, and a Global Alignment Heuristic guide search in weakly connected graphs by modeling semantic-episodic synergistic decision-making. Experimental results demonstrate that NeuroMP significantly outperforms existing planning methods in efficiency and quality while maintaining a high success rate.

#### OpenReview評価

この論文は5件のレビューを受け、 平均4.60/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.444 （内訳：関連性 0.225、 新規性 0.563、 インパクト 0.616） 
レビュアーの信頼度は3.20/5（高い）です。

#### AI評価（内容分析）

この論文は、グラフニューラルネットワークに基づくモーションプランニングに関するものであり、ユーザーの研究興味であるグラフ生成に関連しています。新しい脳にインスパイアされたアプローチを提案しているため、新規性も高いですが、実用性はまだ実験結果に依存しているため、やや低めです。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=1KXST1ksJ2)
- [PDF](https://openreview.net/pdf?id=1KXST1ksJ2)

---

### 12. Reinforcement learning for one-shot DAG scheduling with comparability identification and dense reward

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.595** |
| OpenReview総合 | 0.436 |
| 　├ 関連性 | 0.375 |
| 　├ 新規性 | 0.338 |
| 　└ インパクト | 0.617 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.25/10 |

**著者**: Xumai Qi, Dongdong Zhang, Taotao Liu, Hongcheng Wang

**キーワード**: DAG scheduling, graph theory, combinatorial optimization problem, reinforcement learning

#### 概要

In recent years, many studies proposed to generate solutions for Directed Acyclic Graph (DAG) scheduling problem in one shot by combining reinforcement learning and list scheduling heuristic. However, these existing methods suffer from biased estimation of sampling probabilities and inefficient guidance in training, due to redundant comparisons among node priorities and the sparse reward challenge. To address these issues, we analyze of the limitations of these existing methods, and propose a novel one-shot DAG scheduling method with comparability identification and dense reward signal, based on the policy gradient framework. In our method, a comparable antichain identification mechanism is proposed to eliminate the problem of redundant nodewise priority comparison. We also propose a dense reward signal for node level decision-making optimization in training, effectively addressing the sparse reward challenge. The experimental results show that the proposed method can yield superior results of scheduling objectives compared to other learning-based DAG scheduling methods.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.25/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.436 （内訳：関連性 0.375、 新規性 0.338、 インパクト 0.617） 
レビュアーの信頼度は3.50/5（高い）です。

#### AI評価（内容分析）

この論文はDAGスケジューリングに関するものであり、Graph Generationに関連するため、関連性は高いです。新規性については、従来の手法の限界を克服する新しいアプローチを提案しているため、一定の新規性がありますが、既存の研究に基づいているため完璧ではありません。実用性は、提案された手法が実験結果で優れた性能を示しているものの、実際の応用における具体的な利点や制約が不明なため、やや低めです。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=KDKddNgeKo)
- [PDF](https://openreview.net/pdf?id=KDKddNgeKo)

---

### 13. Differentially Private Gomory-Hu Trees

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.594** |
| OpenReview総合 | 0.436 |
| 　├ 関連性 | 0.312 |
| 　├ 新規性 | 0.383 |
| 　└ インパクト | 0.653 |
| AI評価（関連性） | 0.700 |
| AI評価（新規性） | 0.800 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.33/10 |

**著者**: Anders Aamand, Justin Y. Chen, Mina Dalirrooyfard, Slobodan Mitrović, Yuriy Nevmyvaka 他2名

**キーワード**: differential privacy, graph algorithms, cuts

#### 概要

Given an undirected, weighted $n$-vertex graph $G = (V, E, w)$, a Gomory-Hu tree $T$ is a weighted tree on $V$ that preserves the Min-$s$-$t$-Cut between any pair of vertices $s, t \in V$. Finding cuts in graphs is a key primitive in problems such as bipartite matching, spectral and correlation clustering, and community detection. We design a differentially private (DP) algorithm that computes an approximate Gomory-Hu tree. Our algorithm is $\varepsilon$-DP, runs in polynomial time, and can be used to compute $s$-$t$ cuts that are $\tilde{O}(n/\varepsilon)$-additive approximations of the Min-$s$-$t$-Cuts in $G$ for all distinct $s, t \in V$ with high probability. Our error bound is essentially optimal, since [Dalirrooyfard, Mitrovic and Nevmyvaka, Neurips 2023] showed that privately outputting a single Min-$s$-$t$-Cut requires $\Omega(n)$ additive error even with $(\varepsilon, \delta)$-DP and allowing for multiplicative error. Prior to our work, the best additive error bounds for approximate all-pairs Min-$s$-$t$-Cuts were $O(n^{3/2}/\varepsilon)$ for $\varepsilon$-DP [Gupta, Roth, Ullman, TCC 2009] and $\tilde{O}(\sqrt{mn}/ \varepsilon)$  for $(\varepsilon, \delta)$-DP [Liu, Upadhyay and Zou, SODA 2024], both achieved by DP algorithms that preserve all cuts in the graph. To achieve our result, we develop an $\varepsilon$-DP algorithm for the Minimum Isolating Cuts problem with near-linear error, and introduce a novel privacy composition technique combining elements of both parallel and basic composition to handle `bounded overlap' computational branches in recursive algorithms, which maybe of independent interest.

#### OpenReview評価

この論文は3件のレビューを受け、 平均4.33/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.436 （内訳：関連性 0.312、 新規性 0.383、 インパクト 0.653） 
レビュアーの信頼度は4.33/5（非常に高い）です。

#### AI評価（内容分析）

この論文はグラフアルゴリズムに関連しており、特にGomory-Hu木に焦点を当てているため、Graph Generationに興味がある研究者にとって関連性があります。新規性については、差分プライバシーを考慮した新しいアルゴリズムを提案しており、特にエラー境界の最適性が強調されていますが、実用性は理論的な結果に依存しているため、やや低めです。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=7rBeyE4nie)
- [PDF](https://openreview.net/pdf?id=7rBeyE4nie)

---

### 14. Low-degree evidence for computational transition of recovery rate in stochastic block model

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.586** |
| OpenReview総合 | 0.466 |
| 　├ 関連性 | 0.213 |
| 　├ 新規性 | 0.500 |
| 　└ インパクト | 0.770 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.500 |
| OpenReview評価 | 5.00/10 |

**著者**: Jingqiu Ding, Yiding Hua, Lucas Slot, David Steurer

**キーワード**: low-degree lower bound, stochastic block model, computational complexity

#### 概要

We investigate implications of the (extended) low-degree conjecture (recently formalized in [moitra et al2023]) in the context of the symmetric stochastic block model. Assuming the conjecture holds, we establish that no polynomial-time algorithm can weakly recover community labels below the Kesten-Stigum (KS) threshold. In particular, we rule out polynomial-time estimators that, with constant probability, achieve $n^{-0.49}$ correlation with the true communities. 
Whereas, above the KS threshold, polynomial-time algorithms are known to achieve constant correlation with the true communities with high probability  [massoulie et al 2014,abbe et al 2015]. 

To our knowledge, we provide the first rigorous evidence for such sharp transition in recovery rate for polynomial-time algorithms at the KS threshold. 
Notably, under a stronger version of the low-degree conjecture, our lower bound remains valid even when the number of blocks diverges. 
Furthermore, our results provide evidence of a computational-to-statistical gap in learning the parameters of stochastic block models.

In contrast, prior work either (i) rules out polynomial-time algorithms with $1 - o(1)$ success probability [Hopkins 18, bandeira et al 2021] under the low-degree conjecture, or (ii) degree-$\text{poly}(k)$ polynomials for learning the stochastic block model [Luo et al 2023].

For this, we design a hypothesis test which succeeeds with constant probability under symmetric stochastic block model, and $1-o(1)$ probability under the distribution of \Erdos \Renyi random graphs.
Our proof combines low-degree lower bounds from [Hopkins 18, bandeira et al 2021]  with graph splitting and cross-validation techniques. 
In order to rule out general recovery algorithms, we employ the correlation preserving projection method developed in [Hopkins et al 17].

#### OpenReview評価

この論文は4件のレビューを受け、 平均5.00/10の評価を獲得しました。 採択判定は「Accept (spotlight)」で、特に高く評価されています。 

【評価スコアの詳細】 総合スコア：0.466 （内訳：関連性 0.213、 新規性 0.500、 インパクト 0.770） 
レビュアーの信頼度は3.00/5（高い）です。

#### AI評価（内容分析）

この論文は、確率的ブロックモデルにおける計算的遷移に関する新しい証拠を提供しており、Graph Generationに関連する研究にとって重要です。新規性はあるものの、実用性は限られており、理論的な結果が実際の応用にどのように結びつくかは不明です。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=fBNaGVMDD9)
- [PDF](https://openreview.net/pdf?id=fBNaGVMDD9)

---

### 15. Venus-MAXWELL: Efficient Learning of Protein-Mutation Stability Landscapes using Protein Language Models

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.586** |
| OpenReview総合 | 0.415 |
| 　├ 関連性 | 0.213 |
| 　├ 新規性 | 0.487 |
| 　└ インパクト | 0.612 |
| AI評価（関連性） | 0.600 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.800 |
| OpenReview評価 | 4.75/10 |

**著者**: Yuanxi Yu, Fan Jiang, Xinzhu Ma, Liang Zhang, Bozitao Zhong 他5名

**キーワード**: Deep learning, Protein language model, Protein stability, Protein engineering

#### 概要

In-silico prediction of protein mutant stability, measured by the difference in Gibbs free energy change ($\Delta \Delta G$), is fundamental for protein engineering.
Current sequence-to-label methods typically employ two-stage pipelines: (i) encoding mutant sequences using neural networks (e.g., transformers), followed by (ii) the  $\Delta \Delta G$ regression from the latent representations.
Although these methods have demonstrated promising performance, their dependence on specialized neural network encoders significantly increases the complexity.
Additionally, the requirement to compute latent representations individually for each mutant sequence negatively impacts computational efficiency and poses the risk of overfitting.
This work proposes the Venus-MAXWELL framework, which reformulates mutation $\Delta \Delta G$ prediction as a sequence-to-landscape task.
In Venus-MAXWELL, mutations of a protein and their corresponding $\Delta \Delta G$ values are organized into a landscape matrix, allowing our framework to learn the $\Delta \Delta G$ landscape of a protein with a single forward and backward pass during training. To this end, we curated a new  $\Delta \Delta G$ benchmark dataset with strict controls on data leakage and redundancy to ensure robust evaluation.
Leveraging the zero-shot scoring capability of protein language models (PLMs), Venus-MAXWELL effectively utilizes the evolutionary patterns learned by PLMs during pre-training.
More importantly, Venus-MAXWELL is compatible with multiple protein language models.
For example, when integrated with the ESM-IF, Venus-MAXWELL achieves higher accuracy than ThermoMPNN with 10$\times$ faster in inference speed (despite having 50$\times$ more parameters than ThermoMPNN).
The training codes, model weights, and datasets are publicly available at https://github.com/ai4protein/Venus-MAXWELL.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.75/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.415 （内訳：関連性 0.213、 新規性 0.487、 インパクト 0.612） 
レビュアーの信頼度は3.00/5（高い）です。

#### AI評価（内容分析）

この論文は、タンパク質の変異安定性予測に関するものであり、Graph Generationに直接関連する内容ではないが、タンパク質の構造や変異に関する情報を扱っているため、一定の関連性がある。新しいフレームワークを提案しており、従来の手法に比べて効率的であるため、新規性が高い。実用性も高く、公開されたコードとデータセットにより、他の研究者が容易に利用できる点が評価される。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=w7hiWakSAq)
- [PDF](https://openreview.net/pdf?id=w7hiWakSAq)

---

### 16. Beyond Pairwise Connections: Extracting High-Order Functional Brain Network Structures under Global Constraints

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.580** |
| OpenReview総合 | 0.400 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.425 |
| 　└ インパクト | 0.607 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.25/10 |

**著者**: Ling Zhan, Junjie Huang, Xiaoyao Yu, Wenyu Chen, Tao Jia

**キーワード**: functional brain networks, high-order interactions, graph structure learning, pairwise network modeling

#### 概要

Functional brain network (FBN) modeling often relies on local pairwise interactions, whose limitation in capturing high-order dependencies is theoretically analyzed in this paper. Meanwhile, the computational burden and heuristic nature of current hypergraph modeling approaches hinder end-to-end learning of FBN structures directly from data distributions. To address this, we propose to extract high-order FBN structures under global constraints, and implement this as a Global Constraints oriented Multi-resolution (GCM) FBN structure learning framework. It incorporates 4 types of global constraint (signal synchronization, subject identity, expected edge numbers, and data labels) to enable learning FBN structures for 4 distinct levels (sample/subject/group/project) of modeling resolution. Experimental results demonstrate that GCM achieves up to a 30.6% improvement in relative accuracy and a 96.3% reduction in computational time across 5 datasets and 2 task settings, compared to 9 baselines and 10 state-of-the-art methods. Extensive experiments validate the contributions of individual components and highlight the interpretability of GCM. This work offers a novel perspective on FBN structure learning and provides a foundation for interdisciplinary applications in cognitive neuroscience. Code is publicly available on https://github.com/lzhan94swu/GCM.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.25/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.400 （内訳：関連性 0.225、 新規性 0.425、 インパクト 0.607） 
レビュアーの信頼度は3.25/5（高い）です。

#### AI評価（内容分析）

この論文は高次の機能的脳ネットワーク構造の抽出に関するものであり、Graph Generationに関連する新しいアプローチを提案しています。新規性はあるものの、既存の手法との比較が多いため、完全に独自とは言えません。また、実用性は改善が見られるものの、実際の応用における具体的な利点が不明瞭なため、やや低めに評価しました。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=ybH0avRV4n)
- [PDF](https://openreview.net/pdf?id=ybH0avRV4n)

---

### 17. GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.550** |
| OpenReview総合 | 0.324 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.175 |
| 　└ インパクト | 0.605 |
| AI評価（関連性） | 0.800 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 3.50/10 |

**著者**: Jialong Zhou, Lichao Wang, Xiao Yang

**キーワード**: LLM-based agent, defense, safety

#### 概要

The emergence of large language models (LLMs) enables the development of intelligent agents capable of engaging in complex and multi-turn dialogues. However, multi-agent collaboration faces critical safety challenges, such as hallucination amplification and error injection and propagation. This paper presents GUARDIAN, a unified method for detecting and mitigating multiple safety concerns in GUARDing Intelligent Agent collaboratioNs. By modeling the multi-agent collaboration process as a discrete-time temporal attributed graph, GUARDIAN explicitly captures the propagation dynamics of hallucinations and errors. The unsupervised encoder-decoder architecture incorporating an incremental training paradigm learns to reconstruct node attributes and graph structures from latent embeddings, enabling the identification of anomalous nodes and edges with unparalleled precision. Moreover, we introduce a graph abstraction mechanism based on the Information Bottleneck Theory, which compresses temporal interaction graphs while preserving essential patterns. Extensive experiments demonstrate GUARDIAN's effectiveness in safeguarding LLM multi-agent collaborations against diverse safety vulnerabilities, achieving state-of-the-art accuracy with efficient resource utilization. The code is available at https://github.com/JialongZhou666/GUARDIAN.

#### OpenReview評価

この論文は4件のレビューを受け、 平均3.50/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.324 （内訳：関連性 0.225、 新規性 0.175、 インパクト 0.605） 
レビュアーの信頼度は3.75/5（高い）です。

#### AI評価（内容分析）

この論文は、グラフモデルを用いてLLMベースのエージェントの安全性を確保する方法を提案しており、Graph Generationに関連する要素が含まれています。新規性はあるものの、既存の手法との比較が不明瞭であり、実用性は実験結果に依存するためやや低めです。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=6j9xJ9pBjm)
- [PDF](https://openreview.net/pdf?id=6j9xJ9pBjm)

---

### 18. Searching Latent Program Spaces

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.546** |
| OpenReview総合 | 0.466 |
| 　├ 関連性 | 0.213 |
| 　├ 新規性 | 0.487 |
| 　└ インパクト | 0.782 |
| AI評価（関連性） | 0.500 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.600 |
| OpenReview評価 | 4.75/10 |

**著者**: Matthew Macfarlane, Clément Bonnet

**キーワード**: Test-Time Compute, Latent Search, Deep Learning, Meta-Learning

#### 概要

General intelligence requires systems that acquire new skills efficiently and generalize beyond their training distributions.
Although program synthesis approaches have strong generalization power, they face scaling issues due to large combinatorial spaces that quickly make them impractical and require human-generated DSLs or pre-trained priors to narrow this search space.
On the other hand, deep learning methods have had high successes, but they lack structured test-time adaptation and rely on heavy stochastic sampling or expensive gradient updates for fine-tuning.
In this work, we propose the Latent Program Network (LPN), a new architecture that builds in test-time search directly into neural models.
LPN learns a latent space of implicit programs---neurally mapping inputs to outputs---through which it can search using gradients at test time.
LPN combines the adaptability of symbolic approaches and the scalability of neural methods.
It searches through a compact latent space at test time and bypasses the need for pre-defined domain-specific languages.
On a range of programming-by-examples tasks, LPN either outperforms or matches performance compared to in-context learning and test-time training methods.
Tested on the ARC-AGI benchmark, we demonstrate that LPN can both learn a compact program space and search through it at test time to adapt to novel tasks.
LPN doubles its performance on out-of-distribution tasks when test-time search is switched on.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.75/10の評価を獲得しました。 採択判定は「Accept (spotlight)」で、特に高く評価されています。 

【評価スコアの詳細】 総合スコア：0.466 （内訳：関連性 0.213、 新規性 0.487、 インパクト 0.782） 
レビュアーの信頼度は3.50/5（高い）です。

#### AI評価（内容分析）

この論文はプログラム合成と深層学習の融合に関するものであり、Graph Generationに直接関連する内容ではないため、関連性は中程度と評価しました。新しいアーキテクチャであるLatent Program Networkは、テスト時の検索を組み込むという新規性があり、特にプログラミングタスクにおいて有望です。実用性については、提案手法が特定のタスクでの性能向上を示しているものの、一般的な応用可能性はまだ不明確であるため、やや控えめに評価しました。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=CsXKGIqZtr)
- [PDF](https://openreview.net/pdf?id=CsXKGIqZtr)

---

### 19. Association-Focused Path Aggregation for Graph Fraud Detection

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.542** |
| OpenReview総合 | 0.454 |
| 　├ 関連性 | 0.213 |
| 　├ 新規性 | 0.613 |
| 　└ インパクト | 0.617 |
| AI評価（関連性） | 0.700 |
| AI評価（新規性） | 0.600 |
| AI評価（実用性） | 0.500 |
| OpenReview評価 | 4.25/10 |

**著者**: Tian Qiu, Wenda Li, Zunlei Feng, Jie Lei, Tao Wang 他3名

**キーワード**: deep learning, path aggregation, graph fraud detection

#### 概要

Fraudulent activities have caused substantial negative social impacts and are exhibiting emerging characteristics such as intelligence and industrialization, posing challenges of high-order interactions, intricate dependencies, and the sparse yet concealed nature of fraudulent entities. Existing graph fraud detectors are limited by their narrow "receptive fields", as they focus only on the relations between an entity and its neighbors while neglecting longer-range structural associations hidden between entities. To address this issue, we propose a novel fraud detector based on Graph Path Aggregation (GPA). It operates through variable-length path sampling, semantic-associated path encoding, path interaction and aggregation, and aggregation-enhanced fraud detection. To further facilitate interpretable association analysis, we synthesize G-Internet, the first benchmark dataset in the field of internet fraud detection. Extensive experiments across datasets in multiple fraud scenarios demonstrate that the proposed GPA outperforms mainstream fraud detectors by up to +15% in Average Precision (AP). Additionally, GPA exhibits enhanced robustness to noisy labels and provides excellent interpretability by uncovering implicit fraudulent patterns across broader contexts. Code is available at https://github.com/horrible-dong/GPA.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.25/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.454 （内訳：関連性 0.213、 新規性 0.613、 インパクト 0.617） 
レビュアーの信頼度は3.50/5（高い）です。

#### AI評価（内容分析）

この論文はグラフに基づく詐欺検出に関するものであり、Graph Generationに関連する要素が含まれていますが、直接的な関連性はやや低いです。新規性については、提案された手法が既存の手法に対して改善を示しているものの、特に革新的なアプローチとは言えません。実用性は、実験結果が示すように一定の効果を持つものの、実際の応用においてはさらなる検証が必要です。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=TiE8aTc3Zg)
- [PDF](https://openreview.net/pdf?id=TiE8aTc3Zg)

---

### 20. Noisy Multi-Label Learning through Co-Occurrence-Aware Diffusion

#### スコア

| 項目 | スコア |
|------|--------|
| **最終スコア** | **0.537** |
| OpenReview総合 | 0.442 |
| 　├ 関連性 | 0.225 |
| 　├ 新規性 | 0.558 |
| 　└ インパクト | 0.615 |
| AI評価（関連性） | 0.600 |
| AI評価（新規性） | 0.700 |
| AI評価（実用性） | 0.500 |
| OpenReview評価 | 4.50/10 |

**著者**: Senyu Hou, Yuru Ren, Gaoxia Jiang, Wenjian Wang

**キーワード**: Multi-label classification; Noisy multi-label learning; Diffusion model; Co-occurrence-aware

#### 概要

Noisy labels often compel models to overfit, especially in multi-label classification tasks. Existing methods for noisy multi-label learning (NML) primarily follow a discriminative paradigm, which relies on noise transition matrix estimation or small-loss strategies to correct noisy labels. However, they remain substantial optimization difficulties compared to noisy single-label learning. In this paper, we propose a Co-Occurrence-Aware Diffusion (CAD) model, which reformulates NML from a generative perspective. We treat features as conditions and multi-labels as diffusion targets, optimizing the diffusion model for multi-label learning with theoretical guarantees. Benefiting from the diffusion model's strength in capturing multi-object semantics and structured label matrix representation, we can effectively learn the posterior mapping from features to true multi-labels. To mitigate the interference of noisy labels in the forward process, we guide generation using pseudo-clean labels reconstructed from the latent neighborhood space, replacing original point-wise estimates with neighborhood-based proxies. In the reverse process, we further incorporate label co-occurrence constraints to enhance the model's awareness of incorrect generation directions, thereby promoting robust optimization. Extensive experiments on both synthetic (Pascal-VOC, MS-COCO) and real-world (NUS-WIDE) noisy datasets demonstrate that our approach outperforms state-of-the-art methods.

#### OpenReview評価

この論文は4件のレビューを受け、 平均4.50/10の評価を獲得しました。 採択判定は「Accept (poster)」です。 

【評価スコアの詳細】 総合スコア：0.442 （内訳：関連性 0.225、 新規性 0.558、 インパクト 0.615） 
レビュアーの信頼度は3.25/5（高い）です。

#### AI評価（内容分析）

この論文はマルチラベル分類におけるノイズの影響を扱っており、特に生成モデルの観点からアプローチしているため、Graph Generationに関連する可能性があります。しかし、直接的な関連性は薄く、主にマルチラベル学習に焦点を当てています。新規性は高く、従来の手法とは異なる視点を提供していますが、実用性は実験結果に依存するため、やや低めに評価しました。

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=zft0zTOFkN)
- [PDF](https://openreview.net/pdf?id=zft0zTOFkN)

---

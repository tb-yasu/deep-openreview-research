# 論文レビューレポート

**生成日時**: 2025年11月08日 14:31

## 検索条件

- **学会**: NeurIPS 2025
- **キーワード**: 指定なし
- **検索論文数**: 5526件
- **評価論文数**: 5526件
- **ランク対象論文数**: 35件

## 評価基準

- **研究興味**: graph generation, drug discovery, computational biology, machine learning, bioinformatics, network analysis, molecular modeling
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
- graph construction
- gg

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

### machine learning

**同義語**:
- artificial intelligence
- ai
- deep learning
- data mining
- predictive analytics

### bioinformatics

**同義語**:
- computational biology
- genomic data analysis
- biological data mining
- bioinfo
- bioinformatics tools

### network analysis

**同義語**:
- graph analysis
- social network analysis
- network modeling
- network theory
- nwa

### molecular modeling

**同義語**:
- molecular simulation
- computational chemistry
- molecular dynamics
- quantum chemistry
- 3d molecular modeling

## 統計情報

- **平均総合スコア**: 0.432
- **最高スコア**: 0.515
- **最低スコア**: 0.368
- **平均レビュー評価**: 4.40/10

## トップ論文

### 1. Toward a Unified Geometry Understanding : Riemannian Diffusion Framework for Graph Generation and Prediction

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.818** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.800 |
| 　├ インパクト | 0.750 |
| OpenReview評価 | 4.75/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Yisen Gao, Xingcheng Fu, Qingyun Sun, Jianxin Li, Xianxian LI

**キーワード**: Graph Generation, Hyperbolic Graph Learning, Riemannian Manifold, Graph Learning

#### 概要

Graph diffusion models have made significant progress in learning structured graph data and have demonstrated strong potential for predictive tasks. Existing approaches typically embed node, edge, and graph-level features into a unified latent space, modeling prediction tasks including classification and regression as a form of conditional generation.  However, due to the non-Euclidean nature of graph data, features of different curvatures are entangled in the same latent space without releasing their geometric potential. To address this issue, we aim to construt an ideal Riemannian diffusion model to capture distinct manifold signatures of complex graph data and learn their distribution.  This goal faces two challenges: numerical instability caused by exponential mapping during the encoding proces and manifold deviation during diffusion generation.  To address these challenges, we propose **GeoMancer**: a novel Riemannian graph diffusion framework for both generation and prediction tasks. To mitigate numerical instability, we replace exponential mapping with an isometric-invariant Riemannian gyrokernel approach and decouple multi-level features onto their respective task-specific manifolds to learn optimal representations.  To address manifold deviation, we introduce a manifold-constrained diffusion method and a self-guided strategy for unconditional generation, ensuring that the generated data remains aligned with the manifold signature. Extensive experiments validate the effectiveness of our approach, demonstrating superior performance across a variety of tasks.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 3)

**要約:**
Graph diffusion models offer a unified framework for tackling both graph generation and prediction tasks. To account for the non-Euclidean nature of graph-structured data, this paper proposes GeoMancer, a Riemannian diffusion approach that captures the intrinsic geometric nature of multi-level featu...

**レビュー 2** (評価: 5, 確信度: 4)

**要約:**
This paper proposed a method GeoMancer which approaches graph-structured data from a fundamentally different angle by modeling multi-level latent geometry in curved space.

**レビュー 3** (評価: 5, 確信度: 5)

**要約:**
This work introduces a geometry-based diffusion mechanism that adapts to the intrinsic curvature of multi-level graph features, from node, edge and graph feature. It first embeds multi-level graph data into a product manifold using a specially designed gyroscopic kernel.

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=QQqDBRRslp)
- [PDF](https://openreview.net/pdf?id=QQqDBRRslp)

---

### 2. JAMUN: Bridging Smoothed Molecular Dynamics and Score-Based Learning for Conformational Ensemble Generation

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.808** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.50/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Ameya Daigavane, Bodhi P. Vani, Darcy Davidson, Saeed Saremi, Joshua A Rackers 他1名

**キーワード**: Conformational Ensembles, Molecules, Protein Dynamics, Drug Discovery, Statistical Mechanics, Generative Models

#### 概要

Conformational ensembles of protein structures are immensely important both for understanding protein function and drug discovery in novel modalities such as cryptic pockets. Current techniques for sampling ensembles such as molecular dynamics (MD) are computationally inefficient, while many recent machine learning methods do not transfer to systems outside their training data. We propose JAMUN which performs MD in a smoothed, noised space of all-atom 3D conformations of molecules by utilizing the framework of walk-jump sampling. JAMUN enables ensemble generation for small peptides at rates of an order of magnitude faster than traditional molecular dynamics. The physical priors in JAMUN enables transferability to systems outside of its training data, even to peptides that are longer than those originally trained on. Our model, code and weights are available at https://github.com/prescient-design/jamun.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 2)

**要約:**
The authors present JAMUN, a walk-jump sampling model for generating ensembles of molecular conformations, outperforming the state-of-the-art TBG model, and competitive with the performance of MDGen with no protein-specific parametrization. This is an application paper which applys Walk-Jump samplin...

**レビュー 2** (評価: 5, 確信度: 3)

**要約:**
The paper presents JAMUN, a walk-jump sampler that couples smoothed Langevin dynamics on a latent space with an denoiser. On short-peptide benchmarks the method delivers $10^{1}$–$10^{2}$ accelerations while preserving ensemble fidelity, as verified by Jensen–Shannon distances and MSM state populati...

**レビュー 3** (評価: 5, 確信度: 4)

**要約:**
This paper addresses issues encountered when sampling the Boltzmann distribution using machine learning models, specifically limited sampling speed and transferability. Instead of performing Langevin dynamics directly in molecular conformation space, the authors propose performing Langevin dynamics ...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=8Z3KnaYtw9)
- [PDF](https://openreview.net/pdf?id=8Z3KnaYtw9)

---

### 3. Flatten Graphs as Sequences: Transformers are Scalable Graph Generators

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.808** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.50/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Dexiong Chen, Markus Krimmel, Karsten Borgwardt

**キーワード**: graph generation, transformers, autoregressive modeling, language models, LLMs

#### 概要

We introduce AutoGraph, a scalable autoregressive model for attributed graph generation using decoder-only transformers. By flattening graphs into random sequences of tokens through a reversible process, AutoGraph enables modeling graphs as sequences without relying on additional node features that are expensive to compute, in contrast to diffusion-based approaches. This results in sampling complexity and sequence lengths that scale optimally linearly with the number of edges, making it scalable and efficient for large, sparse graphs. A key success factor of AutoGraph is that its sequence prefixes represent induced subgraphs, creating a direct link to sub-sentences in language modeling. Empirically, AutoGraph achieves state-of-the-art performance on synthetic and molecular benchmarks, with up to 100x faster generation and 3x faster training than leading diffusion models. It also supports substructure-conditioned generation without fine-tuning and shows promising transferability, bridging language modeling and graph generation to lay the groundwork for graph foundation models. Our code is available at https://github.com/BorgwardtLab/AutoGraph.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 4)

**要約:**
This paper presents AUTOGRAPH, an autoregressive framework for generating graphs using decoder-only transformers. The core contribution lies in a reversible flattening procedure that transforms graphs into sequences, enabling the use of language modeling. The authors claim that this approach is scal...

**レビュー 2** (評価: 5, 確信度: 3)

**要約:**
This paper is focused on the topic of graph generation using Transformers and uses an existing principle of recent interest that transforms graphs to sequences. It presents AUTOGRAPH, an autoregressive graph generation framework that converts graphs into token sequences to enable direct use of decod...

**レビュー 3** (評価: 5, 確信度: 3)

**要約:**
This paper proposes a graph generation method called AutoGraph, which serializes graph structures by treating sampled graph trials as tokens, enabling the use of decoder-only Transformer models for graph generation tasks. The authors introduce a newly designed structure named SENT (Segmented Euleria...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=eszmES7j1F)
- [PDF](https://openreview.net/pdf?id=eszmES7j1F)

---

### 4. Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.802** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.700 |
| 　├ インパクト | 0.750 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Lianghong Chen, Dongkyu Eugene Kim, Mike Domaratzki, Pingzhao Hu

**キーワード**: 3D De Novo Molecular Design, Diffusion Model, Multi-Objective Reinforcement Learning, Uncertainty Quantification, Deep Learning

#### 概要

Designing de novo 3D molecules with desirable properties remains a fundamental challenge in drug discovery and molecular engineering. While diffusion models have demonstrated remarkable capabilities in generating high-quality 3D molecular structures, they often struggle to effectively control complex multi-objective constraints critical for real-world applications. In this study, we propose an uncertainty-aware Reinforcement Learning (RL) framework to guide the optimization of 3D molecular diffusion models toward multiple property objectives while enhancing the overall quality of the generated molecules. Our method leverages surrogate models with predictive uncertainty estimation to dynamically shape reward functions, facilitating balance across multiple optimization objectives. We comprehensively evaluate our framework across three benchmark datasets and multiple diffusion model architectures, consistently outperforming baselines for molecular quality and property optimization. Additionally, Molecular Dynamics (MD) simulations and ADMET profiling of top generated candidates indicate promising drug-like behavior and binding stability, comparable to known Epidermal Growth Factor Receptor (EGFR) inhibitors. Our results demonstrate the strong potential of RL-guided generative diffusion models for advancing automated molecular design.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 4)

**要約:**
The paper presents a methodologically sound pipeline, which systematically integrates three key stages: reward design, data sampling, and optimization using Proximal Policy Optimization (PPO). The authors conduct their analysis within a rigorous experimental setting, ensuring a fair and direct compa...

**レビュー 2** (評価: 5, 確信度: 3)

**要約:**
In this work RL is incorporated in the training of guided-diffusion models to facilitate generation of 3D molecules satisfying 
multiple properties, waiving the need to have differentiable surrogate models of the properties of interest. A conditional diffusion model is first trained on the molecular...

**レビュー 3** (評価: 3, 確信度: 4)

**要約:**
The paper proposes an uncertainty-aware RL framework to post-train a molecular diffusion model towards multiple objectives. A surrogate model predicts uncertainty estimation, which is used to reweight reward functions to keep objectives balanced. Experiments on three molecule generation benchmarks a...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=lu4cGylISh)
- [PDF](https://openreview.net/pdf?id=lu4cGylISh)

---

### 5. E2Former: An Efficient and Equivariant Transformer with Linear-Scaling Tensor Products

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 5.25/10 |

**採択判定**: Accept (spotlight)
  - ✨ **発表形式**: Spotlight Presentation

**著者**: Yunyang Li, Lin Huang, Zhihao Ding, Xinran Wei, Chu Wang 他8名

**キーワード**: Equivariant Neural network, quantum chemistry

#### 概要

Equivariant Graph Neural Networks (EGNNs) have demonstrated significant success in modeling microscale systems, including those in chemistry, biology and materials science. However, EGNNs face substantial computational challenges due to the high cost of constructing edge features via spherical tensor products, making them almost impractical for large-scale systems. 
To address this limitation, we introduce E2Former, an equivariant and efficient transformer architecture that incorporates a Wigner $6j$ convolution (Wigner $6j$ Conv). By shifting the computational burden from edges to nodes, Wigner $6j$ Conv reduces the complexity from $O(| \mathcal{E}|)$ to  $O(| \mathcal{V}|)$ while preserving both the model's expressive power and rotational equivariance.
We show that this approach achieves a 7x–30x speedup compared to conventional $\mathrm{SO}(3)$ convolutions. Furthermore, our empirical results demonstrate that the derived E2Former mitigates the computational challenges of existing approaches without compromising the ability to capture detailed geometric information. This development could suggest a promising direction for scalable molecular modeling.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 2)

**要約:**
This paper introduces a novel graph neural network architecture designed to address computational inefficiencies in Equivariant Graph Neural Networks (EGNNs), particularly for applications in molecular modeling. The key innovation is the introduction the concept of Wigner 6j coupling from physics, w...

**レビュー 2** (評価: 5, 確信度: 3)

**要約:**
The paper introduces the E2Former, a new spherically equivariant architecture for molecular modeling. At its core the model uses the novel Wigner 6j convolution, which is mathematically equivalent to conventional SO(3) convolutions, but scales with the number of nodes instead of the number of edges,...

**レビュー 3** (評価: 6, 確信度: 4)

**要約:**
This paper introduces E2Former, an SO(3)-equivariant transformer architecture designed for molecular modelling. The primary contribution is a new method which the authors call "Wigner 6j convolution", a reformulation of the computationally expensive tensor products that are central to equivariant GN...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=ls5L4IMEwt)
- [PDF](https://openreview.net/pdf?id=ls5L4IMEwt)

---

### 6. Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Jingyuan Zhou, Hao Qian, Shikui Tu, Lei Xu

**キーワード**: Structure-based drug design, Molecule Generation, Generative models, Flow Matching, Computational Biology

#### 概要

Structure-based drug design (SBDD), aiming to generate 3D molecules with high binding affinity toward target proteins, is a vital approach in novel drug discovery. Although recent generative models have shown great potential, they suffer from unstable probability dynamics and mismatch between generated molecule size and the protein pockets geometry, resulting in inconsistent quality and off-target effects. We propose PAFlow, a novel target-aware molecular generation model featuring prior interaction guidance and a learnable atom number predictor. PAFlow adopts the efficient flow matching framework to model the generation process and constructs a new form of conditional flow matching for discrete atom types. A protein–ligand interaction predictor is incorporated to guide the vector field toward higher-affinity regions during generation, while an atom number predictor based on protein pocket information is designed to better align generated molecule size with target geometry. Extensive experiments on the CrossDocked2020 benchmark show that PAFlow achieves a new state-of-the-art in binding affinity (up to -8.31 Avg. Vina Score), simultaneously maintains favorable molecular properties.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 4)

**要約:**
This paper proposes PAFlow, a novel structure-based drug design (SBDD) model built on the flow matching (FM) framework. The method separately models continuous 3D atomic coordinates and discrete atom types using tailored probability paths and incorporates two key components: a protein-ligand interac...

**レビュー 2** (評価: 3, 確信度: 5)

**要約:**
This paper introduces PAFlow, a novel target-aware generative model for structure-based drug design. PAFlow integrates prior protein-ligand interaction knowledge and a learnable atom number predictor into a conditional flow matching framework to generate 3D molecules with improved binding affinity. ...

**レビュー 3** (評価: 5, 確信度: 3)

**要約:**
The authors aim to solve the problem of structure-based drug design. Upon this, the authors noticed limitations of existing generative methods, such as unstable probability dynamics and molecular size mismatches. To address such limitations, the authors introduce PAFlow, a novel target-aware molecul...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=yh1t1yFtXG)
- [PDF](https://openreview.net/pdf?id=yh1t1yFtXG)

---

### 7. Accelerating 3D Molecule Generative Models with Trajectory Diagnosis

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 5.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Zhilong Zhang, Yuxuan Song, Yichun Wang, Jingjing Gong, Hanlin Wu 他3名

**キーワード**: 3D Molecule Generation, Fast Generation, Drug Design

#### 概要

Geometric molecule generative models have found expanding applications across various scientific domains, but their generation inefficiency has become a critical bottleneck. Through a systematic investigation of the generative trajectory, we discover a unique challenge for molecule geometric graph generation: generative models require determining the permutation order of atoms in the molecule before refining its atomic feature values. Based on this insight, we decompose the generation process into permutation phase and adjustment phase, and propose a geometric-informed prior and consistency parameter objective to accelerate each phase. Extensive experiments demonstrate that our approach achieves competitive performance with approximately 10 sampling steps, 7.5 × faster than previous state-of-the-art models and approximately 100 × faster than diffusion-based models, offering a significant step towards scalable molecular generation.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 5)

**要約:**
In this paper, the authors introduce MOLTD, a novel approach to accelerating 3D molecule generative models by addressing geometric generation challenges. Through theoretical and empirical analysis, they identify a two-phase generative pattern—permutation reordering and atomic feature adjustment—and ...

**レビュー 2** (評価: 6, 確信度: 5)

**要約:**
Proposes a 3D molecule generation model that can generate molecules in a less number of sampling steps compared to existing flow-matching and diffusion-based models. The generation process split into permutation phase and adjustment phase. A geometric-informed prior is introduced to reduce the numbe...

**レビュー 3** (評価: 5, 確信度: 3)

**要約:**
This paper addresses the problem of 3D molecular generation. The authors empirically demonstrate that diffusion- and flow-based generative models tend to establish atom permutations early in the sampling trajectories. To analyze this phenomenon, they decompose the generation process into two distinc...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=ATewcZPbDj)
- [PDF](https://openreview.net/pdf?id=ATewcZPbDj)

---

### 8. Topology-aware Graph Diffusion Model with Persistent Homology

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.50/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Joonhyuk Park, Donghyun Lee, Yujee Song, Guorong Wu, Won Hwa Kim

**キーワード**: Graph Generation, Diffusion, Topology, Brain Network

#### 概要

Generating realistic graphs faces challenges in estimating accurate distribution of graphs in an embedding space while preserving structural characteristics. However, existing graph generation methods primarily focus on approximating the joint distribution of nodes and edges, often overlooking topological properties such as connected components and loops, hindering accurate representation of global structures. To address this issue, we propose a Topology-Aware diffusion-based Graph Generation (TAGG), which aims to sample synthetic graphs that closely resemble the structural characteristics of the original graph based on persistent homology. Specifically, we suggest two core components: 1) Persistence Diagram Matching (PDM) loss which ensures high topological fidelity of generated graphs, and 2) topology-aware attention module (TAM) which induces the denoising network to capture the homological characteristics of the original graphs. Extensive experiments on conventional graph benchmarks demonstrate the effectiveness of our approach demonstrating high generation performance across various metrics, while achieving closer alignment with the distribution of topological features observed in the original graphs. Furthermore, application to real brain network data showcases its potential for complex and real graph applications.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 3)

**要約:**
The authors introduce a graph generation method based on discrete diffusion, defined as topology-aware because it also preserves the structural features of the original graphs. The approach uses a loss function based on Persistence Diagram Matching (PDM) and integrates a Topology-aware Attention Mod...

**レビュー 2** (評価: 5, 確信度: 4)

**要約:**
Classical methods for operations on graphs, such as random graph generation or graph denoising, sometimes scramble important graph features that we need. One of these features is the topology of the graph. The paper does a good job addressing this problem.

**レビュー 3** (評価: 4, 確信度: 4)

**要約:**
The paper presents TAGG, a topology-aware graph generation framework that integrates persistent homology into diffusion models to better account for topological/structural information of undirected graphs. In particular, the authors introduce a novel loss function ( persistence diagram matching PDM)...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=sye27MizdM)
- [PDF](https://openreview.net/pdf?id=sye27MizdM)

---

### 9. Implicit Generative Property Enhancer

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 3.75/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Pedro O. Pinheiro, Pan Kessel, Aya Abdelsalam Ismail, Sai Pooja Mahajan, Kyunghyun Cho 他2名

**キーワード**: generative modeling, drug discovery, design optimization

#### 概要

Generative modeling is increasingly important for data-driven computational design. Conventional approaches pair a generative model with a discriminative model to select or guide samples toward optimized designs. Yet discriminative models often struggle in data-scarce settings, common in scientific applications, and are unreliable in the tails of the distribution where optimal designs typically lie. We introduce generative property enhancer (GPE), an approach that implicitly guides generation by matching samples with lower property values to higher-value ones. Formulated as conditional density estimation, our framework defines a target distribution with improved properties, compelling the generative model to produce enhanced, diverse designs without auxiliary predictors. GPE is simple, scalable, end-to-end, modality-agnostic, and integrates seamlessly with diverse generative model architectures and losses. We demonstrate competitive empirical results on standard _in silico_ offline (non-sequential) protein fitness optimization benchmarks. Finally, we propose iterative training on a combination of limited real data and self-generated synthetic data, enabling extrapolation beyond the original property ranges.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 3)

**要約:**
This work proposes training conditional generative models on pairs of samples (with one having higher property than the other) in iterative rounds to perform optimization in the design space. The key feature of the proposed method is that it does not rely on surrogate estimates of the property which...

**レビュー 2** (評価: 3, 確信度: 4)

**要約:**
This paper considers the problem of data-driven design optimization, aiming to generate new designs with improved properties over existing ones. The paper presents a method titled Generative Property Enhancer (GPE) where the core idea is to reframe the problem as conditional density estimation. This...

**レビュー 3** (評価: 4, 確信度: 5)

**要約:**
This paper introduces a discriminator-free framework for learning conditional generative models for offline (one-round) optimization, primarily of sequences (though it can be applied to continuous domains). Specifically, it is a framework for learning generative models that “mutate” or manipulate ex...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=PSvsmbCrGs)
- [PDF](https://openreview.net/pdf?id=PSvsmbCrGs)

---

### 10. A Unified Framework for Fair Graph Generation: Theoretical Guarantees and Empirical Advances

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.50/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Zichong Wang, Zhipeng Yin, Wenbin Zhang

**キーワード**: Fairness, Graph Generation, GNN

#### 概要

Graph generation models play pivotal roles in many real-world applications, from data augmentation to privacy-preserving. Despite their deployment successes, existing approaches often exhibit fairness issues, limiting their adoption in high-risk decision-making applications. Most existing fair graph generation works are based on autoregressive models that suffer from ordering sensitivity, while primarily addressing structural bias and overlooking the critical issue of feature bias. To this end, we propose FairGEM, a novel one-shot graph generation framework designed to mitigate both graph structural bias and node feature bias simultaneously. Furthermore, our theoretical analysis establishes that FairGEM delivers substantially stronger fairness guarantees than existing models while preserving generation quality. Extensive experiments across multiple real-world datasets demonstrate that FairGEM achieves superior performance in both generation quality and fairness.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 3)

**要約:**
This paper proposes a one-shot spectral diffusion framework with two fairness regularizers, (1) Structural regularizer that minimizes intra- vs inter-group edge-reconstruction disparity and (2) Feature regularizer that first disentangles sensitive-related vs sensitive-irrelevant attributes (via a VA...

**レビュー 2** (評価: 4, 確信度: 3)

**要約:**
This paper presents FairGEM, a framework for generating synthetic graphs that aim to be both realistic and fair with respect to sensitive attributes. It introduces fairness-aware regularizers on both graph structure and node features, combines them with a disentanglement-based encoder using a VAE an...

**レビュー 3** (評価: 4, 確信度: 4)

**要約:**
In this paper, the authors propose a novel graph generation framework named FairGEM, which can mitigate both graph structural bias and node feature bias simultaneously. Both theoretical analysis and empirical experiments verify the effectiveness of FairGEM.

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=T85ADT8a2y)
- [PDF](https://openreview.net/pdf?id=T85ADT8a2y)

---

### 11. Reinforced Active Learning for Large-Scale Virtual Screening with Learnable Policy Model

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Yicong Chen, Jiahua Rao, Jiancong Xie, Dahao Xu, Zhen WANG 他1名

**キーワード**: Virtual Screening, Active Learning, Molecular Activity, Drug Discovery

#### 概要

Virtual Screening (VS) is vital for drug discovery but struggles with low hit rates and high computational costs. While Active Learning (AL) has shown promise in improving the efficiency of VS, traditional methods rely on inflexible and handcrafted heuristics, limiting adaptability in complex chemical spaces, particularly in balancing molecular diversity and selection accuracy. 
To overcome these challenges, we propose GLARE, a reinforced active learning framework that reformulates VS as a Markov Decision Process (MDP). Using Group Relative Policy Optimization (GRPO), GLARE dynamically balances chemical diversity, biological relevance, and computational constraints, eliminating the need for inflexible heuristics.
Experiments show GLARE outperforms state-of-the-art AL methods, with a 64.8% average improvement in Enrichment Factors (EF). Additionally, GLARE enhances the performance of VS foundation models like DrugCLIP, achieving up to an 8-fold improvement in  EF$_{0.5\\%}$ with as few as 15 active molecules. These results highlight the transformative potential of GLARE for adaptive and efficient drug discovery.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 3)

**要約:**
This paper introduces the GLARE framework (a GRPO-based Learning framework for Active REinforced screening) for virtual screening in drug discovery. It casts virtual screening as a Markov decision process and integrates it into a policy-based GRPO reinforcement learning framework. GLARE achieves com...

**レビュー 2** (評価: 5, 確信度: 4)

**要約:**
This work proposes GLARE, a reinforcement learning-based active learning (AL) framework designed for large-scale virtual screening under limited annotation budgets. GLARE reformulates molecular selection as a Markov Decision Process (MDP) and replaces traditional hand-crafted acquisition strategies ...

**レビュー 3** (評価: 4, 確信度: 2)

**要約:**
This paper presents GLARE, a novel active learning framework for virtual screening. The authors reformulate the selection of molecules as a Markov Decision Process (MDP) and apply Group Relative Policy Optimization (GRPO) to train a learnable policy model for molecule selection. The proposed framewo...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=49EjZytlus)
- [PDF](https://openreview.net/pdf?id=49EjZytlus)

---

### 12. Sampling 3D Molecular Conformers with Diffusion Transformers

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Thorben Frank, Winfried Ripken, Gregor Lied, Klaus Robert Muller, Oliver T. Unke 他1名

**キーワード**: 3D Molecular Graphs, Molecular Modeling, Generative Modeling, Diffusion Transformers, Geometric Deep Learning, Flow Matching

#### 概要

Diffusion Transformers (DiTs) have demonstrated strong performance in generative modeling, particularly in image synthesis, making them a compelling choice for molecular conformer generation. However, applying DiTs to molecules introduces novel challenges, such as integrating discrete molecular graph information with continuous 3D geometry, handling Euclidean symmetries, and designing conditioning mechanisms that generalize across molecules of varying sizes and structures. We propose DiTMC, a framework that adapts DiTs to address these challenges through a modular architecture that separates the processing of 3D coordinates from conditioning on atomic connectivity. To this end, we introduce two complementary graph-based conditioning strategies that integrate seamlessly with the DiT architecture. These are combined with different attention mechanisms, including both standard non-equivariant and SO(3)-equivariant formulations, enabling flexible control over the trade-off between between accuracy and computational efficiency.
Experiments on standard conformer generation benchmarks (GEOM-QM9, -DRUGS, -XL) demonstrate that DiTMC achieves state-of-the-art precision and physical validity. Our results highlight how architectural choices and symmetry priors affect sample quality and efficiency, suggesting promising directions for large-scale generative modeling of molecular structures. Code is available at [https://github.com/ML4MolSim/dit_mc](https://github.com/ML4MolSim/dit_mc).

#### 📊 レビュー詳細

**レビュー 1** (評価: 3, 確信度: 4)

**要約:**
This paper aims to adapt the diffusion transformer, which has shown promise in general domains, to the task of molecular conformer prediction. To this end, several modifications are proposed within a new framework called DiTMC, including two types of molecular graph conditioning strategies. Moreover...

**レビュー 2** (評価: 5, 確信度: 4)

**要約:**
This paper proposes a diffusion transformer (DiT) learning system for generating ensembles of 3D conformers for small molecules. Algorithmic components are introduced to embed a full suite of molecular features within transformer architectures, including atomic and pairwise graph embeddings. Additio...

**レビュー 3** (評価: 4, 確信度: 3)

**要約:**
The authors extend the DiT architecture to irregular molecular geometries. Key moves are (i) two graph-based conditioning schemes (bond-pair vs. all-pair geodesic tokens) and (ii) interchangeable self-attention/positional-embedding blocks that range from standard (non-equivariant) to full SO(3)-equi...

*他 2 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=tmbx9zGVWb)
- [PDF](https://openreview.net/pdf?id=tmbx9zGVWb)

---

### 13. Template-Guided 3D Molecular Pose Generation via Flow Matching and Differentiable Optimization

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Noémie Bergues, Arthur Carré, Paul Join-Lambert, Brice Hoffmann, Arnaud Blondel 他1名

**キーワード**: flow-matching, geometric deep-learning, graph neural networks, differentiable optimization, template-guided docking, computational chemistry, drug-design, 3D pose generation

#### 概要

Predicting the 3D conformation of small molecules within protein binding sites is a key challenge in drug design. When a crystallized reference ligand (template) is available, it provides geometric priors that can guide 3D pose prediction. We present a two-stage method for ligand conformation generation guided by such templates. In the first stage, we introduce a molecular alignment approach based on flow-matching to generate 3D coordinates for the ligand, using the template structure as a reference. In the second stage, a differentiable pose optimization procedure refines this conformation based on shape and pharmacophore similarities, internal energy, and, optionally, the protein binding pocket. We introduce a new benchmark of ligand pairs co-crystallized with the same target to evaluate our approach and show that it outperforms standard docking tools and open-access alignment methods, especially in cases involving low similarity to the template or high ligand flexibility.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 3)

**要約:**
This paper proposes FMA-PO, a two-step method for ligand 3D pose prediction using flow matching for initial alignment and differentiable optimization for pose refinement. It introduces a new benchmark (AlignDockBench) and shows competitive results compared to docking and alignment baselines.

**レビュー 2** (評価: 4, 確信度: 4)

**要約:**
The paper introduces a two-stage framework called FMA-PO, which combines a flow-matching-based model for template-guided ligand pose generation with a differentiable pose refinement module. The method aligns a 2D query molecule to a 3D reference ligand using flow matching, and then refines it based ...

**レビュー 3** (評価: 5, 確信度: 5)

**要約:**
This paper introduces a FM-based method for template-based ligand docking. First, authors formulate the ligand conformer generation task conditioned on the template 3D structure with flow matching. Next, authors propose the pose optimisation (PO) algorithm that refines the generated pose using shape...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=cIYguQc97T)
- [PDF](https://openreview.net/pdf?id=cIYguQc97T)

---

### 14. GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Chunyu Wei, Wenji Hu, Xingjia Hao, Xin Wang, Yifan Yang 他3名

**キーワード**: Large Language Model, Graph Analysis, Tool Learning

#### 概要

Large Language Models (LLMs) face significant limitations when applied to large-scale graphs, struggling with context constraints and inflexible reasoning. We introduce GraphChain, a novel framework enabling LLMs to analyze large graphs by orchestrating dynamic sequences of specialized tools, mimicking human exploratory processes. GraphChain incorporates two core technical contributions: (1) Progressive Graph Distillation, a reinforcement learning approach that learns to generate tool sequences balancing task relevance and intermediate state compression, thereby overcoming LLM context limitations. (2) Structure-aware Test-Time Adaptation (STTA), a mechanism using a lightweight, self-supervised adapter conditioned on graph spectral properties to efficiently adapt a frozen LLM policy to diverse graph structures via soft prompts without retraining. Experiments show GraphChain significantly outperforms prior methods, enabling scalable and adaptive LLM-driven graph analysis.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 4)

**要約:**
This paper introduces GraphChain, which is a method to automatically call graph based API solutions to serve as the intermediate prompt to assist LLM in graph reasoning tasks. To enable the scalability of method to large scale graph, a progressive graph distillation module is proposed to compress gr...

**レビュー 2** (評価: 4, 確信度: 4)

**要約:**
This paper introduces a framework for large-scale graph analysis using LLMs through dynamic tool chaining. The core idea is to decompose complex graph reasoning tasks into sequential tool-based operations. The method includes two main designs: (1) Progressive Graph Distillation, a reinforcement lear...

**レビュー 3** (評価: 4, 確信度: 4)

**要約:**
This paper introduces a framework called GraphChain to enhance the capabilities of Large Language Models (LLMs) in processing large-scale graph data. In this paper, the authors propose two key approaches to allow for a more detailed and adaptive exploration of graph structures: Progressive Graph Dis...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=Rdz6ESQYkK)
- [PDF](https://openreview.net/pdf?id=Rdz6ESQYkK)

---

### 15. Low-Rank Graphon Learning for Networks

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Xinyuan Fan, Feiyan Ma, Chenlei Leng, Weichi Wu

**キーワード**: low-rank graphon, subgraph counts, connection probability matrix, nonparametric statistics, network analysis

#### 概要

Graphons offer a powerful framework for modeling large-scale networks, yet estimation remains challenging. We propose a novel approach that leverages a low-rank additive representation, yielding both a low-rank connection probability matrix and a low-rank graphon--two goals rarely achieved jointly. Our method resolves identification issues and enables an efficient sequential algorithm based on subgraph counts and interpolation. We establish consistency and demonstrate strong empirical performance in terms of computational efficiency and estimation accuracy through simulations and data analysis.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 5)

**要約:**
The author(s) propose a low-rank graphon modelling for networks. The estimation approach reconcile the estimation of the graphon function and the corresponding connection probability matrix. The author(s) propose a sequential efficient algorithm baed on subgraph counts and interpolation. They also p...

**レビュー 2** (評価: 4, 確信度: 3)

**要約:**
The authors proposed a methodology that estimates both the connection probability matrix and its underlying graphon at low computational cost. The approach leveraged the rank-r eigen-decomposition of a graphon into r eigenvalues and eigenfunctions. These quantities were recovered by counting cycles ...

**レビュー 3** (評価: 4, 確信度: 3)

**要約:**
This paper offers a framework for estimating the connectivity matrix and the underlying function of a low-rank graphon through subgraph counts (AKA motifs). The authors' main theorem is an error guarantee in the sup-norm for both the connectivity matrix $P$ (Theorem 3.6) and the graphon function $f$...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=1n5TJh3LEb)
- [PDF](https://openreview.net/pdf?id=1n5TJh3LEb)

---

### 16. Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.801** |
| 　├ 関連性 | 0.900 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 3.75/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Aditya Sengar, Ali Hariri, Daniel Probst, PATRICK BARTH, Pierre Vandergheynst

**キーワード**: Latent Diffusion, Graph Neural Networks, Protein Structure Generation, All-atom modeling, Molecular Dynamics

#### 概要

Generating diverse, all‐atom conformational ensembles of dynamic proteins such as G‐protein‐coupled receptors (GPCRs) is critical for understanding their function, yet most generative models simplify atomic detail or ignore conformational diversity altogether. We present latent diffusion for full protein generation (LD-FPG), a framework that constructs complete all‐atom protein structures, including every side‐chain heavy atom, directly from molecular dynamics (MD) trajectories. LD-FPG employs a Chebyshev graph neural network (ChebNet) to obtain low‐dimensional latent embeddings of protein conformations, which are processed using three pooling strategies: blind, sequential and residue‐based. A diffusion model trained on these latent representations generates new samples that a decoder, optionally regularized by dihedral‐angle losses, maps back to Cartesian coordinates. Using D2R-MD, a $2\mu\text{s}$ MD trajectory (12 000 frames) of the human dopamine D$2$ receptor in a membrane environment, the sequential and residue-based pooling strategies reproduce the reference ensemble with high structural fidelity (all‐atom lDDT \~ $0.7$; $C\alpha$-lDDT \~ $0.8$) and recovers backbone and side‐chain dihedral‐angle distributions with a Jensen–Shannon divergence $<0.03$ compared to the MD data. LD-FPG thereby offers a practical route to system‐specific, all‐atom ensemble generation for large proteins, providing a promising tool for structure‐based therapeutic design on complex, dynamic targets. The D2R-MD dataset and our implementation are freely available to facilitate further research.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 3)

**要約:**
LD-FPG (Latent Diffusion for Full Protein Generation) generates diverse, all-atom protein conformational ensembles (e.g., for GPCRs). It uses a ChebGNN to learn atom-wise latent embeddings, pools them (blind/sequential/residue strategies), samples new conformations via DDPM, and decodes them to coor...

**レビュー 2** (評価: 4, 確信度: 3)

**要約:**
This work proposes a generative model for generating all-atom protein conformations. The model is based on a latent diffusion framework with a ChebNet encoder and is trained on molecular dynamics (MD) simulation data. Experimental results demonstrate the effectiveness of the proposed approach and pr...

**レビュー 3** (評価: 3, 確信度: 4)

**要約:**
This paper introduces "Latent Diffusion for Full-Atom Protein Generation" (LD-FPG), a novel framework designed to generate all-atom conformational ensembles for a specific protein, leveraging data from molecular dynamics (MD) trajectories. The core methodology employs a three-stage architecture: (i)...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=JPjMXgQQxk)
- [PDF](https://openreview.net/pdf?id=JPjMXgQQxk)

---

### 17. BioCG: Constrained Generative Modeling for Biochemical Interaction Prediction

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Amitay Sicherman, Kira Radinsky

**キーワード**: Constrained Generative Modeling, Drug-Target Interaction (DTI), Drug Discovery, Enzyme-Catalyzed Reactions

#### 概要

Predicting interactions between biochemical entities is a core challenge in drug discovery and systems biology, often hindered by limited data and poor generalization to unseen entities. Traditional discriminative models frequently underperform in such settings. We propose BioCG (Biochemical Constrained Generation), a novel framework that reformulates interaction prediction as a constrained sequence generation task. BioCG encodes target entities as unique discrete sequences via Iterative Residual Vector Quantization (I-RVQ) and trains a generative model to produce the sequence of an interacting partner given a query entity. A trie-guided constrained decoding mechanism, built from a catalog of valid target sequences, concentrates the model's learning on the critical distinctions between valid biochemical options, ensuring all outputs correspond to an entity within the pre-defined target catalog. An information-weighted training objective further focuses learning on the most critical decision points. BioCG achieves state-of-the-art (SOTA) performance across diverse tasks, Drug-Target Interaction (DTI), Drug-Drug Interaction (DDI), and Enzyme-Reaction Prediction, especially in data-scarce and cold-start conditions. On the BioSNAP DTI benchmark, for example, BioCG attains an AUC of 89.31\% on unseen proteins, representing a 14.3 percentage point gain over prior SOTA. By directly generating interacting partners from a known biochemical space, BioCG provides a robust and data-efficient solution for in-silico biochemical discovery.

#### 📊 レビュー詳細

**レビュー 1** (評価: 4, 確信度: 4)

**要約:**
The paper presents **BioCG**, a novel framework for biochemical interaction prediction that reframes the problem as a constrained sequence generation task. Unlike traditional discriminative models, BioCG uses Iterative Residual Vector Quantization (I-RVQ) to encode target biochemical entities as uni...

**レビュー 2** (評価: 5, 確信度: 3)

**要約:**
This paper proposes BioCG, a new approach for biochemical interaction prediction (e.g., drug-target, drug-drug, enzyme-reaction) by treating it as a constrained generation task over a fixed set of known targets. It uses an iterative vector quantization (I-RVQ) to encode discrete target codes, and a ...

**レビュー 3** (評価: 4, 確信度: 4)

**要約:**
The paper rethinks the approach to predicting whether two biomolecules interact. Previous methods mostly focus on contrastive learning, while BioCG, the method proposed here, reframes the problem as generating a target molecule conditioned on the query molecule, with the generative process designed ...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=SurremoXPu)
- [PDF](https://openreview.net/pdf?id=SurremoXPu)

---

### 18. Association-Focused Path Aggregation for Graph Fraud Detection

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.781** |
| 　├ 関連性 | 0.850 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.25/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Tian Qiu, Wenda Li, Zunlei Feng, Jie Lei, Tao Wang 他3名

**キーワード**: deep learning, path aggregation, graph fraud detection

#### 概要

Fraudulent activities have caused substantial negative social impacts and are exhibiting emerging characteristics such as intelligence and industrialization, posing challenges of high-order interactions, intricate dependencies, and the sparse yet concealed nature of fraudulent entities. Existing graph fraud detectors are limited by their narrow "receptive fields", as they focus only on the relations between an entity and its neighbors while neglecting longer-range structural associations hidden between entities. To address this issue, we propose a novel fraud detector based on Graph Path Aggregation (GPA). It operates through variable-length path sampling, semantic-associated path encoding, path interaction and aggregation, and aggregation-enhanced fraud detection. To further facilitate interpretable association analysis, we synthesize G-Internet, the first benchmark dataset in the field of internet fraud detection. Extensive experiments across datasets in multiple fraud scenarios demonstrate that the proposed GPA outperforms mainstream fraud detectors by up to +15% in Average Precision (AP). Additionally, GPA exhibits enhanced robustness to noisy labels and provides excellent interpretability by uncovering implicit fraudulent patterns across broader contexts. Code is available at https://github.com/horrible-dong/GPA.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 3)

**要約:**
This paper proposes a novel method called Graph Path Aggregation (GPA) to overcome the “receptive field” limitations commonly observed in existing GNN-based fraud detection models. GPA samples variable-length paths from each user node, encodes these paths by incorporating behavioral features, and le...

**レビュー 2** (評価: 4, 確信度: 5)

**要約:**
The paper presents a graph-based path sampling and aggregation algorithm to detect fraudulent entities online. The authors argue that path sampling is necessary to capture long-range structural dependencies between entities and unveil fraud patterns. A new dataset is also proposed to simulate the on...

**レビュー 3** (評価: 4, 確信度: 3)

**要約:**
This paper introduces a novel fraud detection method called Graph Path Aggregation (GPA), which addresses the limitations of existing graph-based fraud detectors by capturing long-range structural associations between entities. The authors propose a framework that includes variable-length path sampl...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=TiE8aTc3Zg)
- [PDF](https://openreview.net/pdf?id=TiE8aTc3Zg)

---

### 19. AANet: Virtual Screening under Structural Uncertainty via Alignment and Aggregation

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.761** |
| 　├ 関連性 | 0.800 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 4.75/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Wenyu Zhu, Jianhui Wang, Bowen Gao, Yinjun Jia, Haichuan Tan 他3名

**キーワード**: Virtual screening, Drug discovery

#### 概要

Virtual screening (VS) is a critical component of modern drug discovery, yet most existing methods—whether physics-based or deep learning-based—are developed around *holo* protein structures with known ligand-bound pockets. Consequently, their performance degrades significantly on *apo* or predicted structures such as those from AlphaFold2, which are more representative of real-world early-stage drug discovery, where pocket information is often missing. In this paper, we introduce an alignment-and-aggregation framework to enable accurate virtual screening under structural uncertainty. Our method comprises two core components: (1) a tri-modal contrastive learning module that aligns representations of the ligand, the *holo* pocket, and cavities detected from structures, thereby enhancing robustness to pocket localization error; and (2) a cross-attention based adapter for dynamically aggregating candidate binding sites, enabling the model to learn from activity data even without precise pocket annotations. We evaluated our method on a newly curated benchmark of *apo* structures, where it significantly outperforms state-of-the-art methods in blind apo setting, improving the early enrichment factor (EF1\%) from 11.75 to 37.19. Notably, it also maintains strong performance on *holo* structures. These results demonstrate the promise of our approach in advancing first-in-class drug discovery, particularly in scenarios lacking experimentally resolved protein-ligand complexes. Our implementation is publicly available at [https://github.com/Wiley-Z/AANet](https://github.com/Wiley-Z/AANet).

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 3)

**要約:**
This paper tackles the challenge of virtual screening on apo or predicted protein structures, where binding pocket locations are unknown. The authors propose AANet, a two-phase framework to address this structural uncertainty. First, a tri-modal contrastive learning scheme aligns representations of ...

**レビュー 2** (評価: 4, 確信度: 2)

**要約:**
This paper introduces AANet, a novel virtual screening framework designed for structure-based drug discovery under structural uncertainty, particularly when working with apo or predicted protein structures lacking experimentally resolved binding pockets. AANet combines two key components: (1) a tri-...

**レビュー 3** (評価: 5, 確信度: 4)

**要約:**
The paper presents AANet, a contrastive learning-based virtual screening framework designed to handle structural uncertainty in protein structures, particularly apo or AlphaFold2-predicted conformations where pocket annotations are absent or unreliable. AANet consists of two main components: (1) tri...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=TUh4GDposM)
- [PDF](https://openreview.net/pdf?id=TUh4GDposM)

---

### 20. Simultaneous Modeling of Protein Conformation and Dynamics via Autoregression

#### スコア

| 項目 | スコア |
|------|--------|
| **総合スコア** | **0.761** |
| 　├ 関連性 | 0.800 |
| 　├ 新規性 | 0.750 |
| 　├ インパクト | 0.700 |
| OpenReview評価 | 5.00/10 |

**採択判定**: Accept (poster)
  - 📊 **発表形式**: Poster Presentation

**著者**: Yuning Shen, Lihao Wang, Huizhuo Yuan, Yan Wang, Bangji Yang 他1名

**キーワード**: Protein Conformation, Generative Modeling, Molecular Dynamics, Diffusion Models, Language Models

#### 概要

Understanding protein dynamics is critical for elucidating their biological functions. 
The increasing availability of molecular dynamics (MD) data enables the training of deep generative models to efficiently explore the conformational space of proteins.
However, existing approaches either fail to explicitly capture the temporal dependencies between conformations or do not support direct generation of time-independent samples.
To address these limitations, we introduce *ConfRover*, an autoregressive model that simultaneously learns protein conformation and dynamics from MD trajectory data, supporting both time-dependent and time-independent sampling.
At the core of our model is a modular architecture comprising: (i) an *encoding layer*, adapted from protein folding models, that embeds protein-specific information and conformation at each time frame into a latent space; (ii) a *temporal module*, a sequence model that captures conformational dynamics across frames; and (iii) an SE(3) diffusion model as the *structure decoder*, generating conformations in continuous space.
Experiments on ATLAS, a large-scale protein MD dataset of diverse structures, demonstrate the effectiveness of our model in learning conformational dynamics and supporting a wide range of downstream tasks. 
*ConfRover* is the first model to sample both protein conformations and trajectories within a single framework, offering a novel and flexible approach for learning from protein MD data.

#### 📊 レビュー詳細

**レビュー 1** (評価: 5, 確信度: 4)

**要約:**
In this paper, the authors propose a new model for the simulation of protein conformations that can perform multiple tasks: time-independent sampling, trajectory simulation and interpolation between two conformations. This is achieved by defining all tasks as conditioned generation, where the condit...

**レビュー 2** (評価: 5, 確信度: 3)

**要約:**
The authors introduces Confrover, an autoregressive framework that couples a transformer “trajectory module” with an SE(3) diffusion decoder so that one model can (i) roll out protein trajectories, (ii) draw single, time-independent conformations, and (iii) interpolate between two end states. Traini...

**レビュー 3** (評価: 5, 確信度: 4)

**要約:**
The paper presents a method that holistically takes into account both time-independent and time-sequence protein conformations.
This is done by a representation where the encoder is at the level of individual frames, and the sequence are modeled using masked auto-regression. The same model is used f...

*他 1 件のレビューは省略*

**🔗 リンク**:
- [OpenReview](https://openreview.net/forum?id=jj0nJQYFlW)
- [PDF](https://openreview.net/pdf?id=jj0nJQYFlW)

---

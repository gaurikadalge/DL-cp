"""Comprehensive benchmark dataset for reproducible evaluation of retrieval algorithms.

Contains:
- 20 authoritative academic fixture chunks across 6 papers (ResNet, Transformer, AdamW, ViT, LoRA, Diffusion).
- 35 carefully curated and labeled evaluation queries across 5 distinct categories:
  1. Exact Technical Terms & Acronyms
  2. Conceptual Semantic Paraphrases
  3. Multi-Passage Questions
  4. Table & Architectural Specification Queries
  5. Unanswerable / Out-of-Domain Negative Controls
"""

from app.schemas import StableChunkMetadata

# ------------------------------------------------------------------------------
# 1. Authoritative Fixture Document Chunks (20 chunks across 6 academic papers)
# ------------------------------------------------------------------------------

FIXTURE_CHUNKS: list[StableChunkMetadata] = [
    # --------------------------------------------------------------------------
    # Document 1: Deep Residual Learning for Image Recognition (ResNet)
    # --------------------------------------------------------------------------
    StableChunkMetadata(
        chunk_id="resnet#chunk_001",
        document_id="doc_resnet",
        text=(
            "When deeper networks are able to start converging, a degradation problem has been exposed: "
            "with the network depth increasing, accuracy gets saturated and then degrades rapidly. "
            "Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a "
            "suitably deep model leads to higher training error."
        ),
        page_number=1,
        section="1. Introduction",
        content_type="text",
        source_filename="deep_residual_learning.pdf",
    ),
    StableChunkMetadata(
        chunk_id="resnet#chunk_002",
        document_id="doc_resnet",
        text=(
            "We address the degradation problem by introducing a deep residual learning framework. "
            "Instead of hoping each few stacked layers directly fit a desired underlying mapping H(x), "
            "we explicitly let these layers fit a residual mapping F(x) = H(x) - x. The original mapping "
            "is recast into F(x) + x, realized by feedforward networks with identity shortcut connections."
        ),
        page_number=2,
        section="3. Deep Residual Learning",
        content_type="text",
        source_filename="deep_residual_learning.pdf",
    ),
    StableChunkMetadata(
        chunk_id="resnet#chunk_003",
        document_id="doc_resnet",
        text=(
            "The shortcut connections in residual networks simply perform identity mapping, "
            "and their outputs are added to the outputs of the stacked layers. They introduce "
            "neither extra parameters nor computational complexity. The entire network can still "
            "be trained end-to-end by SGD with backpropagation."
        ),
        page_number=3,
        section="3.1. Identity Mapping by Shortcuts",
        content_type="text",
        source_filename="deep_residual_learning.pdf",
    ),
    StableChunkMetadata(
        chunk_id="resnet#chunk_004",
        document_id="doc_resnet",
        text=(
            "As detailed in Table 4, our 152-layer ResNet achieves a 3.57% top-5 error on the ImageNet "
            "test set, winning first place in the ILSVRC 2015 classification competition. The bottleneck "
            "design uses 1x1, 3x3, and 1x1 convolutions to reduce and restore feature dimensions."
        ),
        page_number=6,
        section="4.1. ImageNet Classification",
        content_type="table",
        source_filename="deep_residual_learning.pdf",
    ),
    # --------------------------------------------------------------------------
    # Document 2: Attention Is All You Need (Transformer)
    # --------------------------------------------------------------------------
    StableChunkMetadata(
        chunk_id="transformer#chunk_001",
        document_id="doc_transformer",
        text=(
            "The dominant sequence transduction models are based on complex recurrent or convolutional "
            "neural networks that include an encoder and a decoder. We propose the Transformer, "
            "a model architecture eschewing recurrence and entirely relying on an attention mechanism "
            "to draw global dependencies between input and output."
        ),
        page_number=1,
        section="1. Introduction",
        content_type="text",
        source_filename="attention_is_all_you_need.pdf",
    ),
    StableChunkMetadata(
        chunk_id="transformer#chunk_002",
        document_id="doc_transformer",
        text=(
            "Multi-head attention allows the model to jointly attend to information from different "
            "representation subspaces at different positions. With a single attention head, averaging "
            "inhibits this capability. MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O where "
            "each head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)."
        ),
        page_number=4,
        section="3.2.2. Multi-Head Attention",
        content_type="text",
        source_filename="attention_is_all_you_need.pdf",
    ),
    StableChunkMetadata(
        chunk_id="transformer#chunk_003",
        document_id="doc_transformer",
        text=(
            "As noted in Table 1, a self-attention layer connects all positions with a constant number "
            "of sequentially executed operations O(1), whereas a recurrent layer requires O(n) sequential "
            "operations. Self-attention computational complexity per layer is O(n^2 * d), where n is the "
            "sequence length and d is the representation dimension."
        ),
        page_number=6,
        section="4. Why Self-Attention",
        content_type="table",
        source_filename="attention_is_all_you_need.pdf",
    ),
    StableChunkMetadata(
        chunk_id="transformer#chunk_004",
        document_id="doc_transformer",
        text=(
            "On the WMT 2014 English-to-German translation task, the big transformer model achieves a state-of-the-art "
            "BLEU-4 score of 28.4, outperforming existing best ensembles by over 2.0 BLEU. The model was trained "
            "for 3.5 days on 8 P100 GPUs."
        ),
        page_number=8,
        section="5. Results",
        content_type="text",
        source_filename="attention_is_all_you_need.pdf",
    ),
    # --------------------------------------------------------------------------
    # Document 3: Decoupled Weight Decay Regularization (AdamW)
    # --------------------------------------------------------------------------
    StableChunkMetadata(
        chunk_id="adamw#chunk_001",
        document_id="doc_adamw",
        text=(
            "L2 regularization and weight decay are identical for standard SGD, but they are not "
            "equivalent for adaptive gradient algorithms like Adam. We propose AdamW which decouples "
            "the weight decay from the optimization steps with respect to the loss function."
        ),
        page_number=2,
        section="2. Decoupled Weight Decay",
        content_type="text",
        source_filename="adamw_decoupled_weight_decay.pdf",
    ),
    StableChunkMetadata(
        chunk_id="adamw#chunk_002",
        document_id="doc_adamw",
        text=(
            "In common deep learning libraries, L2 regularization is implemented by adding the product of "
            "the decay coefficient lambda and parameter theta directly to the loss gradient. In Adam, this causes "
            "weights with large historical gradients to be regularized substantially less than weights with small gradients."
        ),
        page_number=3,
        section="2.1. Gradient Disparity in Adaptive Methods",
        content_type="text",
        source_filename="adamw_decoupled_weight_decay.pdf",
    ),
    StableChunkMetadata(
        chunk_id="adamw#chunk_003",
        document_id="doc_adamw",
        text=(
            "Empirical results show that AdamW substantially improves generalization performance over standard Adam with L2, "
            "closing the generalization gap with momentum SGD on image classification benchmarks like CIFAR-10 and ImageNet."
        ),
        page_number=5,
        section="4. Experiments",
        content_type="text",
        source_filename="adamw_decoupled_weight_decay.pdf",
    ),
    # --------------------------------------------------------------------------
    # Document 4: An Image is Worth 16x16 Words: Vision Transformer (ViT)
    # --------------------------------------------------------------------------
    StableChunkMetadata(
        chunk_id="vit#chunk_001",
        document_id="doc_vit",
        text=(
            "While the Transformer architecture has become the de-facto standard for natural language processing, "
            "its applications to computer vision remain limited. We show that a pure transformer applied directly to "
            "sequences of image patches can perform remarkably well on image classification tasks."
        ),
        page_number=1,
        section="1. Introduction",
        content_type="text",
        source_filename="vision_transformer_vit.pdf",
    ),
    StableChunkMetadata(
        chunk_id="vit#chunk_002",
        document_id="doc_vit",
        text=(
            "To handle 2D images, we reshape the image into a sequence of flattened 2D patches of size 16x16 pixels. "
            "The Vision Transformer (ViT) maps each patch to a constant vector dimension D with a trainable linear projection. "
            "Similar to BERT, we prepend a learnable [CLS] classification token to the sequence."
        ),
        page_number=3,
        section="3.1. Patch Embedding & Architecture",
        content_type="text",
        source_filename="vision_transformer_vit.pdf",
    ),
    StableChunkMetadata(
        chunk_id="vit#chunk_003",
        document_id="doc_vit",
        text=(
            "When pre-trained on mid-sized datasets like ImageNet-1K, Vision Transformers yield modest accuracies below "
            "comparably sized ResNets due to lack of inductive biases like translation equivariance. However, when pre-trained "
            "on large datasets like JFT-300M, ViT achieves state-of-the-art results (88.55% top-1 on ImageNet)."
        ),
        page_number=6,
        section="4. Experiments & Scaling",
        content_type="text",
        source_filename="vision_transformer_vit.pdf",
    ),
    # --------------------------------------------------------------------------
    # Document 5: LoRA: Low-Rank Adaptation of Large Language Models
    # --------------------------------------------------------------------------
    StableChunkMetadata(
        chunk_id="lora#chunk_001",
        document_id="doc_lora",
        text=(
            "An important paradigm of NLP involves large-scale pre-training on general domain data followed by task-specific "
            "fine-tuning. We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained model weights and injects "
            "trainable rank decomposition matrices into each layer of the Transformer architecture."
        ),
        page_number=1,
        section="1. Introduction",
        content_type="text",
        source_filename="lora_low_rank_adaptation.pdf",
    ),
    StableChunkMetadata(
        chunk_id="lora#chunk_002",
        document_id="doc_lora",
        text=(
            "For a pre-trained weight matrix W_0 in R^{d x k}, we constrain its update by representing Delta W = B * A, "
            "where B in R^{d x r} and A in R^{r x k}, and the rank r << min(d, k). During training, W_0 is frozen and does not "
            "receive gradient updates, while A and B contain trainable parameters initialized with Gaussian and zero respectively."
        ),
        page_number=3,
        section="3. Problem Formulation & Low-Rank Parametrization",
        content_type="text",
        source_filename="lora_low_rank_adaptation.pdf",
    ),
    StableChunkMetadata(
        chunk_id="lora#chunk_003",
        document_id="doc_lora",
        text=(
            "LoRA reduces the number of trainable parameters by 10,000 times and GPU memory requirements by 3 times on GPT-3 175B. "
            "Crucially, it introduces no additional inference latency compared to full fine-tuning because the learned matrix BA "
            "can be explicitly merged into W_0 at deployment."
        ),
        page_number=5,
        section="4. Efficiency & Latency",
        content_type="text",
        source_filename="lora_low_rank_adaptation.pdf",
    ),
    # --------------------------------------------------------------------------
    # Document 6: Denoising Diffusion Probabilistic Models (DDPM)
    # --------------------------------------------------------------------------
    StableChunkMetadata(
        chunk_id="diffusion#chunk_001",
        document_id="doc_diffusion",
        text=(
            "Deep generative models have seen tremendous progress across GANs and VAEs. We present high quality image synthesis "
            "results using diffusion probabilistic models, which are parameterized Markov chains trained using variational inference "
            "to reverse a process that gradually adds noise to data."
        ),
        page_number=1,
        section="1. Introduction",
        content_type="text",
        source_filename="denoising_diffusion_models.pdf",
    ),
    StableChunkMetadata(
        chunk_id="diffusion#chunk_002",
        document_id="doc_diffusion",
        text=(
            "The forward or diffusion process q(x_t | x_{t-1}) adds Gaussian noise according to a variance schedule beta_1, ..., beta_T. "
            "The reverse process p_theta(x_{t-1} | x_t) is learned using a neural network parametrized as a UNet with self-attention, "
            "trained to predict the added epsilon noise vector."
        ),
        page_number=3,
        section="2. Background & Forward Process",
        content_type="text",
        source_filename="denoising_diffusion_models.pdf",
    ),
    StableChunkMetadata(
        chunk_id="diffusion#chunk_003",
        document_id="doc_diffusion",
        text=(
            "Our simplified loss function L_simple(theta) = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(sqrt(alpha_bar_t) * x_0 + "
            "sqrt(1 - alpha_bar_t) * epsilon, t) ||^2 ] discards weighting terms and yields improved sample quality, achieving an FID "
            "of 3.17 on unconditional CIFAR-10."
        ),
        page_number=5,
        section="3. Simplified Training Objective",
        content_type="text",
        source_filename="denoising_diffusion_models.pdf",
    ),
]


# ------------------------------------------------------------------------------
# 2. Evaluation Benchmark Queries (35 carefully labeled queries)
# ------------------------------------------------------------------------------

BENCHMARK_QUERIES: list[dict] = [
    # ==========================================================================
    # Category 1: Exact Technical Terms & Acronyms (BM25 Precision Focus)
    # ==========================================================================
    {
        "query_id": "q01_adamw_l2",
        "category": "technical_acronym",
        "query": "Why does AdamW decouple weight decay from gradient updates?",
        "relevant_chunks": {"adamw#chunk_001", "adamw#chunk_002"},
        "graded_relevance": {"adamw#chunk_001": 2.0, "adamw#chunk_002": 1.0},
        "expected_documents": {"doc_adamw"},
        "notes": "Targeted exact term AdamW and L2 regularization.",
    },
    {
        "query_id": "q02_self_attn_complexity",
        "category": "technical_acronym",
        "query": "What is the computational complexity per layer of self-attention O(n^2 * d)?",
        "relevant_chunks": {"transformer#chunk_003"},
        "graded_relevance": {"transformer#chunk_003": 2.0},
        "expected_documents": {"doc_transformer"},
        "notes": "Exact mathematical notation search O(n^2 * d).",
    },
    {
        "query_id": "q03_bleu_wmt",
        "category": "technical_acronym",
        "query": "What BLEU-4 score did the Transformer achieve on WMT 2014 English-to-German?",
        "relevant_chunks": {"transformer#chunk_004"},
        "graded_relevance": {"transformer#chunk_004": 2.0},
        "expected_documents": {"doc_transformer"},
        "notes": "Exact metric BLEU-4 on benchmark WMT 2014.",
    },
    {
        "query_id": "q04_lora_rank_r",
        "category": "technical_acronym",
        "query": "How is the low-rank update matrix Delta W = B * A parameterized with rank r in LoRA?",
        "relevant_chunks": {"lora#chunk_002"},
        "graded_relevance": {"lora#chunk_002": 2.0},
        "expected_documents": {"doc_lora"},
        "notes": "Exact formulation Delta W = B * A with rank r.",
    },
    {
        "query_id": "q05_resnet152_imagenet",
        "category": "technical_acronym",
        "query": "What top-5 error rate did the 152-layer ResNet achieve on ImageNet test set?",
        "relevant_chunks": {"resnet#chunk_004"},
        "graded_relevance": {"resnet#chunk_004": 2.0},
        "expected_documents": {"doc_resnet"},
        "notes": "Exact numbers: 152-layer ResNet and 3.57% top-5 error.",
    },
    {
        "query_id": "q06_vit_16x16_patch",
        "category": "technical_acronym",
        "query": "What patch size does Vision Transformer use to reshape 2D images?",
        "relevant_chunks": {"vit#chunk_002"},
        "graded_relevance": {"vit#chunk_002": 2.0},
        "expected_documents": {"doc_vit"},
        "notes": "Exact tokenization dimension 16x16 pixels.",
    },
    {
        "query_id": "q07_ddpm_lsimple",
        "category": "technical_acronym",
        "query": "What is the simplified loss function L_simple for training Denoising Diffusion models?",
        "relevant_chunks": {"diffusion#chunk_003"},
        "graded_relevance": {"diffusion#chunk_003": 2.0},
        "expected_documents": {"doc_diffusion"},
        "notes": "Exact loss notation L_simple(theta) in DDPM.",
    },
    # ==========================================================================
    # Category 2: Conceptual & Semantic Paraphrasing (Dense Retrieval Focus)
    # ==========================================================================
    {
        "query_id": "q08_gradient_vanishing",
        "category": "semantic_paraphrase",
        "query": "How can neural architectures alleviate performance degradation when stacking deeper layers?",
        "relevant_chunks": {"resnet#chunk_001", "resnet#chunk_002"},
        "graded_relevance": {"resnet#chunk_002": 2.0, "resnet#chunk_001": 1.0},
        "expected_documents": {"doc_resnet"},
        "notes": "Conceptual paraphrase avoiding exact terms F(x) = H(x) - x.",
    },
    {
        "query_id": "q09_eschewing_recurrence",
        "category": "semantic_paraphrase",
        "query": "Eliminating sequential time-step dependencies in sequence transduction systems.",
        "relevant_chunks": {"transformer#chunk_001", "transformer#chunk_003"},
        "graded_relevance": {"transformer#chunk_001": 2.0, "transformer#chunk_003": 1.0},
        "expected_documents": {"doc_transformer"},
        "notes": "Conceptual description of replacing RNNs with self-attention.",
    },
    {
        "query_id": "q10_parameter_efficient_tuning",
        "category": "semantic_paraphrase",
        "query": "Methods for fine-tuning massive language models without updating all underlying weights.",
        "relevant_chunks": {"lora#chunk_001", "lora#chunk_003"},
        "graded_relevance": {"lora#chunk_001": 2.0, "lora#chunk_003": 1.0},
        "expected_documents": {"doc_lora"},
        "notes": "General concept of parameter-efficient adaptation.",
    },
    {
        "query_id": "q11_vision_inductive_biases",
        "category": "semantic_paraphrase",
        "query": "Why do convolutional alternatives outperform vision transformers on smaller training sets?",
        "relevant_chunks": {"vit#chunk_003"},
        "graded_relevance": {"vit#chunk_003": 2.0},
        "expected_documents": {"doc_vit"},
        "notes": "Conceptual explanation of translation equivariance and inductive biases.",
    },
    {
        "query_id": "q12_adaptive_learning_rate_gap",
        "category": "semantic_paraphrase",
        "query": "Discrepancy between adaptive optimizer generalization and standard momentum stochastic descent.",
        "relevant_chunks": {"adamw#chunk_002", "adamw#chunk_003"},
        "graded_relevance": {"adamw#chunk_003": 2.0, "adamw#chunk_002": 1.0},
        "expected_documents": {"doc_adamw"},
        "notes": "Conceptual inquiry into Adam vs SGD generalization.",
    },
    {
        "query_id": "q13_reversing_noise_corruption",
        "category": "semantic_paraphrase",
        "query": "Generating synthetic imagery by iteratively removing gradual Gaussian perturbation.",
        "relevant_chunks": {"diffusion#chunk_001", "diffusion#chunk_002"},
        "graded_relevance": {"diffusion#chunk_001": 2.0, "diffusion#chunk_002": 1.0},
        "expected_documents": {"doc_diffusion"},
        "notes": "Conceptual description of generative reverse diffusion.",
    },
    {
        "query_id": "q14_token_subspace_aggregation",
        "category": "semantic_paraphrase",
        "query": "Mechanisms allowing attention to simultaneously focus on multiple distinct linguistic facets.",
        "relevant_chunks": {"transformer#chunk_002"},
        "graded_relevance": {"transformer#chunk_002": 2.0},
        "expected_documents": {"doc_transformer"},
        "notes": "Conceptual paraphrase for multi-head representation subspaces.",
    },
    # ==========================================================================
    # Category 3: Multi-Passage Questions (Requires Multi-Chunk Evidence)
    # ==========================================================================
    {
        "query_id": "q15_resnet_complete_architecture",
        "category": "multi_passage",
        "query": "What causes degradation in deep models and how does residual learning solve it with shortcuts?",
        "relevant_chunks": {"resnet#chunk_001", "resnet#chunk_002", "resnet#chunk_003"},
        "graded_relevance": {
            "resnet#chunk_002": 2.0,
            "resnet#chunk_001": 1.0,
            "resnet#chunk_003": 1.0,
        },
        "expected_documents": {"doc_resnet"},
        "notes": "Requires chunks 1, 2, and 3 for a complete answer.",
    },
    {
        "query_id": "q16_adamw_mechanism_and_benefit",
        "category": "multi_passage",
        "query": "Why is L2 regularization flawed in Adam and how does decoupling weight decay improve results?",
        "relevant_chunks": {"adamw#chunk_001", "adamw#chunk_002", "adamw#chunk_003"},
        "graded_relevance": {
            "adamw#chunk_002": 2.0,
            "adamw#chunk_001": 1.0,
            "adamw#chunk_003": 1.0,
        },
        "expected_documents": {"doc_adamw"},
        "notes": "Spans formulation (chunk 1), gradient mechanics (chunk 2), and benchmarks (chunk 3).",
    },
    {
        "query_id": "q17_transformer_motivation_and_complexity",
        "category": "multi_passage",
        "query": "Why did the Transformer replace recurrent networks and what are its sequential operation advantages?",
        "relevant_chunks": {"transformer#chunk_001", "transformer#chunk_003"},
        "graded_relevance": {"transformer#chunk_001": 2.0, "transformer#chunk_003": 1.0},
        "expected_documents": {"doc_transformer"},
        "notes": "Combines intro rationale (chunk 1) with sequential complexity table (chunk 3).",
    },
    {
        "query_id": "q18_lora_formulation_and_deployment",
        "category": "multi_passage",
        "query": "How is LoRA formulated mathematically and why does it avoid adding inference latency?",
        "relevant_chunks": {"lora#chunk_002", "lora#chunk_003"},
        "graded_relevance": {"lora#chunk_002": 2.0, "lora#chunk_003": 2.0},
        "expected_documents": {"doc_lora"},
        "notes": "Requires matrix formulation (chunk 2) and latency merge property (chunk 3).",
    },
    {
        "query_id": "q19_vit_architecture_and_scaling",
        "category": "multi_passage",
        "query": "How does ViT represent images as tokens and how does dataset scale affect its performance?",
        "relevant_chunks": {"vit#chunk_002", "vit#chunk_003"},
        "graded_relevance": {"vit#chunk_002": 2.0, "vit#chunk_003": 2.0},
        "expected_documents": {"doc_vit"},
        "notes": "Combines patch tokenization (chunk 2) and scaling experiments (chunk 3).",
    },
    {
        "query_id": "q20_diffusion_process_and_loss",
        "category": "multi_passage",
        "query": "Describe the forward Markov chain in diffusion models and the simplified loss objective.",
        "relevant_chunks": {"diffusion#chunk_002", "diffusion#chunk_003"},
        "graded_relevance": {"diffusion#chunk_002": 2.0, "diffusion#chunk_003": 2.0},
        "expected_documents": {"doc_diffusion"},
        "notes": "Requires forward process (chunk 2) and simplified loss objective (chunk 3).",
    },
    # ==========================================================================
    # Category 4: Table & Architectural Specification Queries
    # ==========================================================================
    {
        "query_id": "q21_sequential_operations_comparison",
        "category": "table_specification",
        "query": "According to Table 1, how do self-attention and recurrent layers compare in sequential operations?",
        "relevant_chunks": {"transformer#chunk_003"},
        "graded_relevance": {"transformer#chunk_003": 2.0},
        "expected_documents": {"doc_transformer"},
        "notes": "Direct reference to Table 1 complexity comparisons: O(1) vs O(n).",
    },
    {
        "query_id": "q22_bottleneck_architecture_table",
        "category": "table_specification",
        "query": "How does the bottleneck block design in Table 4 use 1x1 convolutions?",
        "relevant_chunks": {"resnet#chunk_004"},
        "graded_relevance": {"resnet#chunk_004": 2.0},
        "expected_documents": {"doc_resnet"},
        "notes": "Identifies Table 4 bottleneck convolution details.",
    },
    {
        "query_id": "q23_lora_memory_reduction",
        "category": "table_specification",
        "query": "By what factor does LoRA reduce trainable parameters and GPU memory on GPT-3 175B?",
        "relevant_chunks": {"lora#chunk_003"},
        "graded_relevance": {"lora#chunk_003": 2.0},
        "expected_documents": {"doc_lora"},
        "notes": "Specific empirical factors: 10,000x parameters and 3x GPU memory.",
    },
    {
        "query_id": "q24_vit_jft300m_accuracy",
        "category": "table_specification",
        "query": "What top-1 accuracy does Vision Transformer achieve when pre-trained on JFT-300M?",
        "relevant_chunks": {"vit#chunk_003"},
        "graded_relevance": {"vit#chunk_003": 2.0},
        "expected_documents": {"doc_vit"},
        "notes": "Specific empirical metric: 88.55% on ImageNet.",
    },
    {
        "query_id": "q25_ddpm_cifar10_fid",
        "category": "table_specification",
        "query": "What sample quality FID score did Denoising Diffusion models obtain on unconditional CIFAR-10?",
        "relevant_chunks": {"diffusion#chunk_003"},
        "graded_relevance": {"diffusion#chunk_003": 2.0},
        "expected_documents": {"doc_diffusion"},
        "notes": "Specific metric FID = 3.17.",
    },
    {
        "query_id": "q26_cls_token_prepend",
        "category": "table_specification",
        "query": "What special token is prepended to the patch sequence in Vision Transformer?",
        "relevant_chunks": {"vit#chunk_002"},
        "graded_relevance": {"vit#chunk_002": 2.0},
        "expected_documents": {"doc_vit"},
        "notes": "Architectural classification token [CLS].",
    },
    {
        "query_id": "q27_identity_mapping_parameters",
        "category": "table_specification",
        "query": "Do identity shortcut connections add extra computational parameters to deep residual networks?",
        "relevant_chunks": {"resnet#chunk_003"},
        "graded_relevance": {"resnet#chunk_003": 2.0},
        "expected_documents": {"doc_resnet"},
        "notes": "Computational footprint verification: zero extra parameters.",
    },
    # ==========================================================================
    # Category 5: Unanswerable & Out-of-Domain Negative Controls
    # ==========================================================================
    {
        "query_id": "q28_unanswerable_quantum",
        "category": "unanswerable_control",
        "query": "What cryogenic temperature is required to maintain coherence in transmon quantum qubits?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Out-of-domain negative control; system must abstain or return no citations.",
    },
    {
        "query_id": "q29_unanswerable_crispr",
        "category": "unanswerable_control",
        "query": "How does Cas9 endonuclease utilize guide RNA to execute targeted double-strand breaks in DNA?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Biological negative control absent from academic deep learning corpus.",
    },
    {
        "query_id": "q30_unanswerable_tokamak",
        "category": "unanswerable_control",
        "query": "What magnetic field strength in Tesla is generated by superconducting coils in the ITER tokamak?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Plasma physics negative control.",
    },
    {
        "query_id": "q31_unanswerable_photolithography",
        "category": "unanswerable_control",
        "query": "What extreme ultraviolet EUV light wavelength is utilized in advanced semiconductor lithography?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Semiconductor manufacturing negative control.",
    },
    {
        "query_id": "q32_unanswerable_macroeconomics",
        "category": "unanswerable_control",
        "query": "How do central bank quantitative easing policies influence long-term treasury bond yield curves?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Economics negative control.",
    },
    {
        "query_id": "q33_unanswerable_dark_matter",
        "category": "unanswerable_control",
        "query": "What cross-section constraints have Xenon-1T liquid noble gas detectors established for WIMP dark matter?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Astrophysics negative control.",
    },
    {
        "query_id": "q34_unanswerable_batteries",
        "category": "unanswerable_control",
        "query": "What solid-state electrolyte composition prevents lithium dendrite propagation at high current densities?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Materials chemistry negative control.",
    },
    {
        "query_id": "q35_unanswerable_aerodynamics",
        "category": "unanswerable_control",
        "query": "What boundary layer turbulence model accurately predicts shock-wave boundary layer interaction in scramjets?",
        "relevant_chunks": set(),
        "graded_relevance": {},
        "expected_documents": set(),
        "notes": "Hypersonic propulsion negative control.",
    },
]

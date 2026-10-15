"""Synthetic benchmark dataset for reproducible evaluation of retrieval algorithms.

NOTE: This is synthetic fixture test data for automated evaluation and regression testing.
It does not represent production benchmark claims.
"""

from typing import Dict, List
from app.schemas import StableChunkMetadata

# ------------------------------------------------------------------------------
# 1. Authoritative Fixture Document Chunks
# ------------------------------------------------------------------------------

FIXTURE_CHUNKS: List[StableChunkMetadata] = [
    # ResNet Paper
    StableChunkMetadata(
        chunk_id="resnet#chunk_001",
        document_id="doc_resnet",
        text=(
            "When deeper networks are able to start converging, a degradation problem has been exposed: "
            "with network depth increasing, accuracy gets saturated and then degrades rapidly."
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
            "Instead of hoping each few stacked layers directly fit a desired underlying mapping, "
            "we explicitly let these layers fit a residual mapping F(x) = H(x) - x."
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
            "neither extra parameters nor computational complexity."
        ),
        page_number=3,
        section="3.1. Identity Mapping by Shortcuts",
        content_type="text",
        source_filename="deep_residual_learning.pdf",
    ),
    # Attention Paper
    StableChunkMetadata(
        chunk_id="transformer#chunk_001",
        document_id="doc_transformer",
        text=(
            "The dominant sequence transduction models are based on complex recurrent or convolutional "
            "neural networks that include an encoder and a decoder. We propose the Transformer, "
            "a model architecture eschewing recurrence and entirely relying on an attention mechanism."
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
            "inhibits this capability."
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
            "operations. Self-attention complexity per layer is O(n^2 * d)."
        ),
        page_number=6,
        section="4. Why Self-Attention",
        content_type="text",
        source_filename="attention_is_all_you_need.pdf",
    ),
    # AdamW Paper
    StableChunkMetadata(
        chunk_id="adamw#chunk_001",
        document_id="doc_adamw",
        text=(
            "L2 regularization and weight decay are identical for standard SGD, but they are not "
            "equivalent for adaptive gradient algorithms like Adam. We propose AdamW which decouples "
            "the weight decay from the optimization steps."
        ),
        page_number=2,
        section="2. Decoupled Weight Decay",
        content_type="text",
        source_filename="adamw_decoupled_weight_decay.pdf",
    ),
]


# ------------------------------------------------------------------------------
# 2. Evaluation Benchmark Queries with Ground Truth
# ------------------------------------------------------------------------------

BENCHMARK_QUERIES: List[Dict] = [
    {
        "query_id": "q1",
        "query": "How do residual networks solve the degradation problem?",
        "relevant_chunks": {"resnet#chunk_001", "resnet#chunk_002", "resnet#chunk_003"},
        "graded_relevance": {
            "resnet#chunk_002": 2.0,
            "resnet#chunk_001": 1.0,
            "resnet#chunk_003": 1.0,
        },
        "expected_documents": {"doc_resnet"},
        "notes": "Targeted conceptual query matching ResNet formulation.",
    },
    {
        "query_id": "q2",
        "query": "What is the computational complexity per layer of self-attention?",
        "relevant_chunks": {"transformer#chunk_003"},
        "graded_relevance": {
            "transformer#chunk_003": 2.0,
        },
        "expected_documents": {"doc_transformer"},
        "notes": "Specific numerical/mathematical complexity query O(n^2 * d).",
    },
    {
        "query_id": "q3",
        "query": "Why does AdamW decouple weight decay from gradient updates?",
        "relevant_chunks": {"adamw#chunk_001"},
        "graded_relevance": {
            "adamw#chunk_001": 2.0,
        },
        "expected_documents": {"doc_adamw"},
        "notes": "Domain terminology query on AdamW and L2 regularization.",
    },
    {
        "query_id": "q4",
        "query": "What are the benefits of multi-head attention compared to single-head?",
        "relevant_chunks": {"transformer#chunk_002"},
        "graded_relevance": {
            "transformer#chunk_002": 2.0,
        },
        "expected_documents": {"doc_transformer"},
        "notes": "Transformer architectural mechanism comparison.",
    },
    {
        "query_id": "q5",
        "query": "Do identity shortcut connections add extra computational complexity?",
        "relevant_chunks": {"resnet#chunk_003"},
        "graded_relevance": {
            "resnet#chunk_003": 2.0,
        },
        "expected_documents": {"doc_resnet"},
        "notes": "Verification of computational footprint for identity shortcuts.",
    },
]

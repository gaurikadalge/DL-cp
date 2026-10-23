#!/usr/bin/env python3
"""Generate high-resolution academic visualization charts from verified evaluation results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
EVAL_RESULTS_PATH = REPO_ROOT / "evaluation" / "retrieval_benchmark_results.json"


def generate_benchmark_comparison_chart():
    """Generate overall retrieval metrics comparison bar chart."""
    with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    overall = data["overall_results"]
    configs = list(overall.keys())  # ["BM25 Only", "Dense Only", "Hybrid RRF", "Hybrid + Reranker"]
    metrics = ["Hit@K", "Recall@K", "MRR", "NDCG@K"]
    metric_labels = ["Hit@5", "Recall@5", "MRR", "NDCG@5"]

    # Colors tailored for academic publication (clean, distinguishable palette)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    x = np.arange(len(configs))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    plt.subplots_adjust(bottom=0.15)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels, strict=False)):
        values = [overall[cfg][metric] for cfg in configs]
        rects = ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=label,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        # Add value labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title(
        "Retrieval Benchmark Performance by Configuration (N=35 Queries, K=5)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="medium")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=10)

    output_path = DOCS_DIR / "benchmark_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved benchmark comparison chart to: {output_path}")


def generate_category_performance_chart():
    """Generate per-category Recall@5 comparison chart."""
    with open(EVAL_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    categories_data = data["category_results"]
    cat_keys = [
        "technical_acronym",
        "semantic_paraphrase",
        "multi_passage",
        "table_specification",
    ]
    cat_names = [
        "Technical Acronym\n(N=7)",
        "Semantic Paraphrase\n(N=7)",
        "Multi-Passage Synthesis\n(N=6)",
        "Table Specification\n(N=7)",
    ]

    configs = ["BM25 Only", "Dense Only", "Hybrid RRF", "Hybrid + Reranker"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    x = np.arange(len(cat_keys))
    width = 0.2

    # Subplot 1: Recall@5 across categories
    for i, (cfg, color) in enumerate(zip(configs, colors, strict=False)):
        recalls = [categories_data[cat][cfg]["Recall@K"] for cat in cat_keys]
        ax1.bar(
            x + (i - 1.5) * width,
            recalls,
            width,
            label=cfg,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax1.set_title("Recall@5 by Query Category", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Recall@5", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_names, fontsize=9)
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)

    # Subplot 2: NDCG@5 across categories
    for i, (cfg, color) in enumerate(zip(configs, colors, strict=False)):
        ndcgs = [categories_data[cat][cfg]["NDCG@K"] for cat in cat_keys]
        ax2.bar(
            x + (i - 1.5) * width,
            ndcgs,
            width,
            label=cfg,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax2.set_title("NDCG@5 by Query Category", fontsize=12, fontweight="bold")
    ax2.set_ylabel("NDCG@5", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=9)

    plt.suptitle(
        "Category-Level Performance Analysis on Research Papers Benchmark",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    output_path = DOCS_DIR / "category_performance.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved category performance chart to: {output_path}")


def generate_architecture_diagram():
    """Render a clean architecture diagram as docs/architecture.png."""
    fig, ax = plt.subplots(figsize=(14, 9), dpi=300)
    ax.axis("off")

    # Define subsystem boxes
    boxes = [
        # (title, subtitle, x, y, w, h, bg_color, border_color)
        (
            "User Interaction Layer",
            "Streamlit Portal (app/frontend.py)\nResearch QA • Ingestion Library • Benchmarks",
            0.05,
            0.78,
            0.40,
            0.16,
            "#e8f4f8",
            "#2b5c8f",
        ),
        (
            "FastAPI Service Layer",
            "app/main.py • app/api/documents.py • app/api/query.py\nPydantic Schemas • Centralized Settings • CORS",
            0.55,
            0.78,
            0.40,
            0.16,
            "#e8f4f8",
            "#2b5c8f",
        ),
        (
            "Multimodal Ingestion Pipeline",
            "DocumentValidator • PDFExtractor (PyMuPDF)\nScannedPageDetector & OCREngine (Tesseract/EasyOCR)\nTableExtractor (find_tables) • FigureExtractor • SemanticChunker",
            0.05,
            0.45,
            0.40,
            0.26,
            "#fff2e6",
            "#d95f02",
        ),
        (
            "Hybrid Retrieval & RAG Engine",
            "DenseRetriever (FAISS IndexFlatIP)\nBM25Retriever (Okapi BM25 Sparse Search)\nHybridRetriever (Reciprocal Rank Fusion k=60)\nCross-Encoder Reranker (ms-marco-MiniLM)",
            0.55,
            0.45,
            0.40,
            0.26,
            "#f3f0f7",
            "#7570b3",
        ),
        (
            "Persistence & Storage",
            "SQLite DB (data/research_assistant.db)\nFAISS Index (indexes/dense_faiss.index)\nBM25 Index (indexes/bm25_store.json)",
            0.05,
            0.12,
            0.40,
            0.24,
            "#eafaf1",
            "#1b9e77",
        ),
        (
            "Grounded Generation & Guardrails",
            "ContextBuilder (Token Budget & Prompt Guard)\nAnswerGenerator (OpenAI gpt-4o-mini / MockLLM)\nCitationValidator (Authoritative Provenance & Refusal)",
            0.55,
            0.12,
            0.40,
            0.24,
            "#fdfefe",
            "#4a235a",
        ),
    ]

    for title, subtitle, bx, by, bw, bh, bg, border in boxes:
        rect = plt.Rectangle(
            (bx, by),
            bw,
            bh,
            facecolor=bg,
            edgecolor=border,
            linewidth=2,
            linestyle="-",
            transform=ax.transAxes,
            zorder=2,
            clip_on=False,
        )
        ax.add_patch(rect)
        ax.text(
            bx + 0.02,
            by + bh - 0.04,
            title,
            fontsize=12,
            fontweight="bold",
            color=border,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.text(
            bx + 0.02,
            by + 0.03,
            subtitle,
            fontsize=9.5,
            color="#2c3e50",
            va="bottom",
            linespacing=1.4,
            transform=ax.transAxes,
            zorder=3,
        )

    # Connecting Flow Arrows
    arrows = [
        # (x1, y1, x2, y2, label)
        (0.45, 0.86, 0.55, 0.86, "REST API"),
        (0.75, 0.78, 0.75, 0.71, "Search Request"),
        (0.25, 0.78, 0.25, 0.71, "PDF Upload"),
        (0.25, 0.45, 0.25, 0.36, "Store Meta & Chunks"),
        (0.45, 0.58, 0.55, 0.58, "Index Sync"),
        (0.75, 0.45, 0.75, 0.36, "RRF Ranked Chunks"),
        (0.55, 0.24, 0.45, 0.24, "Vector / Sparse Load"),
    ]

    for x1, y1, x2, y2, lbl in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#34495e"},
            zorder=4,
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(
            mx,
            my + 0.015,
            lbl,
            fontsize=8.5,
            fontweight="bold",
            color="#34495e",
            ha="center",
            va="bottom",
            zorder=5,
        )

    plt.title(
        "Multimodal RAG Research Assistant — End-to-End System Architecture",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    output_path = DOCS_DIR / "architecture.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved architecture diagram to: {output_path}")


if __name__ == "__main__":
    generate_benchmark_comparison_chart()
    generate_category_performance_chart()
    generate_architecture_diagram()

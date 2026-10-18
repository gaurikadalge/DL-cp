"""Unit tests for table extraction and structure formatting."""

from ingestion.table_extractor import ExtractedTable, TableExtractor


def test_markdown_table_formatting():
    headers = ["Model", "Accuracy (%)", "Latency (ms)"]
    rows = [
        ["Baseline CNN", "82.4", "14.2"],
        ["ResNet-50", "91.8", "28.5"],
        ["Ours (ViT)", "94.6", "22.1"],
    ]

    md = TableExtractor._cells_to_markdown(headers, rows)
    lines = md.splitlines()

    assert len(lines) == 5  # Header + separator + 3 data rows
    assert "| Model | Accuracy (%) | Latency (ms) |" in lines[0]
    assert "| --- | --- | --- |" in lines[1]
    assert "| Baseline CNN | 82.4 | 14.2 |" in lines[2]
    assert "| Ours (ViT) | 94.6 | 22.1 |" in lines[4]


def test_extracted_table_chunk_representation():
    table = ExtractedTable(
        table_id="p1_tbl_1",
        page_number=2,
        title_or_caption="Table 1: Benchmark Results on GLUE",
        headers=["Dataset", "Score"],
        rows=[["SST-2", "93.5"], ["MNLI", "86.2"]],
        markdown="| Dataset | Score |\n| --- | --- |\n| SST-2 | 93.5 |\n| MNLI | 86.2 |",
    )

    chunk_text = table.to_chunk_text()
    assert "[Table: Table 1: Benchmark Results on GLUE]" in chunk_text
    assert "| SST-2 | 93.5 |" in chunk_text
    assert table.to_dict()["table_id"] == "p1_tbl_1"

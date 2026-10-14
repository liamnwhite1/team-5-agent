"""Exercise table extraction with the LLM and platform I/O mocked."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent as agent_module
from agent import Team5TablePlotAgent, _clean_cell, find_tables

FIXTURES = Path(__file__).parent
REPORT = FIXTURES / "lunar_cheese_annual_report.md"


def test_find_tables_reads_captions_columns_and_rows():
    tables = find_tables(REPORT.read_text())

    assert [table["caption"][:8] for table in tables] == ["Table 1.", "Table 2.", "Table 3."]
    census = tables[1]
    assert census["section"] == "3. Settlements"
    assert census["caption"].endswith("living in the maintenance tunnels.")
    assert census["columns"][:2] == ["Year", "Crater Gouda (residents)"]
    assert len(census["rows"]) == 10
    assert census["rows"][0] == ["2042", "412", "188", "—", "350"]


def test_caption_is_the_bold_part_of_its_paragraph():
    text = "## Part I\n\n**Table 1. Population.** The total includes tinkers.\n\n| A |\n|---|\n| 1 |\n"

    (table,) = find_tables(text)

    assert table["caption"] == "Table 1. Population."
    assert table["section"] == "Part I"


def test_a_caption_belongs_only_to_the_table_right_below_it():
    text = (
        "**Table 1. First.**\n\n| A |\n|---|\n| 1 |\n\n"
        "| B |\n|---|\n| 2 |\n\n"
        "**Table 3. Third.**\n\n- a list in between\n\n| C |\n|---|\n| 3 |\n"
    )

    assert [table["caption"] for table in find_tables(text)] == ["Table 1. First.", "", ""]


def test_short_rows_are_padded_and_escaped_pipes_kept():
    text = "| A | B |\n|---|---|\n| x \\| y |\n"

    (table,) = find_tables(text)

    assert table["rows"] == [["x | y", ""]]


@pytest.mark.parametrize(
    ("cell", "expected"),
    [("412", 412), ("1,020", 1020), ("12.0", 12.0), ("-3.5", -3.5), ("—", None), ("", None),
     ("N/A", None), ("Good", "Good"), ("WHEY-01", "WHEY-01"), ("inf", "inf")],
)
def test_clean_cell(cell, expected):
    assert _clean_cell(cell) == expected


@pytest.fixture
def extract_agent(monkeypatch):
    agent = Team5TablePlotAgent(agent_name="Team-5 Table Plot Agent")
    monkeypatch.setattr(agent, "get_artifact_file_path", AsyncMock(return_value=REPORT))
    monkeypatch.setattr(agent, "resolve_artifact_reference", MagicMock(return_value=None))
    monkeypatch.setattr(agent, "send_status_update", AsyncMock())
    monkeypatch.setattr(agent, "send_json_artifact", AsyncMock(return_value="table-1"))
    monkeypatch.setattr(agent, "_pick_table", AsyncMock(return_value=(1, "It is the census.")))
    return agent


async def test_extract_publishes_the_chosen_table_as_json(extract_agent, context, event_queue):
    params = {"document_artifact_id": "doc", "table_description": "the population data"}

    result = await extract_agent._extract_table(params, context, event_queue)

    assert result["status"] == "success"
    assert result["table_artifact_id"] == "table-1"
    assert result["caption"].startswith("Table 2. Census")
    assert result["numeric_columns"] == result["columns"]
    assert result["row_count"] == 10
    assert result["why_this_table"] == "It is the census."

    payload = extract_agent.send_json_artifact.await_args.args[2]
    assert payload["rows"][0] == [2042, 412, 188, None, 350]
    assert payload["rows"][5][4] == 1020  # written as "1,020"
    name = extract_agent.send_json_artifact.await_args.kwargs["name"]
    assert name.startswith("table_2_census") and "." not in name  # read back as .json


async def test_no_matching_table_is_an_error_listing_the_captions(extract_agent, context, event_queue):
    extract_agent._pick_table.return_value = (-1, "Nothing about rainfall.")
    params = {"document_artifact_id": "doc", "table_description": "the rainfall data"}

    result = await extract_agent._extract_table(params, context, event_queue)

    assert result["status"] == "error"
    assert "matches 'the rainfall data'" in result["error"]
    assert "Table 1. Registry" in result["error"] and "Table 3. Daily" in result["error"]
    extract_agent.send_json_artifact.assert_not_awaited()


async def test_bad_inputs_are_refused_before_asking_the_llm(extract_agent, context, event_queue, tmp_path):
    no_tables = tmp_path / "notes.md"
    no_tables.write_text("# Notes\n\nNo tables here.\n")
    binary = tmp_path / "scan.md"
    binary.write_bytes(b"\xff\xfe\x00not text")

    missing_description = await extract_agent._extract_table(
        {"document_artifact_id": "doc"}, context, event_queue
    )
    extract_agent.get_artifact_file_path.return_value = FIXTURES / "cheese_mouse_sightings.csv"
    csv_file = await extract_agent._extract_table(
        {"document_artifact_id": "doc", "table_description": "x"}, context, event_queue
    )
    extract_agent.get_artifact_file_path.return_value = no_tables
    empty = await extract_agent._extract_table(
        {"document_artifact_id": "doc", "table_description": "x"}, context, event_queue
    )
    extract_agent.get_artifact_file_path.return_value = binary
    unreadable = await extract_agent._extract_table(
        {"document_artifact_id": "doc", "table_description": "x"}, context, event_queue
    )
    extract_agent.get_artifact_file_path.side_effect = FileNotFoundError
    unknown = await extract_agent._extract_table(
        {"document_artifact_id": "nope", "table_description": "x"}, context, event_queue
    )

    assert "table_description is required" in missing_description["error"]
    assert "not a markdown document" in csv_file["error"] and "team_5_plot_table" in csv_file["error"]
    assert "No tables were found" in empty["error"]
    assert "not a readable text file" in unreadable["error"]
    assert "'nope' was not found" in unknown["error"]
    extract_agent._pick_table.assert_not_awaited()


async def test_pick_table_asks_the_small_llm_and_checks_its_answer(monkeypatch, context):
    agent = Team5TablePlotAgent(agent_name="Team-5 Table Plot Agent")
    tables = find_tables(REPORT.read_text())

    @asynccontextmanager
    async def fake_client(_context, config):
        yield "llm"

    calls = AsyncMock(return_value=SimpleNamespace(arguments={"index": 1, "reason": "Census."}))
    monkeypatch.setattr(agent, "llm_client", fake_client)
    monkeypatch.setattr(agent_module, "acall_tool", calls)

    assert await agent._pick_table(context, "the population data", tables) == (1, "Census.")
    prompt = calls.await_args.args[1]
    assert calls.await_args.kwargs["model"] == "small"
    assert '"the population data"' in prompt
    assert "[2] Section: 4. The Great Wheel of Emmental" in prompt
    assert "Columns: Year, Crater Gouda (residents)" in prompt

    # Anything that is not the index of a real table counts as "none fits".
    for answer in (-1, 3, "1", True, None):
        calls.return_value = SimpleNamespace(arguments={"index": answer})
        index, _ = await agent._pick_table(context, "x", tables)
        assert index == -1


async def test_the_plotter_reads_the_extracted_table(extract_agent, context, event_queue, tmp_path, monkeypatch):
    await extract_agent._extract_table(
        {"document_artifact_id": "doc", "table_description": "population"}, context, event_queue
    )
    # The platform materializes an inline JSON artifact as <name>.json.
    table_file = tmp_path / "table_2_census.json"
    table_file.write_text(json.dumps(extract_agent.send_json_artifact.await_args.args[2]))

    extract_agent.get_artifact_file_path.return_value = table_file
    monkeypatch.setattr(extract_agent, "get_provided_inputs", MagicMock(return_value={}))
    monkeypatch.setattr(extract_agent, "load_context_state", lambda _: {})
    monkeypatch.setattr(extract_agent, "save_context_state", lambda *_: None)
    monkeypatch.setattr("agent.ConnectionConfig.from_env", MagicMock(return_value=object()))
    upload = AsyncMock(return_value=SimpleNamespace(artifact_id="plot-1"))
    monkeypatch.setattr(extract_agent, "send_deeplynx_artifact", upload)

    result = await extract_agent._plot_table(
        {"table_artifact_id": "table-1", "x_column": "Year", "y_column": "Brie Basin (residents)"},
        context,
        event_queue,
    )

    assert result["status"] == "success"
    assert upload.await_args.kwargs["description"] == "Brie Basin (residents) vs Year"
    assert upload.await_args.kwargs["metadata"]["source_caption"].startswith("Table 2. Census")


# Malformed tables: the markdown rules would pad, cut or end them quietly.
HEADER = "**Table 2. Census.**\n\n| Year | Gouda | Brie | Mice |\n|---|---|---|---|\n"


@pytest.mark.parametrize(
    ("row", "cells"),
    [
        ("| 2042 | 412 | 350 |", 3),  # a missing cell would shift values left
        ("| 2042 | 412 | 188 | 350 | 99 |", 5),  # an extra cell would be dropped
        ("| 2042 | 412 | 1|88 | 350 |", 5),  # an unescaped pipe splits a cell
    ],
)
def test_rows_with_the_wrong_number_of_cells_are_flagged(row, cells):
    (table,) = find_tables(HEADER + row + "\n")

    assert table["problem"] == f"row 1 (line 5) has {cells} cells but the header has 4"


def test_a_blank_line_inside_a_table_is_flagged():
    text = HEADER + "| 2042 | 412 | 188 | 350 |\n\n| 2043 | 467 | 203 | 420 |\n"

    (table,) = find_tables(text)

    assert table["problem"].startswith("a blank line at line 6 cuts it short")


def test_well_formed_tables_have_no_problem():
    assert [table["problem"] for table in find_tables(REPORT.read_text())] == ["", "", ""]
    # An escaped pipe belongs to its cell, and prose may follow a table.
    (table,) = find_tables(HEADER + "| 2042 | 4\\|12 | 188 | 350 |\n\nSome prose.\n")
    assert table["problem"] == ""


async def test_a_malformed_table_is_an_error_only_when_picked(extract_agent, context, event_queue, tmp_path):
    document = tmp_path / "report.md"
    document.write_text(HEADER + "| 2042 | 412 | 350 |\n\n**Table 3. Other.**\n\n| A |\n|---|\n| 1 |\n")
    extract_agent.get_artifact_file_path.return_value = document
    params = {"document_artifact_id": "doc", "table_description": "the census"}

    extract_agent._pick_table.return_value = (0, "Census.")
    broken = await extract_agent._extract_table(params, context, event_queue)
    extract_agent._pick_table.return_value = (1, "Other.")
    fine = await extract_agent._extract_table(params, context, event_queue)

    assert broken["status"] == "error"
    assert "'Table 2. Census.'" in broken["error"] and "not well-formed markdown" in broken["error"]
    assert "row 1 (line 5) has 3 cells but the header has 4" in broken["error"]
    assert fine["status"] == "success" and fine["caption"] == "Table 3. Other."

"""Exercise CSV/JSON loading and real rendering with platform I/O mocked."""

import csv
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from matplotlib import image as mpimg
from matplotlib.figure import Figure

from agent import Team5TablePlotAgent, _csv_problem

FIXTURES = Path(__file__).parent
OUTPUT = FIXTURES / "output"
COLUMNS = [
    "night", "mice_sighted", "cheese_lost_kg", "tunnel_temperature_C",
    "traps_set", "traps_sprung",
]


@pytest.fixture(params=["csv", "json"])
def table_path(request):
    return FIXTURES / f"cheese_mouse_sightings.{request.param}"


@pytest.fixture
def plot_agent(monkeypatch, table_path, request):
    agent = Team5TablePlotAgent(agent_name="Team-5 Table Plot Agent")
    monkeypatch.setattr(agent, "get_artifact_file_path", AsyncMock(return_value=table_path))
    monkeypatch.setattr(agent, "get_provided_inputs", MagicMock(return_value={}))
    monkeypatch.setattr(agent, "send_input_required", AsyncMock())
    state = {}

    def save_state(context_id, value):
        state.clear()
        state.update(value)

    monkeypatch.setattr(agent, "load_context_state", lambda _: state.copy())
    monkeypatch.setattr(agent, "save_context_state", save_state)
    monkeypatch.setattr("agent.ConnectionConfig.from_env", MagicMock(return_value=object()))

    # Keep the real Figure so tests can inspect the plotted values and labels.
    figures = []

    def make_figure(*args, **kwargs):
        figure = Figure(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr("agent.Figure", make_figure)
    uploads = []

    async def upload(context, event_queue, **kwargs):
        path = kwargs["file_path"]
        # Preserve a preview before the handler removes its temporary PNG.
        OUTPUT.mkdir(exist_ok=True)
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name)
        shutil.copyfile(path, OUTPUT / f"{filename}.png")
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        pixels = mpimg.imread(path)
        assert pixels.shape[0] > 0 and pixels.shape[1] > 0
        assert pixels[:, :, :3].min() < 1  # Image contains visible marks.
        uploads.append(kwargs)
        return SimpleNamespace(artifact_id="plot-result")

    monkeypatch.setattr(agent, "send_deeplynx_artifact", AsyncMock(side_effect=upload))
    return SimpleNamespace(agent=agent, figures=figures, uploads=uploads, state=state)


def test_json_fixture_matches_csv():
    with (FIXTURES / "cheese_mouse_sightings.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        columns = next(reader)
        rows = [[json.loads(value) for value in row] for row in reader]
    table = json.loads((FIXTURES / "cheese_mouse_sightings.json").read_text())
    assert table["caption"]
    assert table["columns"] == columns
    assert table["rows"] == rows


@pytest.mark.parametrize("kind", ["line", "scatter", "bar"])
async def test_plot_data_and_png(plot_agent, context, event_queue, kind):
    agent = plot_agent.agent
    result = await agent._plot_table({
        "table_artifact_id": "mouse-table", "x_column": "night",
        "y_column": "mice_sighted", "kind": kind, "title": "Mouse sightings",
    }, context, event_queue)

    assert result == {
        "status": "success", "artifact_id": "plot-result",
        "x_column": "night", "y_column": "mice_sighted", "kind": kind,
    }
    agent.get_artifact_file_path.assert_awaited_once_with(
        "mouse-table", context=context, artifacts=None,
    )
    axes = plot_agent.figures[0].axes[0]
    expected_y = [14, 19, 23, 31, 28, 35, 42, 39, 47, 52, 49, 58, 61, 66]
    if kind == "line":
        assert list(axes.lines[0].get_xdata()) == list(range(1, 15))
        assert list(axes.lines[0].get_ydata()) == expected_y
    elif kind == "scatter":
        expected_points = [[x, y] for x, y in zip(range(1, 15), expected_y)]
        assert axes.collections[0].get_offsets().tolist() == expected_points
    else:
        assert [bar.get_height() for bar in axes.patches] == expected_y
    assert axes.get_xlabel() == "night"
    assert axes.get_ylabel() == "mice_sighted"
    assert axes.get_title() == "Mouse sightings"
    assert plot_agent.uploads[0]["description"] == "Mouse sightings"
    assert plot_agent.uploads[0]["mime_type"] == "image/png"
    assert not plot_agent.uploads[0]["file_path"].exists()  # Cleaned up after upload.


async def test_decimal_values_and_column_matching(plot_agent, context, event_queue):
    result = await plot_agent.agent._plot_table({
        "table_artifact_id": "mouse-table", "x_column": "NIGHT", "y_column": "temperature",
    }, context, event_queue)
    assert result["y_column"] == "tunnel_temperature_C"
    values = plot_agent.figures[0].axes[0].lines[0].get_ydata()
    assert list(values[:3]) == [4.1, 4.3, 4.0]


@pytest.mark.parametrize("title", [None, "", "   "], ids=["omitted", "empty", "whitespace"])
async def test_default_title_describes_selected_columns(
    plot_agent, table_path, context, event_queue, title,
):
    params = {
        "table_artifact_id": "mouse-table", "x_column": "night", "y_column": "temperature",
    }
    if title is not None:
        params["title"] = title
    result = await plot_agent.agent._plot_table(params, context, event_queue)

    assert result["status"] == "success"
    expected_title = "tunnel_temperature_C vs night"
    assert plot_agent.figures[0].axes[0].get_title() == expected_title
    upload = plot_agent.uploads[0]
    assert upload["description"] == expected_title
    assert upload["metadata"]["source_artifact_id"] == "mouse-table"
    expected_caption = (
        json.loads(table_path.read_text())["caption"] if table_path.suffix == ".json" else ""
    )
    assert upload["metadata"]["source_caption"] == expected_caption


async def test_bar_chart_with_text_categories(plot_agent, table_path, context, event_queue):
    agent = plot_agent.agent
    agent.get_artifact_file_path.return_value = FIXTURES / f"cheese_inventory{table_path.suffix}"
    result = await agent._plot_table({
        "table_artifact_id": "inventory", "x_column": "cheese",
        "y_column": "stock_kg", "kind": "bar",
    }, context, event_queue)

    assert result["status"] == "success"
    assert result["kind"] == "bar"
    axes = plot_agent.figures[0].axes[0]
    assert [label.get_text() for label in axes.get_xticklabels()] == ["Gouda", "Brie", "Cheddar"]
    assert [bar.get_height() for bar in axes.patches] == [1020, 650.5, 0]
    assert axes.get_xlabel() == "cheese"
    assert axes.get_ylabel() == "stock_kg"


@pytest.mark.parametrize("case", ["repeated-label", "missing-categories"])
async def test_category_bar_rows(
    plot_agent, table_path, tmp_path, context, event_queue, case,
):
    # Repeated labels are separate observations, not an instruction to aggregate.
    if case == "repeated-label":
        rows = [["Gouda", 10], ["Brie", 20], ["Gouda", 30]]
        labels, heights = ["Gouda", "Brie", "Gouda"], [10, 20, 30]
    else:
        rows = [[" Gouda ", 10], [None, 20], ["—", 30], ["  ", 40], ["Brie", 50]]
        labels, heights = ["Gouda", "Brie"], [10, 50]

    path = tmp_path / f"categories{table_path.suffix}"
    columns = ["cheese", "stock_kg"]
    if path.suffix == ".csv":
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
    else:
        path.write_text(json.dumps({"caption": "Inventory", "columns": columns, "rows": rows}))
    plot_agent.agent.get_artifact_file_path.return_value = path
    result = await plot_agent.agent._plot_table({
        "table_artifact_id": "categories", "x_column": "cheese",
        "y_column": "stock_kg", "kind": "bar",
    }, context, event_queue)

    assert result["status"] == "success"
    axes = plot_agent.figures[0].axes[0]
    assert [label.get_text() for label in axes.get_xticklabels()] == labels
    assert [bar.get_height() for bar in axes.patches] == heights
    centers = [bar.get_x() + bar.get_width() / 2 for bar in axes.patches]
    assert centers == list(range(len(labels)))


async def test_missing_columns_form_and_resume(plot_agent, context, event_queue):
    agent = plot_agent.agent
    result = await agent._plot_table({"table_artifact_id": "mouse-table"}, context, event_queue)
    assert result == {}
    fields = agent.send_input_required.await_args.kwargs["missing_inputs"]
    assert [field.name for field in fields] == ["x_column", "y_column"]
    assert all(field.options == COLUMNS for field in fields)
    agent.send_deeplynx_artifact.assert_not_awaited()

    agent.get_provided_inputs.return_value = {"x_column": "night", "y_column": "mice_sighted"}
    result = await agent._plot_table({}, context, event_queue)
    assert result["status"] == "success"
    assert "pending_plot" not in plot_agent.state


@pytest.mark.parametrize("params, expected", [
    ({"x_column": "unknown", "y_column": "mice_sighted"}, "Available columns"),
    ({"x_column": "night", "y_column": "traps"}, "Cannot uniquely match"),
    ({"kind": "pie"}, "kind must be"),
])
async def test_invalid_options(plot_agent, context, event_queue, params, expected):
    result = await plot_agent.agent._plot_table(
        {"table_artifact_id": "mouse-table", **params}, context, event_queue,
    )
    assert result["status"] == "error"
    assert expected in result["error"]
    plot_agent.agent.send_deeplynx_artifact.assert_not_awaited()


async def test_missing_file(plot_agent, context, event_queue):
    plot_agent.agent.get_artifact_file_path.side_effect = FileNotFoundError
    result = await plot_agent.agent._plot_table(
        {"table_artifact_id": "missing"}, context, event_queue,
    )
    assert result["status"] == "error"
    assert "Could not read" in result["error"]
    plot_agent.agent.send_deeplynx_artifact.assert_not_awaited()


@pytest.mark.parametrize("contents, suffix, expected", [
    ("night,count\n", ".csv", "no data rows"),
    ("night,count\n1,mouse\n", ".csv", "numeric values"),
    ('{"columns": ["night"]}', ".json", "columns' and 'rows"),
    ("not JSON", ".json", "Could not read"),
    ("not a table", ".pdf", "CSV file or a JSON table"),
])
async def test_bad_input_data(plot_agent, tmp_path, context, event_queue, contents, suffix, expected):
    path = tmp_path / f"bad{suffix}"
    path.write_text(contents)
    plot_agent.agent.get_artifact_file_path.return_value = path
    result = await plot_agent.agent._plot_table({
        "table_artifact_id": "bad-table", "x_column": "night", "y_column": "count",
    }, context, event_queue)
    assert result["status"] == "error"
    assert expected in result["error"]
    plot_agent.agent.send_deeplynx_artifact.assert_not_awaited()


@pytest.mark.parametrize("contents, problem", [
    # A field too few or too many: pandas would pad the row or shift whole columns.
    ("night,mice,temp\n1,14,4.1\n2,4.3\n", "row 2 (line 3) has 2 fields but the header has 3"),
    ("night,mice,temp\n1,14,4.1,99\n2,19,4.3\n", "row 1 (line 2) has 4 fields but the header has 3"),
    ("night,mice,temp\n1,14,4.1,\n2,19,4.3,\n", "row 1 (line 2) has 4 fields but the header has 3"),
    ("year,population\n2042,1,019\n", "row 1 (line 2) has 3 fields but the header has 2"),
    ("night,mice\n1,14\n\n2,19,7\n", "row 2 (line 4) has 3 fields but the header has 2"),
    ("night,mice,temp\n1,14,4.1\noops\n", "row 2 (line 3) has 1 field but the header has 3"),
    ("Mouse sightings\nnight,mice\n1,14\n", "row 1 (line 2) has 2 fields but the header has 1"),
    # Read as one column, or under column names pandas would invent.
    ("night;temp\n1;4,1\n", "its columns are separated by semicolons, not commas"),
    ("night\tmice\n1\t14\n", "its columns are separated by tabs, not commas"),
    ("night,temp,temp \n1,4.1,9.0\n", "more than one column is named 'temp'"),
    ("1,14,4.1\n2,19,4.3\n", "its first row is data, not column names"),
    # Not readable as CSV text.
    ("", "it is empty"),
    ("\n  \n", "it is empty"),
    ('night,mice\n1,"14\n', "it is not valid CSV (unexpected end of data)"),
    (
        "night,temp \xb0C\n1,4.1\n".encode("latin-1"),
        'it is not UTF-8 text (in Excel, save it as "CSV UTF-8")',
    ),
])
def test_csv_problems_are_named(tmp_path, contents, problem):
    path = tmp_path / "data.csv"
    path.write_bytes(contents if isinstance(contents, bytes) else contents.encode())
    assert _csv_problem(path) == problem


@pytest.mark.parametrize("contents", [
    'night,mice\n1,"1,014"\n',  # a quoted thousands separator
    "night,mice,,\n1,14,,\n",  # trailing commas: blank column names, not repeated ones
    "﻿night,mice\r\n1,14\r\n\r\n  \r\n2,19\r\n",  # BOM, CRLF, blank lines
    "night,mice\n1,14\n2,\n",  # an empty field is a missing value
    "night,2042\n1,14\n",  # a number may be one of the column names
])
def test_well_formed_csv_has_no_problem(tmp_path, contents):
    path = tmp_path / "data.csv"
    path.write_bytes(contents.encode())
    assert _csv_problem(path) == ""


async def test_a_malformed_csv_is_an_error_not_a_shifted_plot(
    plot_agent, tmp_path, context, event_queue,
):
    path = tmp_path / "sightings.csv"
    path.write_text("night,mice,temp\n1,14,4.1\n2,4.3\n")  # pandas would plot 4.3 mice on night 2
    plot_agent.agent.get_artifact_file_path.return_value = path
    result = await plot_agent.agent._plot_table({
        "table_artifact_id": "sightings", "x_column": "night", "y_column": "mice",
    }, context, event_queue)
    assert result == {
        "status": "error",
        "error": "The CSV file cannot be read as a table: row 2 (line 3) has 2 fields "
                 "but the header has 3. Fix the file and upload it again.",
    }
    plot_agent.agent.send_deeplynx_artifact.assert_not_awaited()


@pytest.mark.parametrize("contents, expected_y", [
    ('night,mice\n1,"1,014"\n2,"1,019"\n', [1014, 1019]),
    ("night,mice,,\n1,14,,\n2,19,,\n", [14, 19]),
    ("﻿night,mice\r\n1,14\r\n\r\n  \r\n2,19\r\n", [14, 19]),
])
async def test_csv_quirks_that_are_not_errors_still_plot(
    plot_agent, tmp_path, context, event_queue, contents, expected_y,
):
    path = tmp_path / "sightings.csv"
    path.write_bytes(contents.encode())
    plot_agent.agent.get_artifact_file_path.return_value = path
    result = await plot_agent.agent._plot_table({
        "table_artifact_id": "sightings", "x_column": "night", "y_column": "mice",
    }, context, event_queue)
    assert result["status"] == "success"
    assert list(plot_agent.figures[0].axes[0].lines[0].get_ydata()) == expected_y


@pytest.mark.parametrize("partial", [False, True], ids=["complete-form", "partial-form"])
async def test_real_form_answers_survive_replayed_call(
    plot_agent, context, event_queue, monkeypatch, partial,
):
    """Use actual A2A data parts, as the orchestrator sends on resume."""
    from a2a.types import Message, Part
    from google.protobuf.json_format import ParseDict
    from google.protobuf.struct_pb2 import Value

    agent = plot_agent.agent
    monkeypatch.setattr(
        agent, "get_provided_inputs",
        Team5TablePlotAgent.get_provided_inputs.__get__(agent),
    )
    context.message = Message()
    params = {
        "table_artifact_id": "mouse-table", "x_column": None, "y_column": "",
    }
    assert await agent._plot_table(params, context, event_queue) == {}

    def answer(values):
        context.message = Message(parts=[Part(data=ParseDict({"provided_inputs": values}, Value()))])

    if partial:
        answer({"x_column": "night"})
        assert await agent._plot_table(params, context, event_queue) == {}
        fields = agent.send_input_required.await_args.kwargs["missing_inputs"]
        assert [field.name for field in fields] == ["y_column"]
        answer({"y_column": "mice_sighted"})
    else:
        answer({"x_column": "night", "y_column": "mice_sighted"})

    result = await agent._plot_table(params, context, event_queue)
    assert result["status"] == "success"
    assert agent.send_input_required.await_count == (2 if partial else 1)

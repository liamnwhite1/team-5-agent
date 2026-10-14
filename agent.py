# This file is part of Prometheus
# https://inl.gov/super-intelligence/prometheus/
#
# Copyright 2026 Battelle Energy Alliance, LLC
# All rights reserved, see NOTICE.txt for full restrictions
# https://github.com/idaholab/prometheus_core/blob/main/NOTICE.txt
#
# Licensed under the MIT License, please see LICENSE for details
# https://github.com/idaholab/prometheus_core/blob/main/LICENSE
#
# SPDX-License-Identifier: MIT

"""Team-5 agent for extracting tables and plotting tabular artifacts.

Extracting a table: markdown-it-py finds every table in the document together
with the caption written above it, and the small LLM picks the one the user
described. Plotting: see ``_plot_table``.
"""

import json
import logging
import math
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from markdown_it import MarkdownIt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from prometheus_api.agent import EventQueue, MissingInput, PrometheusAgent, RequestContext
from prometheus_api.llm_client import LLMConfig, ToolDefinition, acall_tool
from prometheus_api.storage_client.connection import ConnectionConfig
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

EXTRACT_SKILL = "team_5_extract_table"
PLOT_SKILL = "team_5_plot_table"

# Uploads the extract skill can read as markdown.
MARKDOWN_SUFFIXES = {".md", ".markdown", ".txt"}
# Cell text that means "no value".
MISSING_MARKERS = {"", "—", "–", "-", "NA", "N/A", "nan", "None"}
# Block types that separate a caption from the table below it.
CAPTION_BREAKERS = {
    "table_close", "bullet_list_open", "ordered_list_open", "blockquote_open",
    "fence", "code_block", "hr", "html_block",
}

PICK_TABLE_PROMPT = """\
A user wants one table from a document. They described it as: "{description}"

The tables in the document, by index:
{listing}

Call pick_table with the index of the table the user means. Match on meaning, \
not exact wording. If no table fits the description, use -1."""


class TablePick(BaseModel):
    index: int = Field(description="Index of the table the user means, or -1 if none fits.")
    reason: str = Field(description="One short sentence explaining the choice.")


PICK_TABLE_TOOL = ToolDefinition.from_pydantic(
    "pick_table", TablePick, description="Choose the table the user described."
)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error(message: str) -> dict[str, Any]:
    """A problem the user or the orchestrator can fix: reported, not raised.

    The orchestrator hands the model only the ``error`` text, so everything it
    needs to correct the call (real captions, real column names) goes in there.
    """
    return {"status": "error", "error": message}


def _plain_text(tokens: list) -> str:
    """The visible text of inline markdown tokens, without formatting marks."""
    parts = [
        token.content if token.type in ("text", "code_inline") else " "
        for token in tokens
        if token.type in ("text", "code_inline", "softbreak", "hardbreak")
    ]
    return " ".join("".join(parts).split())


def _caption(paragraph) -> str:
    """The bold text a paragraph starts with, or the whole paragraph if not bold.

    Captions look like "**Table 2. Census of ...** An optional note."
    """
    children = [token for token in paragraph.children or [] if token.content or token.type != "text"]
    if children and children[0].type == "strong_open":
        end = next(i for i, token in enumerate(children) if token.type == "strong_close")
        return _plain_text(children[1:end])
    return _plain_text(children)


def _cells_as_written(line: str) -> int:
    """How many cells a table row has in the source, before the parser pads or cuts it.

    Cells are split on pipes, except escaped ones (\\|) and the optional outer pipes.
    """
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    return len(re.split(r"(?<!\\)\|", text))


def find_tables(markdown: str) -> list[dict[str, Any]]:
    """Every table in a markdown document, with the caption written above it.

    Each table is a dict with its caption, section (the heading it sits under),
    columns and rows. Cells keep the text as written in the document.

    The markdown rules quietly pad a short row, cut a long one, and end a
    table at a blank line, which would hand back wrong numbers without any
    error. Those cases are recorded in the table's "problem" instead.
    """
    lines = markdown.splitlines()
    tokens = MarkdownIt("commonmark").enable("table").parse(markdown)
    tables: list[dict[str, Any]] = []
    section = caption = ""
    row = row_line = None
    for i, token in enumerate(tokens):
        if token.type == "table_open":
            tables.append(
                {"caption": caption, "section": section, "columns": [], "rows": [], "problem": ""}
            )
        elif token.type == "tr_open":
            row, row_line = [], token.map[0]
        elif token.type == "inline" and row is not None:  # one table cell
            row.append(_plain_text(token.children or []))
        elif token.type == "tr_close":
            table = tables[-1]
            if not table["columns"]:
                table["columns"] = row
            else:
                table["rows"].append(row)
                written = _cells_as_written(lines[row_line])
                if written != len(table["columns"]) and not table["problem"]:
                    table["problem"] = (
                        f"row {len(table['rows'])} (line {row_line + 1}) has {written} cells "
                        f"but the header has {len(table['columns'])}"
                    )
            row = None
        elif token.level == 0 and token.type == "heading_open":
            section = caption = _plain_text(tokens[i + 1].children or [])
        elif token.level == 0 and token.type == "paragraph_open":
            first_line = lines[token.map[0]].lstrip()
            if tokens[i - 1].type == "table_close" and first_line.startswith("|"):
                tables[-1]["problem"] = tables[-1]["problem"] or (
                    f"a blank line at line {token.map[0]} cuts it short, so the rows "
                    "after it are not part of the table"
                )
            caption = _caption(tokens[i + 1])
        if token.level == 0 and token.type in CAPTION_BREAKERS:
            caption = ""  # a caption belongs only to the table right below it
    return tables


def _unique_names(names: list[str]) -> list[str]:
    """Name blank headers and number repeated ones: the plotter needs unique names."""
    unique: list[str] = []
    for position, name in enumerate(names, start=1):
        name = name or f"Column {position}"
        candidate, count = name, 2
        while candidate in unique:
            candidate, count = f"{name} ({count})", count + 1
        unique.append(candidate)
    return unique


def _clean_cell(text: str) -> int | float | str | None:
    """'1,020' -> 1020 and '—' -> None. Text that is not a number stays as it is."""
    if text in MISSING_MARKERS:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return text
    if not math.isfinite(number):
        return text
    return int(number) if number.is_integer() and "." not in text else number


def _numeric_columns(columns: list[str], rows: list[list[Any]]) -> list[str]:
    """Columns whose values are all numbers or missing, with at least one number."""
    numeric = []
    for i, column in enumerate(columns):
        values = [row[i] for row in rows if row[i] is not None]
        if values and all(isinstance(value, (int, float)) for value in values):
            numeric.append(column)
    return numeric


def _artifact_name(caption: str, position: int) -> str:
    """A short name with no file extension, so the plotter reads the table as JSON."""
    slug = re.sub(r"[^a-z0-9]+", "_", caption.lower()).strip("_")[:50].rstrip("_")
    return slug or f"table_{position}"


class Team5TablePlotAgent(PrometheusAgent):
    """Team-5 Table Plot Agent."""

    async def _execute(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        """Route a request to the selected skill handler."""
        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Processing your request…",
        )

        # Dispatch on the skill the orchestrator selected.
        handlers = {
            EXTRACT_SKILL: "_extract_table",
            PLOT_SKILL: "_plot_table",
        }
        skill_id = _text(params.get("skill_id"))
        if not skill_id:
            # A direct call without routing metadata (the /api/chat debug route).
            skill_id = EXTRACT_SKILL if params.get("document_artifact_id") else PLOT_SKILL
        if skill_id not in handlers:
            return _error(f"Unknown skill '{skill_id}'. This agent offers: {', '.join(handlers)}.")
        handler = getattr(self, handlers[skill_id], None)
        if handler is None:
            return _error(f"Skill '{skill_id}' is not implemented yet.")

        return await handler(params, context, event_queue)

    async def _extract_table(
        self, params: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:
        """Find the table the user described in a markdown document and publish it as JSON."""
        document_id = _text(params.get("document_artifact_id"))
        description = _text(params.get("table_description"))
        if not document_id:
            return _error("document_artifact_id is required (the uploaded markdown document).")
        if not description:
            return _error("table_description is required: how the user referred to the table.")

        try:
            path = await self.get_artifact_file_path(
                document_id, context=context, artifacts=params.get("artifacts")
            )
        except FileNotFoundError:
            return _error(f"Artifact '{document_id}' was not found in this conversation.")
        reference = self.resolve_artifact_reference(str(context.context_id), document_id)
        source = reference.name if reference else path.name
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            return _error(
                f"'{source}' is not a markdown document. "
                f"A CSV file can be plotted directly with {PLOT_SKILL}."
            )
        try:
            tables = [table for table in find_tables(path.read_text(encoding="utf-8")) if table["rows"]]
        except UnicodeDecodeError:
            return _error(f"'{source}' is not a readable text file.")
        if not tables:
            return _error(f"No tables were found in '{source}'.")

        await self.send_status_update(
            context,
            event_queue,
            message=f"Found {len(tables)} table(s); asking which one matches '{description}'.",
        )
        index, reason = await self._pick_table(context, description, tables)
        if index < 0:
            captions = "; ".join(
                table["caption"] or f"table {position} (no caption)"
                for position, table in enumerate(tables, start=1)
            )
            return _error(f"No table in '{source}' matches '{description}'. Its tables are: {captions}")

        table = tables[index]
        caption = table["caption"]
        if table["problem"]:
            return _error(
                f"The table '{caption or f'table {index + 1}'}' in '{source}' is not well-formed "
                f"markdown: {table['problem']}. Fix the table and upload the document again."
            )
        columns = _unique_names(table["columns"])
        rows = [[_clean_cell(cell) for cell in row] for row in table["rows"]]
        artifact_id = await self.send_json_artifact(
            context,
            event_queue,
            {"caption": caption, "columns": columns, "rows": rows},
            name=_artifact_name(caption, index + 1),
            description=caption or f"Table {index + 1} of {source}",
            catalog_artifact=True,
        )
        return {
            "status": "success",
            "table_artifact_id": artifact_id,
            "caption": caption,
            "columns": columns,
            "numeric_columns": _numeric_columns(columns, rows),
            "row_count": len(rows),
            "why_this_table": reason,
        }

    async def _pick_table(
        self, context: RequestContext, description: str, tables: list[dict[str, Any]]
    ) -> tuple[int, str]:
        """Ask the small LLM which table the description means.

        Returns (index, reason), with index -1 when no table fits. If the LLM
        cannot be reached the error is left to raise: that is infrastructure,
        not something the user can fix.
        """
        listing = "\n".join(
            f"[{i}] Section: {table['section'] or '-'} | "
            f"Caption: {table['caption'][:300] or '(none)'} | "
            f"Columns: {', '.join(table['columns'])}"
            for i, table in enumerate(tables)
        )
        config = LLMConfig.from_env()
        if config.proxy_url:  # only the platform's own LiteLLM proxy needs the master key
            config = config.with_api_key(os.getenv("LITELLM_MASTER_KEY"))
        async with self.llm_client(context, config=config) as llm:
            call = await acall_tool(
                llm,
                PICK_TABLE_PROMPT.format(description=description, listing=listing),
                tool=PICK_TABLE_TOOL,
                model="small",
                max_tokens=400,
            )
        index = call.arguments.get("index")
        if type(index) is not int or not 0 <= index < len(tables):
            index = -1  # "none fits", or an answer that names no real table
        return index, _text(call.arguments.get("reason"))

    async def _plot_table(
        self, params: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> dict[str, Any]:
        """Load CSV or JSON {columns, rows}, ask for axes, and publish a PNG."""
        ctx = str(context.context_id)
        state = self.load_context_state(ctx)
        provided = self.get_provided_inputs(context)
        # Retain the file and any chosen options while the user answers a form.
        pending = state.get("pending_plot", {})
        incoming_id = params.get("table_artifact_id")
        if incoming_id and incoming_id != pending.get("table_artifact_id"):
            pending = {}
            state.pop("pending_plot", None)
            self.save_context_state(ctx, state)
        # Resumes replay the original call, which may still contain empty axes.
        # Those empty placeholders must not erase answers from an earlier form.
        options = dict(pending)
        for values in (params, provided):
            for key, value in values.items():
                if key in {"table_artifact_id", "x_column", "y_column", "title", "kind"} and not _text(value):
                    continue
                options[key] = value
        table_id = _text(options.get("table_artifact_id"))
        if not table_id:
            return _error("table_artifact_id is required (a CSV or JSON table artifact).")
        kind = options.get("kind") or "line"
        if kind not in {"line", "scatter", "bar"}:
            return _error("kind must be line, scatter, or bar.")

        try:
            path = await self.get_artifact_file_path(
                table_id, context=context, artifacts=options.get("artifacts")
            )
            caption = ""
            if path.suffix.lower() == ".json":
                body = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(body, dict) or "columns" not in body or "rows" not in body:
                    return _error("JSON table must contain 'columns' and 'rows'.")
                frame = pd.DataFrame(body["rows"], columns=body["columns"])
                caption = _text(body.get("caption"))
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            else:
                return _error("Please provide a CSV file or a JSON table with columns and rows.")
        except (OSError, ValueError, TypeError):
            return _error("Could not read the table artifact. Check the file and artifact ID.")
        if frame.empty:
            return _error("The table has no data rows to plot.")
        columns = list(frame.columns)
        if (
            any(not isinstance(c, str) or not c.strip() for c in columns)
            or len(set(columns)) != len(columns)
        ):
            return _error("The table must have unique, non-empty text column names.")

        missing = []
        selected = {}
        for key, label in (("x_column", "x axis"), ("y_column", "y axis")):
            requested = _text(options.get(key))
            if not requested:
                missing.append(
                    MissingInput(
                        name=key,
                        description=f"Column for the {label}",
                        user_prompt=f"Which column goes on the {label}?",
                        type="enum",
                        options=columns,
                    )
                )
                continue
            # Exact names preserve meaningful case distinctions such as t/T.
            matches = [c for c in columns if requested.casefold() in c.casefold()]
            if requested in columns:
                selected[key] = requested
            elif len(matches) == 1:
                selected[key] = matches[0]
            else:
                return _error(f"Cannot uniquely match '{requested}'. Available columns: {', '.join(columns)}.")
        if missing:
            state["pending_plot"] = options
            self.save_context_state(ctx, state)
            await self.send_input_required(
                context, event_queue, missing_inputs=missing, hint="Choose the columns to plot."
            )
            return {}

        x_col, y_col = selected["x_column"], selected["y_column"]
        # Thousands separators and common missing markers occur in the practice data.
        y = pd.to_numeric(frame[y_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
        y = y.replace([float("inf"), float("-inf")], float("nan"))
        invalid = y.isna() & ~frame[y_col].fillna("").astype(str).str.strip().isin(
            ["", "—", "–", "-", "NA", "N/A", "nan", "None"]
        )
        if invalid.any() or not y.notna().any():
            return _error(f"Column '{y_col}' must contain numeric values. Available columns: {', '.join(columns)}.")
        x_raw = frame[x_col].replace(["", "—", "–", "-", "NA", "N/A"], float("nan"))
        x_numeric = pd.to_numeric(x_raw.astype(str).str.replace(",", "", regex=False), errors="coerce")
        x = x_numeric if x_numeric.notna().sum() == x_raw.notna().sum() else x_raw
        x = x.replace([float("inf"), float("-inf")], float("nan"))
        valid = x.notna() & y.notna()
        if not valid.any():
            return _error("The selected columns have no rows with both x and y values.")

        title = _text(options.get("title")) or caption or f"{y_col} vs {x_col}"
        figure = Figure(figsize=(8, 5), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        getattr(axes, "plot" if kind == "line" else kind)(x[valid], y[valid])
        axes.set(xlabel=x_col, ylabel=y_col, title=title)
        axes.tick_params(axis="x", labelrotation=30)
        try:
            storage_config = ConnectionConfig.from_env()
        except ValueError:
            return _error("DeepLynx must be configured to publish plot images.")
        with TemporaryDirectory(prefix="team5-plot-") as directory:
            output = Path(directory) / "plot.png"
            figure.savefig(output, format="png", dpi=150)
            artifact = await self.send_deeplynx_artifact(
                context,
                event_queue,
                file_path=output,
                name="plot.png",
                description=title,
                storage_config=storage_config,
                mime_type="image/png",
            )
        state.pop("pending_plot", None)
        self.save_context_state(ctx, state)
        return {
            "status": "success",
            "artifact_id": artifact.artifact_id,
            "x_column": x_col,
            "y_column": y_col,
            "kind": kind,
        }

if __name__ == "__main__":
    agent = Team5TablePlotAgent(agent_name="Team-5 Table Plot Agent")
    agent.run()

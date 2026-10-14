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

"""Team-5 agent for extracting tables and plotting tabular artifacts."""

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from prometheus_api.agent import EventQueue, MissingInput, PrometheusAgent, RequestContext
from prometheus_api.storage_client.connection import ConnectionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

EXTRACT_SKILL = "team_5_extract_table"
PLOT_SKILL = "team_5_plot_table"

def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _error(message: str) -> dict[str, Any]:
    """A problem the user or the orchestrator can fix: reported, not raised.

    The orchestrator hands the model only the ``error`` text, so everything it
    needs to correct the call (real captions, real column names) goes in there.
    """
    return {"status": "error", "error": message}


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

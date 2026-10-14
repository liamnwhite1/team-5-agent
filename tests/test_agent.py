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

"""Tests for the Team-5 Table Plot agent's `_execute` logic.

Extend these as you implement your agent: every input-validation branch and
every distinct result shape deserves a test. Run with `uv run pytest`.
"""

from unittest.mock import AsyncMock

from agent import PLOT_SKILL, Team5TablePlotAgent


def _make_agent() -> Team5TablePlotAgent:
    # agent_card.json is loaded lazily by run()/build_app(), not by the
    # constructor, so unit tests can instantiate the agent directly.
    return Team5TablePlotAgent(agent_name="Team-5 Table Plot Agent")


async def test_execute_returns_success_for_valid_input(context, event_queue):
    agent = _make_agent()

    agent.send_status_update = AsyncMock()
    agent._plot_table = AsyncMock(return_value={"status": "success", "artifact_id": "plot-1"})
    params = {"skill_id": PLOT_SKILL, "table_artifact_id": "table-1"}
    result = await agent._execute(params, context, event_queue)

    assert result["status"] == "success"
    assert result["artifact_id"] == "plot-1"
    agent._plot_table.assert_awaited_once_with(params, context, event_queue)
    agent.send_status_update.assert_awaited_once()


async def test_execute_rejects_missing_input(context, event_queue):
    agent = _make_agent()
    agent.send_status_update = AsyncMock()

    result = await agent._execute({}, context, event_queue)

    assert result["status"] == "error"
    assert "table_artifact_id" in result["error"]

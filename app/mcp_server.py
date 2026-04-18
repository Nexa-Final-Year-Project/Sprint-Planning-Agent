"""
MCP Server for NEXA Sprint Planner Agent.

Exposes sprint planning and sprint validation as MCP tools via FastMCP
(Streamable HTTP transport). Mounted on the existing FastAPI app under /mcp
so all original REST endpoints stay completely intact.
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "nexa-sprint-planner",
    instructions=(
        "Plans optimal sprints by analysing team capacity, task priorities, and "
        "historical velocity using AI (Gemini). "
        "Use plan_sprint to generate a full sprint plan. "
        "Use validate_sprint to run a predictive blocker check on a proposed plan "
        "before committing to it."
    ),
)


@mcp.tool()
async def plan_sprint(
    project_id: str,
    members: List[Dict[str, Any]],
    tasks: List[Dict[str, Any]],
    sprint_config: Dict[str, Any],
    max_tasks_per_member: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate an AI-powered sprint plan for a project.

    project_id: MongoDB project _id string.
    members: List of ProjectMember dicts. Required fields per member —
        _id (str), name (str), role (str), baseWeeklyHours (float),
        reliabilityScore (float 0-1), availabilityPct (float 0-1),
        velocity (float), unavailableDates (list).
    tasks: List of task dicts. Required fields per task —
        taskId (str), title (str), estimatedHours (float),
        priority ('low'|'medium'|'high'|'critical'),
        dependencies (list[str]).
    sprint_config: Required fields —
        sprintLengthDays (int), workHoursPerDay (float),
        sprintGoals (list[str]), startDate (str YYYY-MM-DD).
    max_tasks_per_member: Optional hard cap on tasks assigned per member.

    Returns: success, sprintId, summary, goals, selectedTasks, deferredTasks,
             capacity, riskAnalysis, recommendations, predictedVelocity,
             burndownForecast, sprintRiskScore, fairnessReport,
             memberWorkloadSummary, totalEffort.
    """
    from app.core.planner_engine import plan_single_sprint

    result = await plan_single_sprint(
        project_id=project_id,
        members=members,
        tasks=tasks,
        sprint_config=sprint_config,
        max_tasks_per_member=max_tasks_per_member,
    )
    return result


@mcp.tool()
async def validate_sprint(
    project_id: str,
    members: List[Dict[str, Any]],
    selected_tasks: List[Dict[str, Any]],
    sprint_config: Dict[str, Any],
    auth_token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate a proposed sprint plan BEFORE creation using predictive blocker
    analysis. Calls the Blocker Detection Agent internally.

    project_id: MongoDB project _id.
    members: Same member shape as plan_sprint.
    selected_tasks: List of task dicts the PM wants to include (must include
        _id, title, status, priority, estimatedHours, dependencies, assignedTo).
    sprint_config: Dict with startDate, sprintLengthDays, workHoursPerDay.
    auth_token: Optional Bearer token to forward to the Blocker Agent.

    Returns: success, healthScore, blockerCount, blockers, status,
             summary, actions, warnings.
    """
    blocker_url = os.getenv("BLOCKER_AGENT_URL", "").rstrip("/")
    if not blocker_url:
        return {
            "success": False,
            "error": "BLOCKER_AGENT_URL is not configured in the Sprint Planner environment.",
        }

    payload: Dict[str, Any] = {
        "tasks": selected_tasks,
        "members": members,
        "signals": {
            "context": "predictive_validation",
            "sprintConfig": sprint_config,
        },
    }

    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if auth_token:
        token = auth_token.strip()
        headers["Authorization"] = token if token.startswith("Bearer ") else f"Bearer {token}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{blocker_url}/api/blockers/detect",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

    return {"success": True, **data}

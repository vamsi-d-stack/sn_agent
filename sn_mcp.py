"""
Read-only ServiceNow MCP server.

Exposes ONLY fetch/search tools for:
  - incident        (INC...)
  - change_request  (CHG...)
  - change_task     (CTASK...)

Safety by design:
  * Only HTTP GET is ever issued (no create/update/delete code exists).
  * Table names are hard-coded in an allowlist.
  * Field lists are restricted per table.
  * User input is validated so it can't inject extra encoded-query clauses ('^').

Run:  python server.py            (stdio transport, for Claude Desktop / Claude Code)
"""

import os
import re
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

INSTANCE = os.environ["SERVICENOW_INSTANCE"].rstrip("/")  # e.g. https://dev12345.service-now.com
USERNAME = os.environ["SERVICENOW_USERNAME"]
PASSWORD = os.environ["SERVICENOW_PASSWORD"]
TIMEOUT = float(os.environ.get("SERVICENOW_TIMEOUT", "30"))
MAX_LIMIT = 50

mcp = MCPServer("servicenow-readonly")

# --------------------------------------------------------------------------
# Config: allowed tables and the fields we return for each
# --------------------------------------------------------------------------
TABLES: dict[str, dict[str, Any]] = {
    "incident": {
        "prefix": "INC",
        "fields": [
            "number", "short_description", "description", "state", "priority",
            "urgency", "impact", "category", "subcategory", "caller_id",
            "assignment_group", "assigned_to", "opened_by", "opened_at",
            "resolved_at", "closed_at", "close_code", "close_notes",
            "cmdb_ci", "sys_updated_on", "sys_id",
        ],
    },
    "change_request": {
        "prefix": "CHG",
        "fields": [
            "number", "short_description", "description", "state", "type",
            "priority", "risk", "impact", "category", "assignment_group",
            "assigned_to", "requested_by", "start_date", "end_date",
            "justification", "implementation_plan", "backout_plan",
            "test_plan", "cmdb_ci", "approval", "opened_at", "closed_at",
            "sys_updated_on", "sys_id",
        ],
    },
    "change_task": {
        "prefix": "CTASK",
        "fields": [
            "number", "short_description", "description", "state", "priority",
            "change_request", "change_task_type", "assignment_group",
            "assigned_to", "planned_start_date", "planned_end_date",
            "work_start", "work_end", "close_notes", "opened_at", "closed_at",
            "sys_updated_on", "sys_id",
        ],
    },
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_SAFE_VALUE = re.compile(r"^[\w\s\-.,:@/()#&']{1,200}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?$")


def _clean(value: str, label: str) -> str:
    """Reject anything that could break out of an encoded query (^, newlines, etc.)."""
    value = value.strip()
    if not _SAFE_VALUE.match(value):
        raise ValueError(f"Invalid characters or length in '{label}'.")
    return value


def _number(table: str, number: str) -> str:
    number = number.strip().upper()
    prefix = TABLES[table]["prefix"]
    if not re.fullmatch(rf"{prefix}\d{{4,12}}", number):
        raise ValueError(f"Expected a number like {prefix}0010001, got '{number}'.")
    return number


async def _get(table: str, query: str, limit: int) -> list[dict]:
    """The ONLY function that talks to ServiceNow. GET only."""
    if table not in TABLES:
        raise ValueError("Table not allowed.")
    params = {
        "sysparm_query": query,
        "sysparm_limit": max(1, min(limit, MAX_LIMIT)),
        "sysparm_fields": ",".join(TABLES[table]["fields"]),
        "sysparm_display_value": "true",          # human-readable state/priority/etc.
        "sysparm_exclude_reference_link": "true",  # drop noisy link objects
    }
    async with httpx.AsyncClient(timeout=TIMEOUT, auth=(USERNAME, PASSWORD)) as client:
        resp = await client.get(
            f"{INSTANCE}/api/now/table/{table}",
            params=params,
            headers={"Accept": "application/json"},
        )
    if resp.status_code == 401:
        raise RuntimeError("ServiceNow authentication failed (401).")
    if resp.status_code == 403:
        raise RuntimeError("ServiceNow user lacks read access to this table (403).")
    resp.raise_for_status()
    return resp.json().get("result", [])


def _build_query(
    keyword: Optional[str],
    state: Optional[str],
    priority: Optional[str],
    assignment_group: Optional[str],
    assigned_to: Optional[str],
    opened_after: Optional[str],
    extra: Optional[list[str]] = None,
) -> str:
    parts: list[str] = list(extra or [])
    if keyword:
        parts.append(f"short_descriptionLIKE{_clean(keyword, 'keyword')}")
    if state:
        parts.append(f"state={_clean(state, 'state')}")  # internal value, e.g. 1,2,3
    if priority:
        parts.append(f"priority={_clean(priority, 'priority')}")
    if assignment_group:
        parts.append(f"assignment_group.name={_clean(assignment_group, 'assignment_group')}")
    if assigned_to:
        parts.append(f"assigned_to.name={_clean(assigned_to, 'assigned_to')}")
    if opened_after:
        if not _DATE.match(opened_after.strip()):
            raise ValueError("opened_after must be YYYY-MM-DD or YYYY-MM-DD HH:MM:SS.")
        parts.append(f"opened_at>={opened_after.strip()}")
    parts.append("ORDERBYDESCsys_updated_on")
    return "^".join(parts)


async def _fetch_one(table: str, number: str) -> dict:
    rows = await _get(table, f"number={_number(table, number)}", 1)
    if not rows:
        return {"error": f"{number} not found (or not visible to the service account)."}
    return rows[0]


# --------------------------------------------------------------------------
# Incident tools
# --------------------------------------------------------------------------
@mcp.tool()
async def get_incident(number: str) -> dict:
    """Get full details of one incident by number, e.g. INC0010001."""
    return await _fetch_one("incident", number)


@mcp.tool()
async def search_incidents(
    keyword: Optional[str] = None,
    state: Optional[str] = None,
    priority: Optional[str] = None,
    assignment_group: Optional[str] = None,
    assigned_to: Optional[str] = None,
    opened_after: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Search incidents. keyword matches short description; state/priority use
    ServiceNow internal values (e.g. priority 1=Critical, state 1=New, 2=In Progress,
    6=Resolved, 7=Closed). assignment_group/assigned_to are display names.
    opened_after is YYYY-MM-DD. Max 50 results, newest-updated first."""
    q = _build_query(keyword, state, priority, assignment_group, assigned_to, opened_after)
    return await _get("incident", q, limit)


# --------------------------------------------------------------------------
# Change request tools
# --------------------------------------------------------------------------
@mcp.tool()
async def get_change_request(number: str) -> dict:
    """Get full details of one change request by number, e.g. CHG0030001."""
    return await _fetch_one("change_request", number)


@mcp.tool()
async def search_change_requests(
    keyword: Optional[str] = None,
    state: Optional[str] = None,
    priority: Optional[str] = None,
    assignment_group: Optional[str] = None,
    assigned_to: Optional[str] = None,
    opened_after: Optional[str] = None,
    change_type: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Search change requests. change_type: normal, standard or emergency.
    Other filters behave like search_incidents."""
    extra = []
    if change_type:
        extra.append(f"type={_clean(change_type, 'change_type').lower()}")
    q = _build_query(keyword, state, priority, assignment_group, assigned_to, opened_after, extra)
    return await _get("change_request", q, limit)


# --------------------------------------------------------------------------
# Change task (CTASK) tools
# --------------------------------------------------------------------------
@mcp.tool()
async def get_change_task(number: str) -> dict:
    """Get full details of one change task by number, e.g. CTASK0040001."""
    return await _fetch_one("change_task", number)


@mcp.tool()
async def get_tasks_for_change(change_number: str, limit: int = 25) -> list[dict]:
    """List all change tasks belonging to a change request (e.g. CHG0030001)."""
    chg = _number("change_request", change_number)
    return await _get("change_task", f"change_request.number={chg}^ORDERBYnumber", limit)


@mcp.tool()
async def search_change_tasks(
    keyword: Optional[str] = None,
    state: Optional[str] = None,
    priority: Optional[str] = None,
    assignment_group: Optional[str] = None,
    assigned_to: Optional[str] = None,
    opened_after: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Search change tasks. Filters behave like search_incidents."""
    q = _build_query(keyword, state, priority, assignment_group, assigned_to, opened_after)
    return await _get("change_task", q, limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")
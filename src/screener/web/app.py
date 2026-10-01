"""The web UI (§17). Localhost-only, read-only, no authentication.

Three deliberate constraints:

* **Read-only.** Every connection is opened with SQLite's `mode=ro`. Browsing must never be able
  to corrupt the store the pipeline writes to.
* **Localhost.** The app is intended to be reached over an SSH tunnel or on the same machine.
  There is no auth because there is no exposure; binding it anywhere else without a token would
  publish the whole paper history, so `screener web` refuses a non-loopback host unless
  `--allow-remote` is passed explicitly.
* **No JavaScript required.** Server-rendered pages, so the UI works with scripting off and has
  no build step. The one concession is a `<details>` element for expanding long abstracts.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from screener.web import queries

TEMPLATES = Path(__file__).parent / "templates"
STATIC = Path(__file__).parent / "static"


def create_app(db_path: str | Path) -> FastAPI:
    """Build the app for one database path. A factory so tests get an isolated DB."""
    # Fail at startup with a readable reason rather than mid-render with a SQL error.
    _probe = queries.connect(db_path)
    try:
        queries.require_schema(_probe)
    finally:
        _probe.close()

    app = FastAPI(title="Agent Papers Daily", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    templates = Jinja2Templates(directory=str(TEMPLATES))

    # `period.html` offers quick 7d/30d/90d ranges, which need date arithmetic in the template.
    def days_ago(value: object, n: int) -> str:
        return (date.fromisoformat(str(value)) - timedelta(days=int(n))).isoformat()

    templates.env.filters["days_ago"] = days_ago

    def render(request: Request, name: str, **context: Any) -> HTMLResponse:
        context.setdefault("today", date.today().isoformat())
        return templates.TemplateResponse(request, name, context)

    def period(start: str | None, end: str | None) -> tuple[str, str]:
        default_start, default_end = queries.default_range()
        return (start or default_start), (end or default_end)

    @app.get("/healthz", response_class=HTMLResponse)
    def healthz() -> str:
        return "ok"

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        conn = queries.connect(db_path)
        try:
            return render(
                request,
                "dashboard.html",
                page="dashboard",
                overview=queries.overview(conn),
                funnel=queries.funnel(conn),
                days=queries.days(conn, limit=30),
            )
        finally:
            conn.close()

    @app.get("/day/{day}", response_class=HTMLResponse)
    def day_view(request: Request, day: str) -> HTMLResponse:
        if not _is_date(day):
            raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")
        conn = queries.connect(db_path)
        try:
            detail = queries.day_detail(conn, day)
            return render(request, "day.html", page="days", **detail)
        finally:
            conn.close()

    @app.get("/days", response_class=HTMLResponse)
    def days_view(request: Request) -> HTMLResponse:
        conn = queries.connect(db_path)
        try:
            return render(request, "days.html", page="days", days=queries.days(conn, limit=200))
        finally:
            conn.close()

    @app.get("/papers", response_class=HTMLResponse)
    def papers_view(
        request: Request,
        start: str | None = None,
        end: str | None = None,
        q: str = "",
        limit: int = Query(default=100, ge=1, le=500),
    ) -> HTMLResponse:
        start, end = period(start, end)
        conn = queries.connect(db_path)
        try:
            return render(
                request,
                "papers.html",
                page="papers",
                start=start,
                end=end,
                q=q,
                limit=limit,
                items=queries.papers(conn, start, end, q, limit),
            )
        finally:
            conn.close()

    @app.get("/reactions", response_class=HTMLResponse)
    def reactions_view(
        request: Request, start: str | None = None, end: str | None = None
    ) -> HTMLResponse:
        start, end = period(start, end)
        conn = queries.connect(db_path)
        try:
            rows = queries.reactions(conn, start, end)
            return render(
                request,
                "reactions.html",
                page="reactions",
                start=start,
                end=end,
                rows=rows,
                top=queries.top_by_users(conn, start, end, limit=10),
                overview=queries.overview(conn),
            )
        finally:
            conn.close()

    @app.get("/top", response_class=HTMLResponse)
    def top_view(
        request: Request,
        start: str | None = None,
        end: str | None = None,
        by: str = "score",
        limit: int = Query(default=25, ge=1, le=100),
    ) -> HTMLResponse:
        start, end = period(start, end)
        by = by if by in {"score", "users"} else "score"
        conn = queries.connect(db_path)
        try:
            items = (
                queries.top_by_score(conn, start, end, limit)
                if by == "score"
                else queries.top_by_users(conn, start, end, limit)
            )
            return render(
                request,
                "top.html",
                page="top",
                start=start,
                end=end,
                by=by,
                limit=limit,
                items=items,
                overview=queries.overview(conn),
            )
        finally:
            conn.close()

    @app.get("/paper/{arxiv_id}", response_class=HTMLResponse)
    def paper_view(request: Request, arxiv_id: str) -> HTMLResponse:
        conn = queries.connect(db_path)
        try:
            detail = queries.paper_detail(conn, arxiv_id)
            if detail is None:
                raise HTTPException(status_code=404, detail=f"no paper {arxiv_id}")
            return render(request, "paper.html", page="papers", **detail)
        finally:
            conn.close()

    return app


def _is_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True

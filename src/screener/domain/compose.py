"""Ranking -> Telegram-safe HTML chunks (§8).

Pure. Three properties are load-bearing and therefore asserted rather than hoped for:

* no chunk exceeds Telegram's 4096-character limit;
* splitting happens **on item boundaries only**, never mid-item — which is what makes the
  item->chunk map meaningful, and that map is how a reply gets attributed to a paper (§10);
* **exactly one paper per message** (§8.4).

The third is not a formatting preference. A Telegram *reaction* is attached to a message, not to
a region of one, so a message carrying two papers makes the reaction impossible to attribute:
there is no way to know which of them the reader meant. The digest used to pack up to five items
per message, which meant most messages were ambiguous — in the live database four of five were.
One paper per message makes a rating exact by construction, for reactions and replies alike.

`compose` takes the papers by key rather than storing a `Paper` on every `Ranking`: a
`Ranking` is a judgment about a paper, not a copy of it, and keeping it lean keeps the
`rankings` table (§10) from duplicating the `papers` table.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from html import escape

from screener.domain.models import Digest, DigestItem, Paper, Ranking
from screener.domain.types import PaperKey

TELEGRAM_LIMIT = 4096

_GATE_LABELS = {
    "pure_survey": "surveys",
    "no_technical_contribution": "position/short",
    "not_agentic": "not agentic",
    "marketing_whitepaper": "whitepapers",
    "withdrawn": "withdrawn",
}


def render_header(
    now: datetime,
    *,
    picks: int,
    scanned: int,
    relevant: int,
    cost_usd: float,
    unsent_notice: bool = False,
) -> str:
    when = now.strftime("%a %Y-%m-%d")
    head = (
        f"🧠 <b>Agent Papers — Daily</b>\n{when} · {picks} picks · {scanned} scanned · "
        f"{relevant} relevant · ${cost_usd:.2f}"
    )
    if unsent_notice:
        head += "\n<i>(previously unsent digest)</i>"
    return head


def render_footer(
    gated_by_reason: Mapping[str, int], below_threshold: int, best_below: float | None
) -> str:
    parts = [
        f"{count} {_GATE_LABELS.get(reason, reason)}"
        for reason, count in sorted(gated_by_reason.items(), key=lambda kv: -kv[1])
        if count
    ]
    if below_threshold:
        suffix = f" (best {best_below:.1f})" if best_below is not None else ""
        parts.append(f"{below_threshold} below threshold{suffix}")
    skipped = " · ".join(parts) if parts else "nothing skipped"
    return (
        f"<i>Skipped: {escape(skipped)}</i>\n"
        f"<i>Reply 👍 / 👎 / 🔥 to rate an item and tune tomorrow's ranking.</i>"
    )


def render_item(r: Ranking, paper: Paper, index: int) -> str:
    """One item block. Every prose field comes from the stored Review (§7)."""
    header = f"<b>{index}. {escape(paper.title)}</b>  🔥 {r.score:.1f}"

    meta = f"<code>{escape(r.arxiv_id)}</code>"
    if paper.primary_category:
        meta += f" · {escape(paper.primary_category)}"
    if paper.comment and "accept" in paper.comment.lower():
        meta += f" · {escape(paper.comment[:60])}"

    # Every link is conditional on a non-empty URL. `Paper.abs_url` defaults to "" and the
    # `papers.abs_url` column is NOT NULL DEFAULT '', so an unguarded link emits
    # `<a href="">`, which Telegram rejects with a 400 (an empty href is not a valid entity).
    links: list[str] = []
    if paper.abs_url:
        links.append(f'<a href="{escape(paper.abs_url)}">abs</a>')
    if paper.pdf_url:
        links.append(f'<a href="{escape(paper.pdf_url)}">pdf</a>')
    if paper.code_url:
        links.append(f'<a href="{escape(paper.code_url)}">code</a>')
    if not links:
        links.append(f"<code>{escape(paper.arxiv_id)}</code>")

    lines = [
        header,
        meta,
        " · ".join(links) if links else meta,
        "",
        f"<b>TL;DR</b> {escape(r.review.tldr)}",
        "",
        f"<b>What they did</b> {escape(r.review.what_they_did)}",
        "",
        _lens_block(r),
        "",
        f"<b>Weak spot</b> {escape(r.review.caveats)}",
    ]
    if r.basis is not None:
        lines += ["", f"<b>Basis</b> {escape(r.basis.render_compact())}"]
    elif signals := _signals_line(r):
        lines += ["", f"<b>Signals</b> {signals}"]
    return "\n".join(lines)


def _lens_block(r: Ranking) -> str:
    lens = ", ".join(r.review.lenses) if r.review.lenses else "none claimed"
    return f"<b>Why it matters</b> <i>[{escape(lens)}]</i> {escape(r.review.why_it_matters)}"


def _signals_line(r: Ranking) -> str:
    bits: list[str] = []
    if paper_flags := [f.value for f in r.soft_flags]:
        bits.append(", ".join(paper_flags[:2]))
    if r.topics:
        bits.append(escape(r.topics[0]))
    if r.lab:
        bits.append(escape(r.lab))
    return " · ".join(bits)


def compose(
    picks: Sequence[Ranking],
    papers: Mapping[PaperKey, Paper],
    now: datetime,
    *,
    scanned: int,
    relevant: int,
    cost_usd: float,
    gated_by_reason: Mapping[str, int] | None = None,
    below_threshold: int = 0,
    best_below: float | None = None,
    unsent_notice: bool = False,
) -> Digest:
    """Build the digest: one message per paper, so a reaction identifies a paper (§8.4)."""
    header = render_header(
        now,
        picks=len(picks),
        scanned=scanned,
        relevant=relevant,
        cost_usd=cost_usd,
        unsent_notice=unsent_notice,
    )
    footer = render_footer(gated_by_reason or {}, below_threshold, best_below)

    chunks: list[str] = []
    items: list[DigestItem] = []
    total = len(picks)

    for index, r in enumerate(picks, start=1):
        paper = papers.get((r.arxiv_id, r.version))
        if paper is None:
            raise KeyError(f"no Paper for {r.arxiv_id}v{r.version}; compose needs it to render")
        parts: list[str] = []
        if index == 1:
            parts.append(header)  # the header rides on the first paper's message
        parts.append(render_item(r, paper, index))
        if index == total:
            parts.append(footer)  # so does the footer, on the last
        chunks.append("\n\n".join(parts))
        items.append(DigestItem(arxiv_id=r.arxiv_id, version=r.version, chunk_index=index - 1))

    if not picks:
        chunks.append("\n\n".join([header, footer]))

    for i, chunk in enumerate(chunks):
        if len(chunk) > TELEGRAM_LIMIT:
            raise AssertionError(
                f"chunk {i} is {len(chunk)} chars, over the {TELEGRAM_LIMIT} limit"
            )

    return Digest(chunks=chunks, items=items)

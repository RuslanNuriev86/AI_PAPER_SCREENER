"""The soft budget cap (§11, §13.1).

Not a port: there is no external system here and no adapter, so this is a plain class over a
price table plus whatever has already been charged.

**The cap is soft on purpose.** §12.1 requires that on breach the run "stops reviewing
further papers and ships what is verified". A context manager that raises can only signal
breach by unwinding the stack, which cannot ship anything — so `soft_cap` sets
`exhausted` instead and each spending stage asks `affordable()` before its next call. The
run is then recorded `degraded`, not `failed`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from screener.domain.models import LedgerEntry
from screener.domain.types import Status

#: Per-1M-token prices in USD as (input, output).
#:
#: The ledger's job is to enforce a ceiling, not to bill, so these are deliberately
#: conservative. DeepSeek prices vary by time of day (off-peak is half of peak) and by cache
#: hit (a hit is ~2% of a miss); we use the **peak, cache-miss** rate, which over-estimates
#: real spend — the safe direction for a cap. Source: DeepSeek "Models & Pricing".
PRICES: dict[str, tuple[float, float]] = {
    "deepseek-flash": (0.30, 1.20),  # peak, cache miss / output
    "deepseek-v4-pro": (1.32, 3.96),
    # Legacy ids still accepted by the API and served by DeepSeek-V4.1-Flash.
    "deepseek-v4-flash": (0.30, 1.20),
    "deepseek-v4-flash-vision-exp": (0.30, 1.20),
    # Kept so a non-DeepSeek deployment still gets a sane estimate.
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
}
#: Fallback for an unrecognised model. High on purpose: an unknown price must not make the
#: cap look more generous than it is.
DEFAULT_PRICE = (2.0, 8.0)


def estimate(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES.get(model, DEFAULT_PRICE)
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


class Ledger:
    def __init__(self, cap_usd: float, already_spent: float = 0.0) -> None:
        self.cap_usd = cap_usd
        self.spent = already_spent
        self.exhausted = False
        self.entries: list[LedgerEntry] = []

    @property
    def status(self) -> Status:
        return "degraded" if self.exhausted else "ok"

    def affordable(self, projected_usd: float) -> bool:
        """True if another call costing `projected_usd` stays inside the cap."""
        if self.spent + projected_usd > self.cap_usd:
            self.exhausted = True
            return False
        return True

    def charge(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
        self.spent += entry.usd
        if self.spent > self.cap_usd:
            self.exhausted = True

    @contextmanager
    def soft_cap(self, cap_usd: float) -> Iterator[Ledger]:
        """Enter the budget. Never raises on breach — only records it.

        Yields `self` so the caller's `ledger` and `deps.ledger` are the same object, which
        is what lets the failure path still report `deps.ledger.spent`.
        """
        self.cap_usd = cap_usd
        self.exhausted = self.spent > cap_usd
        try:
            yield self
        finally:
            if self.spent > cap_usd:
                self.exhausted = True

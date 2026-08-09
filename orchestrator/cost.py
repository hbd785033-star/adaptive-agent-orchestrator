"""
Token cost estimator — heuristic pricing table.

Prices are approximate and change frequently; treat as ORDER OF MAGNITUDE only.
Replace with a live pricing API call if you need precision.

Usage:
    from orchestrator.cost import estimate_cost
    usd = estimate_cost("claude-sonnet-4", input_tokens=1000, output_tokens=300)
"""
from __future__ import annotations

# Price per 1M tokens (USD).  Source: public pricing pages, 2026-Q3 snapshots.
# fmt: off
_PRICE_TABLE: dict[str, tuple[float, float]] = {
    # model-key           (input $/1M, output $/1M)
    "claude-opus-4":      (15.00, 75.00),
    "claude-sonnet-4":    ( 3.00, 15.00),
    "claude-sonnet-4-5":  ( 3.00, 15.00),
    "claude-haiku-4":     ( 0.80,  4.00),
    "gpt-4o":             ( 5.00, 15.00),
    "gpt-4o-mini":        ( 0.15,  0.60),
    "gemini-2-flash":     ( 0.075, 0.30),
    "gemini-2-pro":       ( 1.25,  5.00),
    # fallback for unknown models
    "__default__":        ( 3.00, 15.00),
}
# fmt: on


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return estimated USD cost (never negative; rounds to 6 decimal places)."""
    # Normalise model name: lowercase, strip date suffixes like -20250101
    raw_key = model.lower()
    key = raw_key.split(":", 1)[-1] if ":" in raw_key else raw_key
    key = "-".join(                             # drop pure-digit trailing parts
        p for p in key.split("-") if not p.isdigit()
    )
    # Partial key match: pick first entry whose key is contained in model name
    price_in, price_out = _PRICE_TABLE.get("__default__")
    for table_key in sorted(
        (name for name in _PRICE_TABLE if name != "__default__"),
        key=len,
        reverse=True,
    ):
        if table_key in key or key in table_key:
            price_in, price_out = _PRICE_TABLE[table_key]
            break

    usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return round(usd, 6)


# Expose pricing table for CLI / stats display
def known_models() -> list[str]:
    return [k for k in _PRICE_TABLE if k != "__default__"]

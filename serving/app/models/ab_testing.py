"""Deterministic A/B routing for champion/challenger traffic splitting.

Routing is based on a hash of the transaction_id so the same transaction
always goes to the same model variant — required for consistent debugging
and fair comparison.
"""

from __future__ import annotations

import hashlib


def route_to_challenger(transaction_id: str, challenger_fraction: float) -> bool:
    """Return True if this transaction should be routed to the challenger model.

    Uses MD5 of the transaction_id modulo 100 to deterministically assign
    each ID to a bucket. IDs whose bucket falls below challenger_fraction * 100
    go to the challenger; the rest go to the champion.

    Args:
        transaction_id: Unique identifier for the transaction.
        challenger_fraction: Fraction of traffic for the challenger (0.0–1.0).
    """
    digest = hashlib.md5(transaction_id.encode(), usedforsecurity=False).hexdigest()
    bucket = int(digest, 16) % 100
    return bucket < int(challenger_fraction * 100)

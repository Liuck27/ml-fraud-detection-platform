"""Tests for A/B routing: determinism and approximate traffic distribution."""

from __future__ import annotations

import uuid

from serving.app.models.ab_testing import route_to_challenger


def test_routing_is_deterministic() -> None:
    """Same transaction_id must always route to the same model."""
    txn_id = "deterministic-test-id-12345"
    result = route_to_challenger(txn_id, challenger_fraction=0.20)
    for _ in range(100):
        assert route_to_challenger(txn_id, 0.20) == result


def test_full_challenger_fraction_routes_all() -> None:
    """fraction=1.0 → every transaction goes to challenger."""
    for _ in range(50):
        assert route_to_challenger(str(uuid.uuid4()), 1.0) is True


def test_zero_challenger_fraction_routes_none() -> None:
    """fraction=0.0 → no transaction goes to challenger."""
    for _ in range(50):
        assert route_to_challenger(str(uuid.uuid4()), 0.0) is False


def test_approximate_split_ratio() -> None:
    """With fraction=0.20 and 10,000 IDs the challenger rate should be ~20% ± 2%."""
    n = 10_000
    fraction = 0.20
    challenger_count = sum(
        route_to_challenger(str(uuid.uuid4()), fraction) for _ in range(n)
    )
    actual = challenger_count / n
    assert abs(actual - fraction) < 0.02, f"Expected ~{fraction:.0%}, got {actual:.2%}"


def test_different_ids_can_split() -> None:
    """Two different IDs must not always route the same way (not trivially constant)."""
    results = {route_to_challenger(str(i), 0.50) for i in range(100)}
    assert len(results) == 2, "Expected both True and False with fraction=0.50"

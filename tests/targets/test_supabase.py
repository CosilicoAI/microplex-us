"""Tests for loading US calibration targets from Supabase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from microplex_us.supabase_targets import SupabaseTargetLoader

SUPABASE_URL = "https://test.supabase.co"
SUPABASE_KEY = "test-key"


@dataclass
class MockResponse:
    payload: list[dict[str, Any]]

    def json(self) -> list[dict[str, Any]]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


@pytest.fixture
def loader() -> SupabaseTargetLoader:
    return SupabaseTargetLoader(SUPABASE_URL, SUPABASE_KEY)


@pytest.fixture
def request_queue(monkeypatch: pytest.MonkeyPatch):
    calls = []
    responses: list[MockResponse] = []

    def fake_get(
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, Any],
        timeout: int,
    ) -> MockResponse:
        calls.append(
            {
                "url": url,
                "headers": headers,
                "params": params,
                "timeout": timeout,
            }
        )
        return responses.pop(0)

    def queue(*payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        responses[:] = [MockResponse(payload) for payload in payloads]
        return calls

    monkeypatch.setattr("microplex_us.supabase_targets.requests.get", fake_get)
    return queue


def test_missing_service_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COSILICO_SUPABASE_SERVICE_KEY", raising=False)

    with pytest.raises(ValueError, match="COSILICO_SUPABASE_SERVICE_KEY"):
        SupabaseTargetLoader(SUPABASE_URL)


def test_load_all_targets(loader: SupabaseTargetLoader, request_queue) -> None:
    calls = request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "t2",
                "variable": "snap_spending",
                "value": 103100000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "USDA SNAP", "institution": "USDA"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    targets = loader.load_all()

    assert [target["variable"] for target in targets] == [
        "employment_income",
        "snap_spending",
    ]
    assert calls[0]["url"] == f"{SUPABASE_URL}/rest/v1/targets"
    assert calls[0]["params"]["limit"] == 1000
    assert calls[0]["params"]["offset"] == 0


def test_load_by_institution(
    loader: SupabaseTargetLoader,
    request_queue,
) -> None:
    request_queue(
        [{"id": "src-1", "institution": "IRS", "name": "IRS SOI"}],
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            }
        ],
    )

    targets = loader.load_by_institution("IRS")

    assert len(targets) == 1
    assert targets[0]["source"]["institution"] == "IRS"


def test_load_by_period(loader: SupabaseTargetLoader, request_queue) -> None:
    calls = request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            }
        ]
    )

    targets = loader.load_by_period(2024)

    assert len(targets) == 1
    assert calls[0]["params"]["period"] == "eq.2024"


def test_cps_column_mapping(loader: SupabaseTargetLoader) -> None:
    mapping = loader.get_cps_column_map()

    assert mapping["employment_income"] == "employment_income"
    assert mapping["self_employment_income"] == "self_employment_income"
    assert mapping["dividend_income"] == "dividend_income"
    assert mapping["snap_spending"] == "snap"
    assert mapping["ssi_spending"] == "ssi"
    assert mapping["eitc_spending"] == "eitc"


def test_build_continuous_targets(
    loader: SupabaseTargetLoader,
    request_queue,
) -> None:
    request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "t2",
                "variable": "snap_spending",
                "value": 103100000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "USDA SNAP", "institution": "USDA"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    constraints = loader.build_calibration_constraints()

    assert constraints["employment_income"] == 9022400000000
    assert constraints["snap"] == 103100000000


def test_build_state_targets(
    loader: SupabaseTargetLoader,
    request_queue,
) -> None:
    request_queue(
        [
            {
                "id": "t1",
                "variable": "medicaid_enrollment",
                "value": 14000000,
                "target_type": "count",
                "period": 2024,
                "source": {"name": "CMS Medicaid", "institution": "HHS"},
                "stratum": {"name": "California", "jurisdiction": "us-ca"},
            }
        ]
    )

    constraints = loader.build_calibration_constraints(include_states=True)

    assert constraints["medicaid_ca"] == 14000000


def test_get_summary(loader: SupabaseTargetLoader, request_queue) -> None:
    request_queue(
        [
            {
                "id": "t1",
                "variable": "employment_income",
                "value": 9022400000000,
                "target_type": "amount",
                "period": 2024,
                "source": {"name": "IRS SOI", "institution": "IRS"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
            {
                "id": "t2",
                "variable": "person_count",
                "value": 330000000,
                "target_type": "count",
                "period": 2024,
                "source": {"name": "Census", "institution": "Census"},
                "stratum": {"name": "National", "jurisdiction": "us"},
            },
        ]
    )

    summary = loader.get_summary()

    assert summary == {
        "total": 2,
        "by_institution": {"IRS": 1, "Census": 1},
        "by_variable": {"employment_income": 1, "person_count": 1},
        "by_type": {"amount": 1, "count": 1},
    }

"""US Supabase calibration target loader."""

from __future__ import annotations

import os
from typing import Any

import requests


class SupabaseTargetLoader:
    """Load US calibration targets from the microplex Supabase schema."""

    # Mapping from Supabase variable names to CPS column names.
    CPS_COLUMN_MAP = {
        "employment_income": "employment_income",
        "self_employment_income": "self_employment_income",
        "dividend_income": "dividend_income",
        "interest_income": "interest_income",
        "rental_income": "rental_income",
        "social_security": "social_security",
        "unemployment_compensation": "unemployment_compensation",
        "taxable_pension_income": "taxable_pension_income",
        "tax_exempt_pension_income": "tax_exempt_pension_income",
        "long_term_capital_gains": "long_term_capital_gains",
        "short_term_capital_gains": "short_term_capital_gains",
        "partnership_s_corp_income": "partnership_s_corp_income",
        "farm_income": "farm_income",
        "alimony_income": "alimony_income",
        "snap_spending": "snap",
        "ssi_spending": "ssi",
        "eitc_spending": "eitc",
        "social_security_spending": "social_security",
        "unemployment_spending": "unemployment_compensation",
        "medicaid_enrollment": "medicaid",
        "aca_enrollment": "aca",
        "snap_households": "snap",
        "health_insurance_premiums": "health_insurance_premiums",
        "other_medical_expenses": "medical_expenses",
    }

    STATE_FIPS = {
        "01": "al",
        "02": "ak",
        "04": "az",
        "05": "ar",
        "06": "ca",
        "08": "co",
        "09": "ct",
        "10": "de",
        "11": "dc",
        "12": "fl",
        "13": "ga",
        "15": "hi",
        "16": "id",
        "17": "il",
        "18": "in",
        "19": "ia",
        "20": "ks",
        "21": "ky",
        "22": "la",
        "23": "me",
        "24": "md",
        "25": "ma",
        "26": "mi",
        "27": "mn",
        "28": "ms",
        "29": "mo",
        "30": "mt",
        "31": "ne",
        "32": "nv",
        "33": "nh",
        "34": "nj",
        "35": "nm",
        "36": "ny",
        "37": "nc",
        "38": "nd",
        "39": "oh",
        "40": "ok",
        "41": "or",
        "42": "pa",
        "44": "ri",
        "45": "sc",
        "46": "sd",
        "47": "tn",
        "48": "tx",
        "49": "ut",
        "50": "vt",
        "51": "va",
        "53": "wa",
        "54": "wv",
        "55": "wi",
        "56": "wy",
    }

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        schema: str = "microplex",
    ) -> None:
        """Initialize the loader.

        Args:
            url: Supabase URL. Defaults to SUPABASE_URL env var.
            key: Supabase key. Defaults to COSILICO_SUPABASE_SERVICE_KEY env var.
            schema: Schema to use. Defaults to 'microplex'.
        """
        self.url = url or os.environ.get(
            "SUPABASE_URL",
            "https://nsupqhfchdtqclomlrgs.supabase.co",
        )
        self.key = key or os.environ.get("COSILICO_SUPABASE_SERVICE_KEY")
        if not self.key:
            raise ValueError(
                "Supabase service key must be provided via the key argument or "
                "COSILICO_SUPABASE_SERVICE_KEY."
            )
        self.base_url = f"{self.url}/rest/v1"
        self.headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Accept-Profile": schema,
            "Content-Profile": schema,
        }
        self._cache = {}

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        paginate: bool = True,
    ) -> list[dict[str, Any]]:
        """Make a GET request to Supabase with optional pagination."""
        url = f"{self.base_url}/{endpoint}"
        params = params or {}

        if not paginate:
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        all_results = []
        offset = 0
        limit = 1000

        while True:
            page_params = {**params, "limit": limit, "offset": offset}
            response = requests.get(
                url,
                headers=self.headers,
                params=page_params,
                timeout=30,
            )
            response.raise_for_status()
            results = response.json()

            if not results:
                break

            all_results.extend(results)
            offset += limit

            if len(results) < limit:
                break

        return all_results

    def load_all(self, period: int | None = None) -> list[dict[str, Any]]:
        """Load all targets with source and stratum info."""
        params = {
            "select": "id,variable,value,target_type,period,notes,source:sources(id,name,institution),stratum:strata(id,name,jurisdiction)",
        }
        if period:
            params["period"] = f"eq.{period}"

        return self._get("targets", params)

    def load_by_institution(
        self,
        institution: str,
        period: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load targets from a specific source institution."""
        sources = self._get("sources", {"institution": f"eq.{institution}"})
        source_ids = [source["id"] for source in sources]

        if not source_ids:
            return []

        params = {
            "select": "id,variable,value,target_type,period,notes,source:sources(id,name,institution),stratum:strata(id,name,jurisdiction)",
            "source_id": f"in.({','.join(source_ids)})",
        }
        if period:
            params["period"] = f"eq.{period}"

        return self._get("targets", params)

    def load_by_period(self, period: int) -> list[dict[str, Any]]:
        """Load targets for a specific year."""
        return self.load_all(period=period)

    def get_cps_column_map(self) -> dict[str, str]:
        """Get the mapping from Supabase variable names to CPS columns."""
        return self.CPS_COLUMN_MAP.copy()

    def _parse_jurisdiction(self, jurisdiction: str) -> str | None:
        """Parse jurisdiction to get the state code when applicable."""
        if jurisdiction in {"us", "us-national"}:
            return None

        if jurisdiction.startswith("us-") and len(jurisdiction) == 5:
            state = jurisdiction[3:].lower()
            if len(state) == 2:
                return state

        if jurisdiction.startswith("us-") and len(jurisdiction) == 5:
            fips = jurisdiction[3:]
            return self.STATE_FIPS.get(fips)

        return None

    def build_calibration_constraints(
        self,
        period: int = 2024,
        include_states: bool = False,
        target_types: list[str] | None = None,
    ) -> dict[str, float]:
        """Build a CPS-column calibration constraint dict from Supabase targets."""
        targets = self.load_all(period=period)
        constraints = {}

        for target in targets:
            variable = target["variable"]
            value = target["value"]
            target_type = target.get("target_type", "amount")
            stratum = target.get("stratum", {})
            jurisdiction = stratum.get("jurisdiction", "us")

            if target_types and target_type not in target_types:
                continue

            cps_col = self.CPS_COLUMN_MAP.get(variable)
            if not cps_col:
                continue

            state = self._parse_jurisdiction(jurisdiction)

            if state and include_states:
                constraints[f"{cps_col}_{state}"] = value
            elif not state and cps_col not in constraints:
                constraints[cps_col] = value

        return constraints

    def get_summary(self) -> dict[str, Any]:
        """Get summary counts for available targets in Supabase."""
        targets = self.load_all()

        by_institution = {}
        by_variable = {}
        by_type = {}

        for target in targets:
            institution = target.get("source", {}).get("institution", "Unknown")
            by_institution[institution] = by_institution.get(institution, 0) + 1

            variable = target["variable"]
            by_variable[variable] = by_variable.get(variable, 0) + 1

            target_type = target.get("target_type", "amount")
            by_type[target_type] = by_type.get(target_type, 0) + 1

        return {
            "total": len(targets),
            "by_institution": by_institution,
            "by_variable": by_variable,
            "by_type": by_type,
        }


__all__ = ["SupabaseTargetLoader"]

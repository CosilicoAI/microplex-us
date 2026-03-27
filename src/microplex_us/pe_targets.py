"""PolicyEngine-parity calibration targets."""

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

# US States
STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL",
    "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME",
    "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
    "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI",
    "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

# State FIPS codes
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"
}


class PETargets:
    """PolicyEngine calibration targets."""

    # IRS SOI income variables mapped to CPS columns
    INCOME_MAP = {
        'employment_income': 'employment_income',
        'self_employment_income': 'self_employment_income',
        'social_security': 'social_security_income',
        'ssi': 'ssi_income',
        'unemployment_compensation': 'unemployment_income',
        'dividend_income': 'dividend_income',
        'interest_income': 'interest_income',
        'rental_income': 'rental_income',
        'pension_income': 'pension_income',
        'capital_gains': 'capital_gains',
    }

    # Benefit programs
    BENEFIT_MAP = {
        'snap_participation': 'snap_enrolled',
        'ssi_participation': 'ssi_enrolled',
        'social_security_participation': 'social_security_enrolled',
        'medicaid_enrollment': 'medicaid_enrolled',
    }

    def __init__(self, pe_path: str | None = None):
        """Initialize PE targets loader.

        Args:
            pe_path: Path to PE-US calibration folder. If None, uses installed package.
        """
        if pe_path is None:
            # Default to installed package location
            import sys
            for path in sys.path:
                test_path = Path(path) / "policyengine_us" / "parameters" / "calibration"
                if test_path.exists():
                    pe_path = test_path
                    break

        self.pe_path = Path(pe_path) if pe_path else None
        self._targets = None

    def load_all(self) -> pd.DataFrame:
        """Load all PE calibration targets."""
        if self._targets is not None:
            return self._targets

        if self.pe_path is None:
            raise ValueError("PE calibration path not found")

        targets = []

        for yaml_file in self.pe_path.rglob("*.yaml"):
            targets.extend(self._parse_yaml(yaml_file))

        self._targets = pd.DataFrame(targets)
        return self._targets

    def _parse_yaml(self, yaml_file: Path) -> list[dict]:
        """Parse a PE calibration YAML file."""
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        if not data:
            return []

        rel_path = yaml_file.relative_to(self.pe_path)
        category = str(rel_path.parent).replace("/", ".")
        name = yaml_file.stem

        metadata = data.get('metadata', {})
        unit = metadata.get('unit', 'unknown')
        description = data.get('description', '')

        targets = []

        if 'values' in data:
            # National target
            values = data['values']
            latest_date = max(values.keys())
            latest_value = values[latest_date]

            targets.append({
                'name': name,
                'category': category,
                'value': latest_value,
                'year': latest_date.year if isinstance(latest_date, date) else int(str(latest_date)[:4]),
                'unit': unit,
                'geography': 'national',
                'state_code': None,
                'state_fips': None,
                'description': description
            })
        else:
            # State-level data
            for key, val in data.items():
                if key in STATES and isinstance(val, dict):
                    latest_date = max(val.keys())
                    latest_value = val[latest_date]

                    targets.append({
                        'name': f"{name}_{key}",
                        'category': category,
                        'value': latest_value,
                        'year': latest_date.year if isinstance(latest_date, date) else int(str(latest_date)[:4]),
                        'unit': unit,
                        'geography': 'state',
                        'state_code': key,
                        'state_fips': STATE_FIPS.get(key),
                        'description': description
                    })

        return targets

    def get_national_targets(self) -> pd.DataFrame:
        """Get national-level targets."""
        df = self.load_all()
        return df[df['geography'] == 'national']

    def get_state_targets(self, state: str | None = None) -> pd.DataFrame:
        """Get state-level targets.

        Args:
            state: State code (e.g., 'CA') or None for all states
        """
        df = self.load_all()
        state_df = df[df['geography'] == 'state']

        if state:
            state_df = state_df[state_df['state_code'] == state]

        return state_df

    def get_income_targets(self) -> pd.DataFrame:
        """Get IRS SOI income targets."""
        df = self.load_all()
        return df[df['category'].str.startswith('gov.irs.soi')]

    def get_benefit_targets(self) -> pd.DataFrame:
        """Get benefit program targets (SNAP, SSI, SS, Medicaid, etc.)."""
        df = self.load_all()
        benefit_cats = ['gov.usda.snap', 'gov.ssa.ssi', 'gov.ssa.social_security',
                        'gov.hhs.medicaid', 'gov.hhs.cms.chip', 'gov.aca']
        return df[df['category'].str.startswith(tuple(benefit_cats))]

    def summary(self) -> dict:
        """Get summary of available targets."""
        df = self.load_all()

        return {
            'total': len(df),
            'national': len(df[df['geography'] == 'national']),
            'state': len(df[df['geography'] == 'state']),
            'by_category': df.groupby('category').size().to_dict(),
            'income_targets': len(self.get_income_targets()),
            'benefit_targets': len(self.get_benefit_targets()),
        }


def get_pe_targets() -> PETargets:
    """Get PolicyEngine targets instance."""
    return PETargets()


def create_calibration_targets(
    synthetic_df: pd.DataFrame,
    target_types: list[str] = None
) -> dict[str, float]:
    """Create calibration target dict from PE targets.

    Args:
        synthetic_df: Synthetic population DataFrame with income/benefit columns
        target_types: List of target types to include. Options:
            - 'income': IRS SOI income totals
            - 'benefits': Benefit program participation
            - 'population': Census population by state
            - 'all': All targets

    Returns:
        Dict mapping target name to target value
    """
    pe = get_pe_targets()

    if target_types is None:
        target_types = ['income', 'benefits']

    targets = {}

    if 'income' in target_types or 'all' in target_types:
        income_df = pe.get_income_targets()
        for _, row in income_df.iterrows():
            targets[f"income_{row['name']}"] = row['value']

    if 'benefits' in target_types or 'all' in target_types:
        benefit_df = pe.get_benefit_targets()
        # Only national for now
        national_benefits = benefit_df[benefit_df['geography'] == 'national']
        for _, row in national_benefits.iterrows():
            targets[f"benefit_{row['name']}"] = row['value']

    return targets

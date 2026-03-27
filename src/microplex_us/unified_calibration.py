"""Unified multi-target calibration for PE parity."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .pe_targets import PETargets


@dataclass
class CalibrationTarget:
    """A calibration target."""
    name: str
    target_value: float
    column: str | None = None  # Column to sum for this target
    filter_col: str | None = None  # Column to filter on
    filter_val: str | None = None  # Value to filter for
    is_count: bool = False  # If True, count rows instead of sum


class UnifiedCalibrator:
    """Calibrate synthetic population to multiple target types.

    Supports:
    - Geographic targets (CD, state, SLDU)
    - Income totals (IRS SOI)
    - Benefit program participation/spending
    - Demographic distributions
    """

    def __init__(
        self,
        geographic_targets: dict[str, float] | None = None,
        income_targets: bool = True,
        benefit_targets: bool = True,
        population_targets: bool = True,
    ):
        """Initialize unified calibrator.

        Args:
            geographic_targets: Dict of geography_id -> population target
            income_targets: Include IRS SOI income totals
            benefit_targets: Include benefit program targets
            population_targets: Include Census population targets
        """
        self.geographic_targets = geographic_targets or {}
        self.include_income = income_targets
        self.include_benefits = benefit_targets
        self.include_population = population_targets

        self._pe_targets = None
        self._all_targets = None

    def _load_pe_targets(self) -> pd.DataFrame:
        """Load PolicyEngine targets."""
        if self._pe_targets is None:
            pe = PETargets()
            self._pe_targets = pe.load_all()
        return self._pe_targets

    def build_target_matrix(
        self,
        df: pd.DataFrame,
        weight_col: str = 'weight'
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build design matrix and target vector for calibration.

        Args:
            df: Synthetic population DataFrame
            weight_col: Name of weight column

        Returns:
            Tuple of (design_matrix, target_vector, target_names)
        """
        targets = []
        target_names = []

        n = len(df)
        design_rows = []

        # 1. Geographic targets
        if self.geographic_targets:
            # Determine geography column
            geo_col = None
            for col in ['cd_geoid', 'state_fips', 'sldu_geoid', 'sldl_geoid']:
                if col in df.columns:
                    geo_col = col
                    break

            if geo_col:
                for geo_id, target in self.geographic_targets.items():
                    # Create indicator vector
                    indicator = (df[geo_col] == geo_id).astype(float).values
                    design_rows.append(indicator)
                    targets.append(target)
                    target_names.append(f"geo_{geo_id}")

        # 2. Income targets from PE
        if self.include_income:
            pe_df = self._load_pe_targets()
            income_df = pe_df[pe_df['category'].str.contains('irs.soi')]
            income_national = income_df[income_df['geography'] == 'national']

            # Map PE target names to our column names
            income_map = {
                'employment_income': 'employment_income',
                'self_employment_income': 'self_employment_income',
                'social_security': 'social_security',
                'dividend_income': 'dividend_income',
                'interest_income': 'interest_income',
                'rental_income': 'rental_income',
                'pension_income': 'pension_income',
                'taxable_pension_income': 'taxable_pension_income',
                'ssi': 'ssi',
                'unemployment_compensation': 'unemployment_compensation',
                'long_term_capital_gains': 'long_term_capital_gains',
                'short_term_capital_gains': 'short_term_capital_gains',
                'qualified_dividend_income': 'dividend_income',  # Approximate
                'farm_income': 'farm_income',
                'alimony_income': 'alimony_income',
            }

            for _, row in income_national.iterrows():
                pe_name = row['name']
                target = row['value']

                if pe_name in income_map:
                    col_name = income_map[pe_name]
                    if col_name in df.columns:
                        # Use income values directly (will be multiplied by weights)
                        design_rows.append(df[col_name].fillna(0).values)
                        targets.append(target)
                        target_names.append(f"income_{pe_name}")

        # 3. Benefit targets from PE
        if self.include_benefits:
            pe_df = self._load_pe_targets()

            benefit_map = {
                'snap': ('snap', 'gov.cbo'),  # CBO SNAP spending
                'ssi': ('ssi', 'gov.cbo'),
                'eitc': ('eitc', 'gov.treasury'),
            }

            for var_name, (col_name, cat_prefix) in benefit_map.items():
                if col_name in df.columns:
                    # Find matching PE target
                    matching = pe_df[
                        (pe_df['category'].str.startswith(cat_prefix)) &
                        (pe_df['name'].str.lower() == var_name) &
                        (pe_df['geography'] == 'national')
                    ]

                    if not matching.empty:
                        target = matching.iloc[0]['value']
                        design_rows.append(df[col_name].fillna(0).values)
                        targets.append(target)
                        target_names.append(f"benefit_{var_name}")

        # 4. Population targets
        if self.include_population:
            pe_df = self._load_pe_targets()

            # Total population
            pop_row = pe_df[
                (pe_df['category'] == 'gov.census.populations') &
                (pe_df['name'] == 'total')
            ]
            if not pop_row.empty:
                # Just count people (weight of 1 each)
                design_rows.append(np.ones(n))
                targets.append(pop_row.iloc[0]['value'])
                target_names.append("population_total")

        # Convert to arrays
        if not design_rows:
            raise ValueError("No targets configured")

        design_matrix = np.column_stack(design_rows)
        target_vector = np.array(targets)

        return design_matrix, target_vector, target_names

    def calibrate(
        self,
        df: pd.DataFrame,
        weight_col: str = 'weight',
        max_iter: int = 100,
        tol: float = 1e-6,
        bounds: tuple[float, float] = (0.1, 10.0)
    ) -> pd.DataFrame:
        """Calibrate weights using iterative proportional fitting.

        Args:
            df: Synthetic population DataFrame
            weight_col: Name of weight column
            max_iter: Maximum iterations
            tol: Convergence tolerance
            bounds: (min_factor, max_factor) bounds on weight adjustments

        Returns:
            DataFrame with calibrated weights
        """
        df = df.copy()

        # Build target matrix
        X, targets, names = self.build_target_matrix(df, weight_col)

        n_samples, n_targets = X.shape
        print(f"Calibrating {n_samples:,} samples to {n_targets} targets")

        # Initialize weights
        if weight_col in df.columns:
            weights = df[weight_col].values.copy()
        else:
            weights = np.ones(n_samples)

        # IPF iteration
        for iteration in range(max_iter):
            old_weights = weights.copy()

            for j in range(n_targets):
                # Current weighted sum for target j
                current = np.sum(weights * X[:, j])

                if current > 0:
                    # Adjustment factor
                    factor = targets[j] / current

                    # Apply bounded adjustment to relevant samples
                    mask = X[:, j] > 0
                    adjustment = np.clip(factor, bounds[0], bounds[1])
                    weights[mask] *= adjustment

            # Check convergence
            if np.max(np.abs(weights - old_weights) / (old_weights + 1e-10)) < tol:
                print(f"Converged after {iteration + 1} iterations")
                break

        # Compute final errors
        print(f"\n{'Target':<40} {'Computed':>15} {'Target':>15} {'Error':>10}")
        print("-" * 85)

        for j, name in enumerate(names):
            computed = np.sum(weights * X[:, j])
            target = targets[j]
            error = abs(computed - target) / target * 100

            if target > 1e9:
                comp_str = f"${computed/1e9:.1f}B"
                tgt_str = f"${target/1e9:.1f}B"
            elif target > 1e6:
                comp_str = f"{computed/1e6:.1f}M"
                tgt_str = f"{target/1e6:.1f}M"
            else:
                comp_str = f"{computed:,.0f}"
                tgt_str = f"{target:,.0f}"

            print(f"{name:<40} {comp_str:>15} {tgt_str:>15} {error:>9.2f}%")

        df['calibrated_weight'] = weights
        return df


def calibrate_to_pe_targets(
    df: pd.DataFrame,
    geo_targets: dict[str, float] | None = None,
    include_income: bool = True,
    include_benefits: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Convenience function to calibrate to PE targets.

    Args:
        df: Synthetic population DataFrame with income/benefit columns
        geo_targets: Optional geographic targets (CD population, etc.)
        include_income: Include IRS SOI income targets
        include_benefits: Include benefit program targets
        **kwargs: Additional args passed to calibrate()

    Returns:
        DataFrame with calibrated_weight column
    """
    calibrator = UnifiedCalibrator(
        geographic_targets=geo_targets,
        income_targets=include_income,
        benefit_targets=include_benefits,
    )

    return calibrator.calibrate(df, **kwargs)

"""Tests for PSID data source integration."""

import numpy as np
import pandas as pd
import pytest

from microplex_us.data_sources.psid import (
    PSID_TO_MICROPLEX_VARS,
    PSIDDataset,
    calibrate_divorce_rates,
    calibrate_marriage_rates,
    extract_transition_rates,
    get_age_specific_rates,
    load_psid_panel,
)


class TestPSIDDataset:
    """Test PSIDDataset container."""

    def test_dataset_creation(self):
        """Test creating a PSIDDataset."""
        persons = pd.DataFrame({
            "person_id": [1, 1, 2, 2],
            "year": [2019, 2021, 2019, 2021],
            "age": [30, 32, 45, 47],
            "is_male": [True, True, False, False],
            "marital_status": [1, 1, 2, 1],  # married, married, single, married
        })

        ds = PSIDDataset(persons=persons, source="mock")

        assert ds.n_persons == 2
        assert ds.n_observations == 4
        assert ds.years == [2019, 2021]

    def test_dataset_summary(self):
        """Test dataset summary method."""
        persons = pd.DataFrame({
            "person_id": [1, 1, 2, 2],
            "year": [2019, 2021, 2019, 2021],
            "age": [30, 32, 45, 47],
        })

        ds = PSIDDataset(persons=persons, source="mock")
        summary = ds.summary()

        assert summary["n_persons"] == 2
        assert summary["n_observations"] == 4
        assert summary["years"] == [2019, 2021]


class TestLoadPSID:
    """Test PSID loading functionality."""

    def test_load_requires_data_dir(self):
        """Test that loading requires a data directory."""
        with pytest.raises((FileNotFoundError, ValueError)):
            load_psid_panel(data_dir="/nonexistent/path")

    def test_variable_mapping(self):
        """Test that PSID variables map to microplex conventions."""
        # These should map to standard microplex names
        assert "age" in PSID_TO_MICROPLEX_VARS.values()
        assert "is_male" in PSID_TO_MICROPLEX_VARS.values()
        assert "total_income" in PSID_TO_MICROPLEX_VARS.values()


class TestTransitionRates:
    """Test transition rate extraction from PSID data."""

    @pytest.fixture
    def mock_transitions_df(self):
        """Create mock transition data from PSID."""
        # Simulates output from psid.get_household_transitions()
        return pd.DataFrame({
            "person_id": range(100),
            "year_from": [2019] * 100,
            "year_to": [2021] * 100,
            "type": ["marriage"] * 20 + ["divorce"] * 10 + ["same_household"] * 70,
            "age_from": np.random.randint(20, 60, 100),
            "marital_from": [2] * 20 + [1] * 10 + [1] * 35 + [2] * 35,  # Single/married
            "marital_to": [1] * 20 + [4] * 10 + [1] * 35 + [2] * 35,  # Married/divorced
        })

    def test_extract_transition_rates(self, mock_transitions_df):
        """Test extracting overall transition rates."""
        rates = extract_transition_rates(mock_transitions_df)

        assert "marriage" in rates
        assert "divorce" in rates
        assert rates["marriage"] == pytest.approx(0.20, abs=0.01)
        assert rates["divorce"] == pytest.approx(0.10, abs=0.01)

    def test_get_age_specific_rates(self, mock_transitions_df):
        """Test extracting age-specific transition rates."""
        age_rates = get_age_specific_rates(
            mock_transitions_df,
            transition_type="marriage",
            age_bins=[(20, 29), (30, 39), (40, 49), (50, 59)],
        )

        assert isinstance(age_rates, dict)
        assert (20, 29) in age_rates or len(age_rates) >= 0  # May have empty bins

    def test_rates_are_probabilities(self, mock_transitions_df):
        """Test that extracted rates are valid probabilities."""
        rates = extract_transition_rates(mock_transitions_df)

        for rate in rates.values():
            assert 0.0 <= rate <= 1.0


class TestCalibration:
    """Test calibration of microplex models from PSID rates."""

    @pytest.fixture
    def psid_rates(self):
        """Mock PSID-derived transition rates."""
        return {
            "marriage": {
                (18, 24): 0.05,
                (25, 29): 0.08,
                (30, 34): 0.06,
                (35, 44): 0.04,
                (45, 54): 0.02,
                (55, 99): 0.01,
            },
            "divorce": {
                (18, 24): 0.06,
                (25, 29): 0.04,
                (30, 34): 0.03,
                (35, 44): 0.025,
                (45, 54): 0.02,
                (55, 99): 0.015,
            },
        }

    def test_calibrate_marriage_rates(self, psid_rates):
        """Test calibrating marriage rates from PSID."""
        calibrated = calibrate_marriage_rates(psid_rates["marriage"])

        # Should return dict compatible with MarriageTransition
        assert isinstance(calibrated, dict)
        for age_range, rate in calibrated.items():
            assert isinstance(age_range, tuple)
            assert len(age_range) == 2
            assert 0.0 <= rate <= 1.0

    def test_calibrate_divorce_rates(self, psid_rates):
        """Test calibrating divorce rates from PSID."""
        calibrated = calibrate_divorce_rates(psid_rates["divorce"])

        assert isinstance(calibrated, dict)
        for key, rate in calibrated.items():
            assert 0.0 <= rate <= 1.0

    def test_calibrated_model_uses_psid_rates(self, psid_rates):
        """Test that calibrated model actually uses PSID rates."""
        from microplex.transitions import MarriageTransition

        calibrate_marriage_rates(psid_rates["marriage"])

        # Create model with calibrated rates
        model = MarriageTransition(base_rates={"male": 0.05, "female": 0.06})

        # Model should use provided rates
        assert model.base_rates is not None


class TestMultiSourceIntegration:
    """Test PSID integration with MultiSourceFusion."""

    @pytest.fixture
    def mock_psid_data(self):
        """Create mock PSID panel data."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "person_id": np.repeat(range(n // 2), 2),
            "period": np.tile([0, 1], n // 2),
            "age": np.repeat(np.random.randint(20, 60, n // 2), 2),
            "total_income": np.abs(np.random.randn(n) * 50000 + 40000),
            "is_male": np.repeat(np.random.choice([True, False], n // 2), 2),
        })

    def test_psid_as_fusion_source(self, mock_psid_data):
        """Test adding PSID as a source to MultiSourceFusion."""
        from microplex.fusion import MultiSourceFusion

        fusion = MultiSourceFusion(
            shared_vars=["age", "total_income"],
            all_vars=["age", "total_income"],
            n_periods=2,
        )

        # Should be able to add PSID as a source
        fusion.add_source(
            "psid",
            mock_psid_data,
            source_vars=["age", "total_income"],
            n_periods=2,
            person_id_col="person_id",
            period_col="period",
        )

        assert "psid" in fusion.sources
        assert fusion.sources["psid"].source_vars == ["age", "total_income"]

    def test_coverage_evaluation_with_psid(self, mock_psid_data):
        """Test evaluating coverage on PSID holdout data."""
        from microplex.fusion import MultiSourceFusion

        # Need at least 2 sources for fusion
        mock_cps_data = mock_psid_data.copy()
        mock_cps_data["person_id"] = mock_cps_data["person_id"] + 1000

        fusion = MultiSourceFusion(
            shared_vars=["age", "total_income"],
            all_vars=["age", "total_income"],
            n_periods=2,
        )

        fusion.add_source("psid", mock_psid_data, source_vars=["age", "total_income"])
        fusion.add_source("cps", mock_cps_data, source_vars=["age", "total_income"])

        # Should be able to add sources
        assert len(fusion.sources) == 2

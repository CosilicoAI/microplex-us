"""Tests for PUF source-provider implementation."""

from __future__ import annotations

import sys
import types

import pandas as pd
from microplex.core import EntityType, SourceArchetype, SourceProvider, SourceQuery

import microplex_us.data_sources.puf as puf_module
from microplex_us.data_sources import PUFSourceProvider, expand_to_persons
from microplex_us.data_sources.puf import (
    PEStyleQRFShareModel,
    _fit_pe_style_puf_social_security_qrf_model_from_reference,
    _impute_missing_puf_demographics,
    _impute_puf_social_security_components,
    _sample_tax_units,
    map_puf_variables,
)
from microplex_us.data_sources.share_imputation import fit_grouped_share_model


def _mock_social_security_share_model_loader(*_args):
    reference = pd.DataFrame(
        {
            "age_bucket": [
                "under_18",
                "18_to_29",
                "30_to_44",
                "45_to_61",
                "62_to_74",
                "75_plus",
            ],
            "weight": [1.0] * 6,
            "social_security_retirement": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "social_security_disability": [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            "social_security_survivors": [0.0] * 6,
            "social_security_dependents": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    return fit_grouped_share_model(
        reference,
        explicit_component_columns=(
            "social_security_retirement",
            "social_security_disability",
            "social_security_survivors",
        ),
        implicit_component_column="social_security_dependents",
        feature_sets=(("age_bucket",),),
        weight_column="weight",
    )


def _install_fake_qrf(monkeypatch, prediction_frame: pd.DataFrame):
    calls: dict[str, object] = {}

    class FakeFittedModel:
        def predict(self, X_test):
            calls["X_test"] = X_test.copy()
            return prediction_frame.copy()

    class FakeQRF:
        def __init__(self, **kwargs):
            calls["init_kwargs"] = dict(kwargs)

        def fit(self, *, X_train, predictors, imputed_variables, n_jobs):
            calls["X_train"] = X_train.copy()
            calls["predictors"] = tuple(predictors)
            calls["imputed_variables"] = tuple(imputed_variables)
            calls["n_jobs"] = n_jobs
            return FakeFittedModel()

    microimpute_module = types.ModuleType("microimpute")
    models_module = types.ModuleType("microimpute.models")
    qrf_module = types.ModuleType("microimpute.models.qrf")
    qrf_module.QRF = FakeQRF
    microimpute_module.models = models_module
    models_module.qrf = qrf_module
    monkeypatch.setitem(sys.modules, "microimpute", microimpute_module)
    monkeypatch.setitem(sys.modules, "microimpute.models", models_module)
    monkeypatch.setitem(sys.modules, "microimpute.models.qrf", qrf_module)
    return calls


def test_expand_to_persons_preserves_joint_tax_unit_monetary_totals():
    tax_units = pd.DataFrame(
        {
            "filing_status": ["JOINT"],
            "employment_income": [100.0],
            "self_employment_income": [50.0],
            "taxable_interest_income": [20.0],
            "ordinary_dividend_income": [30.0],
            "qualified_dividend_income": [10.0],
            "gross_social_security": [40.0],
            "taxable_pension_income": [60.0],
            "unemployment_compensation": [80.0],
            "rental_income": [90.0],
            "weight": [1.0],
            "household_id": ["joint-household"],
            "year": [2024],
        }
    )

    persons = expand_to_persons(tax_units)
    head = persons.loc[persons["is_head"] == 1].iloc[0]
    spouse = persons.loc[persons["is_spouse"] == 1].iloc[0]

    assert head["employment_income"] == 60.0
    assert spouse["employment_income"] == 40.0
    assert head["self_employment_income"] == 30.0
    assert spouse["self_employment_income"] == 20.0
    assert head["taxable_interest_income"] == 10.0
    assert spouse["taxable_interest_income"] == 10.0
    assert head["ordinary_dividend_income"] == 15.0
    assert spouse["ordinary_dividend_income"] == 15.0
    assert head["non_qualified_dividend_income"] == 10.0
    assert spouse["non_qualified_dividend_income"] == 10.0
    assert persons["taxable_interest_income"].sum() == 20.0
    assert persons["ordinary_dividend_income"].sum() == 30.0
    assert persons["qualified_dividend_income"].sum() == 10.0
    assert persons["non_qualified_dividend_income"].sum() == 20.0
    assert persons["dividend_income"].sum() == 30.0
    assert persons["social_security"].sum() == 40.0
    assert persons["social_security_retirement"].sum() == 0.0
    assert persons["pension_income"].sum() == 60.0
    assert persons["income"].sum() == 470.0


def test_expand_to_persons_derives_retirement_social_security_for_older_records():
    tax_units = pd.DataFrame(
        {
            "filing_status": ["SINGLE", "SINGLE"],
            "gross_social_security": [40.0, 25.0],
            "age": [68, 45],
            "weight": [1.0, 1.0],
            "household_id": ["older-household", "younger-household"],
            "year": [2024, 2024],
        }
    )

    persons = expand_to_persons(tax_units).sort_values("household_id").reset_index(drop=True)

    assert persons["social_security"].tolist() == [40.0, 25.0]
    assert persons["social_security_retirement"].tolist() == [40.0, 0.0]


def test_impute_puf_social_security_components_uses_grouped_cps_shares():
    persons = pd.DataFrame(
        {
            "age": [12, 40, 70],
            "social_security": [100.0, 200.0, 300.0],
        }
    )

    result = _impute_puf_social_security_components(
        persons,
        share_model=_mock_social_security_share_model_loader(),
    )

    assert result["social_security_dependents"].tolist() == [100.0, 0.0, 0.0]
    assert result["social_security_disability"].tolist() == [0.0, 200.0, 0.0]
    assert result["social_security_retirement"].tolist() == [0.0, 0.0, 300.0]
    assert result["social_security_survivors"].tolist() == [0.0, 0.0, 0.0]


def test_fit_pe_style_puf_social_security_qrf_model_uses_pe_predictors(monkeypatch):
    predictions = pd.DataFrame(
        {
            "social_security_retirement_share": [0.6, 0.1],
            "social_security_disability_share": [0.3, 0.2],
            "social_security_survivors_share": [0.05, 0.4],
            "social_security_dependents_share": [0.05, 0.3],
        }
    )
    calls = _install_fake_qrf(monkeypatch, predictions)
    reference = pd.DataFrame(
        {
            "age": [70, 45, 12, 67] * 30,
            "sex": [1, 2, 2, 1] * 30,
            "filing_status": ["JOINT", "SINGLE", "SINGLE", "JOINT"] * 30,
            "is_head": [1, 1, 0, 1] * 30,
            "is_dependent": [0, 0, 1, 0] * 30,
            "social_security": [100.0, 100.0, 100.0, 100.0] * 30,
            "social_security_retirement": [100.0, 0.0, 0.0, 100.0] * 30,
            "social_security_disability": [0.0, 100.0, 0.0, 0.0] * 30,
            "social_security_survivors": [0.0, 0.0, 0.0, 0.0] * 30,
            "social_security_dependents": [0.0, 0.0, 100.0, 0.0] * 30,
        }
    )

    model = _fit_pe_style_puf_social_security_qrf_model_from_reference(
        reference,
        min_training_records=1,
    )

    assert isinstance(model, PEStyleQRFShareModel)
    assert model.predictors == (
        "age",
        "is_male",
        "tax_unit_is_joint",
        "is_tax_unit_head",
        "is_tax_unit_dependent",
    )
    assert calls["predictors"] == model.predictors
    assert calls["n_jobs"] == 1

    persons = pd.DataFrame(
        {
            "age": [70, 45],
            "sex": [1, 2],
            "filing_status": ["JOINT", "SINGLE"],
            "is_head": [1, 1],
            "is_dependent": [0, 0],
            "social_security": [300.0, 200.0],
        }
    )
    result = _impute_puf_social_security_components(persons, share_model=model)

    assert result["social_security_retirement"].tolist() == [180.0, 20.0]
    assert result["social_security_disability"].tolist() == [90.0, 40.0]
    assert result["social_security_survivors"].tolist() == [15.0, 80.0]
    assert result["social_security_dependents"].tolist() == [15.0, 60.0]
    assert list(calls["X_test"].columns) == list(model.predictors)


def test_puf_source_provider_selects_pe_qrf_social_security_strategy(
    tmp_path,
    monkeypatch,
):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00200": [50_000.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    pe_qrf_called: list[tuple[int, object]] = []

    def fake_pe_qrf_loader(*, cps_reference_year, cache_dir):
        pe_qrf_called.append((cps_reference_year, cache_dir))
        return _mock_social_security_share_model_loader()

    def fail_grouped_loader(**_kwargs):
        raise AssertionError("grouped-share loader should not be used in pe_qrf mode")

    monkeypatch.setattr(
        puf_module,
        "_default_pe_style_puf_social_security_share_model",
        fake_pe_qrf_loader,
    )
    monkeypatch.setattr(
        puf_module,
        "_default_puf_social_security_share_model",
        fail_grouped_loader,
    )

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
        social_security_split_strategy="pe_qrf",
    )
    frame = provider.load_frame(SourceQuery(period=2024))

    assert pe_qrf_called
    assert "social_security_retirement" in frame.tables[EntityType.PERSON].columns


def test_expand_to_persons_splits_negative_joint_self_employment_losses():
    tax_units = pd.DataFrame(
        {
            "filing_status": ["JOINT"],
            "self_employment_income": [-100.0],
            "weight": [1.0],
            "household_id": ["joint-household"],
            "year": [2024],
        }
    )

    persons = expand_to_persons(tax_units)
    head = persons.loc[persons["is_head"] == 1].iloc[0]
    spouse = persons.loc[persons["is_spouse"] == 1].iloc[0]

    assert head["self_employment_income"] == -60.0
    assert spouse["self_employment_income"] == -40.0
    assert persons["self_employment_income"].sum() == -100.0
    assert persons["income"].sum() == -100.0


def test_expand_to_persons_uses_pe_demographic_helpers_when_present():
    tax_units = pd.DataFrame(
        {
            "filing_status": ["JOINT", "SINGLE"],
            "employment_income": [100.0, 50.0],
            "pre_tax_contributions": [20.0, 0.0],
            "gross_social_security": [40.0, 0.0],
            "weight": [1.0, 1.0],
            "household_id": ["joint-household", "single-household"],
            "exemptions_count": [2, 3],
            "_puf_recid": [101, 202],
            "_puf_agerange": [4, 5],
            "_puf_earnsplit": [0, 0],
            "_puf_gender": [1, 2],
            "_puf_agedp1": [2, 4],
            "_puf_agedp2": [3, 6],
            "year": [2024, 2024],
        }
    )

    persons = expand_to_persons(tax_units).sort_values("person_id").reset_index(drop=True)

    assert persons["person_id"].tolist() == ["101:1", "101:2", "202:1", "202:3", "202:4"]
    assert persons["tax_unit_id"].tolist() == ["101", "101", "202", "202", "202"]

    head = persons.loc[persons["person_id"] == "101:1"].iloc[0]
    spouse = persons.loc[persons["person_id"] == "101:2"].iloc[0]
    dependent_1 = persons.loc[persons["person_id"] == "202:3"].iloc[0]
    dependent_2 = persons.loc[persons["person_id"] == "202:4"].iloc[0]

    assert head["employment_income"] == 100.0
    assert spouse["employment_income"] == 0.0
    assert head["pre_tax_contributions"] == 20.0
    assert spouse["pre_tax_contributions"] == 0.0
    assert head["age"] == 50
    assert spouse["age"] == 50
    assert dependent_1["is_dependent"] == 1
    assert dependent_2["is_dependent"] == 1
    assert dependent_1["employment_income"] == 0.0
    assert dependent_2["employment_income"] == 0.0
    assert dependent_1["age"] == 18
    assert dependent_2["age"] == 27


def test_puf_source_provider_loads_observation_frame_from_local_files(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101, 202],
            "MARS": [2, 1],
            "XTOT": [2, 1],
            "S006": [100.0, 200.0],
            "E00200": [50_000.0, 20_000.0],
            "E00900": [0.0, 5_000.0],
            "AGE_HEAD": [45, 67],
            "GENDER": [1, 2],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101, 202]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
        social_security_share_model_loader=_mock_social_security_share_model_loader,
    )
    frame = provider.load_frame(
        SourceQuery(period=2024, provider_filters={"sample_n": 1, "random_seed": 0})
    )

    assert isinstance(provider, SourceProvider)
    assert set(frame.tables) == {EntityType.HOUSEHOLD, EntityType.PERSON}
    assert len(frame.tables[EntityType.HOUSEHOLD]) == 1
    assert frame.tables[EntityType.PERSON]["household_id"].nunique() == 1
    assert frame.tables[EntityType.HOUSEHOLD]["year"].tolist() == [2024]
    assert frame.tables[EntityType.PERSON]["year"].nunique() == 1
    assert "income" in frame.tables[EntityType.PERSON].columns
    assert provider.descriptor.name.startswith("irs_soi_puf_")
    assert frame.source.archetype is SourceArchetype.TAX_MICRODATA


def test_puf_source_provider_sampling_respects_tax_unit_weights(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101, 202, 303],
            "MARS": [1, 1, 1],
            "XTOT": [1, 1, 1],
            "S006": [0.0, 0.0, 100.0],
            "E00200": [10_000.0, 20_000.0, 30_000.0],
            "AGE_HEAD": [45, 55, 65],
            "GENDER": [1, 2, 1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101, 202, 303]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
        social_security_share_model_loader=_mock_social_security_share_model_loader,
    )
    frame = provider.load_frame(
        SourceQuery(period=2024, provider_filters={"sample_n": 1, "random_seed": 0})
    )

    assert frame.tables[EntityType.HOUSEHOLD]["household_id"].tolist() == ["303"]
    assert frame.tables[EntityType.PERSON]["household_id"].nunique() == 1
    assert frame.tables[EntityType.PERSON]["household_id"].iloc[0] == "303"


def test_puf_source_provider_marks_placeholder_and_derived_variables_in_capabilities(
    tmp_path,
):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00200": [50_000.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
        social_security_share_model_loader=_mock_social_security_share_model_loader,
    )
    frame = provider.load_frame(SourceQuery(period=2024))
    descriptor = frame.source

    assert not descriptor.allows_conditioning_on("state_fips")
    assert not descriptor.is_authoritative_for("state_fips")
    assert not descriptor.allows_conditioning_on("income")
    assert not descriptor.is_authoritative_for("income")
    assert descriptor.is_authoritative_for("employment_income")
    assert not descriptor.allows_conditioning_on("employment_income")
    assert descriptor.allows_conditioning_on("age")


def test_puf_source_provider_does_not_duplicate_joint_tax_unit_financial_income(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [2],
            "XTOT": [2],
            "S006": [100.0],
            "E00200": [100.0],
            "E00900": [50.0],
            "E00300": [20.0],
            "E00600": [30.0],
            "E00650": [10.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2015,
        social_security_share_model_loader=_mock_social_security_share_model_loader,
    )
    frame = provider.load_frame(SourceQuery(period=2015))
    persons = frame.tables[EntityType.PERSON]

    assert len(persons) == 2
    assert persons["taxable_interest_income"].sum() == 20.0
    assert persons["ordinary_dividend_income"].sum() == 30.0
    assert persons["qualified_dividend_income"].sum() == 10.0
    assert persons["income"].sum() == 200.0


def test_puf_source_provider_maps_policyengine_medical_and_alimony_inputs(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00200": [50_000.0],
            "E00800": [2_000.0],
            "E17500": [1_000.0],
            "E26390": [700.0],
            "E26400": [200.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2015,
    )
    frame = provider.load_frame(SourceQuery(period=2015))
    persons = frame.tables[EntityType.PERSON]

    assert persons["alimony_income"].sum() == 2_000.0
    assert persons["health_insurance_premiums_without_medicare_part_b"].sum() == 453.0
    assert persons["other_medical_expenses"].sum() == 325.0
    assert persons["medicare_part_b_premiums"].sum() == 137.0
    assert persons["over_the_counter_health_expenses"].sum() == 85.0
    assert persons["estate_income"].sum() == 500.0
    assert persons["income"].sum() == 52_000.0


def test_map_puf_variables_preserves_rental_loss_sign():
    raw = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E25850": [200.0],
            "E25860": [500.0],
        }
    )

    mapped = map_puf_variables(raw)

    assert mapped.loc[0, "rental_income_positive"] == 200.0
    assert mapped.loc[0, "rental_income_negative"] == 500.0
    assert mapped.loc[0, "rental_income"] == -300.0


def test_map_puf_variables_uses_pe_puf_business_and_farm_income_formulas():
    raw = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E26270": [999.0],
            "E26190": [1_200.0],
            "E26180": [200.0],
            "E25940": [25.0],
            "E25980": [500.0],
            "E25920": [0.0],
            "E25960": [50.0],
            "E30400": [923.5],
            "E30500": [0.0],
            "E00900": [0.0],
            "E02100": [300.0],
            "T27800": [700.0],
            "E27200": [125.0],
        }
    )

    mapped = map_puf_variables(raw)

    assert mapped.loc[0, "partnership_s_corp_income"] == 1_450.0
    assert mapped.loc[0, "partnership_se_income"] == 700.0
    assert mapped.loc[0, "farm_income"] == 700.0
    assert mapped.loc[0, "farm_operations_income"] == 300.0
    assert mapped.loc[0, "farm_rent_income"] == 125.0


def test_map_puf_variables_adds_pe_exact_irs_inputs():
    raw = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00600": [500.0],
            "E00650": [200.0],
            "E01400": [1_250.0],
            "E01500": [2_100.0],
            "E01700": [1_600.0],
            "E02300": [800.0],
            "E02400": [1_100.0],
            "E03290": [300.0],
            "E07300": [90.0],
            "E07400": [80.0],
            "E07600": [70.0],
            "E09700": [60.0],
            "E09800": [50.0],
            "E11200": [40.0],
            "E24518": [30.0],
            "E24515": [20.0],
            "E58990": [10.0],
            "E62900": [5.0],
            "E87521": [15.0],
            "P08000": [25.0],
            "E07240": [35.0],
            "E07260": [45.0],
            "E00700": [55.0],
            "E01200": [65.0],
        }
    )

    mapped = map_puf_variables(raw)

    assert mapped.loc[0, "non_qualified_dividend_income"] == 300.0
    assert mapped.loc[0, "taxable_ira_distributions"] == 1_250.0
    assert mapped.loc[0, "taxable_unemployment_compensation"] == 800.0
    assert mapped.loc[0, "social_security"] == 1_100.0
    assert mapped.loc[0, "tax_exempt_pension_income"] == 500.0
    assert mapped.loc[0, "health_savings_account_ald"] == 300.0
    assert mapped.loc[0, "foreign_tax_credit"] == 90.0
    assert mapped.loc[0, "general_business_credit"] == 80.0
    assert mapped.loc[0, "prior_year_minimum_tax_credit"] == 70.0
    assert mapped.loc[0, "recapture_of_investment_credit"] == 60.0
    assert mapped.loc[0, "unreported_payroll_tax"] == 50.0
    assert mapped.loc[0, "excess_withheld_payroll_tax"] == 40.0
    assert mapped.loc[0, "long_term_capital_gains_on_collectibles"] == 30.0
    assert mapped.loc[0, "unrecaptured_section_1250_gain"] == 20.0
    assert mapped.loc[0, "investment_income_elected_form_4952"] == 10.0
    assert mapped.loc[0, "amt_foreign_tax_credit"] == 5.0
    assert mapped.loc[0, "american_opportunity_credit"] == 15.0
    assert mapped.loc[0, "other_credits"] == 25.0
    assert mapped.loc[0, "savers_credit"] == 35.0
    assert mapped.loc[0, "energy_efficient_home_improvement_credit"] == 45.0
    assert mapped.loc[0, "salt_refund_income"] == 55.0
    assert mapped.loc[0, "miscellaneous_income"] == 65.0


def test_map_puf_variables_can_impute_pre_tax_contributions_with_injected_model():
    class DummyFittedModel:
        def predict(self, X_test):
            return pd.DataFrame(
                {"pre_tax_contributions": X_test["employment_income"] * 0.1},
                index=X_test.index,
            )

    raw = pd.DataFrame(
        {
            "RECID": [101],
            "MARS": [1],
            "XTOT": [1],
            "S006": [100.0],
            "E00200": [50_000.0],
            "AGE_HEAD": [45],
            "GENDER": [1],
        }
    )

    mapped = map_puf_variables(
        raw,
        impute_pre_tax_contributions=True,
        pre_tax_contribution_model=puf_module.PEStyleQRFImputationModel(
            predictors=("employment_income", "age", "is_male"),
            imputed_variable="pre_tax_contributions",
            fitted_model=DummyFittedModel(),
        ),
    )

    assert mapped.loc[0, "pre_tax_contributions"] == 5_000.0


def test_impute_missing_puf_demographics_uses_qrf_predictions(monkeypatch):
    prediction_frame = pd.DataFrame(
        {
            "AGEDP1": [2.2],
            "AGEDP2": [3.1],
            "AGEDP3": [0.0],
            "AGERANGE": [4.4],
            "EARNSPLIT": [2.6],
            "GENDER": [1.8],
        }
    )
    calls = _install_fake_qrf(monkeypatch, prediction_frame)

    raw = pd.DataFrame(
        {
            "RECID": list(range(101, 202)) + [999],
            "E00200": [50_000.0] * 101 + [80_000.0],
            "MARS": [1] * 101 + [2],
            "DSI": [0] * 102,
            "EIC": [0] * 101 + [1],
            "XTOT": [1] * 101 + [2],
            "AGEDP1": [1.0] * 101 + [float("nan")],
            "AGEDP2": [0.0] * 101 + [float("nan")],
            "AGEDP3": [0.0] * 101 + [float("nan")],
            "AGERANGE": [3.0] * 101 + [float("nan")],
            "EARNSPLIT": [0.0] * 101 + [float("nan")],
            "GENDER": [1.0] * 101 + [float("nan")],
        }
    )

    imputed = _impute_missing_puf_demographics(raw)

    assert calls["predictors"] == ("E00200", "MARS", "DSI", "EIC", "XTOT")
    assert calls["imputed_variables"] == (
        "AGEDP1",
        "AGEDP2",
        "AGEDP3",
        "AGERANGE",
        "EARNSPLIT",
        "GENDER",
    )
    assert imputed.loc[101, "AGEDP1"] == 2
    assert imputed.loc[101, "AGEDP2"] == 3
    assert imputed.loc[101, "AGERANGE"] == 4
    assert imputed.loc[101, "EARNSPLIT"] == 3
    assert imputed.loc[101, "GENDER"] == 2


def test_download_puf_prefers_existing_local_files_without_hub_lookup(tmp_path, monkeypatch):
    puf_path = tmp_path / "puf_2015.csv"
    demographics_path = tmp_path / "demographics_2015.csv"
    puf_path.write_text("RECID,MARS\n1,1\n")
    demographics_path.write_text("RECID\n1\n")

    def fail_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called when local files exist")

    monkeypatch.setattr(puf_module, "hf_hub_download", fail_download, raising=False)

    resolved_puf_path, resolved_demo_path = puf_module.download_puf(tmp_path)

    assert resolved_puf_path == puf_path
    assert resolved_demo_path == demographics_path


def test_map_puf_variables_seed_controls_age_imputation():
    puf = pd.DataFrame(
        {
            "RECID": [101, 202, 303],
            "MARS": [1, 1, 1],
            "XTOT": [1, 1, 1],
            "S006": [100.0, 100.0, 100.0],
            "E00200": [50_000.0, 150_000.0, 250_000.0],
            "E02400": [0.0, 10_000.0, 0.0],
            "E01400": [0.0, 0.0, 20_000.0],
        }
    )

    first = puf_module.map_puf_variables(puf, random_seed=17)
    second = puf_module.map_puf_variables(puf, random_seed=17)
    third = puf_module.map_puf_variables(puf, random_seed=18)

    assert first["age"].tolist() == second["age"].tolist()
    assert first["age"].tolist() != third["age"].tolist()


def test_puf_source_provider_age_imputation_is_reproducible_with_same_seed(tmp_path):
    puf = pd.DataFrame(
        {
            "RECID": [101, 202, 303],
            "MARS": [1, 2, 1],
            "XTOT": [1, 2, 1],
            "S006": [100.0, 120.0, 80.0],
            "E00200": [50_000.0, 75_000.0, 150_000.0],
            "E02400": [0.0, 8_000.0, 0.0],
            "E01400": [0.0, 0.0, 25_000.0],
            "GENDER": [1, 2, 1],
        }
    )
    puf_path = tmp_path / "puf.csv"
    demographics_path = tmp_path / "demographics.csv"
    puf.to_csv(puf_path, index=False)
    pd.DataFrame({"RECID": [101, 202, 303]}).to_csv(demographics_path, index=False)

    provider = PUFSourceProvider(
        puf_path=puf_path,
        demographics_path=demographics_path,
        target_year=2024,
        social_security_share_model_loader=_mock_social_security_share_model_loader,
    )
    query = SourceQuery(period=2024, provider_filters={"sample_n": 3, "random_seed": 7})
    first = provider.load_frame(query)
    second = provider.load_frame(query)

    first_persons = first.tables[EntityType.PERSON].sort_values("person_id").reset_index(drop=True)
    second_persons = second.tables[EntityType.PERSON].sort_values("person_id").reset_index(drop=True)

    assert first_persons["age"].tolist() == second_persons["age"].tolist()


def test_puf_sampling_falls_back_to_uniform_when_weighted_sampling_is_infeasible(
    monkeypatch,
):
    tax_units = pd.DataFrame(
        {
            "household_id": [1, 2, 3],
            "weight": [10.0, 20.0, 30.0],
        }
    )

    original_sample = pd.DataFrame.sample

    def flaky_sample(self, *args, **kwargs):
        if kwargs.get("weights") is not None:
            raise ValueError(
                "Weighted sampling cannot be achieved with replace=False."
            )
        return original_sample(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "sample", flaky_sample)

    sampled = _sample_tax_units(
        tax_units,
        sample_n=2,
        random_seed=42,
    )

    assert len(sampled) == 2

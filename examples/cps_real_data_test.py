"""
CPS Real Data Test for HierarchicalSynthesizer

Complete workflow testing microplex on real CPS ASEC data:
1. Load CPS data using load_cps_for_synthesis()
2. Fit HierarchicalSynthesizer on real data
3. Generate synthetic households and persons
4. Evaluate quality using multivariate metrics:
   - Distance correlation (dCor) preservation
   - Energy distance
   - Nearest neighbor authenticity
5. Compare derived HH income distribution to real
6. Test reweighting to state-level targets

Run with: python examples/cps_real_data_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Add paths for local development
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from microplex import HierarchicalSynthesizer, HouseholdSchema, Reweighter

from microplex_us.data import create_sample_data, load_cps_for_synthesis


def setup_cps_schema():
    """Create a schema tailored for CPS ASEC data."""
    return HouseholdSchema(
        hh_vars=["n_persons", "n_adults", "n_children", "state_fips", "tenure"],
        person_vars=["age", "sex", "income", "employment_status", "education"],
        person_condition_vars=[
            "n_persons",
            "n_adults",
            "n_children",
            "state_fips",
            "tenure",
            "person_number",
            "is_first_adult",
            "is_child_slot",
        ],
        derived_vars={
            "hh_income": "sum:income",
            "n_workers": "count:employment_status==1",
        },
        hh_id_col="household_id",
        person_id_col="person_id",
    )


# ============================================================================
# MULTIVARIATE METRICS
# ============================================================================

def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute Distance Correlation between two variables.

    dCor = 0 iff X and Y are independent. Unlike Pearson correlation,
    distance correlation captures ALL types of dependence (linear,
    nonlinear, non-monotonic).
    """
    n = len(X)
    if n < 2:
        return 0.0

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)

    a = np.abs(X[:, None] - X[None, :])
    b = np.abs(Y[:, None] - Y[None, :])

    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = (A * B).sum() / (n * n)
    dvar_x = (A * A).sum() / (n * n)
    dvar_y = (B * B).sum() / (n * n)

    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0

    dcor = np.sqrt(dcov2 / np.sqrt(dvar_x * dvar_y))
    return float(np.clip(dcor, 0, 1))


def compute_dcor_preservation(
    original: pd.DataFrame,
    synthetic: pd.DataFrame,
    variables: list,
    sample_size: int = 500,
) -> dict:
    """
    Compare distance correlation structure between original and synthetic.

    Returns dict with:
    - original_dcors: dCor matrix for original data
    - synthetic_dcors: dCor matrix for synthetic data
    - dcor_errors: absolute error per pair
    - mean_dcor_error: average error across all pairs
    """
    # Sample for efficiency
    n_sample = min(sample_size, len(original), len(synthetic))
    np.random.seed(42)
    orig_sample = original.sample(n=n_sample, random_state=42)
    synth_sample = synthetic.sample(n=n_sample, random_state=42)

    available_vars = [
        v for v in variables
        if v in orig_sample.columns and v in synth_sample.columns
    ]

    dcor_errors = {}
    orig_dcors = {}
    synth_dcors = {}

    for i, var1 in enumerate(available_vars):
        for j, var2 in enumerate(available_vars):
            if i < j:
                pair = f"{var1}_vs_{var2}"
                orig_dcor = distance_correlation(
                    orig_sample[var1].values, orig_sample[var2].values
                )
                synth_dcor = distance_correlation(
                    synth_sample[var1].values, synth_sample[var2].values
                )
                orig_dcors[pair] = orig_dcor
                synth_dcors[pair] = synth_dcor
                dcor_errors[pair] = abs(orig_dcor - synth_dcor)

    return {
        "original_dcors": orig_dcors,
        "synthetic_dcors": synth_dcors,
        "dcor_errors": dcor_errors,
        "mean_dcor_error": np.mean(list(dcor_errors.values())) if dcor_errors else 0,
    }


def compute_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute energy distance between two distributions.

    Energy distance is the MULTIVARIATE GENERALIZATION OF CRPS.
    D = 0 iff distributions are identical.
    """
    n_x = len(X)
    n_y = len(Y)

    dist_XX = cdist(X, X, metric="euclidean")
    dist_YY = cdist(Y, Y, metric="euclidean")
    dist_XY = cdist(X, Y, metric="euclidean")

    term1 = 2 * np.mean(dist_XY)
    term2 = (np.sum(dist_XX) - np.trace(dist_XX)) / (n_x * (n_x - 1))
    term3 = (np.sum(dist_YY) - np.trace(dist_YY)) / (n_y * (n_y - 1))

    return float(term1 - term2 - term3)


def compute_nearest_neighbor_authenticity(
    synthetic: np.ndarray,
    holdout: np.ndarray,
    train: np.ndarray = None,
) -> dict:
    """
    Compute nearest neighbor authenticity metrics.

    For each synthetic record, find distance to nearest real record.
    Also checks for overfitting by comparing to training data.

    Returns:
    - mean_distance: average nearest neighbor distance
    - min_distance: minimum distance (privacy check)
    - max_distance: maximum distance (outliers)
    - privacy_ratio: ratio of holdout distance to train distance (if train provided)
    """
    # Synthetic -> Holdout distances
    dist_to_holdout = cdist(synthetic, holdout, metric="euclidean")
    nn_distances = np.min(dist_to_holdout, axis=1)

    result = {
        "mean_distance": float(np.mean(nn_distances)),
        "median_distance": float(np.median(nn_distances)),
        "min_distance": float(np.min(nn_distances)),
        "max_distance": float(np.max(nn_distances)),
        "q25_distance": float(np.percentile(nn_distances, 25)),
        "q75_distance": float(np.percentile(nn_distances, 75)),
    }

    # Privacy/overfitting check if training data provided
    if train is not None:
        dist_to_train = cdist(synthetic, train, metric="euclidean")
        nn_distances_train = np.min(dist_to_train, axis=1)

        # Ratio > 1 means synthetic is closer to holdout (good generalization)
        # Ratio < 1 means synthetic is closer to train (overfitting)
        ratios = (nn_distances + 1e-10) / (nn_distances_train + 1e-10)
        closer_to_train = np.mean(nn_distances_train < nn_distances)

        result["privacy_ratio"] = float(np.mean(ratios))
        result["fraction_closer_to_train"] = float(closer_to_train)

    return result


def evaluate_multivariate_quality(
    original_persons: pd.DataFrame,
    synthetic_persons: pd.DataFrame,
    person_vars: list,
    sample_size: int = 1000,
) -> dict:
    """
    Comprehensive multivariate quality evaluation.
    """
    print("\n" + "=" * 60)
    print("MULTIVARIATE QUALITY EVALUATION")
    print("=" * 60)

    results = {}

    # 1. Distance Correlation Preservation
    print("\n1. Computing distance correlation (dCor) preservation...")
    dcor_results = compute_dcor_preservation(
        original_persons, synthetic_persons, person_vars, sample_size
    )
    results["dcor"] = dcor_results
    print(f"   Mean dCor error: {dcor_results['mean_dcor_error']:.4f}")

    # 2. Energy Distance
    print("\n2. Computing energy distance...")
    numeric_vars = [v for v in person_vars if v in original_persons.columns]

    orig_subset = original_persons[numeric_vars].dropna()
    synth_subset = synthetic_persons[numeric_vars].dropna()

    n_sample = min(sample_size, len(orig_subset), len(synth_subset))
    orig_sample = orig_subset.sample(n=n_sample, random_state=42)
    synth_sample = synth_subset.sample(n=n_sample, random_state=42)

    scaler = StandardScaler()
    orig_norm = scaler.fit_transform(orig_sample)
    synth_norm = scaler.transform(synth_sample)

    energy_dist = compute_energy_distance(orig_norm, synth_norm)
    results["energy_distance"] = energy_dist
    print(f"   Energy distance: {energy_dist:.4f}")

    # 3. Nearest Neighbor Authenticity
    print("\n3. Computing nearest neighbor authenticity...")
    # Split original into train/holdout for privacy check
    n_holdout = len(orig_norm) // 2
    holdout_norm = orig_norm[:n_holdout]
    train_norm = orig_norm[n_holdout:]

    nn_results = compute_nearest_neighbor_authenticity(
        synth_norm, holdout_norm, train_norm
    )
    results["nearest_neighbor"] = nn_results
    print(f"   Mean NN distance: {nn_results['mean_distance']:.4f}")
    print(f"   Min NN distance: {nn_results['min_distance']:.4f} (privacy check)")
    if "privacy_ratio" in nn_results:
        print(f"   Privacy ratio: {nn_results['privacy_ratio']:.4f} (>1 = good)")
        print(f"   Closer to train: {nn_results['fraction_closer_to_train']:.1%}")

    return results


# ============================================================================
# HOUSEHOLD INCOME COMPARISON
# ============================================================================

def compare_hh_income_distributions(
    original_persons: pd.DataFrame,
    synthetic_persons: pd.DataFrame,
    synthetic_hh: pd.DataFrame,
) -> dict:
    """
    Compare derived household income distribution between original and synthetic.
    """
    print("\n" + "=" * 60)
    print("HOUSEHOLD INCOME DISTRIBUTION COMPARISON")
    print("=" * 60)

    # Compute original HH income
    orig_hh_income = original_persons.groupby("household_id")["income"].sum()

    # Synthetic HH income is already computed as hh_income
    synth_hh_income = synthetic_hh["hh_income"] if "hh_income" in synthetic_hh.columns else (
        synthetic_persons.groupby("household_id")["income"].sum()
    )

    # Summary statistics
    results = {
        "original": {
            "mean": float(orig_hh_income.mean()),
            "median": float(orig_hh_income.median()),
            "std": float(orig_hh_income.std()),
            "q25": float(orig_hh_income.quantile(0.25)),
            "q75": float(orig_hh_income.quantile(0.75)),
            "zero_fraction": float((orig_hh_income == 0).mean()),
        },
        "synthetic": {
            "mean": float(synth_hh_income.mean()),
            "median": float(synth_hh_income.median()),
            "std": float(synth_hh_income.std()),
            "q25": float(synth_hh_income.quantile(0.25)),
            "q75": float(synth_hh_income.quantile(0.75)),
            "zero_fraction": float((synth_hh_income == 0).mean()),
        },
    }

    # KS test
    ks_stat, ks_pval = stats.ks_2samp(
        orig_hh_income.dropna().values,
        synth_hh_income.dropna().values,
    )
    results["ks_statistic"] = float(ks_stat)
    results["ks_pvalue"] = float(ks_pval)

    # Variance ratio
    orig_var = orig_hh_income.var()
    synth_var = synth_hh_income.var()
    results["variance_ratio"] = float(synth_var / orig_var) if orig_var > 0 else 0

    # Print results
    print("\nOriginal HH Income:")
    print(f"  Mean: ${results['original']['mean']:,.0f}")
    print(f"  Median: ${results['original']['median']:,.0f}")
    print(f"  Std: ${results['original']['std']:,.0f}")
    print(f"  Zero fraction: {results['original']['zero_fraction']:.2%}")

    print("\nSynthetic HH Income:")
    print(f"  Mean: ${results['synthetic']['mean']:,.0f}")
    print(f"  Median: ${results['synthetic']['median']:,.0f}")
    print(f"  Std: ${results['synthetic']['std']:,.0f}")
    print(f"  Zero fraction: {results['synthetic']['zero_fraction']:.2%}")

    print("\nDistribution Comparison:")
    print(f"  KS statistic: {ks_stat:.4f}")
    print(f"  Variance ratio: {results['variance_ratio']:.4f}")

    status = "Good" if ks_stat < 0.1 else "Fair" if ks_stat < 0.2 else "Poor"
    print(f"  Status: {status}")

    return results


# ============================================================================
# STATE-LEVEL REWEIGHTING
# ============================================================================

def test_state_level_reweighting(
    synthetic_hh: pd.DataFrame,
    original_hh: pd.DataFrame,
) -> dict:
    """
    Test reweighting synthetic data to match state-level population targets.
    """
    print("\n" + "=" * 60)
    print("STATE-LEVEL REWEIGHTING TEST")
    print("=" * 60)

    # Compute original state distribution as targets
    # Weight by person count per household to approximate population
    if "state_fips" not in original_hh.columns:
        print("WARNING: state_fips not in original household data, skipping reweighting test")
        return {"error": "state_fips not available"}

    state_populations = original_hh.groupby("state_fips")["n_persons"].sum()

    # Scale to match synthetic data size
    scale_factor = len(synthetic_hh) / len(original_hh)
    state_targets = (state_populations * scale_factor).round().astype(int).to_dict()

    print("\nTarget state distribution (scaled):")
    top_states = dict(sorted(state_targets.items(), key=lambda x: -x[1])[:5])
    for state, count in top_states.items():
        print(f"  State {int(state)}: {count:,}")

    # Add state column to synthetic households if needed
    if "state_fips" not in synthetic_hh.columns:
        print("\nWARNING: state_fips not in synthetic_hh, adding from n_persons-based imputation")
        # Impute state - for demo purposes use uniform distribution
        np.random.seed(42)
        synthetic_hh = synthetic_hh.copy()
        synthetic_hh["state_fips"] = np.random.choice(
            list(state_targets.keys()),
            size=len(synthetic_hh),
            p=np.array(list(state_targets.values())) / sum(state_targets.values()),
        )

    # Ensure state_fips types match
    synthetic_hh["state_fips"] = synthetic_hh["state_fips"].astype(int)

    # Filter to states that exist in both datasets
    synth_states = set(synthetic_hh["state_fips"].unique())
    target_states = set(state_targets.keys())
    common_states = synth_states & target_states

    if len(common_states) < len(target_states):
        print(f"\nFiltering to {len(common_states)} common states")
        state_targets = {k: v for k, v in state_targets.items() if k in common_states}
        synthetic_hh = synthetic_hh[synthetic_hh["state_fips"].isin(common_states)].copy()

    # Before reweighting distribution
    before_counts = synthetic_hh["state_fips"].value_counts().sort_index()

    results = {"targets": state_targets, "before": before_counts.to_dict()}

    # NOTE: The synthesizer treats state_fips as continuous, which produces
    # out-of-range values. We filter to valid states for the reweighting test.
    valid_states = set(state_targets.keys())
    synth_hh_filtered = synthetic_hh[synthetic_hh["state_fips"].isin(valid_states)].copy()

    print(f"\nFiltered to {len(synth_hh_filtered)} households with valid state FIPS")
    if len(synth_hh_filtered) < 100:
        print("  WARNING: Too few valid households for meaningful reweighting")
        print("  NOTE: state_fips is treated as continuous by the synthesizer")
        print("  Consider using discrete_vars for categorical variables")
        return {
            "error": "insufficient_valid_states",
            "n_valid": len(synth_hh_filtered),
            "targets": state_targets,
        }

    # Test different sparsity settings
    for sparsity in ["l1", "l2", "l0"]:
        print(f"\n--- {sparsity.upper()} Reweighting ---")

        try:
            reweighter = Reweighter(sparsity=sparsity)
            weighted_hh = reweighter.fit_transform(
                synth_hh_filtered,
                {"state_fips": state_targets},
                drop_zeros=False,
            )

            # Check results
            stats_dict = reweighter.get_sparsity_stats()
            print(f"  Records used: {stats_dict['n_nonzero']:,} / {stats_dict['n_records']:,}")
            print(f"  Sparsity: {stats_dict['sparsity']:.1%}")
            print(f"  Max weight: {stats_dict['max_weight']:.2f}")

            # Check target matching
            weighted_counts = weighted_hh.groupby("state_fips")["weight"].sum()

            max_error = 0
            for state in list(state_targets.keys())[:5]:
                actual = weighted_counts.get(state, 0)
                target = state_targets[state]
                error = abs(actual - target) / target if target > 0 else 0
                max_error = max(max_error, error)

            print(f"  Max target error: {max_error:.2%}")

            results[f"{sparsity}_stats"] = stats_dict
            results[f"{sparsity}_max_error"] = max_error

        except Exception as e:
            print(f"  ERROR: {e}")
            results[f"{sparsity}_error"] = str(e)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete CPS real data test."""

    print("=" * 80)
    print("MICROPLEX CPS REAL DATA TEST")
    print("HierarchicalSynthesizer on CPS ASEC Data")
    print("=" * 80)

    # Configuration
    epochs = 50
    sample_fraction = 0.3  # Use 30% of CPS data
    n_synthetic_households = 5000

    results = {}

    # ========================================================================
    # STEP 1: Load CPS Data
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading CPS ASEC Data")
    print("=" * 80)

    try:
        households, persons = load_cps_for_synthesis(
            sample_fraction=sample_fraction,
            random_state=42,
        )
        print("\nLoaded CPS ASEC data:")
        print(f"  Households: {len(households):,}")
        print(f"  Persons: {len(persons):,}")
        print(f"  Avg HH size: {len(persons) / len(households):.2f}")

        results["data"] = {
            "n_households": len(households),
            "n_persons": len(persons),
            "avg_hh_size": len(persons) / len(households),
        }

    except FileNotFoundError:
        print("\nWARNING: CPS data not found. Using synthetic sample data.")
        households, persons = create_sample_data(n_households=3000, seed=42)
        print(f"Generated sample data: {len(households)} HH, {len(persons)} persons")
        results["data"] = {"source": "synthetic_sample"}

    # ========================================================================
    # STEP 2: Fit HierarchicalSynthesizer
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Training HierarchicalSynthesizer")
    print("=" * 80)

    schema = setup_cps_schema()
    print("\nSchema:")
    print(f"  HH vars: {schema.hh_vars}")
    print(f"  Person vars: {schema.person_vars}")

    synth = HierarchicalSynthesizer(
        schema=schema,
        hh_flow_kwargs={"n_layers": 4, "hidden_dim": 64},
        person_flow_kwargs={"n_layers": 6, "hidden_dim": 128},
        random_state=42,
    )

    print(f"\nTraining with {epochs} epochs...")
    start_time = time.time()

    synth.fit(
        households,
        persons,
        hh_weight_col="hh_weight" if "hh_weight" in households.columns else None,
        epochs=epochs,
        verbose=True,
    )

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds")
    results["training_time"] = train_time

    # ========================================================================
    # STEP 3: Generate Synthetic Households
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"STEP 3: Generating {n_synthetic_households:,} Synthetic Households")
    print("=" * 80)

    start_time = time.time()
    synthetic_hh, synthetic_persons = synth.generate(
        n_households=n_synthetic_households,
        verbose=True,
    )
    generate_time = time.time() - start_time

    print(f"\nGeneration completed in {generate_time:.1f} seconds")
    print(f"  Synthetic households: {len(synthetic_hh):,}")
    print(f"  Synthetic persons: {len(synthetic_persons):,}")
    print(f"  Avg HH size: {len(synthetic_persons) / len(synthetic_hh):.2f}")

    results["generation"] = {
        "n_synthetic_hh": len(synthetic_hh),
        "n_synthetic_persons": len(synthetic_persons),
        "generation_time": generate_time,
    }

    # ========================================================================
    # STEP 4: Multivariate Quality Evaluation
    # ========================================================================
    person_vars = ["age", "sex", "income", "employment_status", "education"]

    multivariate_results = evaluate_multivariate_quality(
        persons,
        synthetic_persons,
        person_vars,
        sample_size=1000,
    )
    results["multivariate"] = multivariate_results

    # ========================================================================
    # STEP 5: Household Income Comparison
    # ========================================================================
    hh_income_results = compare_hh_income_distributions(
        persons,
        synthetic_persons,
        synthetic_hh,
    )
    results["hh_income"] = hh_income_results

    # ========================================================================
    # STEP 6: State-Level Reweighting
    # ========================================================================
    reweighting_results = test_state_level_reweighting(
        synthetic_hh,
        households,
    )
    results["reweighting"] = reweighting_results

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nKey Quality Metrics:")
    print(f"  dCor preservation error: {multivariate_results['dcor']['mean_dcor_error']:.4f}")
    print(f"  Energy distance: {multivariate_results['energy_distance']:.4f}")
    print(f"  NN mean distance: {multivariate_results['nearest_neighbor']['mean_distance']:.4f}")
    print(f"  HH income KS stat: {hh_income_results['ks_statistic']:.4f}")
    print(f"  HH income variance ratio: {hh_income_results['variance_ratio']:.4f}")

    # Interpretation
    print("\nInterpretation Guide:")
    dcor_err = multivariate_results['dcor']['mean_dcor_error']
    dcor_status = "Excellent" if dcor_err < 0.05 else "Good" if dcor_err < 0.1 else "Fair"
    print(f"  dCor: {dcor_status} (< 0.05 excellent, < 0.1 good)")

    ks = hh_income_results['ks_statistic']
    ks_status = "Excellent" if ks < 0.1 else "Good" if ks < 0.2 else "Fair"
    print(f"  HH Income KS: {ks_status} (< 0.1 excellent, < 0.2 good)")

    var_ratio = hh_income_results['variance_ratio']
    var_status = "Good" if 0.8 <= var_ratio <= 1.2 else "Fair"
    print(f"  Variance Ratio: {var_status} (0.8-1.2 is good)")

    print("\nReweighting Results:")
    for sparsity in ["l1", "l2", "l0"]:
        if f"{sparsity}_stats" in reweighting_results:
            stats_dict = reweighting_results[f"{sparsity}_stats"]
            max_err = reweighting_results.get(f"{sparsity}_max_error", 0)
            print(f"  {sparsity.upper()}: {stats_dict['n_nonzero']} records, max error {max_err:.2%}")

    print("\nTimings:")
    print(f"  Training: {results['training_time']:.1f}s")
    print(f"  Generation: {results['generation']['generation_time']:.1f}s")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()

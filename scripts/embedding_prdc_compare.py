"""Compare raw-feature PRDC vs learned-embedding PRDC on the stage-1 methods.

The scale-up-protocol doc flagged that PRDC in ~50 dimensions may be
degenerate (curse of dimensionality: k-NN distances concentrate and the
metric becomes noise-dominated). This script settles the question.

Procedure:

1. Fit each of (ZI-QRF, ZI-MAF, ZI-QDNN) on 40k x 50 real ECPS.
2. Generate synthetic records from each.
3. Train a 16-dim autoencoder on the holdout's raw features only.
4. Compute PRDC in the raw 50-dim feature space (unchanged from stage 1).
5. Compute PRDC in the 16-dim learned latent space.
6. Report both side-by-side. If the ordering changes, the stage-1
   finding was metric-driven not method-driven; if it's preserved, the
   finding is robust.

Usage:
    uv run python scripts/embedding_prdc_compare.py \
        --output artifacts/embedding_prdc_compare.json

Runs in ~5 minutes on 40 k rows x 50 cols (driven by ZI-MAF fit time).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prdc import compute_prdc
from sklearn.preprocessing import StandardScaler

from microplex.eval.benchmark import ZIMAFMethod, ZIQDNNMethod, ZIQRFMethod
from microplex_us.bakeoff import (
    DEFAULT_CONDITION_COLS,
    DEFAULT_TARGET_COLS,
    ScaleUpRunner,
    ScaleUpStageConfig,
    stage1_config,
)

LOGGER = logging.getLogger(__name__)


class Autoencoder(nn.Module):
    """Tiny autoencoder for dimensionality reduction on tabular features."""

    def __init__(self, n_features: int, latent_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def fit_autoencoder(
    x: np.ndarray, latent_dim: int = 16, epochs: int = 200, lr: float = 1e-3
) -> Autoencoder:
    """Fit an autoencoder on standardized features."""
    n_features = x.shape[1]
    model = Autoencoder(n_features=n_features, latent_dim=latent_dim)
    x_t = torch.tensor(x, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batch_size = 256
    ds = torch.utils.data.TensorDataset(x_t)
    g = torch.Generator()
    g.manual_seed(42)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if (epoch + 1) % 50 == 0:
            LOGGER.info("  AE epoch %d loss=%.4f", epoch + 1, total / len(x))
    model.eval()
    return model


def encode(model: Autoencoder, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.encode(torch.tensor(x, dtype=torch.float32)).numpy()


def compute_prdc_both_spaces(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    encoder: Autoencoder,
    scaler: StandardScaler,
    k: int = 5,
    max_samples: int = 15_000,
    seed: int = 42,
) -> dict:
    """Return {raw: ..., embed: ...} PRDC tuples."""
    rng = np.random.default_rng(seed)
    cols = [c for c in real.columns if c in synthetic.columns]
    r = real[cols].to_numpy(dtype=np.float64)
    s = synthetic[cols].to_numpy(dtype=np.float64)
    if len(r) > max_samples:
        r = r[rng.choice(len(r), size=max_samples, replace=False)]
    if len(s) > max_samples:
        s = s[rng.choice(len(s), size=max_samples, replace=False)]

    raw_r = scaler.transform(r)
    raw_s = scaler.transform(s)
    raw_metrics = compute_prdc(raw_r, raw_s, nearest_k=k)

    emb_r = encode(encoder, raw_r.astype(np.float32))
    emb_s = encode(encoder, raw_s.astype(np.float32))
    emb_metrics = compute_prdc(emb_r, emb_s, nearest_k=k)

    return {
        "raw": {k: float(v) for k, v in raw_metrics.items()},
        "embed": {k: float(v) for k, v in emb_metrics.items()},
    }


def build_method(name: str):
    registry = {
        "ZI-QRF": ZIQRFMethod,
        "ZI-MAF": ZIMAFMethod,
        "ZI-QDNN": ZIQDNNMethod,
    }
    return registry[name]()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=40_000)
    parser.add_argument(
        "--methods", nargs="+", default=["ZI-QRF", "ZI-MAF", "ZI-QDNN"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embedding_prdc_compare.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--ae-epochs", type=int, default=200)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    base = stage1_config()
    cfg = ScaleUpStageConfig(
        stage="embedding_prdc",
        n_rows=args.n_rows,
        methods=tuple(args.methods),
        condition_cols=DEFAULT_CONDITION_COLS,
        target_cols=DEFAULT_TARGET_COLS,
        holdout_frac=0.2,
        seed=args.seed,
        k=5,
        data_path=base.data_path,
        year=base.year,
        rare_cell_checks=(),
        prdc_max_samples=15_000,
    )

    runner = ScaleUpRunner(cfg)
    df = runner.load_frame()
    train, holdout = runner.split(df)
    LOGGER.info(
        "loaded: train=%d holdout=%d cols=%d", len(train), len(holdout), len(df.columns)
    )

    scaler = StandardScaler().fit(holdout.to_numpy(dtype=np.float64))

    LOGGER.info("fitting autoencoder on holdout...")
    t0 = time.time()
    encoder = fit_autoencoder(
        scaler.transform(holdout.to_numpy(dtype=np.float64)).astype(np.float32),
        latent_dim=args.latent_dim,
        epochs=args.ae_epochs,
    )
    LOGGER.info("  autoencoder fit=%.1fs", time.time() - t0)

    results = []
    for method_name in args.methods:
        LOGGER.info("== %s ==", method_name)
        method = build_method(method_name)
        t0 = time.time()
        method.fit(sources={"ecps": train.copy()}, shared_cols=list(DEFAULT_CONDITION_COLS))
        fit_s = time.time() - t0

        t0 = time.time()
        synth = method.generate(len(train), seed=args.seed)
        gen_s = time.time() - t0

        metrics = compute_prdc_both_spaces(
            holdout, synth, encoder, scaler, k=5, seed=args.seed
        )
        LOGGER.info(
            "  raw:   prec=%.3f dens=%.3f cov=%.3f",
            metrics["raw"]["precision"],
            metrics["raw"]["density"],
            metrics["raw"]["coverage"],
        )
        LOGGER.info(
            "  embed: prec=%.3f dens=%.3f cov=%.3f  (fit=%.1fs gen=%.1fs)",
            metrics["embed"]["precision"],
            metrics["embed"]["density"],
            metrics["embed"]["coverage"],
            fit_s,
            gen_s,
        )
        results.append(
            {
                "method": method_name,
                "fit_wall_seconds": fit_s,
                "generate_wall_seconds": gen_s,
                **metrics,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str))

    print()
    print("== Raw-feature PRDC (50-dim) ==")
    for r in sorted(results, key=lambda x: -x["raw"]["coverage"]):
        print(
            f"  {r['method']:8s}: cov={r['raw']['coverage']:.3f} "
            f"prec={r['raw']['precision']:.3f} dens={r['raw']['density']:.3f}"
        )
    print()
    print(f"== Learned-embedding PRDC ({args.latent_dim}-dim) ==")
    for r in sorted(results, key=lambda x: -x["embed"]["coverage"]):
        print(
            f"  {r['method']:8s}: cov={r['embed']['coverage']:.3f} "
            f"prec={r['embed']['precision']:.3f} dens={r['embed']['density']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train, evaluate and save the price model.

    python src/train.py

Reads data/cleaned_item_details.csv, splits it 80/20 with a fixed seed, fits
the preprocessing on the training split only, compares plain linear
regression with ridge regression (alpha chosen by 5-fold cross-validation on
the training split), reports R2 / MAE / RMSE in Toman on the test split next
to a median-price baseline, writes docs/results.png and saves the winner to
models/price_model.joblib.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import CATEGORICAL, FEATURES, TARGET, build_preprocessor, load_clean  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "price_model.joblib"
PLOT_PATH = REPO_ROOT / "docs" / "results.png"
TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def make_model(regressor) -> TransformedTargetRegressor:
    """Preprocessing + regressor, fitted on log(price) and predicting Toman."""
    pipeline = Pipeline([("prep", build_preprocessor()), ("reg", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def evaluate(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def format_row(name: str, m: dict) -> str:
    return f"{name:22s} R2 = {m['R2']:6.3f}   MAE = {m['MAE'] / 1e6:6.2f}M   RMSE = {m['RMSE'] / 1e6:6.2f}M Toman"


def plot_results(y_true, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true / 1e6, y_pred / 1e6, s=14, alpha=0.6)
    lo, hi = 0.4, 350
    ax.plot([lo, hi], [lo, hi], color="tab:red", linewidth=1.5, label="perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual price (million Toman)")
    ax.set_ylabel("Predicted price (million Toman)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"train rows: {len(X_train)}   test rows: {len(X_test)}\n")

    candidates = {
        "LinearRegression": make_model(LinearRegression()),
        "Ridge": make_model(RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
    }

    # Pick the model on the training split only (5-fold CV), never on the test split.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_mae = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_mae[name] = -scores.mean()
        print(f"{name:22s} 5-fold CV MAE on train = {cv_mae[name] / 1e6:6.2f}M Toman")
    winner_name = min(cv_mae, key=cv_mae.get)
    print(f"\nselected: {winner_name}\n")

    print("Test split:")
    baseline = np.full(len(y_test), y_train.median())
    metrics = {"median baseline": evaluate(y_test, baseline)}
    print(format_row("median baseline", metrics["median baseline"]))
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics[name] = evaluate(y_test, model.predict(X_test))
        print(format_row(name, metrics[name]))

    winner = candidates[winner_name]
    if winner_name == "Ridge":
        alpha = winner.regressor_.named_steps["reg"].alpha_
        print(f"\nRidge alpha chosen by CV: {alpha:.4g}")

    plot_results(y_test, winner.predict(X_test), f"{winner_name}: actual vs predicted (test split)", PLOT_PATH)
    print(f"saved {PLOT_PATH.relative_to(REPO_ROOT)}")

    bundle = {
        "model": winner,
        "model_name": winner_name,
        "features": FEATURES,
        "categories": {col: sorted(X_train[col].unique()) for col in CATEGORICAL},
        "test_metrics": metrics[winner_name],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"saved {MODEL_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

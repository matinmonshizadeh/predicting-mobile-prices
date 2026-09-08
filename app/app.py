"""Tkinter front end for the saved price model.

    python app/app.py

Loads models/price_model.joblib (created by ``python src/train.py``) and
predicts a listing price from brand, condition, origin, colour, SIM count,
storage and RAM. The model is not retrained here.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.features import CATEGORICAL, FEATURES  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "price_model.joblib"

NUMERIC_FIELDS = {
    # column name: (label, default value)
    "SIM Count": ("SIM count", "2"),
    "Internal Storage(GB)": ("Internal storage (GB)", "128"),
    "RAM(GB)": ("RAM (GB)", "4"),
}
CATEGORICAL_LABELS = {
    "Brand": "Brand",
    "Status": "Condition",
    "Brand Origin": "Brand origin",
    "Color": "Colour",
}


class PricePredictorApp:
    def __init__(self, root: tk.Tk, bundle: dict):
        self.root = root
        self.model = bundle["model"]
        self.categories = bundle["categories"]
        self.inputs: dict[str, tk.Widget] = {}

        root.title("Mobile Price Predictor")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=12)
        frame.grid()

        row = 0
        for col in CATEGORICAL:
            ttk.Label(frame, text=CATEGORICAL_LABELS[col]).grid(row=row, column=0, sticky="w", pady=4)
            box = ttk.Combobox(frame, values=self.categories[col], state="readonly", width=22)
            box.current(0)
            box.grid(row=row, column=1, pady=4, padx=(8, 0))
            self.inputs[col] = box
            row += 1

        for col, (label, default) in NUMERIC_FIELDS.items():
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = ttk.Entry(frame, width=24)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=4, padx=(8, 0))
            self.inputs[col] = entry
            row += 1

        ttk.Button(frame, text="Predict price", command=self.predict).grid(
            row=row, column=0, columnspan=2, pady=(12, 4)
        )
        self.result = ttk.Label(frame, text="", font=("TkDefaultFont", 11, "bold"))
        self.result.grid(row=row + 1, column=0, columnspan=2, pady=(4, 0))
        ttk.Label(
            frame,
            text=f"{bundle['model_name']} on log(price), "
            f"test MAE {bundle['test_metrics']['MAE'] / 1e6:.1f}M Toman",
            foreground="gray",
        ).grid(row=row + 2, column=0, columnspan=2, pady=(8, 0))

    def read_inputs(self) -> pd.DataFrame:
        values = {}
        for col in CATEGORICAL:
            values[col] = self.inputs[col].get()
        for col in NUMERIC_FIELDS:
            text = self.inputs[col].get().strip()
            try:
                number = float(text)
            except ValueError:
                raise ValueError(f"{NUMERIC_FIELDS[col][0]} must be a number, got {text!r}") from None
            if number <= 0:
                raise ValueError(f"{NUMERIC_FIELDS[col][0]} must be positive")
            values[col] = number
        return pd.DataFrame([values])[FEATURES]

    def predict(self) -> None:
        try:
            features = self.read_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return
        price = float(self.model.predict(features)[0])
        self.result.config(text=f"Predicted price: {price:,.0f} Toman")


def main() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"{MODEL_PATH} not found. Run `python src/train.py` first.")
    bundle = joblib.load(MODEL_PATH)
    root = tk.Tk()
    PricePredictorApp(root, bundle)
    root.mainloop()


if __name__ == "__main__":
    main()

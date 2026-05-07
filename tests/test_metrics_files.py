from pathlib import Path

import pandas as pd


REQUIRED = {
    "summary.csv": {"model_type", "test_accuracy", "test_macro_f1"},
    "benchmark_seq_len.csv": {"model_type", "sequence_length", "mean_forward_ms"},
    "noise_sweep.csv": {"model_type", "noise_level", "test_accuracy"},
    "gradient_variance.csv": {"depth", "grad_variance", "grad_norm"},
    "attention_alignment.csv": {"left_model", "right_model", "linear_cka"},
}


def test_metric_schema_when_files_exist():
    metrics_dir = Path("results/metrics")
    for filename, columns in REQUIRED.items():
        path = metrics_dir / filename
        if path.exists():
            df = pd.read_csv(path)
            assert columns.issubset(df.columns)


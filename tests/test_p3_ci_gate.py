import json
import os

from src.mlops.drift_monitor import analyze_prediction_batch


def test_dvc_dataset_tracking_exists():
    """Verifies that the dataset is tracked via DVC pointer."""
    dvc_pointer = os.path.join("data", "raw", "tickets_raw.json.dvc")
    assert os.path.exists(dvc_pointer), "DVC tracking pointer file missing!"


def test_processed_splits_exist():
    """Verifies train/validation splits were generated properly."""
    train_path = os.path.join("data", "processed", "train.json")
    val_path = os.path.join("data", "processed", "val.json")
    assert os.path.exists(train_path)
    assert os.path.exists(val_path)

    with open(train_path) as f:
        train_data = json.load(f)
        assert len(train_data) >= 500


def test_serving_benchmarks_report():
    """Asserts that vLLM serving benchmark exists and verifies throughput gain."""
    report_file = os.path.join("results", "serving_benchmark.json")
    assert os.path.exists(report_file), "Serving benchmark report missing!"

    with open(report_file) as f:
        benchmarks = json.load(f)

    vllm_bench = next(b for b in benchmarks if "vLLM" in b["engine"])
    pytorch_bench = next(b for b in benchmarks if "Vanilla PyTorch" in b["engine"])

    # vLLM must deliver at least 3x throughput over PyTorch
    assert vllm_bench["tokens_per_second"] > pytorch_bench["tokens_per_second"] * 3


def test_drift_monitor_alert_trigger():
    """Asserts that drift monitor properly catches malformed payloads."""
    malformed_batch = [
        {"broken_field": "error"},
        {"ticket_id": "T1", "category": "alien_invasion", "urgency": "extreme"},
    ] * 20
    report = analyze_prediction_batch(malformed_batch, [100.0] * 40)
    assert report.drift_detected is True
    assert report.alert_level in ["WARNING", "CRITICAL"]

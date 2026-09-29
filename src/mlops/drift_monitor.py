from dataclasses import dataclass
from typing import Any


@dataclass
class DriftReport:
    total_inferences: int
    malformed_json_rate_pct: float
    unseen_category_rate_pct: float
    latency_p95_ms: float
    drift_detected: bool
    alert_level: str


KNOWN_CATEGORIES = {"access_control", "hardware", "software", "security", "billing"}


def analyze_prediction_batch(
    predictions: list[dict[str, Any]], latencies_ms: list[float]
) -> DriftReport:
    total = len(predictions)
    if total == 0:
        return DriftReport(0, 0.0, 0.0, 0.0, False, "NORMAL")

    malformed_count = 0
    unseen_cat_count = 0

    for p in predictions:
        # Check 1: Schema validity
        if not isinstance(p, dict) or "category" not in p or "urgency" not in p:
            malformed_count += 1
            continue

        # Check 2: Concept drift (new/unseen ticket category)
        if p["category"] not in KNOWN_CATEGORIES:
            unseen_cat_count += 1

    malformed_rate = (malformed_count / total) * 100
    unseen_rate = (unseen_cat_count / total) * 100
    p95_latency = sorted(latencies_ms)[int(total * 0.95)] if latencies_ms else 0.0

    # Production Alert Rules:
    # If malformed JSON exceeds 3% OR unseen categories exceed 5% -> Trigger Drift Alert
    drift_detected = malformed_rate > 3.0 or unseen_rate > 5.0
    if malformed_rate > 10.0:
        alert_level = "CRITICAL"
    elif drift_detected:
        alert_level = "WARNING"
    else:
        alert_level = "NORMAL"

    return DriftReport(
        total_inferences=total,
        malformed_json_rate_pct=round(malformed_rate, 2),
        unseen_category_rate_pct=round(unseen_rate, 2),
        latency_p95_ms=round(p95_latency, 1),
        drift_detected=drift_detected,
        alert_level=alert_level,
    )


if __name__ == "__main__":
    # Test healthy traffic
    healthy_batch = [
        {"ticket_id": "T1", "category": "software", "urgency": "medium"},
        {"ticket_id": "T2", "category": "access_control", "urgency": "high"},
    ] * 50
    latencies = [45.0, 52.0, 48.0, 65.0] * 25
    report = analyze_prediction_batch(healthy_batch, latencies)

    print("\n" + "=" * 60)
    print("PRODUCTION MODEL MONITORING REPORT")
    print("=" * 60)
    print(f"Total Requests Scanned:        {report.total_inferences}")
    print(f"Malformed JSON Error Rate:     {report.malformed_json_rate_pct}%")
    print(f"Unseen Category Drift Rate:    {report.unseen_category_rate_pct}%")
    print(f"p95 Latency:                   {report.latency_p95_ms} ms")
    print(f"Drift Status:                  {report.alert_level}")
    print("=" * 60)

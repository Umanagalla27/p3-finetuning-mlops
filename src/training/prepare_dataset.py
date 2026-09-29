import json
import os
import random
from sklearn.model_selection import train_test_split

CATEGORIES = ["access_control", "hardware", "software", "security", "billing"]
URGENCIES = ["low", "medium", "high", "critical"]

TEMPLATES = [
    ("Need access to {system} for {reason}.", "access_control", "high", "grant_temporary_access"),
    ("My {hardware_item} is completely broken and {hw_issue}.", "hardware", "high", "dispatch_technician"),
    ("Application {app_name} keeps throwing error {err_code} on launch.", "software", "medium", "restart_service"),
    ("Urgent: Detected unauthorized login attempt from IP {ip_addr}.", "security", "critical", "quarantine_account"),
    ("Monthly cloud invoice discrepancy of ${amount} on account {acc_id}.", "billing", "low", "audit_billing"),
]

SYSTEMS = ["Production Postgres DB", "AWS Production S3", "Kubernetes Staging Cluster", "Snowflake Data Warehouse"]
REASONS = ["Q3 financial audit", "debugging live checkout outage", "compliance review", "data migration"]
HARDWARE = ["MacBook Pro M2", "Lenovo ThinkPad", "Dell 4K Monitor", "YubiKey 5C"]
HW_ISSUES = ["screen won't turn on", "battery swollen", "keyboard unresponsive", "device overheating"]
APPS = ["Docker Desktop", "Slack", "Postman", "Internal CRM"]
ERRORS = ["ERR_503_GATEWAY", "FATAL_OOM_KILL", "CONNECTION_REFUSED", "AUTH_EXPIRED"]


def generate_synthetic_corpus(num_samples: int = 1200):
    """Generates 1,200 domain-specific instruction samples for IT ticket JSON extraction."""
    samples = []

    for i in range(num_samples):
        tmpl, cat, base_urgency, tool = random.choice(TEMPLATES)
        text = tmpl.format(
            system=random.choice(SYSTEMS),
            reason=random.choice(REASONS),
            hardware_item=random.choice(HARDWARE),
            hw_issue=random.choice(HW_ISSUES),
            app_name=random.choice(APPS),
            err_code=random.choice(ERRORS),
            ip_addr=f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            amount=random.randint(50, 4500),
            acc_id=f"ACC-{random.randint(1000, 9999)}"
        )

        expected_json = {
            "ticket_id": f"TKT-{10000 + i}",
            "category": cat,
            "urgency": base_urgency,
            "recommended_tool": tool,
            "requires_human_approval": base_urgency in ["high", "critical"],
        }

        sample = {
            "instruction": "Extract the structured IT ticket metadata and output ONLY a valid JSON object matching the enterprise schema.",
            "input": text,
            "output": json.dumps(expected_json)
        }
        samples.append(sample)

    return samples


def prepare_and_save():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("[Data Prep] Generating 1,200 instruction-tuning samples...")
    corpus = generate_synthetic_corpus(1200)

    train_data, val_data = train_test_split(corpus, test_size=0.15, random_state=42)

    # Save raw
    with open("data/raw/tickets_raw.json", "w") as f:
        json.dump(corpus, f, indent=2)

    # Save processed train/val
    with open("data/processed/train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("data/processed/val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"[Data Prep] Saved: {len(train_data)} train samples -> data/processed/train.json")
    print(f"[Data Prep] Saved: {len(val_data)} validation samples -> data/processed/val.json")


if __name__ == "__main__":
    prepare_and_save()

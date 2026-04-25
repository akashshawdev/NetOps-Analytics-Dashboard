"""
Stage 2: Data Simulation
========================
Generates realistic cloud network metrics for a multi-region AWS/Azure-style infrastructure.
Simulates 30 days of data across 5 cloud regions with realistic patterns.

WHY these metrics?
- Latency: Directly impacts user experience and SLA compliance
- Uptime: Core availability metric used in 99.9%/99.99% SLA commitments
- Packet Loss: Indicates network congestion or hardware faults
- Incidents: Used for MTTR (Mean Time to Resolve) and RCA tracking
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import os

# ── Config ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

REGIONS = ["us-east-1", "eu-west-1", "ap-southeast-1", "us-west-2", "eu-central-1"]
DEVICES = {
    "us-east-1":     ["router-01", "firewall-01", "switch-core-01", "lb-01"],
    "eu-west-1":     ["router-02", "firewall-02", "switch-core-02", "lb-02"],
    "ap-southeast-1":["router-03", "firewall-03", "switch-core-03", "lb-03"],
    "us-west-2":     ["router-04", "firewall-04", "switch-core-04", "lb-04"],
    "eu-central-1":  ["router-05", "firewall-05", "switch-core-05", "lb-05"],
}
INCIDENT_TYPES = [
    "High Latency Spike",
    "Packet Loss Detected",
    "Link Down",
    "BGP Route Flap",
    "DDoS Suspected",
    "Interface Error Rate High",
    "MTU Mismatch",
    "Firewall CPU Overload",
]
SEVERITIES    = ["Critical", "High", "Medium", "Low"]
SEVERITY_W    = [0.10,       0.20,   0.45,    0.25]   # weighted probability
DAYS          = 30
START_DATE    = datetime(2025, 3, 1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helper: inject anomalies ─────────────────────────────────────────────────
def inject_anomalies(series: np.ndarray, anomaly_ratio: float = 0.03) -> np.ndarray:
    """Randomly spike ~3% of values to simulate real network events."""
    n = len(series)
    idx = np.random.choice(n, size=int(n * anomaly_ratio), replace=False)
    series[idx] *= np.random.uniform(3.0, 6.0, size=len(idx))
    return series


# ── 1. Network Metrics (latency, packet loss, bandwidth) ────────────────────
def simulate_network_metrics() -> pd.DataFrame:
    records = []
    timestamps = [START_DATE + timedelta(hours=h) for h in range(DAYS * 24)]

    for region in REGIONS:
        # Base latency varies by region (intercontinental > intra-region)
        base_latency = {"us-east-1": 18, "eu-west-1": 25, "ap-southeast-1": 45,
                        "us-west-2": 22, "eu-central-1": 28}[region]

        latency     = np.random.normal(base_latency, base_latency * 0.15, len(timestamps))
        packet_loss = np.random.exponential(0.3, len(timestamps))          # % loss
        bandwidth   = np.random.normal(850, 120, len(timestamps))          # Mbps
        jitter      = np.random.exponential(2.5, len(timestamps))          # ms

        # Business-hours traffic bump (higher load 08:00–18:00)
        for i, ts in enumerate(timestamps):
            if 8 <= ts.hour <= 18:
                latency[i]    *= 1.15
                bandwidth[i]  *= 1.30
                packet_loss[i] *= 1.10

        latency     = inject_anomalies(latency,     0.03)
        packet_loss = inject_anomalies(packet_loss, 0.02)

        # Clip to realistic bounds
        latency     = np.clip(latency,     1,   500)
        packet_loss = np.clip(packet_loss, 0,   20)
        bandwidth   = np.clip(bandwidth,   50,  1000)
        jitter      = np.clip(jitter,      0.1, 50)

        for i, ts in enumerate(timestamps):
            records.append({
                "timestamp":   ts.strftime("%Y-%m-%d %H:%M:%S"),
                "region":      region,
                "latency_ms":  round(latency[i], 2),
                "packet_loss_pct": round(packet_loss[i], 3),
                "bandwidth_mbps":  round(bandwidth[i], 1),
                "jitter_ms":       round(jitter[i], 2),
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "network_metrics.csv"), index=False)
    print(f"[✓] network_metrics.csv  → {len(df):,} rows")
    return df


# ── 2. Device Uptime Logs ────────────────────────────────────────────────────
def simulate_uptime_logs() -> pd.DataFrame:
    records = []
    for region, devices in DEVICES.items():
        for device in devices:
            # Each device has ~99.5 % target uptime
            total_hours   = DAYS * 24
            downtime_hrs  = max(0, np.random.normal(total_hours * 0.005, 1.5))
            uptime_hrs    = total_hours - downtime_hrs
            uptime_pct    = round((uptime_hrs / total_hours) * 100, 3)

            # Random downtime window
            down_start = START_DATE + timedelta(hours=random.randint(0, total_hours - 4))
            down_end   = down_start + timedelta(hours=round(downtime_hrs, 1))

            records.append({
                "region":       region,
                "device":       device,
                "total_hours":  total_hours,
                "uptime_hours": round(uptime_hrs, 2),
                "downtime_hours": round(downtime_hrs, 2),
                "uptime_pct":   uptime_pct,
                "downtime_start": down_start.strftime("%Y-%m-%d %H:%M:%S"),
                "downtime_end":   down_end.strftime("%Y-%m-%d %H:%M:%S"),
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "uptime_logs.csv"), index=False)
    print(f"[✓] uptime_logs.csv      → {len(df):,} rows")
    return df


# ── 3. Incident Logs ─────────────────────────────────────────────────────────
def simulate_incidents() -> pd.DataFrame:
    records = []
    incident_id = 1000

    for day in range(DAYS):
        # Weekends have fewer incidents
        date     = START_DATE + timedelta(days=day)
        n_events = random.randint(2, 6) if date.weekday() < 5 else random.randint(0, 3)

        for _ in range(n_events):
            region   = random.choice(REGIONS)
            devices  = DEVICES[region]
            severity = random.choices(SEVERITIES, weights=SEVERITY_W)[0]

            # MTTR: Critical faster (pager alerts), Low slower (ticket queue)
            mttr_map = {"Critical": (10, 45), "High": (30, 120),
                        "Medium": (60, 300), "Low": (120, 720)}
            mttr_min, mttr_max = mttr_map[severity]
            mttr = random.randint(mttr_min, mttr_max)

            open_time  = date + timedelta(hours=random.randint(0, 23),
                                          minutes=random.randint(0, 59))
            close_time = open_time + timedelta(minutes=mttr)

            records.append({
                "incident_id":   f"INC-{incident_id}",
                "timestamp":     open_time.strftime("%Y-%m-%d %H:%M:%S"),
                "resolved_at":   close_time.strftime("%Y-%m-%d %H:%M:%S"),
                "region":        region,
                "device":        random.choice(devices),
                "incident_type": random.choice(INCIDENT_TYPES),
                "severity":      severity,
                "mttr_minutes":  mttr,
                "status":        "Resolved",
            })
            incident_id += 1

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(OUTPUT_DIR, "incidents.csv"), index=False)
    print(f"[✓] incidents.csv        → {len(df):,} rows")
    return df


# ── 4. Simulation Summary ────────────────────────────────────────────────────
def write_summary(metrics_df, uptime_df, incidents_df):
    summary = {
        "simulation_run":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period_days":       DAYS,
        "regions":           REGIONS,
        "total_metric_rows": len(metrics_df),
        "total_devices":     len(uptime_df),
        "total_incidents":   len(incidents_df),
        "avg_latency_ms":    round(metrics_df["latency_ms"].mean(), 2),
        "avg_packet_loss":   round(metrics_df["packet_loss_pct"].mean(), 3),
        "avg_uptime_pct":    round(uptime_df["uptime_pct"].mean(), 3),
    }
    with open(os.path.join(OUTPUT_DIR, "simulation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[✓] simulation_summary.json written")
    print(f"\n{'─'*45}")
    print(f"  Simulation complete — {DAYS} days, {len(REGIONS)} regions")
    print(f"  Avg Latency : {summary['avg_latency_ms']} ms")
    print(f"  Avg Uptime  : {summary['avg_uptime_pct']} %")
    print(f"  Incidents   : {summary['total_incidents']}")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    print("\n=== NetOps Data Simulator ===\n")
    m = simulate_network_metrics()
    u = simulate_uptime_logs()
    i = simulate_incidents()
    write_summary(m, u, i)

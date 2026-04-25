"""
Stage 3: Data Processing Pipeline
===================================
Ingests raw CSVs → cleans → enriches → stores structured output.

Pipeline steps:
  1. Load raw files with validation
  2. Clean: parse timestamps, drop nulls, clip outliers
  3. Enrich: add time features, severity scores, SLA flags
  4. Aggregate: hourly/daily summaries
  5. Store: processed CSVs for analytics layer
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# SLA thresholds (industry-standard for cloud infra)
SLA = {
    "latency_ms_warn":    50,    # ms — alert if avg > 50 ms
    "latency_ms_crit":    150,   # ms — critical if avg > 150 ms
    "packet_loss_warn":   1.0,   # % — alert threshold
    "packet_loss_crit":   5.0,   # % — critical threshold
    "uptime_target":      99.9,  # % — standard cloud SLA
}


# ── STEP 1 & 2: Load + Clean Network Metrics ────────────────────────────────
def process_network_metrics() -> pd.DataFrame:
    print("[1/3] Processing network_metrics.csv ...")
    df = pd.read_csv(os.path.join(RAW_DIR, "network_metrics.csv"))

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Drop rows with any null in key columns
    key_cols = ["latency_ms", "packet_loss_pct", "bandwidth_mbps"]
    before = len(df)
    df.dropna(subset=key_cols, inplace=True)
    print(f"    Dropped {before - len(df)} null rows")

    # Clip extreme outliers (IQR method per region)
    for region, grp in df.groupby("region"):
        q_lo = grp["latency_ms"].quantile(0.01)
        q_hi = grp["latency_ms"].quantile(0.99)
        df.loc[df["region"] == region, "latency_ms"] = \
            df.loc[df["region"] == region, "latency_ms"].clip(q_lo, q_hi)

    # ── STEP 3: Enrich ──────────────────────────────────────────────────────
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.day_name()
    df["date"]         = df["timestamp"].dt.date
    df["is_business_hours"] = df["hour"].between(8, 18).astype(int)

    # SLA flags
    df["latency_breach"] = (df["latency_ms"] > SLA["latency_ms_warn"]).astype(int)
    df["loss_breach"]    = (df["packet_loss_pct"] > SLA["packet_loss_warn"]).astype(int)

    # Health score per row: 100 = perfect, decreases with latency & loss
    df["health_score"] = 100 \
        - (df["latency_ms"] / SLA["latency_ms_crit"] * 40).clip(0, 40) \
        - (df["packet_loss_pct"] / SLA["packet_loss_crit"] * 30).clip(0, 30)
    df["health_score"] = df["health_score"].clip(0, 100).round(1)

    df.to_csv(os.path.join(PROC_DIR, "metrics_clean.csv"), index=False)
    print(f"    ✓ metrics_clean.csv saved ({len(df):,} rows)")

    # ── STEP 4: Hourly aggregation ──────────────────────────────────────────
    hourly = df.groupby(["date", "hour", "region"]).agg(
        avg_latency   =("latency_ms",      "mean"),
        max_latency   =("latency_ms",      "max"),
        avg_loss      =("packet_loss_pct", "mean"),
        avg_bandwidth =("bandwidth_mbps",  "mean"),
        avg_jitter    =("jitter_ms",       "mean"),
        health_score  =("health_score",    "mean"),
        breach_count  =("latency_breach",  "sum"),
    ).round(2).reset_index()

    hourly.to_csv(os.path.join(PROC_DIR, "metrics_hourly.csv"), index=False)
    print(f"    ✓ metrics_hourly.csv  saved ({len(hourly):,} rows)")

    # ── Daily aggregation ───────────────────────────────────────────────────
    daily = df.groupby(["date", "region"]).agg(
        avg_latency   =("latency_ms",      "mean"),
        p95_latency   =("latency_ms",      lambda x: x.quantile(0.95)),
        avg_loss      =("packet_loss_pct", "mean"),
        avg_bandwidth =("bandwidth_mbps",  "mean"),
        health_score  =("health_score",    "mean"),
        breach_hours  =("latency_breach",  "sum"),
    ).round(2).reset_index()

    daily.to_csv(os.path.join(PROC_DIR, "metrics_daily.csv"), index=False)
    print(f"    ✓ metrics_daily.csv   saved ({len(daily):,} rows)\n")
    return df


# ── Process Uptime ────────────────────────────────────────────────────────────
def process_uptime() -> pd.DataFrame:
    print("[2/3] Processing uptime_logs.csv ...")
    df = pd.read_csv(os.path.join(RAW_DIR, "uptime_logs.csv"))

    df["downtime_start"] = pd.to_datetime(df["downtime_start"])
    df["downtime_end"]   = pd.to_datetime(df["downtime_end"])
    df["sla_met"]        = (df["uptime_pct"] >= SLA["uptime_target"]).astype(int)
    df["sla_gap"]        = (SLA["uptime_target"] - df["uptime_pct"]).clip(lower=0).round(3)

    # Region-level uptime summary
    region_summary = df.groupby("region").agg(
        avg_uptime_pct  =("uptime_pct",      "mean"),
        min_uptime_pct  =("uptime_pct",      "min"),
        devices_meeting_sla=("sla_met",       "sum"),
        total_devices   =("device",           "count"),
        total_downtime_hrs=("downtime_hours", "sum"),
    ).round(3).reset_index()
    region_summary["sla_compliance_pct"] = (
        region_summary["devices_meeting_sla"] / region_summary["total_devices"] * 100
    ).round(1)

    df.to_csv(os.path.join(PROC_DIR, "uptime_clean.csv"), index=False)
    region_summary.to_csv(os.path.join(PROC_DIR, "uptime_by_region.csv"), index=False)
    print(f"    ✓ uptime_clean.csv + uptime_by_region.csv saved\n")
    return df


# ── Process Incidents ─────────────────────────────────────────────────────────
def process_incidents() -> pd.DataFrame:
    print("[3/3] Processing incidents.csv ...")
    df = pd.read_csv(os.path.join(RAW_DIR, "incidents.csv"))

    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["resolved_at"] = pd.to_datetime(df["resolved_at"])
    df["date"]        = df["timestamp"].dt.date
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()

    # Severity score for sorting/weighting
    sev_score = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
    df["severity_score"] = df["severity"].map(sev_score)

    # Daily summary
    daily_inc = df.groupby(["date", "severity"]).agg(
        count        =("incident_id",  "count"),
        avg_mttr_min =("mttr_minutes", "mean"),
    ).round(1).reset_index()

    # Region summary
    region_inc = df.groupby("region").agg(
        total_incidents =("incident_id",  "count"),
        avg_mttr_min    =("mttr_minutes", "mean"),
        critical_count  =("severity",     lambda x: (x == "Critical").sum()),
    ).round(1).reset_index()

    # Type breakdown
    type_breakdown = df.groupby("incident_type").agg(
        count        =("incident_id",  "count"),
        avg_mttr_min =("mttr_minutes", "mean"),
    ).round(1).reset_index().sort_values("count", ascending=False)

    df.to_csv(os.path.join(PROC_DIR, "incidents_clean.csv"), index=False)
    daily_inc.to_csv(os.path.join(PROC_DIR, "incidents_daily.csv"), index=False)
    region_inc.to_csv(os.path.join(PROC_DIR, "incidents_by_region.csv"), index=False)
    type_breakdown.to_csv(os.path.join(PROC_DIR, "incident_types.csv"), index=False)
    print(f"    ✓ incidents_clean + daily + region + types saved\n")
    return df


# ── Pipeline Runner ───────────────────────────────────────────────────────────
def run_pipeline():
    print("\n=== NetOps Processing Pipeline ===\n")
    start = datetime.now()

    metrics   = process_network_metrics()
    uptime    = process_uptime()
    incidents = process_incidents()

    elapsed = (datetime.now() - start).total_seconds()
    log = {
        "pipeline_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 2),
        "files_generated": [
            "metrics_clean.csv", "metrics_hourly.csv", "metrics_daily.csv",
            "uptime_clean.csv", "uptime_by_region.csv",
            "incidents_clean.csv", "incidents_daily.csv",
            "incidents_by_region.csv", "incident_types.csv",
        ]
    }
    with open(os.path.join(PROC_DIR, "pipeline_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"=== Pipeline complete in {elapsed:.2f}s ===\n")


if __name__ == "__main__":
    run_pipeline()

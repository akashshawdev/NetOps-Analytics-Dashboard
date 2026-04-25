"""
Stage 5: NetOps Analytics Dashboard (Streamlit)
================================================
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NetOps Analytics Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
RPT  = os.path.join(BASE, "reports", "analytics_report.json")

REGION_LABELS = {
    "us-east-1":      "US East (N. Virginia)",
    "eu-west-1":      "EU West (Ireland)",
    "ap-southeast-1": "AP Southeast (Singapore)",
    "us-west-2":      "US West (Oregon)",
    "eu-central-1":   "EU Central (Frankfurt)",
}

COLOR_MAP = {
    "us-east-1":      "#4f8ef7",
    "eu-west-1":      "#f7954f",
    "ap-southeast-1": "#4ff7a0",
    "us-west-2":      "#f74f7a",
    "eu-central-1":   "#b04ff7",
}

SEVERITY_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f97316",
    "Medium":   "#eab308",
    "Low":      "#22c55e",
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_report():
    with open(RPT) as f:
        return json.load(f)

@st.cache_data
def load_csv(name):
    return pd.read_csv(os.path.join(PROC, name))

try:
    report    = load_report()
    metrics   = load_csv("metrics_clean.csv")
    daily     = load_csv("metrics_daily.csv")
    incidents = load_csv("incidents_clean.csv")
    uptime    = load_csv("uptime_clean.csv")
    data_ok   = True
except Exception as e:
    data_ok = False
    st.error(f"⚠️ Data not found. Run the pipeline first.\n\n`python pipeline/simulate_data.py && python pipeline/process_data.py && python analytics/compute_kpis.py`\n\nError: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/network.png", width=60)
    st.title("NetOps Analytics")
    st.caption(f"Period: {report['period']}")
    st.divider()

    selected_regions = st.multiselect(
        "Filter Regions",
        options=list(REGION_LABELS.keys()),
        default=list(REGION_LABELS.keys()),
        format_func=lambda x: REGION_LABELS[x],
    )
    severity_filter = st.multiselect(
        "Incident Severity",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"],
    )
    st.divider()
    st.caption(f"Generated: {report['generated_at']}")
    st.caption("Simulated cloud infra · 5 regions · 30 days")

# ── Filter data ───────────────────────────────────────────────────────────────
if selected_regions:
    metrics_f   = metrics[metrics["region"].isin(selected_regions)]
    daily_f     = daily[daily["region"].isin(selected_regions)]
    incidents_f = incidents[
        incidents["region"].isin(selected_regions) &
        incidents["severity"].isin(severity_filter)
    ]
    uptime_f = uptime[uptime["region"].isin(selected_regions)]
else:
    metrics_f, daily_f, incidents_f, uptime_f = metrics, daily, incidents, uptime

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("## 🌐 NetOps Analytics Dashboard")
st.markdown("Cloud infrastructure monitoring · AWS/Azure-style multi-region deployment")
st.divider()

# ── KPI TILES ─────────────────────────────────────────────────────────────────
kup  = report["kpi_uptime"]
klat = report["kpi_latency"]
kinc = report["kpi_incidents"]
kimp = report["impact"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🟢 Network Uptime",    f"{kup['overall_uptime_pct']}%",
          delta="SLA ✓" if kup["sla_met"] else "SLA ✗")
c2.metric("⚡ Avg Latency",      f"{klat['overall_avg_ms']} ms",
          delta=f"P95: {klat['overall_p95_ms']} ms")
c3.metric("📦 Packet Loss",      f"{report['kpi_loss']['overall_avg_pct']}%")
c4.metric("🚨 Total Incidents",  kinc["total_incidents"],
          delta=f"MTTR: {kinc['overall_avg_mttr']} min")
c5.metric("💚 Health Score",     f"{report['kpi_health']['overall_health_score']}/100")

st.divider()

# ── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Latency & Loss", "📡 Uptime", "🚨 Incidents", "🔍 Anomalies", "💼 Impact"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Latency & Loss
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Daily Average Latency by Region")
        daily_f["date"] = pd.to_datetime(daily_f["date"])
        fig = px.line(
            daily_f[daily_f["region"].isin(selected_regions)],
            x="date", y="avg_latency", color="region",
            color_discrete_map=COLOR_MAP,
            labels={"avg_latency": "Avg Latency (ms)", "date": "Date", "region": "Region"},
            template="plotly_dark",
        )
        fig.update_traces(line_width=2)
        fig.add_hline(y=50, line_dash="dash", line_color="#eab308",
                      annotation_text="Warn 50ms", annotation_position="bottom right")
        fig.add_hline(y=150, line_dash="dash", line_color="#ef4444",
                      annotation_text="Critical 150ms", annotation_position="bottom right")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Avg Latency by Region")
        lat_by_region = metrics_f.groupby("region")["latency_ms"].mean().round(2).reset_index()
        lat_by_region["label"] = lat_by_region["region"].map(REGION_LABELS)
        fig2 = px.bar(lat_by_region, x="latency_ms", y="label", orientation="h",
                      color="latency_ms", color_continuous_scale="RdYlGn_r",
                      template="plotly_dark",
                      labels={"latency_ms": "Avg Latency (ms)", "label": ""})
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Hourly Latency Heatmap (Business Hours vs Off-Hours)")
    hour_region = metrics_f.groupby(["hour", "region"])["latency_ms"].mean().reset_index()
    pivot = hour_region.pivot(index="region", columns="hour", values="latency_ms").round(1)
    fig3 = px.imshow(pivot, color_continuous_scale="RdYlGn_r", aspect="auto",
                     labels={"color": "Latency (ms)", "x": "Hour of Day", "y": "Region"},
                     template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Packet Loss % Over Time")
        loss_daily = metrics_f.groupby(["date", "region"])["packet_loss_pct"].mean().reset_index()
        loss_daily["date"] = pd.to_datetime(loss_daily["date"])
        fig4 = px.area(loss_daily, x="date", y="packet_loss_pct", color="region",
                       color_discrete_map=COLOR_MAP, template="plotly_dark",
                       labels={"packet_loss_pct": "Packet Loss (%)", "date": "Date"})
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.subheader("Bandwidth Utilization")
        bw = metrics_f.groupby("region")["bandwidth_mbps"].mean().round(1).reset_index()
        fig5 = px.bar(bw, x="region", y="bandwidth_mbps", color="region",
                      color_discrete_map=COLOR_MAP, template="plotly_dark",
                      labels={"bandwidth_mbps": "Avg Bandwidth (Mbps)", "region": "Region"})
        fig5.update_layout(showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Uptime
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Device Uptime % by Region")
        fig = px.box(uptime_f, x="region", y="uptime_pct", color="region",
                     color_discrete_map=COLOR_MAP, template="plotly_dark",
                     labels={"uptime_pct": "Uptime (%)", "region": "Region"})
        fig.add_hline(y=99.9, line_dash="dash", line_color="#eab308",
                      annotation_text="SLA Target 99.9%")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("SLA Compliance Overview")
        total   = len(uptime_f)
        sla_met = (uptime_f["uptime_pct"] >= 99.9).sum()
        fig2 = go.Figure(go.Pie(
            labels=["Meeting SLA", "Below SLA"],
            values=[sla_met, total - sla_met],
            hole=0.55,
            marker_colors=["#22c55e", "#ef4444"],
        ))
        fig2.update_layout(template="plotly_dark",
                           annotations=[{"text": f"{sla_met}/{total}", "showarrow": False,
                                         "font": {"size": 22}}])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Device-Level Uptime Table")
    display = uptime_f[["region", "device", "uptime_pct", "downtime_hours", "sla_met"]].copy()
    display["sla_met"] = display["sla_met"].map({1: "✅ Yes", 0: "❌ No"})
    display.columns = ["Region", "Device", "Uptime %", "Downtime (hrs)", "SLA Met"]
    st.dataframe(display.sort_values("Uptime %").reset_index(drop=True),
                 use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Incidents
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Incident Frequency Over Time")
        inc_daily = incidents_f.copy()
        inc_daily["date"] = pd.to_datetime(inc_daily["timestamp"]).dt.date
        inc_ts = inc_daily.groupby(["date", "severity"]).size().reset_index(name="count")
        inc_ts["date"] = pd.to_datetime(inc_ts["date"])
        fig = px.bar(inc_ts, x="date", y="count", color="severity",
                     color_discrete_map=SEVERITY_COLORS, template="plotly_dark",
                     labels={"count": "Incidents", "date": "Date"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("By Severity")
        sev_counts = incidents_f["severity"].value_counts().reset_index()
        fig2 = px.pie(sev_counts, names="severity", values="count",
                      color="severity", color_discrete_map=SEVERITY_COLORS,
                      template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("MTTR by Severity (minutes)")
        mttr = incidents_f.groupby("severity")["mttr_minutes"].mean().round(1).reset_index()
        fig3 = px.bar(mttr, x="severity", y="mttr_minutes", color="severity",
                      color_discrete_map=SEVERITY_COLORS, template="plotly_dark",
                      labels={"mttr_minutes": "Avg MTTR (min)", "severity": "Severity"})
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Top Incident Types")
        types = incidents_f["incident_type"].value_counts().head(6).reset_index()
        fig4 = px.bar(types, x="count", y="incident_type", orientation="h",
                      template="plotly_dark",
                      labels={"count": "Count", "incident_type": "Type"},
                      color="count", color_continuous_scale="Blues")
        fig4.update_coloraxes(showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Recent Incidents")
    recent = incidents_f.sort_values("timestamp", ascending=False).head(15)[
        ["timestamp", "incident_id", "region", "device", "incident_type", "severity", "mttr_minutes"]
    ]
    st.dataframe(recent.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: Anomalies
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    anomalies = report["anomalies"]
    st.subheader(f"🔍 Anomaly Detection — {len(anomalies)} flags detected (Z-Score > 3σ)")
    st.info("Z-Score method: flags data points more than 3 standard deviations from the regional mean. "
            "In real NOCs, these trigger PagerDuty / OpsGenie alerts automatically.")

    if anomalies:
        adf = pd.DataFrame(anomalies)
        adf = adf[adf["region"].isin(selected_regions)]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(adf, x="timestamp", y="value", color="severity",
                             symbol="metric", size="z_score",
                             color_discrete_map={"Critical":"#ef4444","High":"#f97316"},
                             template="plotly_dark",
                             labels={"value": "Metric Value", "timestamp": "Time"},
                             title="Anomaly Events")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            anom_region = adf["region"].value_counts().reset_index()
            fig2 = px.bar(anom_region, x="region", y="count", color="region",
                          color_discrete_map=COLOR_MAP, template="plotly_dark",
                          title="Anomalies by Region",
                          labels={"count": "Count", "region": "Region"})
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Anomaly Log")
        st.dataframe(adf.sort_values("z_score", key=abs, ascending=False).reset_index(drop=True),
                     use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Impact
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    imp = report["impact"]
    st.subheader("💼 Business Impact of the NetOps Platform")
    st.caption("Based on industry benchmarks. MTTR reduction modeled on Gartner AIOps research.")

    c1, c2, c3 = st.columns(3)
    c1.metric("MTTR Before", f"{imp['mttr_before_min']} min")
    c1.metric("MTTR After",  f"{imp['mttr_after_min']} min",
              delta=f"-{imp['mttr_reduction_pct']}%", delta_color="inverse")

    c2.metric("Early Detection",  f"{imp['anomaly_early_detection_min']} min faster")
    c2.metric("Auto-flagged",     f"{imp['incidents_auto_flagged']} incidents",
              delta="72% automation rate")

    c3.metric("Est. Cost Saved",         f"${imp['estimated_cost_saved_usd']:,}")
    c3.metric("SLA Breach Reduction",    f"{imp['sla_breach_reduction_pct']}%",
              delta_color="inverse")

    st.divider()
    st.subheader("How these numbers are estimated")
    st.markdown("""
| Metric | Method |
|--------|--------|
| **MTTR reduction (35%)** | Industry avg: automated alerting vs manual log-checking. Source: Gartner AIOps 2024 |
| **Early detection (18 min)** | Z-score alerting fires before human review cycle (~20 min avg) |
| **Cost saved** | Downtime hrs × $5,000/hr (enterprise SLA penalty) × 30% prevention rate |
| **SLA breach reduction** | Proactive threshold alerts catch breaches 1–2 hours before SLA window closes |
    """)

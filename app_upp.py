# app.py
import os
import re
import json
import html
import difflib
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import altair as alt

# Optional OpenAI import (graceful if not installed)
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False

# =========================
# Page Config & Global CSS
# =========================
st.set_page_config(
    page_title="AI Log Analyzer",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Background & container */
.stApp {
    background-image: url("https://img.freepik.com/premium-vector/white-technology-background_23-2148400094.jpg?w=740");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
.css-18e3th9, .main .block-container {
    background-color: rgba(255,255,255,0.90);
    padding: 1.5rem;
    border-radius: 16px;
}

/* Typography & centering */
h1, h2, h3, h4, h5, h6, p, div, table, svg {
    text-align: center !important;
}
.stButton button {
    display: block;
    margin-left: auto;
    margin-right: auto;
}

/* Inputs */
.stTextInput input, .stTextArea textarea {
    background-color: #ffffff !important;
    color: #111 !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
    color: #111 !important;
    font-weight: 700 !important;
}

/* Inline highlight for keyword matches */
.hl { background: #fff3cd; padding: 0 2px; border-radius: 4px; }
.add { background: #e6ffed; }
.rem { background: #ffeef0; }

/* Log viewer styling */
.logbox {
    text-align: left !important;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    white-space: pre-wrap;
    background: #0f172a;
    color: #e2e8f0;
    padding: 12px;
    border-radius: 12px;
    max-height: 420px;
    overflow: auto;
    border: 1px solid #1f2937;
}
.line-num {
    opacity: 0.6;
    display: inline-block;
    width: 38px;
    user-select: none;
}
.section {
    border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; margin: 8px 0;
    background: #fff;
    text-align: left !important;
}
.kpi {
    text-align: center !important;
}
.small {
    font-size: 12px; opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("""
<div style="text-align:center; padding:8px 0 0;">
  <img src="https://images.moneycontrol.com/static-mcnews/2023/07/TCS-LOGO.png?impolicy=website&width=1600&height=900" width="180">
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #0f172a; margin-top: 0;'>🔍 AI Log Analyzer — Ultimate Edition</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="section">
<b>Upload logs or paste text.</b> Get errors, warnings, root causes, suggested fixes, clustering, timelines, and side-by-side diff — with or without an API key.
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar: Mode & Settings
# =========================
st.sidebar.header("⚙️ Mode")
mode = st.sidebar.radio("Select Mode", ["Normal Analysis", "Log Comparison"], index=0)

st.sidebar.header("🤖 AI Settings")
st.sidebar.caption(
    "⚠️ Note: Only chat models like `gpt-4o`, `gpt-4o-mini`, and `gpt-35-turbo` "
    "are guaranteed to produce correct JSON results. "
    "Models like Whisper or Embedding may fail and fall back to local parsing."
)
env_key = os.environ.get("OPENAI_API_KEY", "")
api_key = st.sidebar.text_input("OpenAI API Key (optional)", type="password", value="")
use_key = api_key.strip() or env_key.strip()
model = st.sidebar.selectbox(
    "Model",
    [  "azure/genailab-maas-gpt-35-turbo",
    "azure/genailab-maas-gpt-4o",
    "azure/genailab-maas-gpt-4o-mini",
    "azure/genailab-maas-text-embedding-3-large",
    "azure/genailab-maas-whisper",
    "azure_ai/genailab-maas-DeepSeek-R1",
    "azure_ai/genailab-maas-DeepSeek-V3-0324",
    "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
    "azure_ai/genailab-maas-Llama-3.3-70B-Instruct",
    "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
    "azure_ai/genailab-maas-Phi-3.5-vision-instruct",
    "azure_ai/genailab-maas-Phi-4-reasoning"],
    index=0
)
temperature = st.sidebar.slider("Creativity (temperature)", 0.0, 1.0, 0.2)

st.sidebar.header("🔍 Filters")
search_keyword = st.sidebar.text_input("Keyword / Regex (optional)", value="")
case_sensitive = st.sidebar.checkbox("Case sensitive search", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Built By KolkataCodeWizards")

# =========================
# Utilities
# =========================
TIMESTAMP_PAT = r'(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})'

SEV_PATTERNS = {
    "CRITICAL": re.compile(r"\bCRITICAL|FATAL|PANIC|SEVERE\b", re.I),
    "ERROR": re.compile(r"\bERROR|ERR\b", re.I),
    "WARNING": re.compile(r"\bWARNING|WARN\b", re.I),
    "INFO": re.compile(r"\bINFO\b", re.I),
    "DEBUG": re.compile(r"\bDEBUG|TRACE\b", re.I),
}

def extract_timestamps(text: str) -> List[datetime]:
    matches = re.findall(TIMESTAMP_PAT, text)
    stamps = []
    for m in matches:
        try:
            # support "YYYY-MM-DD HH:MM:SS" and "YYYY-MM-DDTHH:MM:SS"
            m = m.replace("T", " ")
            stamps.append(datetime.strptime(m, "%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
    return stamps

def severity_of_line(line: str) -> str:
    for sev, pat in SEV_PATTERNS.items():
        if pat.search(line):
            return sev
    return "INFO"  # default

def split_lines(text: str) -> List[str]:
    return [ln for ln in text.splitlines()]

def keyword_highlight(line: str, pattern: str, case_sensitive: bool) -> str:
    if not pattern:
        return html.escape(line)
    flags = 0 if case_sensitive else re.I
    try:
        rx = re.compile(pattern, flags)
    except re.error:
        # If invalid regex, fallback to literal search
        escaped = html.escape(line)
        if (pattern in line) if case_sensitive else (pattern.lower() in line.lower()):
            return escaped.replace(
                html.escape(pattern),
                f'<span class="hl">{html.escape(pattern)}</span>'
            )
        return escaped

    def repl(m):
        return f'<span class="hl">{html.escape(m.group(0))}</span>'

    return rx.sub(repl, html.escape(line))

def cluster_similar(items: List[str], threshold: float = 0.82) -> List[Dict[str, Any]]:
    """
    Simple clustering with difflib similarity, O(n^2) but fine for typical log batches.
    Returns list of clusters: {"rep": str, "items": [str], "count": int}
    """
    clusters: List[Dict[str, Any]] = []
    for it in items:
        placed = False
        for c in clusters:
            if difflib.SequenceMatcher(None, c["rep"], it).ratio() >= threshold:
                c["items"].append(it)
                c["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep": it, "items": [it], "count": 1})
    clusters.sort(key=lambda x: (-x["count"], x["rep"]))
    return clusters

def local_smart_parse(text: str) -> Dict[str, Any]:
    """
    Works without AI. Extracts severities, naive root causes, and generic fixes.
    """
    lines = [ln.strip() for ln in split_lines(text) if ln.strip()]
    errors = [ln for ln in lines if severity_of_line(ln) in ("ERROR", "CRITICAL")]
    warnings = [ln for ln in lines if severity_of_line(ln) == "WARNING"]
    infos = [ln for ln in lines if severity_of_line(ln) == "INFO"]
    debugs = [ln for ln in lines if severity_of_line(ln) == "DEBUG"]

    # Heuristic root causes/fixes
    root_causes = []
    fixes = []
    if any("disk" in ln.lower() and "full" in ln.lower() for ln in errors + warnings):
        root_causes.append("Low or full disk space on the host/volume.")
        fixes.append("Clean temporary files, rotate logs, or increase disk allocation.")

    if any("timeout" in ln.lower() for ln in errors + warnings):
        root_causes.append("Network/service timeout under load or dependency unresponsive.")
        fixes.append("Increase timeouts, add retries/backoff, or check dependency health.")

    if any("connection refused" in ln.lower() for ln in errors):
        root_causes.append("Target service not listening or blocked by firewall/security group.")
        fixes.append("Verify service is running, port open, and network rules allow traffic.")

    timestamps = extract_timestamps(text)
    summary = (
        f"Found {len(errors)} errors/criticals, {len(warnings)} warnings, "
        f"{len(infos)} info, and {len(debugs)} debug entries."
    )

    return {
        "errors": errors,
        "warnings": warnings,
        "info": infos,
        "debug": debugs,
        "critical": [ln for ln in lines if "CRITICAL" in ln.upper() or "FATAL" in ln.upper()],
        "root_causes": list(dict.fromkeys(root_causes)) or ["Not identified."],
        "fixes": list(dict.fromkeys(fixes)) or ["No specific fixes suggested."],
        "summary": summary
    }

def call_ai_analysis(text: str, model: str, temperature: float, key: str) -> Dict[str, Any]:
    """
    Ask the model for structured JSON. If anything fails, fall back to local parser.
    """
    if not (OPENAI_AVAILABLE and key):
        return local_smart_parse(text)

    try:
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content":
                 "You are an expert log analyst. Return STRICT JSON with keys: "
                 "errors (list of strings), warnings (list), info (list), debug (list), "
                 "critical (list), root_causes (list), fixes (list), summary (string). "
                 "Do not include any backticks or commentary."},
                {"role": "user", "content": text}
            ]
        )
        txt = resp.choices[0].message.content.strip()
        # Attempt JSON extraction if model added extra text
        # Find first { and last }
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end != -1 and end > start:
            txt = txt[start:end+1]
        data = json.loads(txt)
        # Ensure required keys exist
        for k in ["errors","warnings","info","debug","critical","root_causes","fixes","summary"]:
            data.setdefault(k, [] if k != "summary" else "")
        return data
    except Exception:
        return local_smart_parse(text)

def severity_table(data: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "Severity": ["CRITICAL","ERROR","WARNING","INFO","DEBUG"],
        "Count": [
            len(data.get("critical", [])),
            len(data.get("errors", [])),
            len(data.get("warnings", [])),
            len(data.get("info", [])),
            len(data.get("debug", []))
        ]
    })

def make_html_report(title: str, summary: str, sev_df: pd.DataFrame, clusters: Dict[str, Any]) -> str:
    donut = sev_df.to_json(orient="records")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
h1, h2, h3 {{ margin: 0 0 8px; }}
.card {{ border:1px solid #e5e7eb; border-radius:12px; padding:16px; margin:12px 0; }}
.small {{ opacity:.75; font-size: 13px; }}
pre {{ background:#0f172a; color:#e2e8f0; padding:12px; border-radius: 10px; overflow:auto; }}
.tag {{ display:inline-block; padding:2px 6px; background:#eef2ff; border-radius:6px; margin:2px; }}
</style></head>
<body>
<h1>AI Log Analyzer Report</h1>
<div class="small">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
<div class="card"><h2>Executive Summary</h2><p>{html.escape(summary)}</p></div>
<div class="card"><h2>Severity Overview</h2><pre>{html.escape(donut)}</pre><div class="small">JSON for charting pipelines.</div></div>
<div class="card"><h2>Top Issue Clusters</h2>
{"".join(f"<div class='tag'>{html.escape(k)} &times; {v['count']}</div>" for k,v in clusters.items()) or "<p>No clusters found.</p>"}
</div>
</body></html>"""

def to_dataframe(lst: List[str], col: str) -> pd.DataFrame:
    return pd.DataFrame(lst, columns=[col])

def apply_keyword_filter(lst: List[str], pattern: str, case_sensitive: bool) -> List[str]:
    if not pattern: return lst
    try:
        flags = 0 if case_sensitive else re.I
        rx = re.compile(pattern, flags)
        return [x for x in lst if rx.search(x)]
    except re.error:
        # Invalid regex: fallback to contains
        if case_sensitive:
            return [x for x in lst if pattern in x]
        else:
            p = pattern.lower()
            return [x for x in lst if p in x.lower()]

def render_logbox(text: str, pattern: str):
    lines = split_lines(text)
    buf = []
    for i, ln in enumerate(lines, start=1):
        buf.append(f'<span class="line-num">{i:>3}</span> {keyword_highlight(ln, pattern, case_sensitive)}')
    st.markdown('<div class="logbox">' + "\n".join(buf) + '</div>', unsafe_allow_html=True)

def side_by_side_diff(a: List[str], b: List[str]) -> str:
    """
    Returns HTML with side-by-side diff table.
    """
    sm = difflib.SequenceMatcher(None, a, b)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        while i1 < i2 or j1 < j2:
            left = html.escape(a[i1]) if i1 < i2 else ""
            right = html.escape(b[j1]) if j1 < j2 else ""
            cls_l = cls_r = ""
            if i1 < i2 and j1 < j2:
                if left != right:
                    cls_l = "rem"; cls_r = "add"
            elif i1 < i2:
                cls_l = "rem"
            else:
                cls_r = "add"
            rows.append(f"""
            <tr>
              <td class="{cls_l}">{left}</td>
              <td class="{cls_r}">{right}</td>
            </tr>""")
            if i1 < i2: i1 += 1
            if j1 < j2: j1 += 1
    return f"""
    <div class="section" style="overflow:auto;">
      <table style="width:100%; border-collapse:collapse;">
        <thead>
          <tr>
            <th style="text-align:left; padding:6px; border-bottom:1px solid #e5e7eb;">Log A</th>
            <th style="text-align:left; padding:6px; border-bottom:1px solid #e5e7eb;">Log B</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """

# =========================
# Mode: Normal Analysis
# =========================
if mode == "Normal Analysis":
    uploaded_files = st.file_uploader("📂 Upload one or more logs", type=["log","txt"], accept_multiple_files=True)
    pasted = st.text_area("Or paste log text here:", height=220, value="Sample log:\n[2025-08-28 23:45:01] ERROR: Disk full\n[2025-08-28 23:45:05] WARNING: CPU high")
    run = st.button("🚀 Analyze Logs", type="primary")

    if run:
        # Compose text
        combined = ""
        if uploaded_files:
            for f in uploaded_files:
                combined += f"\n--- File: {f.name} ---\n"
                combined += f.read().decode("utf-8", errors="replace") + "\n"
        else:
            combined = pasted.strip()

        if not combined:
            st.warning("Please upload or paste logs.")
            st.stop()

        with st.spinner("Analyzing…"):
            data = call_ai_analysis(combined, model=model, temperature=temperature, key=use_key)

        # KPI metrics
        sev_df = severity_table(data)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🔥 Critical", int(sev_df.loc[sev_df["Severity"]=="CRITICAL","Count"].values[0]))
        c2.metric("❌ Errors", int(sev_df.loc[sev_df["Severity"]=="ERROR","Count"].values[0]))
        c3.metric("⚠️ Warnings", int(sev_df.loc[sev_df["Severity"]=="WARNING","Count"].values[0]))
        c4.metric("ℹ️ Info", int(sev_df.loc[sev_df["Severity"]=="INFO","Count"].values[0]))
        c5.metric("🐞 Debug", int(sev_df.loc[sev_df["Severity"]=="DEBUG","Count"].values[0]))

        tabs = st.tabs([
            "📊 Dashboard",
            "❌ Errors",
            "⚠️ Warnings",
            "🔥 Critical",
            "ℹ️ Info",
            "🐞 Debug",
            "🧩 Clusters",
            "📄 Raw Logs",
            "📝 JSON Output",
            "⬇️ Export"
        ])

        # Dashboard
        with tabs[0]:
            st.subheader("Executive Summary")
            st.write(data.get("summary", "No summary available."))

            # Donut chart
            donut = alt.Chart(sev_df).mark_arc(innerRadius=60).encode(
                theta="Count",
                color="Severity",
                tooltip=["Severity","Count"]
            ).properties(title="Severity Distribution")
            st.altair_chart(donut, use_container_width=True)

            # Timeline by hour
            stamps = extract_timestamps(combined)
            if stamps:
                df_time = pd.DataFrame({"Timestamp": stamps})
                time_chart = alt.Chart(df_time).mark_bar().encode(
                    x=alt.X("hours(Timestamp):O", title="Hour of Day"),
                    y=alt.Y("count():Q", title="Entries"),
                    tooltip=["count()"]
                ).properties(title="Log Frequency by Hour")
                st.altair_chart(time_chart, use_container_width=True)
            else:
                st.info("No timestamps detected for timeline.")

        # Helper to render lists with filters
        def render_list(lst: List[str], label: str):
            filtered = apply_keyword_filter(lst, search_keyword, case_sensitive)
            if filtered:
                st.dataframe(to_dataframe(filtered, label))
            else:
                st.info(f"No {label.lower()} found (after filter).")

        with tabs[1]:
            st.subheader("Errors")
            render_list(data.get("errors", []), "Errors")

        with tabs[2]:
            st.subheader("Warnings")
            render_list(data.get("warnings", []), "Warnings")

        with tabs[3]:
            st.subheader("Critical")
            render_list(data.get("critical", []), "Critical")

        with tabs[4]:
            st.subheader("Info")
            render_list(data.get("info", []), "Info")

        with tabs[5]:
            st.subheader("Debug")
            render_list(data.get("debug", []), "Debug")

        # Clusters tab (on errors + warnings)
        with tabs[6]:
            st.subheader("Clusters of Similar Issues")
            base_items = (data.get("errors", []) or []) + (data.get("warnings", []) or [])
            base_items = apply_keyword_filter(base_items, search_keyword, case_sensitive)
            if base_items:
                clusters = cluster_similar(base_items)
                for c in clusters[:25]:
                    with st.expander(f"{c['rep']}  —  ×{c['count']}"):
                        st.write(pd.DataFrame(c["items"], columns=["Instance"]))
            else:
                st.info("No errors/warnings (or none match your filter).")

        # Raw logs with highlight
        with tabs[7]:
            st.subheader("Raw Logs (with highlight)")
            render_logbox(combined, search_keyword)

        # JSON output
        with tabs[8]:
            st.subheader("Model / Parser JSON Output")
            st.json(data)

        # Export
        with tabs[9]:
            st.subheader("Export")
            # JSON
            st.download_button(
                "💾 Download JSON Report",
                data=json.dumps(data, indent=2),
                file_name="log_analysis.json",
                mime="application/json"
            )
            # CSVs
            errs = to_dataframe(apply_keyword_filter(data.get("errors", []), search_keyword, case_sensitive), "error")
            warns = to_dataframe(apply_keyword_filter(data.get("warnings", []), search_keyword, case_sensitive), "warning")
            st.download_button("📊 Download Errors CSV", data=errs.to_csv(index=False), file_name="errors.csv", mime="text/csv")
            st.download_button("📊 Download Warnings CSV", data=warns.to_csv(index=False), file_name="warnings.csv", mime="text/csv")

            # HTML report
            clusters_src = cluster_similar((data.get("errors", []) or []) + (data.get("warnings", []) or []))
            cluster_map = {c["rep"]: {"count": c["count"]} for c in clusters_src[:30]}
            html_report = make_html_report("AI Log Analyzer Report", data.get("summary",""), sev_df, cluster_map)
            st.download_button("🧾 Download HTML Report (print to PDF)", data=html_report, file_name="log_report.html", mime="text/html")

# =========================
# Mode: Log Comparison
# =========================
else:
    st.subheader("🔍 Log Comparison (Side-by-Side)")
    colA, colB = st.columns(2)
    with colA:
        file1 = st.file_uploader("Upload Log A", type=["log","txt"], key="logA")
    with colB:
        file2 = st.file_uploader("Upload Log B", type=["log","txt"], key="logB")
    pastedA = st.text_area("Or paste Log A", height=160, key="pasteA")
    pastedB = st.text_area("Or paste Log B", height=160, key="pasteB")
    go = st.button("⚡ Compare", type="primary")

    if go:
        logA = ""
        if file1:
            logA = file1.read().decode("utf-8", errors="replace")
        elif pastedA.strip():
            logA = pastedA.strip()

        logB = ""
        if file2:
            logB = file2.read().decode("utf-8", errors="replace")
        elif pastedB.strip():
            logB = pastedB.strip()

        if not (logA and logB):
            st.warning("Please provide both logs (upload or paste).")
            st.stop()

        linesA = split_lines(logA)
        linesB = split_lines(logB)

        # Quick new/removed lists
        setA = set(linesA)
        setB = set(linesB)
        added = [ln for ln in linesB if ln not in setA]
        removed = [ln for ln in linesA if ln not in setB]

        st.success("Comparison Complete")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("➕ New in B vs A", len(added))
        with c2:
            st.metric("➖ Removed from A", len(removed))

        # Side-by-side diff
        st.markdown("### Inline Diff")
        st.markdown(side_by_side_diff(linesA, linesB), unsafe_allow_html=True)

        # AI / heuristic summary of differences
        st.markdown("### 🤖 Summary of Differences")
        diff_text = (
            "New Entries:\n" + "\n".join(added[:200]) +
            "\n\nRemoved Entries:\n" + "\n".join(removed[:200])
        )
        data_diff = call_ai_analysis(diff_text, model=model, temperature=temperature, key=use_key)
        st.write(data_diff.get("summary", "No summary available."))

        with st.expander("Details: New Entries"):
            render_logbox("\n".join(added) or "No new entries.", search_keyword)
        with st.expander("Details: Removed Entries"):
            render_logbox("\n".join(removed) or "No removed entries.", search_keyword)

        # Export diff
        export_payload = {
            "summary": data_diff.get("summary",""),
            "added": added,
            "removed": removed
        }
        st.download_button(
            "💾 Download Diff JSON",
            data=json.dumps(export_payload, indent=2),
            file_name="log_diff.json",
            mime="application/json"
        )

"""Self-contained HTML report renderer with interactive event distributions."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from event_study_framework.study import EventStudyResult


def _fmt(value: object) -> str:
    """Format a scalar value for display."""

    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))


def _fmt_pct(value: object) -> str:
    """Format a decimal as percentage text."""

    if value is None or value == "" or pd.isna(value):
        return ""
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def html_table(df: pd.DataFrame, columns: Sequence[str], percent_cols: Iterable[str] = ()) -> str:
    """Render a compact HTML table."""

    percent_set = set(percent_cols)
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    rows: List[str] = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row.get(col, "")
            text = _fmt_pct(value) if col in percent_set else _fmt(value)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _json_records(df: pd.DataFrame) -> str:
    """Serialize a DataFrame to JSON records with date fields normalized."""

    if df.empty:
        return "[]"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.replace([np.inf, -np.inf], np.nan)
    return json.dumps(json.loads(out.to_json(orient="records", date_format="iso", force_ascii=False)), ensure_ascii=False)


def _path_payload(result: EventStudyResult) -> str:
    """Serialize event paths for interactive display."""

    payload = {}
    for name, frame in result.event_paths.items():
        if frame.empty:
            payload[name] = []
            continue
        path = frame.reset_index(names="offset")
        payload[name] = json.loads(path.to_json(orient="records", force_ascii=False))
    return json.dumps(payload, ensure_ascii=False)


def _frame_mapping_payload(frames: dict[str, pd.DataFrame]) -> str:
    """Serialize an event-to-DataFrame mapping for browser-side charts."""

    payload = {}
    for name, frame in frames.items():
        if frame.empty:
            payload[name] = []
            continue
        normalized = frame.copy().replace([np.inf, -np.inf], np.nan)
        for column in normalized.columns:
            if pd.api.types.is_datetime64_any_dtype(normalized[column]):
                normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
        payload[name] = json.loads(
            normalized.to_json(orient="records", date_format="iso", force_ascii=False)
        )
    return json.dumps(payload, ensure_ascii=False)


def _factor_group_sections(result: EventStudyResult) -> str:
    """Render static factor-group sections."""

    cards: List[str] = []
    for factor_name, table in result.group_summaries.items():
        cards.append(f"<section><h2>{html.escape(factor_name)} 因子分组标签表现</h2>")
        cards.append(
            html_table(
                table,
                ["factor", "label", "group", "events", "mean", "median", "cvar_5", "excess_mean"],
                ["mean", "median", "cvar_5", "excess_mean"],
            )
        )
        cards.append("</section>")
    return "".join(cards)


def _decision_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the strongest return-horizon evidence for each event and control mode."""

    if summary.empty or "label" not in summary:
        return pd.DataFrame()
    candidates = summary[summary["label"].astype(str).str.startswith("ret_")].copy()
    if candidates.empty:
        return candidates
    if "horizon" not in candidates:
        candidates["horizon"] = pd.to_numeric(
            candidates["label"].astype(str).str.extract(r"_(\d+)d$")[0],
            errors="coerce",
        )
    q_values = candidates["q_value"] if "q_value" in candidates else pd.Series(np.nan, index=candidates.index)
    edges = candidates["signal_edge"] if "signal_edge" in candidates else pd.Series(np.nan, index=candidates.index)
    candidates["_q_sort"] = pd.to_numeric(q_values, errors="coerce").fillna(2.0)
    candidates["_edge_sort"] = -pd.to_numeric(edges, errors="coerce").fillna(-np.inf)
    candidates = candidates.sort_values(
        ["event", "control_mode", "_q_sort", "_edge_sort", "horizon"],
        kind="stable",
    )
    return candidates.groupby(["event", "control_mode"], sort=False, as_index=False).head(1)


def render_event_report(
    result: EventStudyResult,
    output_path: Path,
    title: str = "事件研究可视化报告",
    max_interactive_rows: Optional[int] = 100_000,
    random_state: int = 42,
) -> Path:
    """Render an event-study result to a self-contained interactive HTML report.

    ``max_interactive_rows`` only caps rows embedded in HTML for browser-side
    interaction. CSV outputs and static summary tables can still use the full
    event-study result.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = result.summary.copy()
    percent_cols = [
        "mean",
        "median",
        "control_mean",
        "excess_mean",
        "excess_median",
        "hit_rate_positive",
        "hit_rate_negative",
        "ci_low",
        "ci_high",
        "signal_edge",
        "directional_hit_rate",
        "signal_ci_low",
        "signal_ci_high",
        "positive_year_ratio",
        "recent_signal_edge",
        "prior_signal_edge",
    ]
    summary_cols = [
        "event",
        "signal_role",
        "label",
        "control_mode",
        "events",
        "event_dates",
        "signal_edge",
        "directional_hit_rate",
        "signal_ci_low",
        "signal_ci_high",
        "positive_year_ratio",
        "recent_signal_edge",
        "bootstrap_p",
        "q_value",
        "assessment",
        "action_hint",
    ]
    decision_cols = [
        "event",
        "signal_role",
        "control_mode",
        "horizon",
        "events",
        "event_dates",
        "signal_edge",
        "directional_hit_rate",
        "positive_year_ratio",
        "recent_signal_edge",
        "signal_ci_low",
        "signal_ci_high",
        "bootstrap_p",
        "q_value",
        "assessment",
        "action_hint",
    ]
    decisions = _decision_summary(summary)
    recent_cols = [
        col
        for col in result.recent_events.columns
        if col in {"date", "code", "event", "close", "ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d"}
    ]
    full_observation_count = len(result.event_observations)
    observations = result.event_observations.copy()
    sample_note = ""
    if max_interactive_rows is not None and max_interactive_rows > 0 and len(observations) > max_interactive_rows:
        key_columns = ["event", "date", "code"]
        observation_keys = observations[key_columns].drop_duplicates()
        rows_per_key = max(1.0, len(observations) / max(len(observation_keys), 1))
        max_keys = max(1, int(max_interactive_rows / rows_per_key))
        sampled_keys = observation_keys.sample(n=min(max_keys, len(observation_keys)), random_state=random_state)
        observations = observations.merge(sampled_keys, on=key_columns, how="inner")
        sort_cols = [col for col in ["date", "event", "label", "mode", "code"] if col in observations.columns]
        if sort_cols:
            observations = observations.sort_values(sort_cols)
        sample_note = (
            f"交互图表基于 {len(sampled_keys):,} 个完整事件样本、"
            f"{len(observations):,} / {full_observation_count:,} 行逐笔观测抽样；"
            "完整逐笔数据请查看 event_observations.csv。"
        )
    path_observations = result.event_path_observations.copy()
    if not observations.empty:
        observations["date"] = pd.to_datetime(observations["date"]).dt.strftime("%Y-%m-%d")
    if not path_observations.empty:
        path_observations["date"] = pd.to_datetime(path_observations["date"]).dt.strftime("%Y-%m-%d")
        if not observations.empty:
            keys = observations[["event", "date", "code"]].drop_duplicates()
            path_observations = path_observations.merge(keys, on=["event", "date", "code"], how="inner")
        if max_interactive_rows is not None and max_interactive_rows > 0 and len(path_observations) > max_interactive_rows:
            path_keys = path_observations[["event", "date", "code"]].drop_duplicates()
            rows_per_key = max(1.0, len(path_observations) / max(len(path_keys), 1))
            max_keys = max(1, int(max_interactive_rows / rows_per_key))
            path_keys = path_keys.sample(n=min(max_keys, len(path_keys)), random_state=random_state)
            path_observations = path_observations.merge(path_keys, on=["event", "date", "code"], how="inner")
            path_note = f"路径图基于 {len(path_keys):,} 个事件样本抽样重新聚合。"
            sample_note = f"{sample_note} {path_note}".strip()
    event_options = sorted(observations["event"].dropna().unique().tolist()) if not observations.empty else []
    if not event_options:
        event_options = sorted(result.strategy_backtests)
    mode_options = sorted(observations["mode"].dropna().unique().tolist()) if not observations.empty else []
    label_types = (
        sorted(
            {str(label).split("_", 1)[0] for label in observations["label"].dropna()},
            key=lambda label_type: (label_type != "ret", label_type),
        )
        if not observations.empty
        else ["ret"]
    )
    horizons = sorted(int(h) for h in result.horizons) if result.horizons else [1, 3, 5, 10, 20]
    dates = sorted(observations["date"].dropna().unique().tolist()) if not observations.empty else []
    default_horizon = 10 if 10 in horizons else horizons[0]
    horizon_options = "".join(
        f"<option value='{horizon}'{' selected' if horizon == default_horizon else ''}>{horizon} 日</option>"
        for horizon in horizons
    )
    label_type_options = "".join(
        f"<option value='{html.escape(label_type)}'>{html.escape(label_type.upper())}</option>"
        for label_type in label_types
    )
    fee_rates = sorted(
        {
            float(fee)
            for table in result.strategy_metrics.values()
            if not table.empty and "fee_rate" in table
            for fee in table["fee_rate"].dropna().tolist()
        }
    ) or [0.0, 0.002]
    fee_options = "".join(
        f"<option value='{fee}'>{fee:.2%}</option>" for fee in fee_rates
    )

    styles = """
    body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; background: #f8fafc; color: #111827; }
    header { padding: 14px 24px; background: #111827; color: white; }
    main { padding: 14px 24px 32px; }
    section { background: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 14px; margin-bottom: 14px; overflow-x: auto; }
    h1 { margin: 0; font-size: 22px; }
    h2 { font-size: 16px; margin: 0 0 10px; }
    h3 { font-size: 13px; margin: 0 0 6px; color: #374151; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border-bottom: 1px solid #e5e7eb; padding: 7px 8px; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { background: #f3f4f6; color: #374151; position: sticky; top: 0; }
    .controls { display: grid; grid-template-columns: repeat(8, minmax(110px, 1fr)); gap: 10px; align-items: end; }
    .control label { display: block; font-size: 12px; color: #4b5563; margin-bottom: 4px; }
    select, input[type=range] { width: 100%; }
    .stat-grid { display: grid; grid-template-columns: repeat(8, minmax(100px, 1fr)); gap: 8px; margin: 10px 0; }
    .stat { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px; }
    .stat span { display:block; color:#6b7280; font-size:11px; }
    .stat strong { display:block; margin-top:3px; font-size:15px; }
    .stat.verdict strong { font-size:13px; }
    .supported { color:#047857; }
    .opposite { color:#b91c1c; }
    .uncertain { color:#92400e; }
    .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; min-width: 760px; }
    .chart-card { min-width: 0; }
    .chart-card.wide { grid-column: 1 / -1; }
    svg.chart { width: 100%; height: 205px; border: 1px solid #f3f4f6; background: white; display: block; }
    svg.chart.compact { height: 185px; }
    .note { color: #6b7280; font-size: 12px; line-height: 1.5; margin: 6px 0 10px; }
    .small { font-size: 12px; color: #6b7280; line-height: 1.4; }
    @media (max-width: 1100px) {
      .controls { grid-template-columns: repeat(3, minmax(140px, 1fr)); }
    }
    """

    script = f"""
    const OBS = {_json_records(observations)};
    const SUMMARY = {_json_records(summary)};
    const PATHS = {_path_payload(result)};
    const PATH_OBS = {_json_records(path_observations)};
    const BACKTESTS = {_frame_mapping_payload(result.strategy_backtests)};
    const BACKTEST_METRICS = {_frame_mapping_payload(result.strategy_metrics)};
    const DATES = {json.dumps(dates, ensure_ascii=False)};
    const OBS_INDEX = new Map();
    const OBS_PREFIX_INDEX = new Map();
    const PATH_INDEX = new Map();
    const SUMMARY_INDEX = new Map();

    function observationKey(event, mode, label) {{
      return [event || "", mode || "", label || ""].join("|");
    }}
    function summaryKey(event, mode, label) {{
      return [event || "", mode || "", label || ""].join("|");
    }}
    function appendIndex(index, key, row) {{
      if (!index.has(key)) index.set(key, []);
      index.get(key).push(row);
    }}
    function buildInteractiveIndexes() {{
      OBS.forEach(r => {{
        appendIndex(OBS_INDEX, observationKey(r.event, r.mode, r.label), r);
        const prefix = String(r.label || "").split("_", 1)[0];
        appendIndex(OBS_PREFIX_INDEX, observationKey(r.event, r.mode, prefix), r);
      }});
      [OBS_INDEX, OBS_PREFIX_INDEX].forEach(index => {{
        index.forEach(rows => rows.sort((a, b) => String(a.date).localeCompare(String(b.date))));
      }});
      PATH_OBS.forEach(r => appendIndex(PATH_INDEX, pathKey(r), r));
      SUMMARY.forEach(r => SUMMARY_INDEX.set(summaryKey(r.event, r.control_mode, r.label), r));
    }}
    function lowerDateBound(rows, date) {{
      if (!date) return 0;
      let lo = 0, hi = rows.length;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (String(rows[mid].date) < date) lo = mid + 1;
        else hi = mid;
      }}
      return lo;
    }}
    function upperDateBound(rows, date) {{
      if (!date) return rows.length;
      let lo = 0, hi = rows.length;
      while (lo < hi) {{
        const mid = Math.floor((lo + hi) / 2);
        if (String(rows[mid].date) <= date) lo = mid + 1;
        else hi = mid;
      }}
      return lo;
    }}
    function rowsInDateWindow(rows, startDate, endDate) {{
      return rows.slice(lowerDateBound(rows, startDate), upperDateBound(rows, endDate));
    }}

    function pct(x) {{
      if (x === null || x === undefined || Number.isNaN(x)) return "";
      return (x * 100).toFixed(2) + "%";
    }}
    function decimal3(x) {{
      return Number.isFinite(x) ? Number(x).toFixed(3) : "NA";
    }}
    function mean(arr) {{
      const x = arr.filter(v => Number.isFinite(v));
      if (!x.length) return NaN;
      return x.reduce((a,b) => a+b, 0) / x.length;
    }}
    function quantile(arr, q) {{
      const x = arr.filter(v => Number.isFinite(v)).sort((a,b) => a-b);
      if (!x.length) return NaN;
      const pos = (x.length - 1) * q;
      const lo = Math.floor(pos), hi = Math.ceil(pos);
      if (lo === hi) return x[lo];
      return x[lo] * (hi - pos) + x[hi] * (pos - lo);
    }}
    function std(arr) {{
      const x = arr.filter(v => Number.isFinite(v));
      if (x.length < 2) return NaN;
      const m = mean(x);
      return Math.sqrt(x.reduce((s,v) => s + (v-m)*(v-m), 0) / (x.length - 1));
    }}
    function currentFilters() {{
      const event = document.getElementById("eventSelect").value;
      const mode = document.getElementById("modeSelect").value;
      const type = document.getElementById("labelType").value;
      const horizon = Number(document.getElementById("horizonSelect").value);
      const startIdx = Math.min(Number(document.getElementById("dateStart").value), Math.max(DATES.length - 1, 0));
      const windowLength = Number(document.getElementById("dateWindow").value);
      const endIdx = Math.min(Math.max(DATES.length - 1, 0), startIdx + windowLength);
      return {{
        event,
        mode,
        type,
        horizon,
        startDate: DATES[startIdx],
        endDate: DATES[endIdx],
        windowLength: endIdx >= startIdx ? endIdx - startIdx + 1 : 0
      }};
    }}
    function filteredRows() {{
      const f = currentFilters();
      document.getElementById("windowText").textContent = f.windowLength + " 个交易日";
      document.getElementById("dateText").textContent = (f.startDate || "") + " 至 " + (f.endDate || "");
      const label = f.type + "_" + f.horizon + "d";
      const candidates = OBS_INDEX.get(observationKey(f.event, f.mode, label)) || [];
      return rowsInDateWindow(candidates, f.startDate, f.endDate);
    }}
    function currentSummary() {{
      const f = currentFilters();
      return SUMMARY_INDEX.get(summaryKey(f.event, f.mode, f.type + "_" + f.horizon + "d")) || null;
    }}
    function filteredWindowRows(labelPrefix) {{
      const f = currentFilters();
      const candidates = OBS_PREFIX_INDEX.get(observationKey(f.event, f.mode, labelPrefix)) || [];
      return rowsInDateWindow(candidates, f.startDate, f.endDate);
    }}
    function drawAxes(svg, width, height, pad, xLabelLeft, xLabelRight, yMin, yMax, yFormatter) {{
      const fmtY = yFormatter || pct;
      svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left}}" y1="${{height-pad.bottom}}" x2="${{width-pad.right}}" y2="${{height-pad.bottom}}" stroke="#9ca3af"/>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left}}" y1="${{pad.top}}" x2="${{pad.left}}" y2="${{height-pad.bottom}}" stroke="#9ca3af"/>`);
      if (xLabelLeft !== null && xLabelLeft !== undefined) {{
        svg.insertAdjacentHTML("beforeend", `<text x="${{pad.left}}" y="${{height-7}}" font-size="10" fill="#6b7280">${{xLabelLeft}}</text>`);
      }}
      if (xLabelRight !== null && xLabelRight !== undefined) {{
        svg.insertAdjacentHTML("beforeend", `<text x="${{width-pad.right-52}}" y="${{height-7}}" font-size="10" fill="#6b7280">${{xLabelRight}}</text>`);
      }}
      svg.insertAdjacentHTML("beforeend", `<text x="4" y="${{pad.top+4}}" font-size="10" fill="#6b7280">${{fmtY(yMax)}}</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="4" y="${{height-pad.bottom}}" font-size="10" fill="#6b7280">${{fmtY(yMin)}}</text>`);
    }}
    function niceStep(span, targetTicks) {{
      if (!Number.isFinite(span) || span <= 0) return 1;
      const raw = span / Math.max(targetTicks || 7, 1);
      const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
      const normalized = raw / magnitude;
      const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
      return factor * magnitude;
    }}
    function linearTicks(lo, hi, targetTicks) {{
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [];
      if (lo === hi) return [lo];
      const step = Math.max(1, niceStep(hi - lo, targetTicks));
      const ticks = [];
      const first = Math.ceil(lo / step) * step;
      for (let value = first; value <= hi + step * 1e-9; value += step) {{
        ticks.push(Number(value.toFixed(10)));
      }}
      if (!ticks.length) return [lo, hi];
      return ticks;
    }}
    function relativeDayLabel(offset) {{
      if (Math.abs(offset) < 1e-9) return "T";
      return offset > 0 ? "T+" + offset : "T" + offset;
    }}
    function drawPathXTicks(svg, width, height, pad, xLo, xHi, xScale) {{
      linearTicks(xLo, xHi, 7).forEach(offset => {{
        const x = xScale(offset);
        const isEventDay = Math.abs(offset) < 1e-9;
        svg.insertAdjacentHTML(
          "beforeend",
          `<line x1="${{x.toFixed(1)}}" y1="${{pad.top}}" x2="${{x.toFixed(1)}}" y2="${{height-pad.bottom}}" stroke="${{isEventDay ? "#9ca3af" : "#e5e7eb"}}" stroke-dasharray="${{isEventDay ? "4 3" : "none"}}"/>`
        );
        svg.insertAdjacentHTML("beforeend", `<line x1="${{x.toFixed(1)}}" y1="${{height-pad.bottom}}" x2="${{x.toFixed(1)}}" y2="${{height-pad.bottom+4}}" stroke="#9ca3af"/>`);
        svg.insertAdjacentHTML("beforeend", `<text x="${{x.toFixed(1)}}" y="${{height-pad.bottom+15}}" text-anchor="middle" font-size="10" fill="#6b7280">${{relativeDayLabel(offset)}}</text>`);
      }});
    }}
    function drawHistogram(values, targetId, title) {{
      const svg = document.getElementById(targetId);
      const width = 520, height = 220, pad = {{left: 42, right: 12, top: 24, bottom: 28}};
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `<rect width="${{width}}" height="${{height}}" fill="white"/>`;
      const clean = values.filter(v => Number.isFinite(v));
      svg.insertAdjacentHTML("beforeend", `<text x="${{pad.left}}" y="16" font-size="12" fill="#374151">${{title}}</text>`);
      if (!clean.length) {{
        svg.insertAdjacentHTML("beforeend", `<text x="20" y="44" fill="#6b7280">无数据</text>`);
        return;
      }}
      const lo = Math.min(...clean), hi = Math.max(...clean);
      const bins = 50;
      const span = (hi - lo) || 1;
      const counts = Array(bins).fill(0);
      clean.forEach(v => {{
        const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / span * bins)));
        counts[idx] += 1;
      }});
      const maxCount = Math.max(...counts) || 1;
      counts.forEach((c, i) => {{
        const x = pad.left + i * (width - pad.left - pad.right) / bins;
        const w = Math.max(1, (width - pad.left - pad.right) / bins - 1);
        const h = c / maxCount * (height - pad.top - pad.bottom);
        const y = height - pad.bottom - h;
        const center = lo + (i + 0.5) / bins * span;
        const color = center >= 0 ? "#059669" : "#dc2626";
        svg.insertAdjacentHTML("beforeend", `<rect x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{w.toFixed(1)}}" height="${{h.toFixed(1)}}" fill="${{color}}" opacity="0.75"/>`);
      }});
      drawAxes(svg, width, height, pad, pct(lo), pct(hi), 0, maxCount, v => String(Math.round(v)));
      const zeroX = lo <= 0 && hi >= 0 ? pad.left + (0 - lo) / span * (width - pad.left - pad.right) : null;
      if (zeroX !== null) {{
        svg.insertAdjacentHTML("beforeend", `<line x1="${{zeroX.toFixed(1)}}" y1="${{pad.top}}" x2="${{zeroX.toFixed(1)}}" y2="${{height-pad.bottom}}" stroke="#111827" stroke-dasharray="4 4"/>`);
      }}
    }}
    function drawHorizonEvidence(eventName, mode) {{
      const svg = document.getElementById("cumulativeExcessChart");
      const width = 1040, height = 210, pad = {{left: 54, right: 24, top: 30, bottom: 38}};
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `<rect width="${{width}}" height="${{height}}" fill="white"/>`;
      const points = SUMMARY
        .filter(r => r.event === eventName && r.control_mode === mode && String(r.label || "").startsWith("ret_") && Number.isFinite(r.signal_edge))
        .map(r => ({{horizon:Number(r.horizon), value:r.signal_edge, ciLow:r.signal_ci_low, ciHigh:r.signal_ci_high, assessment:r.assessment || "证据不足", role:r.signal_role || "信号"}}))
        .sort((a,b) => a.horizon - b.horizon);
      if (!points.length) {{
        svg.insertAdjacentHTML("beforeend", `<text x="20" y="44" fill="#6b7280">无持有期证据数据</text>`);
        return;
      }}
      svg.insertAdjacentHTML("beforeend", `<text x="${{pad.left}}" y="18" font-size="12" fill="#374151">${{points[0].role}}方向优势与 95% 聚类 Bootstrap 区间（全样本）</text>`);
      const xLo = Math.min(...points.map(p => p.horizon));
      const xHi = Math.max(...points.map(p => p.horizon));
      const yValues = [0];
      points.forEach(p => [p.value, p.ciLow, p.ciHigh].forEach(v => {{ if (Number.isFinite(v)) yValues.push(v); }}));
      const yLoRaw = Math.min(...yValues), yHiRaw = Math.max(...yValues);
      const yPad = Math.max((yHiRaw - yLoRaw) * 0.08, 0.001);
      const yLo = yLoRaw - yPad, yHi = yHiRaw + yPad;
      const xSpan = (xHi - xLo) || 1, ySpan = (yHi - yLo) || 1;
      function xy(point) {{
        const x = xHi === xLo ? width / 2 : pad.left + (point.horizon - xLo) / xSpan * (width - pad.left - pad.right);
        const y = height - pad.bottom - (point.value - yLo) / ySpan * (height - pad.top - pad.bottom);
        return [x, y];
      }}
      drawAxes(svg, width, height, pad, null, null, yLo, yHi);
      const zeroY = height - pad.bottom - (0 - yLo) / ySpan * (height - pad.top - pad.bottom);
      svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left}}" y1="${{zeroY.toFixed(1)}}" x2="${{width-pad.right}}" y2="${{zeroY.toFixed(1)}}" stroke="#9ca3af" stroke-dasharray="4 4"/>`);
      const pts = points.map(p => xy(p).map(v => v.toFixed(1)).join(",")).join(" ");
      svg.insertAdjacentHTML("beforeend", `<polyline fill="none" stroke="#6d28d9" stroke-width="2.2" points="${{pts}}"/>`);
      points.forEach(p => {{
        const [x, y] = xy(p);
        const color = String(p.assessment).startsWith("支持") ? "#059669" : p.assessment === "方向相反" ? "#dc2626" : "#7c3aed";
        if (Number.isFinite(p.ciLow) && Number.isFinite(p.ciHigh)) {{
          const yLow = height - pad.bottom - (p.ciLow - yLo) / ySpan * (height - pad.top - pad.bottom);
          const yHigh = height - pad.bottom - (p.ciHigh - yLo) / ySpan * (height - pad.top - pad.bottom);
          svg.insertAdjacentHTML("beforeend", `<line x1="${{x.toFixed(1)}}" y1="${{yHigh.toFixed(1)}}" x2="${{x.toFixed(1)}}" y2="${{yLow.toFixed(1)}}" stroke="${{color}}" stroke-width="1.6"/>`);
          svg.insertAdjacentHTML("beforeend", `<line x1="${{(x-4).toFixed(1)}}" y1="${{yHigh.toFixed(1)}}" x2="${{(x+4).toFixed(1)}}" y2="${{yHigh.toFixed(1)}}" stroke="${{color}}"/>`);
          svg.insertAdjacentHTML("beforeend", `<line x1="${{(x-4).toFixed(1)}}" y1="${{yLow.toFixed(1)}}" x2="${{(x+4).toFixed(1)}}" y2="${{yLow.toFixed(1)}}" stroke="${{color}}"/>`);
        }}
        svg.insertAdjacentHTML("beforeend", `<circle cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="4" fill="${{color}}"/>`);
        svg.insertAdjacentHTML("beforeend", `<text x="${{x.toFixed(1)}}" y="${{height-pad.bottom+15}}" text-anchor="middle" font-size="10" fill="#6b7280">${{p.horizon}}d</text>`);
        svg.insertAdjacentHTML("beforeend", `<text x="${{(x+5).toFixed(1)}}" y="${{(y-6).toFixed(1)}}" font-size="10" fill="#4b5563">${{pct(p.value)}}</text>`);
      }});
      svg.insertAdjacentHTML("beforeend", `<text x="${{width/2-24}}" y="${{height-6}}" font-size="10" fill="#6b7280">持有期</text>`);
    }}
    function pathKey(row) {{
      return row.event + "|" + row.date + "|" + row.code;
    }}
    function aggregatePath(rows, eventName, mode) {{
      if (PATH_OBS.length) {{
        const keys = new Set(rows.map(pathKey));
        const grouped = new Map();
        keys.forEach(key => {{
          const candidates = PATH_INDEX.get(key) || [];
          candidates.forEach(r => {{
            const rowMode = r.mode === undefined || r.mode === null ? r.control_mode : r.mode;
            if (rowMode !== undefined && rowMode !== null && String(rowMode) !== String(mode)) return;
            const offset = Number(r.offset);
            if (!grouped.has(offset)) grouped.set(offset, {{eventValues: [], controlValues: []}});
            const bucket = grouped.get(offset);
            const eventValue = Number.isFinite(r.value) ? r.value : r.event_value;
            if (Number.isFinite(eventValue)) bucket.eventValues.push(eventValue);
            if (Number.isFinite(r.control_mean)) bucket.controlValues.push(r.control_mean);
          }});
        }});
        return Array.from(grouped.entries()).sort((a,b) => Number(a[0]) - Number(b[0])).map(([offset, bucket]) => ({{
          offset: Number(offset),
          mean: mean(bucket.eventValues),
          median: quantile(bucket.eventValues, 0.5),
          p25: quantile(bucket.eventValues, 0.25),
          p75: quantile(bucket.eventValues, 0.75),
          control_mean: mean(bucket.controlValues)
        }}));
      }}
      return PATHS[eventName] || [];
    }}
    function drawPath(filtered, eventName, mode) {{
      const svg = document.getElementById("pathChart");
      const rows = aggregatePath(filtered, eventName, mode);
      const width = 520, height = 220, pad = {{left: 48, right: 18, top: 46, bottom: 42}};
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `<rect width="${{width}}" height="${{height}}" fill="white"/>`;
      svg.insertAdjacentHTML("beforeend", `<text x="${{pad.left}}" y="16" font-size="12" fill="#374151">事件前后收益路径</text>`);
      if (!rows.length) {{
        svg.insertAdjacentHTML("beforeend", `<text x="20" y="44" fill="#6b7280">无路径数据</text>`);
        return;
      }}
      const offsets = rows.map(r => Number(r.offset));
      const values = [];
      rows.forEach(r => ["mean", "p25", "p75", "control_mean"].forEach(c => {{
        if (Number.isFinite(r[c])) values.push(r[c]);
      }}));
      if (!values.length) {{
        svg.insertAdjacentHTML("beforeend", `<text x="20" y="44" fill="#6b7280">无有效路径数据</text>`);
        return;
      }}
      const rawLo = Math.min(...values), rawHi = Math.max(...values);
      const yPad = Math.max((rawHi - rawLo) * 0.08, 0.001);
      const lo = rawLo - yPad, hi = rawHi + yPad;
      const ySpan = hi - lo;
      const xLo = Math.min(...offsets), xHi = Math.max(...offsets);
      const xSpan = (xHi - xLo) || 1;
      function xScale(offset) {{
        if (xHi === xLo) return (pad.left + width - pad.right) / 2;
        return pad.left + (offset - xLo) / xSpan * (width - pad.left - pad.right);
      }}
      function xy(offset, value) {{
        const x = xScale(offset);
        const y = height - pad.bottom - (value - lo) / ySpan * (height - pad.top - pad.bottom);
        return [x, y];
      }}
      drawPathXTicks(svg, width, height, pad, xLo, xHi, xScale);
      drawAxes(svg, width, height, pad, null, null, lo, hi);
      const zeroY = lo <= 0 && hi >= 0 ? xy(xLo, 0)[1] : null;
      if (zeroY !== null) svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left}}" y1="${{zeroY.toFixed(1)}}" x2="${{width-pad.right}}" y2="${{zeroY.toFixed(1)}}" stroke="#9ca3af" stroke-dasharray="4 4"/>`);

      const bandRows = rows.filter(r => Number.isFinite(r.p25) && Number.isFinite(r.p75));
      if (bandRows.length > 1) {{
        const upper = bandRows.map(r => xy(Number(r.offset), r.p75).map(v => v.toFixed(1)).join(","));
        const lower = bandRows.slice().reverse().map(r => xy(Number(r.offset), r.p25).map(v => v.toFixed(1)).join(","));
        svg.insertAdjacentHTML("beforeend", `<polygon points="${{upper.concat(lower).join(" ")}}" fill="#93c5fd" opacity="0.28"/>`);
      }}
      const eventPoints = rows.filter(r => Number.isFinite(r.mean)).map(r => xy(Number(r.offset), r.mean).map(v => v.toFixed(1)).join(",")).join(" ");
      if (eventPoints) svg.insertAdjacentHTML("beforeend", `<polyline fill="none" stroke="#2563eb" stroke-width="2.4" stroke-linejoin="round" points="${{eventPoints}}"/>`);
      const controlPoints = rows.filter(r => Number.isFinite(r.control_mean)).map(r => xy(Number(r.offset), r.control_mean).map(v => v.toFixed(1)).join(",")).join(" ");
      if (controlPoints) svg.insertAdjacentHTML("beforeend", `<polyline fill="none" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="7 4" stroke-linejoin="round" points="${{controlPoints}}"/>`);

      svg.insertAdjacentHTML("beforeend", `<rect x="${{pad.left}}" y="25" width="18" height="8" fill="#93c5fd" opacity="0.4"/><text x="${{pad.left+23}}" y="33" font-size="10" fill="#4b5563">事件 P25-P75</text>`);
      svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left+128}}" y1="29" x2="${{pad.left+149}}" y2="29" stroke="#2563eb" stroke-width="2.4"/><text x="${{pad.left+154}}" y="33" font-size="10" fill="#4b5563">事件均值</text>`);
      if (controlPoints) svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left+225}}" y1="29" x2="${{pad.left+246}}" y2="29" stroke="#f59e0b" stroke-width="2.2" stroke-dasharray="7 4"/><text x="${{pad.left+251}}" y="33" font-size="10" fill="#4b5563">同日未触发对照均值</text>`);
      svg.insertAdjacentHTML("beforeend", `<text x="${{width/2-26}}" y="${{height-5}}" font-size="10" fill="#6b7280">相对事件日</text>`);
    }}
    function drawStrategyBacktest(eventName, horizon, feeRate) {{
      const svg = document.getElementById("strategyBacktestChart");
      const width = 1040, height = 285, pad = {{left: 62, right: 25, top: 45, bottom: 48}};
      svg.setAttribute("viewBox", `0 0 ${{width}} ${{height}}`);
      svg.innerHTML = `<rect width="${{width}}" height="${{height}}" fill="white"/>`;
      const rows = (BACKTESTS[eventName] || []).filter(r => Number(r.horizon) === Number(horizon) && Math.abs(Number(r.fee_rate) - Number(feeRate)) < 1e-10);
      const metric = (BACKTEST_METRICS[eventName] || []).find(r => Number(r.horizon) === Number(horizon) && Math.abs(Number(r.fee_rate) - Number(feeRate)) < 1e-10);
      const metricValue = (key, formatter=decimal3) => metric && Number.isFinite(metric[key]) ? formatter(metric[key]) : "-";
      document.getElementById("btAnnualized").textContent = metricValue("annualized_return", pct);
      document.getElementById("btSharpe").textContent = metricValue("sharpe_ratio");
      document.getElementById("btDrawdown").textContent = metricValue("max_drawdown", pct);
      document.getElementById("btCalmar").textContent = metricValue("calmar_ratio");
      document.getElementById("btExcess").textContent = metricValue("excess_annualized_return", pct);
      document.getElementById("btBenchmark").textContent = metricValue("benchmark_annualized_return", pct);
      document.getElementById("btHoldings").textContent = metricValue("average_holdings", v => Number(v).toFixed(1));
      document.getElementById("btTrades").textContent = metric ? `${{metric.entries || 0}} / ${{metric.exits || 0}}` : "-";
      if (!rows.length) {{
        svg.insertAdjacentHTML("beforeend", `<text x="20" y="44" fill="#6b7280">无对应策略回测数据</text>`);
        return;
      }}
      const columns = ["strategy_cumulative", "benchmark_cumulative", "excess_cumulative"];
      const values = [0];
      rows.forEach(r => columns.forEach(column => {{ if (Number.isFinite(r[column])) values.push(r[column]); }}));
      const rawLo = Math.min(...values), rawHi = Math.max(...values);
      const yPadding = Math.max((rawHi - rawLo) * 0.08, 0.01);
      const yLo = rawLo - yPadding, yHi = rawHi + yPadding, ySpan = yHi - yLo;
      const xSpan = Math.max(rows.length - 1, 1);
      const xScale = index => pad.left + index / xSpan * (width - pad.left - pad.right);
      const yScale = value => height - pad.bottom - (value - yLo) / ySpan * (height - pad.top - pad.bottom);
      drawAxes(svg, width, height, pad, null, null, yLo, yHi);
      const zeroY = yScale(0);
      svg.insertAdjacentHTML("beforeend", `<line x1="${{pad.left}}" y1="${{zeroY.toFixed(1)}}" x2="${{width-pad.right}}" y2="${{zeroY.toFixed(1)}}" stroke="#9ca3af" stroke-dasharray="4 4"/>`);
      const tickCount = Math.min(9, rows.length);
      const tickIndexes = Array.from(new Set(Array.from({{length:tickCount}}, (_, i) => Math.round(i * (rows.length - 1) / Math.max(tickCount - 1, 1)))));
      tickIndexes.forEach(index => {{
        const x = xScale(index), date = String(rows[index].date || "").slice(0, 10);
        svg.insertAdjacentHTML("beforeend", `<line x1="${{x.toFixed(1)}}" y1="${{height-pad.bottom}}" x2="${{x.toFixed(1)}}" y2="${{height-pad.bottom+4}}" stroke="#6b7280"/><text x="${{x.toFixed(1)}}" y="${{height-pad.bottom+17}}" text-anchor="end" transform="rotate(-32 ${{x.toFixed(1)}} ${{height-pad.bottom+17}})" font-size="9" fill="#6b7280">${{date}}</text>`);
      }});
      const series = [
        ["strategy_cumulative", "#2563eb", "策略累计收益"],
        ["benchmark_cumulative", "#f59e0b", "市场平均累计收益"],
        ["excess_cumulative", "#059669", "累计超额收益"]
      ];
      series.forEach(([column, color, label], seriesIndex) => {{
        const points = rows.map((row, index) => Number.isFinite(row[column]) ? `${{xScale(index).toFixed(1)}},${{yScale(row[column]).toFixed(1)}}` : "").filter(Boolean).join(" ");
        if (points) svg.insertAdjacentHTML("beforeend", `<polyline fill="none" stroke="${{color}}" stroke-width="2.2" stroke-linejoin="round" points="${{points}}"/>`);
        const legendX = pad.left + seriesIndex * 190;
        svg.insertAdjacentHTML("beforeend", `<line x1="${{legendX}}" y1="25" x2="${{legendX+24}}" y2="25" stroke="${{color}}" stroke-width="3"/><text x="${{legendX+30}}" y="29" font-size="11" fill="#374151">${{label}}</text>`);
      }});
      svg.insertAdjacentHTML("beforeend", `<text x="${{width-245}}" y="29" font-size="10" fill="#6b7280">次日复权收盘价成交；单边费率 ${{pct(Number(feeRate))}}</text>`);
    }}
    function updateInteractive() {{
      const rows = filteredRows();
      const evidence = currentSummary();
      const eventValues = rows.map(r => r.event_value);
      const excessValues = rows.map(r => r.excess_mean);
      const controlValues = rows.map(r => r.control_mean);
      const directionSign = evidence && evidence.direction === "exit" ? -1 : 1;
      const directionalValues = excessValues.filter(Number.isFinite).map(value => value * directionSign);
      const stats = {{
        count: rows.length,
        mean: mean(eventValues),
        excess: mean(excessValues),
        signalEdge: mean(directionalValues),
        directionalHit: directionalValues.filter(value => value > 0).length / Math.max(directionalValues.length, 1)
      }};
      document.getElementById("statCount").textContent = stats.count;
      document.getElementById("statMean").textContent = pct(stats.mean);
      document.getElementById("statExcess").textContent = pct(stats.excess);
      document.getElementById("statSignalEdge").textContent = pct(stats.signalEdge);
      document.getElementById("statDirectionalHit").textContent = pct(stats.directionalHit);
      document.getElementById("statInference").textContent = evidence ? `p=${{decimal3(evidence.bootstrap_p)}} / q=${{decimal3(evidence.q_value)}}` : "";
      const verdict = evidence ? (evidence.assessment || "证据不足") : "";
      const verdictNode = document.getElementById("statVerdict");
      verdictNode.textContent = verdict;
      verdictNode.className = String(verdict).startsWith("支持") ? "supported" : verdict === "方向相反" ? "opposite" : "uncertain";
      document.getElementById("statAction").textContent = evidence ? (evidence.action_hint || "") : "";
      drawHistogram(eventValues, "distChart", "事件后收益分布");
      drawHistogram(excessValues, "excessChart", "相对对照组超额分布");
      drawHistogram(controlValues, "controlChart", "控制组收益均值分布");
      drawPath(
        rows,
        document.getElementById("eventSelect").value,
        document.getElementById("modeSelect").value
      );
      drawHorizonEvidence(
        document.getElementById("eventSelect").value,
        document.getElementById("modeSelect").value
      );
      drawStrategyBacktest(
        document.getElementById("eventSelect").value,
        document.getElementById("horizonSelect").value,
        document.getElementById("feeSelect").value
      );
    }}
    let pendingUpdateFrame = null;
    function scheduleInteractiveUpdate() {{
      if (pendingUpdateFrame !== null) return;
      pendingUpdateFrame = window.requestAnimationFrame(() => {{
        pendingUpdateFrame = null;
        updateInteractive();
      }});
    }}
    window.addEventListener("DOMContentLoaded", () => {{
      buildInteractiveIndexes();
      const ds = document.getElementById("dateStart");
      const dw = document.getElementById("dateWindow");
      ds.max = Math.max(DATES.length - 1, 0);
      ds.value = 0;
      dw.max = Math.max(DATES.length - 1, 0);
      dw.value = Math.max(DATES.length - 1, 0);
      ["eventSelect", "modeSelect", "labelType", "horizonSelect", "feeSelect"].forEach(id => {{
        document.getElementById(id).addEventListener("change", updateInteractive);
      }});
      ["dateStart", "dateWindow"].forEach(id => {{
        document.getElementById(id).addEventListener("input", scheduleInteractiveUpdate);
      }});
      updateInteractive();
    }});
    """

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>{styles}</style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>报告把统计显著性与交易方向分开：方向优势为正表示该事件有利于配置的开仓/平仓用途。</p>
  </header>
  <main>
    <section>
      <h2>信号决策摘要</h2>
      <p class="note">每个事件和控制组展示 FDR q 值最小的收益持有期。只有方向优势的聚类置信区间高于 0，且经过全部检验的 FDR 校正后仍显著，才标记为“支持开仓/平仓”；负值显著则标记为“方向相反”。positive_year_ratio 表示年度方向优势为正的年份占比，recent_signal_edge 是最近三分之一事件日期的方向优势，用来观察信号是否衰减。</p>
      {html_table(decisions, decision_cols, percent_cols) if not decisions.empty else "<p class='note'>无足够收益数据</p>"}
    </section>
    <section>
      <h2>交互式事件分布诊断</h2>
      {f"<p class='note'>{html.escape(sample_note)}</p>" if sample_note else ""}
      <div class="controls">
        <div class="control"><label>事件</label><select id="eventSelect">{"".join(f"<option value='{html.escape(e)}'>{html.escape(e)}</option>" for e in event_options)}</select></div>
        <div class="control"><label>控制组</label><select id="modeSelect">{"".join(f"<option value='{html.escape(m)}'>{html.escape(m)}</option>" for m in mode_options)}</select></div>
        <div class="control"><label>标签类型</label><select id="labelType">{label_type_options}</select></div>
        <div class="control"><label for="horizonSelect">持有期</label><select id="horizonSelect">{horizon_options}</select></div>
        <div class="control"><label for="feeSelect">单边手续费</label><select id="feeSelect">{fee_options}</select></div>
        <div class="control"><label>开始日期</label><input id="dateStart" type="range" min="0" step="1"></div>
        <div class="control"><label>观测窗口长度：<span id="windowText"></span></label><input id="dateWindow" type="range" min="0" step="1"></div>
        <div class="control"><label>观测区间</label><div id="dateText" class="small"></div></div>
      </div>
      <div class="stat-grid">
        <div class="stat"><span>样本数</span><strong id="statCount"></strong></div>
        <div class="stat"><span>事件后收益</span><strong id="statMean"></strong></div>
        <div class="stat"><span>相对控制超额</span><strong id="statExcess"></strong></div>
        <div class="stat"><span>开/平仓方向优势</span><strong id="statSignalEdge"></strong></div>
        <div class="stat"><span>方向命中率</span><strong id="statDirectionalHit"></strong></div>
        <div class="stat"><span>聚类 p / FDR q</span><strong id="statInference"></strong></div>
        <div class="stat verdict"><span>证据结论</span><strong id="statVerdict"></strong><div id="statAction" class="small"></div></div>
      </div>
      <div class="chart-grid">
        <div class="chart-card"><svg id="distChart" class="chart"></svg></div>
        <div class="chart-card"><svg id="excessChart" class="chart"></svg></div>
        <div class="chart-card"><svg id="controlChart" class="chart"></svg></div>
        <div class="chart-card"><svg id="pathChart" class="chart"></svg></div>
        <div class="chart-card wide"><svg id="cumulativeExcessChart" class="chart"></svg></div>
      </div>
      <h3 style="margin-top:14px">事件驱动策略回测</h3>
      <p class="note">信号发生后的下一交易日以复权收盘价建仓；按所选持有期持有，重复信号延长持有期。组合日收益为持仓股票收益的等权平均，股票不在交易池时目标仓位归零。手续费按目标权重单边换手收取。</p>
      <div class="stat-grid">
        <div class="stat"><span>年化收益</span><strong id="btAnnualized"></strong></div>
        <div class="stat"><span>夏普比率</span><strong id="btSharpe"></strong></div>
        <div class="stat"><span>最大回撤</span><strong id="btDrawdown"></strong></div>
        <div class="stat"><span>卡玛比率</span><strong id="btCalmar"></strong></div>
        <div class="stat"><span>年化超额</span><strong id="btExcess"></strong></div>
        <div class="stat"><span>基准年化</span><strong id="btBenchmark"></strong></div>
        <div class="stat"><span>平均持仓数</span><strong id="btHoldings"></strong></div>
        <div class="stat"><span>开仓 / 平仓笔数</span><strong id="btTrades"></strong></div>
      </div>
      <div class="chart-grid">
        <div class="chart-card wide"><svg id="strategyBacktestChart" class="chart" style="height:285px"></svg></div>
      </div>
    </section>
    <section>
      <h2>事件效应总表</h2>
      {html_table(summary, summary_cols, percent_cols) if not summary.empty else "<p class='note'>无数据</p>"}
    </section>
    {_factor_group_sections(result)}
    <section>
      <h2>最近事件明细</h2>
      {html_table(result.recent_events.head(120), recent_cols, [c for c in recent_cols if c.startswith("ret_")]) if not result.recent_events.empty else "<p class='note'>无事件</p>"}
    </section>
    <section>
      <h2>方法说明</h2>
      <p class="note">same_date 控制时间漂移；same_date_style 进一步匹配同日市值风格。显著性使用事件日期聚类 Bootstrap：同一天的股票一起重采样，p 值来自零效应中心化分布，置信区间来自原始聚类分布，并对报告内全部检验做 Benjamini-Hochberg FDR 校正。默认只计算未来收益 ret；MFE/MAE 仅在明确启用时计算。统计证据用于筛选候选规则，不包含交易成本、容量、滑点或组合回测。</p>
    </section>
  </main>
  <script>{script}</script>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path

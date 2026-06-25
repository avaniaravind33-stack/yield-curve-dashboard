// src/App.jsx
// Yield Curve Dashboard — root component
// Four tabs: Yield Surface | NS Calibration | Kalman Factors | Recession Probability

import { useState, useEffect, useRef } from "react";
import Plot from "react-plotly.js";
import { useData } from "./hooks/useData.js";

// ─── Shared layout constants ──────────────────────────────────────────────────
const DARK = {
  paper_bgcolor: "#0d1117",
  plot_bgcolor:  "#0d1117",
  font:          { color: "#c9d1d9", family: "IBM Plex Sans, sans-serif", size: 12 },
  gridcolor:     "#21262d",
  linecolor:     "#30363d",
  zerolinecolor: "#30363d",
};

const TABS = [
  { id: "surface",    label: "3D Yield Surface" },
  { id: "ns",         label: "NS Calibration"   },
  { id: "kalman",     label: "Kalman Factors"   },
  { id: "recession",  label: "Recession Probability" },
];

// ─── Utility ──────────────────────────────────────────────────────────────────
function LoadingCard({ message }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
      height: 400, color: "#8b949e", fontSize: 14, gap: 10 }}>
      <span style={{ animation: "spin 1s linear infinite", display: "inline-block" }}>⟳</span>
      {message || "Loading data…"}
    </div>
  );
}

function ErrorCard({ message }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center",
      height: 400, color: "#f85149", fontSize: 14, gap: 8 }}>
      ⚠ {message}
    </div>
  );
}

function StatPill({ label, value, color }) {
  return (
    <div style={{
      background: "#161b22", border: "1px solid #30363d",
      borderRadius: 8, padding: "10px 16px", minWidth: 140,
    }}>
      <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600, color: color || "#58a6ff",
        fontFamily: "IBM Plex Mono, monospace" }}>{value}</div>
    </div>
  );
}

// ─── TAB 1: 3D Yield Surface ──────────────────────────────────────────────────
function YieldSurfaceTab() {
  const { data, loading, error } = useData("yield_surface.json");

  if (loading) return <LoadingCard message="Building 3D yield surface…" />;
  if (error)   return <ErrorCard message={error} />;
  if (!data)   return null;

  const { dates, maturities, z, meta } = data;

  // Sample every Nth date to keep the 3D surface responsive
  const step = Math.max(1, Math.floor(dates.length / 80));
  const sampledDates = dates.filter((_, i) => i % step === 0);
  const sampledZ     = z.filter((_, i) => i % step === 0);

  const trace = {
    type: "surface",
    x: maturities,
    y: sampledDates,
    z: sampledZ,
    colorscale: [
      [0,   "#0d1117"],
      [0.2, "#0c3460"],
      [0.4, "#185fa5"],
      [0.6, "#58a6ff"],
      [0.8, "#d29922"],
      [1,   "#f85149"],
    ],
    colorbar: {
      title: { text: "Yield (%)", font: { color: "#8b949e", size: 11 } },
      tickfont: { color: "#8b949e", size: 10 },
      thickness: 12, len: 0.6,
    },
    contours: {
      z: { show: true, usecolormap: true, highlightcolor: "#ffffff", project: { z: false } },
    },
    lighting:      { diffuse: 0.9, specular: 0.05, roughness: 0.8 },
    lightposition: { x: 200, y: 200, z: 1000 },
  };

  const layout = {
    ...DARK,
    title: {
      text: "U.S. Treasury Yield Surface — 1993 to Present",
      font: { color: "#ffffff", size: 15 }, x: 0.02,
    },
    scene: {
      xaxis: { title: "Maturity", tickfont: { color: "#8b949e", size: 10 },
        gridcolor: "#21262d", zerolinecolor: "#21262d" },
      yaxis: { title: "Date", tickfont: { color: "#8b949e", size: 9 },
        gridcolor: "#21262d", zerolinecolor: "#21262d", nticks: 8 },
      zaxis: { title: "Yield (%)", tickfont: { color: "#8b949e", size: 10 },
        gridcolor: "#21262d", zerolinecolor: "#21262d" },
      camera: { eye: { x: 1.6, y: -1.6, z: 0.8 } },
      bgcolor: "#0d1117",
    },
    margin: { l: 0, r: 0, t: 50, b: 0 },
    height: 520,
  };

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <StatPill label="Maturities" value={maturities.length} />
        <StatPill label="Date range" value={`${meta.date_range[0].slice(0,4)}–${meta.date_range[1].slice(0,4)}`} color="#3fb950" />
        <StatPill label="Observations" value={meta.n_dates.toLocaleString()} color="#d29922" />
      </div>
      <Plot data={[trace]} layout={layout} config={{ responsive: true, displayModeBar: true }}
        style={{ width: "100%" }} />
      <p style={{ color: "#8b949e", fontSize: 12, marginTop: 12, lineHeight: 1.6 }}>
        Drag to rotate · scroll to zoom · double-click to reset. Each row is a
        single date's yield curve across all maturities. Warm colors = higher yields.
      </p>
    </div>
  );
}

// ─── TAB 2: Nelson-Siegel Calibration ────────────────────────────────────────
function NSCalibrationTab() {
  const { data, loading, error } = useData("ns_calibration.json");
  const [selectedDate, setSelectedDate] = useState(null);
  const [view, setView] = useState("params"); // "params" | "fit"

  if (loading) return <LoadingCard message="Loading Nelson-Siegel calibration…" />;
  if (error)   return <ErrorCard message={error} />;
  if (!data)   return null;

  const { params_history, cross_sections, maturities, meta } = data;
  const dates      = params_history.map(r => r.date);
  const beta1      = params_history.map(r => r.beta1);
  const beta2      = params_history.map(r => r.beta2);
  const beta3      = params_history.map(r => r.beta3);
  const rmse       = params_history.map(r => r.rmse);

  // Cross-section chart for a selected date
  const csDate  = selectedDate || dates[dates.length - 1];
  const cs      = cross_sections[csDate] || {};
  const obsVals = maturities.map(m => cs.observed?.[m] ?? null);
  const fitVals = maturities.map(m => cs.fitted_ns?.[m] ?? null);

  const paramsLayout = {
    ...DARK,
    title: { text: "Nelson-Siegel Factor History (β1 Level · β2 Slope · β3 Curvature)",
      font: { color: "#ffffff", size: 13 }, x: 0.02 },
    xaxis: { gridcolor: "#21262d", linecolor: "#30363d" },
    yaxis: { title: "Value (%)", gridcolor: "#21262d", linecolor: "#30363d" },
    legend: { x: 0.01, y: 0.99, bgcolor: "#161b22", bordercolor: "#30363d", borderwidth: 1 },
    margin: { l: 55, r: 20, t: 50, b: 40 },
    height: 380,
  };

  const paramsTraces = [
    { x: dates, y: beta1, name: "β1 Level",     line: { color: "#58a6ff", width: 1.5 } },
    { x: dates, y: beta2, name: "β2 Slope",     line: { color: "#3fb950", width: 1.5 } },
    { x: dates, y: beta3, name: "β3 Curvature", line: { color: "#d29922", width: 1.5 } },
  ];

  const rmseLayout = {
    ...DARK,
    title: { text: "Calibration RMSE over time", font: { color: "#ffffff", size: 13 }, x: 0.02 },
    xaxis: { gridcolor: "#21262d", linecolor: "#30363d" },
    yaxis: { title: "RMSE (%)", gridcolor: "#21262d", linecolor: "#30363d" },
    margin: { l: 55, r: 20, t: 50, b: 40 },
    height: 200,
    showlegend: false,
  };

  const fitLayout = {
    ...DARK,
    title: { text: `NS model fit — ${csDate}`, font: { color: "#ffffff", size: 13 }, x: 0.02 },
    xaxis: { title: "Maturity", gridcolor: "#21262d", linecolor: "#30363d" },
    yaxis: { title: "Yield (%)", gridcolor: "#21262d", linecolor: "#30363d" },
    legend: { x: 0.65, y: 0.05, bgcolor: "#161b22", bordercolor: "#30363d", borderwidth: 1 },
    margin: { l: 55, r: 20, t: 50, b: 50 },
    height: 380,
  };

  const fitTraces = [
    { x: maturities, y: obsVals, name: "Observed",
      mode: "markers", marker: { color: "#58a6ff", size: 8, symbol: "circle" } },
    { x: maturities, y: fitVals, name: "NS Fitted",
      mode: "lines", line: { color: "#f85149", width: 2, dash: "dot" } },
  ];

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <StatPill label="Dates calibrated" value={meta.n_dates.toLocaleString()} />
        <StatPill label="Avg RMSE" value={`${(rmse.reduce((a,b)=>a+b,0)/rmse.length).toFixed(3)}%`} color="#d29922" />
        <StatPill label="NSS fitted" value={meta.has_nss ? "Yes" : "No"} color="#3fb950" />
      </div>

      {/* Toggle */}
      <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
        {["params","fit"].map(v => (
          <button key={v} onClick={() => setView(v)} style={{
            padding: "6px 16px", borderRadius: 6, fontSize: 13, cursor: "pointer",
            background: view === v ? "#1f6feb" : "#161b22",
            color: view === v ? "#ffffff" : "#8b949e",
            border: `1px solid ${view === v ? "#1f6feb" : "#30363d"}`,
          }}>
            {v === "params" ? "Factor history" : "Cross-section fit"}
          </button>
        ))}
      </div>

      {view === "params" ? (
        <>
          <Plot data={paramsTraces} layout={paramsLayout}
            config={{ responsive: true }} style={{ width: "100%" }} />
          <Plot data={[{ x: dates, y: rmse, type: "scatter", fill: "tozeroy",
            line: { color: "#bc8cff", width: 1 }, fillcolor: "#bc8cff22" }]}
            layout={rmseLayout} config={{ responsive: true }} style={{ width: "100%" }} />
        </>
      ) : (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
            <label style={{ color: "#8b949e", fontSize: 13 }}>Select date:</label>
            <select value={csDate} onChange={e => setSelectedDate(e.target.value)}
              style={{ background: "#161b22", color: "#c9d1d9", border: "1px solid #30363d",
                borderRadius: 6, padding: "5px 10px", fontSize: 13 }}>
              {dates.filter((_, i) => i % Math.max(1, Math.floor(dates.length/60)) === 0)
                .map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>
          <Plot data={fitTraces} layout={fitLayout}
            config={{ responsive: true }} style={{ width: "100%" }} />
          <p style={{ color: "#8b949e", fontSize: 12, marginTop: 8 }}>
            Blue dots = actual Treasury yields · Red dashed = Nelson-Siegel model fit.
            A tight fit means β1/β2/β3 fully capture the curve shape on this date.
          </p>
        </>
      )}
    </div>
  );
}

// ─── TAB 3: Kalman Filter Factors ────────────────────────────────────────────
function KalmanFactorsTab() {
  const { data, loading, error } = useData("kalman_factors.json");
  const [factor, setFactor] = useState("beta1");

  if (loading) return <LoadingCard message="Loading Kalman filter results…" />;
  if (error)   return <ErrorCard message={error} />;
  if (!data)   return null;

  const { ols_factors, kalman_factors, factor_labels, meta } = data;

  const olsSeries    = ols_factors[factor];
  const kalmanSeries = kalman_factors[factor];

  const factorDesc = {
    beta1: "Level — the long-run yield to which the entire curve converges. A high β1 means rates are structurally elevated.",
    beta2: "Slope — the difference between long and short rates. Negative = inverted curve (recession signal). Positive = normal / steepening.",
    beta3: "Curvature — the hump in the middle of the curve. Captures how much medium maturities deviate from a flat interpolation.",
  };

  const layout = {
    ...DARK,
    title: { text: `${factor_labels[factor]} — OLS cross-section vs Kalman-smoothed`,
      font: { color: "#ffffff", size: 13 }, x: 0.02 },
    xaxis: { gridcolor: "#21262d", linecolor: "#30363d" },
    yaxis: { title: "Factor value (%)", gridcolor: "#21262d", linecolor: "#30363d" },
    legend: { x: 0.01, y: 0.99, bgcolor: "#161b22", bordercolor: "#30363d", borderwidth: 1 },
    margin: { l: 60, r: 20, t: 50, b: 40 },
    height: 420,
  };

  const traces = [
    { x: olsSeries.dates, y: olsSeries.values, name: "OLS (cross-section)",
      line: { color: "#8b949e", width: 1 }, opacity: 0.7 },
    { x: kalmanSeries.dates, y: kalmanSeries.values, name: "Kalman-smoothed",
      line: { color: "#58a6ff", width: 2 } },
  ];

  // Spread of beta2 (slope) trace for inversion highlighting
  const spreadTrace = factor === "beta2" ? [{
    x: olsSeries.dates,
    y: olsSeries.values.map(v => Math.min(v, 0)),
    fill: "tozeroy", fillcolor: "#f8514930", line: { width: 0 },
    name: "Inverted (< 0)", showlegend: true,
  }] : [];

  return (
    <div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <StatPill label="Observations" value={meta.n_dates.toLocaleString()} />
        {["beta1","beta2","beta3"].map(f => (
          <button key={f} onClick={() => setFactor(f)} style={{
            padding: "8px 16px", borderRadius: 6, fontSize: 13, cursor: "pointer",
            background: factor === f ? "#1f6feb" : "#161b22",
            color: factor === f ? "#ffffff" : "#8b949e",
            border: `1px solid ${factor === f ? "#1f6feb" : "#30363d"}`,
          }}>
            {factor_labels[f]}
          </button>
        ))}
      </div>

      <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
        padding: "10px 16px", marginBottom: 16, fontSize: 13, color: "#8b949e", lineHeight: 1.6 }}>
        {factorDesc[factor]}
      </div>

      <Plot data={[...spreadTrace, ...traces]} layout={layout}
        config={{ responsive: true }} style={{ width: "100%" }} />

      <p style={{ color: "#8b949e", fontSize: 12, marginTop: 12 }}>
        Gray line = OLS fit each month independently (noisy). Blue line = Kalman filter
        tracks the latent state across time — it sees through noise to the underlying trend.
        {factor === "beta2" && " Red fill = curve inversion (historically a recession leading indicator)."}
      </p>
    </div>
  );
}

// ─── TAB 4: Recession Probability ────────────────────────────────────────────
function RecessionProbabilityTab() {
  const { data, loading, error } = useData("recession_probability.json");

  if (loading) return <LoadingCard message="Loading recession model…" />;
  if (error)   return <ErrorCard message={error} />;
  if (!data)   return null;

  const { dates, probabilities, current, nber_recessions, model_metrics, meta } = data;

  // NBER shading shapes
  const recShapes = (nber_recessions || []).map(r => ({
    type: "rect", xref: "x", yref: "paper",
    x0: r.start, x1: r.end, y0: 0, y1: 1,
    fillcolor: "#f8514920", line: { width: 0 },
  }));

  // Threshold reference line
  const threshold = 0.30;
  const thresholdShape = {
    type: "line", xref: "paper", yref: "y",
    x0: 0, x1: 1, y0: threshold * 100, y1: threshold * 100,
    line: { color: "#d29922", width: 1.2, dash: "dot" },
  };

  // Current prob color
  const prob = current.probability;
  const probColor = prob >= 50 ? "#f85149" : prob >= 30 ? "#d29922" : "#3fb950";
  const regime    = prob >= 50 ? "Elevated risk" : prob >= 30 ? "Moderate risk" : "Expansion";

  const mainTrace = {
    x: dates,
    y: probabilities.map(p => p * 100),
    name: "P(recession, 12M ahead)",
    type: "scatter", mode: "lines",
    line: { color: "#58a6ff", width: 1.8 },
    fill: "tozeroy", fillcolor: "#58a6ff18",
  };

  // Colour the fill red when above threshold
  const highTrace = {
    x: dates,
    y: probabilities.map(p => p * 100 >= threshold * 100 ? p * 100 : null),
    name: "Above threshold (30%)",
    type: "scatter", mode: "lines",
    line: { color: "#f85149", width: 2 },
    fill: "tozeroy", fillcolor: "#f8514930",
  };

  const layout = {
    ...DARK,
    title: { text: "12-month-ahead Recession Probability (Probit Model)",
      font: { color: "#ffffff", size: 13 }, x: 0.02 },
    xaxis: { gridcolor: "#21262d", linecolor: "#30363d" },
    yaxis: { title: "Probability (%)", range: [0, 100],
      gridcolor: "#21262d", linecolor: "#30363d",
      ticksuffix: "%" },
    shapes: [...recShapes, thresholdShape],
    annotations: (nber_recessions || []).map(r => ({
      x: r.start, y: 92, xref: "x", yref: "y",
      text: r.label, showarrow: false,
      font: { color: "#f85149", size: 9 },
      textangle: -45,
    })).concat([{
      x: "paper", y: threshold * 100, xref: "paper", yref: "y",
      text: " 30% threshold", showarrow: false, x: 0.01,
      font: { color: "#d29922", size: 10 }, xanchor: "left",
    }]),
    legend: { x: 0.01, y: 0.99, bgcolor: "#161b22", bordercolor: "#30363d", borderwidth: 1 },
    margin: { l: 60, r: 20, t: 50, b: 40 },
    height: 420,
  };

  const metrics = model_metrics || {};

  return (
    <div>
      {/* Current reading */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
        <div style={{ background: "#161b22", border: `1px solid ${probColor}`,
          borderRadius: 8, padding: "12px 20px", minWidth: 180 }}>
          <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 4 }}>
            Current estimate — {current.date}
          </div>
          <div style={{ fontSize: 28, fontWeight: 700, color: probColor,
            fontFamily: "IBM Plex Mono, monospace" }}>
            {prob.toFixed(1)}%
          </div>
          <div style={{ fontSize: 12, color: probColor, marginTop: 3 }}>{regime}</div>
        </div>
        {metrics.auc_roc   && <StatPill label="ROC-AUC"      value={metrics.auc_roc.toFixed(3)}   color="#bc8cff" />}
        {metrics.pseudo_r2 && <StatPill label="McFadden R²"  value={metrics.pseudo_r2.toFixed(3)} color="#39d353" />}
        {metrics.brier_score && <StatPill label="Brier score" value={metrics.brier_score.toFixed(3)} color="#d29922" />}
        <StatPill label="Observations" value={meta.n_observations.toLocaleString()} />
      </div>

      <Plot data={[mainTrace, highTrace]} layout={layout}
        config={{ responsive: true }} style={{ width: "100%" }} />

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginTop: 16 }}>
        <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
          padding: "10px 14px", fontSize: 12, color: "#8b949e", lineHeight: 1.6, flex: 1, minWidth: 200 }}>
          <strong style={{ color: "#c9d1d9" }}>Red shading</strong> = NBER-dated recessions.<br />
          <strong style={{ color: "#c9d1d9" }}>Blue area</strong> = model probability over time.<br />
          <strong style={{ color: "#d29922" }}>Dashed line</strong> = 30% signal threshold.
        </div>
        <div style={{ background: "#161b22", border: "1px solid #30363d", borderRadius: 8,
          padding: "10px 14px", fontSize: 12, color: "#8b949e", lineHeight: 1.6, flex: 1, minWidth: 200 }}>
          The model uses 2Y–10Y spread, 3M–10Y spread, and DNS β factors as features.
          A reading above 30% has historically preceded recessions by 6–18 months.
        </div>
      </div>
    </div>
  );
}

// ─── ROOT APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("surface");

  return (
    <div style={{ minHeight: "100vh", background: "#0d1117", color: "#c9d1d9",
      fontFamily: "IBM Plex Sans, sans-serif" }}>

      {/* Header */}
      <header style={{ borderBottom: "1px solid #21262d", padding: "0 32px",
        background: "#010409" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex",
          alignItems: "center", gap: 24, height: 56 }}>
          <div>
            <span style={{ fontSize: 15, fontWeight: 600, color: "#ffffff",
              letterSpacing: "-0.02em" }}>Yield Curve Dashboard</span>
            <span style={{ fontSize: 11, color: "#8b949e", marginLeft: 10,
              fontFamily: "IBM Plex Mono, monospace" }}>Nelson-Siegel · Kalman · Probit</span>
          </div>
          <div style={{ marginLeft: "auto", fontSize: 11, color: "#8b949e" }}>
            FRED Data · U.S. Treasuries · 1993–present
          </div>
        </div>
      </header>

      {/* Tab nav */}
      <nav style={{ borderBottom: "1px solid #21262d", padding: "0 32px",
        background: "#010409" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", gap: 0 }}>
          {TABS.map(tab => (
            <button key={tab.id} onClick={() => setActiveTab(tab.id)} style={{
              padding: "12px 20px", fontSize: 13, cursor: "pointer",
              background: "transparent",
              color: activeTab === tab.id ? "#ffffff" : "#8b949e",
              border: "none",
              borderBottom: activeTab === tab.id
                ? "2px solid #1f6feb" : "2px solid transparent",
              transition: "color 0.15s, border-color 0.15s",
            }}>
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 32px" }}>
        {activeTab === "surface"   && <YieldSurfaceTab />}
        {activeTab === "ns"        && <NSCalibrationTab />}
        {activeTab === "kalman"    && <KalmanFactorsTab />}
        {activeTab === "recession" && <RecessionProbabilityTab />}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: "1px solid #21262d", padding: "16px 32px",
        marginTop: 40, fontSize: 11, color: "#8b949e", textAlign: "center" }}>
        Data: FRED · St. Louis Fed · U.S. Treasury · NBER &nbsp;|&nbsp;
        Models: Nelson-Siegel-Svensson · Dynamic NS + Kalman Filter · Probit Regression
      </footer>

      <style>{`
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes spin { to { transform: rotate(360deg); } }
        button:hover { opacity: 0.85; }
        select:focus, button:focus { outline: 2px solid #1f6feb; outline-offset: 2px; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
      `}</style>
    </div>
  );
}

import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from "recharts";
import "./styles.css";

const API = "http://localhost:8000";

const defaultPatient = {
  age: 50, gender: 1, height: 165, weight: 70,
  ap_hi: 120, ap_lo: 80, cholesterol: 1,
  gluc: 1, smoke: 0, alco: 0, active: 1
};

function App() {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [patient, setPatient] = useState(defaultPatient);
  const [prediction, setPrediction] = useState(null);
  const [ai, setAi] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function analyze() {
    setLoading(true); setError("");
    try {
      const fd = new FormData();
      if (file) fd.append("file", file);
      const r = await fetch(`${API}/api/analyze`, { method: "POST", body: fd });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Analysis failed");
      setAnalysis(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }

  async function predict() {
    setError("");
    try {
      const payload = Object.fromEntries(
        Object.entries(patient).map(([k, v]) => [k, Number(v)])
      );
      const r = await fetch(`${API}/api/predict`, {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "Prediction failed");
      setPrediction(data);
    } catch (e) { setError(e.message); }
  }

  async function explain() {
    if (!analysis) return;
    setAi("Generating explanation...");
    try {
      const r = await fetch(`${API}/api/ai-explain`, {
        method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({analysis, prediction})
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "AI explanation failed");
      setAi(data.explanation);
    } catch (e) { setAi(e.message); }
  }

  const modelData = analysis?.models || [];

  return (
    <div className="app">
      <header>
        <div>
          <h1>Cardiovascular Disease Prediction</h1>
          <p className="sub">Preprocessing · EDA · Correlation · Model comparison · Prediction · AI explanation</p>
        </div>
        <button onClick={analyze} disabled={loading}>{loading ? "Analyzing..." : "Run Analysis"}</button>
      </header>

      {error && <div className="error">{error}</div>}

      <section className="card upload">
        <h2>1. Dataset</h2>
        <p>Expected columns: <code>id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio</code></p>
        <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)} />
        <span>{file ? file.name : "No file selected — backend will try data/cardio_train.csv"}</span>
      </section>

      {analysis && <>
        <section className="grid four">
          <Metric title="Rows" value={analysis.summary.rows_after_cleaning}/>
          <Metric title="Duplicates Removed" value={analysis.summary.duplicates_removed}/>
          <Metric title="Best Model" value={analysis.best_model}/>
          <Metric title="Best Accuracy" value={analysis.models[0].accuracy}/>
        </section>

        <section className="card">
          <h2>2. Model Comparison</h2>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={modelData}>
              <CartesianGrid strokeDasharray="3 3"/>
              <XAxis dataKey="model" interval={0} angle={-12} textAnchor="end" height={70}/>
              <YAxis domain={[0,1]}/>
              <Tooltip/>
              <Bar dataKey="accuracy" name="Accuracy"/>
            </BarChart>
          </ResponsiveContainer>
          <div className="tableWrap">
            <table>
              <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead>
              <tbody>{modelData.map(m =>
                <tr key={m.model} className={m.model === analysis.best_model ? "best" : ""}>
                  <td>{m.model}</td><td>{m.accuracy}</td><td>{m.precision}</td>
                  <td>{m.recall}</td><td>{m.f1}</td><td>{m.roc_auc ?? "-"}</td>
                </tr>
              )}</tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <h2>3. Exploratory Data Analysis</h2>
          <div className="plots">
            {Object.entries(analysis.plots).map(([name, src]) =>
              <figure key={name}><img src={src} alt={name}/><figcaption>{name.replaceAll("_"," ")}</figcaption></figure>
            )}
          </div>
        </section>

        <section className="card">
          <h2>4. Correlation Matrix</h2>
          <div className="tableWrap">
            <table>
              <thead><tr><th>Feature</th>{Object.keys(analysis.correlation).map(k => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{Object.entries(analysis.correlation).map(([row, vals]) =>
                <tr key={row}><td>{row}</td>{Object.keys(analysis.correlation).map(col => <td key={col}>{vals[col]}</td>)}</tr>
              )}</tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <h2>5. Individual Prediction</h2>
          <div className="formGrid">
            {Object.keys(defaultPatient).map(k =>
              <label key={k}>{k}
                <input type="number" step="any" value={patient[k]}
                  onChange={e => setPatient({...patient, [k]: e.target.value})}/>
              </label>
            )}
          </div>
          <button onClick={predict}>Predict</button>
          {prediction && <div className={`prediction ${prediction.prediction ? "risk" : "lower"}`}>
            <strong>{prediction.label}</strong>
            <span>Probability: {(prediction.probability * 100).toFixed(1)}%</span>
          </div>}
        </section>

        <section className="card">
          <h2>6. ChatGPT Project Explanation</h2>
          <p>Uses the backend OpenAI API integration to turn the metrics and prediction into a report-friendly explanation.</p>
          <button onClick={explain}>Generate AI Explanation</button>
          {ai && <div className="ai">{ai}</div>}
        </section>
      </>}

      <footer>
        Designed and developed by Sushma Kanna
      </footer>
    </div>
  );
}

function Metric({title, value}) {
  return <div className="metric"><span>{title}</span><strong>{value}</strong></div>;
}

createRoot(document.getElementById("root")).render(<App />);

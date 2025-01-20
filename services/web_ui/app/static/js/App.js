import React, { useState } from "react";

function App() {
  const [configPath, setConfigPath] = useState("");
  const [modelType, setModelType] = useState("LSTM");
  const [response, setResponse] = useState("");

  const runFlow = async () => {
    try {
      const res = await fetch("http://localhost:4243/run_flow/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config_path: configPath, model_type: modelType }),
      });
      const data = await res.json();
      setResponse(JSON.stringify(data, null, 2));
    } catch (err) {
      setResponse(`Error: ${err.message}`);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Run Flow</h1>
      <div>
        <label>Config Path:</label>
        <input
          type="text"
          value={configPath}
          onChange={(e) => setConfigPath(e.target.value)}
          placeholder="e.g., configs/full_flow_config.yaml"
          style={{ margin: "10px" }}
        />
      </div>
      <div>
        <label>Model Type:</label>
        <select
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
          style={{ margin: "10px" }}
        >
          <option value="LSTM">LSTM</option>
          <option value="GRU">GRU</option>
          <option value="BiLSTM">BiLSTM</option>
          <option value="Conv1D-BiLSTM">Conv1D-BiLSTM</option>
        </select>
      </div>
      <button onClick={runFlow}>Run Flow</button>
      <pre style={{ background: "#f4f4f4", padding: "10px" }}>{response}</pre>
    </div>
  );
}

export default App;

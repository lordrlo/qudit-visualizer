// src/App.tsx
import React, { useEffect, useRef, useState } from "react";
import { runSimulation } from "./api";
import type {
  SimulationResponse,
  InitialStateType,
  ComplexNumber,
} from "./api";
import { WignerHeatmap } from "./components/WignerHeatmap";

const containerStyle: React.CSSProperties = {
  display: "flex",
  height: "100vh",
  fontFamily:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
  background: "#020617", // dark background
  color: "#e5e7eb",
};

const sidebarStyle: React.CSSProperties = {
  width: 320,
  padding: 16,
  boxSizing: "border-box",
  borderRight: "1px solid #374151",
};

const mainStyle: React.CSSProperties = {
  flex: 1,
  padding: 16,
  boxSizing: "border-box",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const labelStyle: React.CSSProperties = {
  display: "block",
  fontSize: 14,
  marginBottom: 4,
};

const smallTextStyle: React.CSSProperties = {
  fontSize: 12,
  color: "#9ca3af",
};

const buttonStyle: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: 6,
  border: "none",
  background: "#10b981",
  color: "#022c22",
  fontWeight: 600,
  cursor: "pointer",
};

const playButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  background: "#38bdf8",
  color: "#0f172a",
};

const rangeStyle: React.CSSProperties = {
  width: "100%",
};

const DEFAULT_T_MAX = 10;
const DEFAULT_STEPS = 201;

export const App: React.FC = () => {
  const [d, setD] = useState(3);

  const [initialType, setInitialType] =
    useState<InitialStateType>("basis");
  const [basisIndex, setBasisIndex] = useState(0);

  // custom state amplitudes: arrays of length d
  const [psiRe, setPsiRe] = useState<number[]>([1, 0, 0]); // default |0>
  const [psiIm, setPsiIm] = useState<number[]>([0, 0, 0]);

  const [simData, setSimData] = useState<SimulationResponse | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const lastTimeRef = useRef<number | null>(null);
  const frameIdRef = useRef<number | null>(null);

  const nSteps = simData?.ts.length ?? DEFAULT_STEPS;

  // Resize psi arrays when d changes
  useEffect(() => {
    setBasisIndex(0);

    setPsiRe((old) => {
      const arr = [...old];
      if (arr.length < d) {
        while (arr.length < d) arr.push(0);
      } else if (arr.length > d) {
        arr.length = d;
      }

      const allZero = arr.reduce((acc, v) => acc && v === 0, true);
      if (allZero && arr.length > 0) {
        arr[0] = 1;
      }
      return arr;
    });

    setPsiIm((old) => {
      const arr = [...old];
      if (arr.length < d) {
        while (arr.length < d) arr.push(0);
      } else if (arr.length > d) {
        arr.length = d;
      }
      return arr;
    });
  }, [d]);

  // Animation loop over frames
  useEffect(() => {
    if (!playing || !simData) {
      if (frameIdRef.current != null) {
        cancelAnimationFrame(frameIdRef.current);
        frameIdRef.current = null;
      }
      lastTimeRef.current = null;
      return;
    }

    const loop = (timestamp: number) => {
      if (lastTimeRef.current == null) {
        lastTimeRef.current = timestamp;
      } else {
        const dtMs = timestamp - lastTimeRef.current;
        lastTimeRef.current = timestamp;

        const dt = (dtMs / 1000) * speed; // physical time increment
        const totalTime =
          simData.ts[simData.ts.length - 1] || 1;
        const framesPerSecond =
          (simData.ts.length - 1) / totalTime;
        const dFrame = dt * framesPerSecond;

        setFrame((prev) => {
          let next = prev + dFrame;
          if (next >= nSteps) next -= nSteps;
          return next;
        });
      }

      frameIdRef.current = requestAnimationFrame(loop);
    };

    frameIdRef.current = requestAnimationFrame(loop);

    return () => {
      if (frameIdRef.current != null) {
        cancelAnimationFrame(frameIdRef.current);
        frameIdRef.current = null;
      }
    };
  }, [playing, speed, simData, nSteps]);

  async function handleRun() {
    setLoading(true);
    setErrorMsg(null);
    setPlaying(false);
    setFrame(0);
    try {
      // build psi_custom if needed
      let psi_custom: ComplexNumber[] | undefined = undefined;
      if (initialType === "custom") {
        psi_custom = Array.from({ length: d }).map((_, q) => ({
          re: psiRe[q] ?? 0,
          im: psiIm[q] ?? 0,
        }));
      }

      const res = await runSimulation({
        d,
        hamiltonian: "diagonal_quadratic",
        initial_state: initialType,
        basis_index: basisIndex,
        t_max: DEFAULT_T_MAX,
        n_steps: DEFAULT_STEPS,
        psi_custom,
      });

      setSimData(res);
    } catch (err: any) {
      setErrorMsg(err.message ?? String(err));
      setSimData(null);
    } finally {
      setLoading(false);
    }
  }

  const currentIndex =
    simData && simData.ts.length > 0
      ? Math.floor(frame) % simData.ts.length
      : 0;

  const currentW =
    simData && simData.W.length > 0
      ? simData.W[currentIndex]
      : null;

  const currentT =
    simData && simData.ts.length > 0
      ? simData.ts[currentIndex]
      : 0;

  return (
    <div style={containerStyle}>
      {/* Sidebar / controls */}
      <div style={sidebarStyle}>
        <h1 style={{ fontSize: 20, fontWeight: 600, marginBottom: 12 }}>
          Discrete Wigner Visualizer
        </h1>

        {/* Dimension */}
        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Dimension d (odd)</label>
          <select
            value={d}
            onChange={(e) => {
              const nd = parseInt(e.target.value, 10);
              setD(nd);
              setSimData(null);
              setFrame(0);
            }}
            style={{
              width: "100%",
              padding: "4px 6px",
              borderRadius: 4,
              border: "1px solid #4b5563",
              background: "#020617",
              color: "#e5e7eb",
            }}
          >
            {[3, 5, 7].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>

        {/* Initial state */}
        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Initial state</label>
          <select
            value={initialType}
            onChange={(e) =>
              setInitialType(e.target.value as InitialStateType)
            }
            style={{
              width: "100%",
              padding: "4px 6px",
              borderRadius: 4,
              border: "1px solid #4b5563",
              background: "#020617",
              color: "#e5e7eb",
            }}
          >
            <option value="basis">Basis state |q⟩</option>
            <option value="equal_superposition">
              Equal superposition
            </option>
            <option value="custom">Custom amplitudes</option>
          </select>

          {initialType === "basis" && (
            <div style={{ marginTop: 8 }}>
              <label style={{ ...labelStyle, fontSize: 12 }}>
                q index: {basisIndex}
              </label>
              <input
                type="range"
                min={0}
                max={d - 1}
                value={basisIndex}
                onChange={(e) =>
                  setBasisIndex(parseInt(e.target.value, 10))
                }
                style={rangeStyle}
              />
            </div>
          )}

          {initialType === "custom" && (
            <div style={{ marginTop: 8 }}>
              <div
                style={{
                  ...smallTextStyle,
                  marginBottom: 4,
                }}
              >
                Enter amplitudes ψ = Σ₍q₎ (a₍q₎ + i b₍q₎) |q⟩. The state is
                normalized automatically.
              </div>
              {Array.from({ length: d }).map((_, q) => (
                <div
                  key={q}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    marginBottom: 4,
                  }}
                >
                  <span style={{ fontSize: 12 }}>|{q}⟩:</span>
                  <input
                    type="number"
                    step="0.01"
                    value={psiRe[q]}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value || "0");
                      setPsiRe((prev) => {
                        const arr = [...prev];
                        arr[q] = isNaN(v) ? 0 : v;
                        return arr;
                      });
                    }}
                    style={{
                      width: 70,
                      padding: "2px 4px",
                      borderRadius: 4,
                      border: "1px solid #4b5563",
                      background: "#020617",
                      color: "#e5e7eb",
                      fontSize: 12,
                    }}
                    placeholder="Re"
                  />
                  <input
                    type="number"
                    step="0.01"
                    value={psiIm[q]}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value || "0");
                      setPsiIm((prev) => {
                        const arr = [...prev];
                        arr[q] = isNaN(v) ? 0 : v;
                        return arr;
                      });
                    }}
                    style={{
                      width: 70,
                      padding: "2px 4px",
                      borderRadius: 4,
                      border: "1px solid #4b5563",
                      background: "#020617",
                      color: "#e5e7eb",
                      fontSize: 12,
                    }}
                    placeholder="Im"
                  />
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Run button */}
        <div style={{ marginBottom: 12 }}>
          <button
            onClick={handleRun}
            disabled={loading}
            style={{
              ...buttonStyle,
              opacity: loading ? 0.6 : 1,
            }}
          >
            {loading ? "Running..." : "Run simulation"}
          </button>
        </div>

        {/* Time + play controls */}
        {simData && (
          <>
            <div style={{ marginBottom: 12 }}>
              <label style={labelStyle}>
                Time t = {currentT.toFixed(2)}
              </label>
              <input
                type="range"
                min={0}
                max={(simData.ts.length || 1) - 1}
                value={currentIndex}
                onChange={(e) =>
                  setFrame(parseInt(e.target.value, 10))
                }
                style={rangeStyle}
              />
            </div>

            <div
              style={{
                display: "flex",
                gap: 8,
                alignItems: "center",
                marginBottom: 8,
              }}
            >
              <button
                onClick={() => setPlaying((p) => !p)}
                style={playButtonStyle}
              >
                {playing ? "Pause" : "Play"}
              </button>
              <div style={{ fontSize: 12 }}>
                Speed
                <input
                  type="range"
                  min={0.1}
                  max={4}
                  step={0.1}
                  value={speed}
                  onChange={(e) =>
                    setSpeed(parseFloat(e.target.value))
                  }
                  style={{ marginLeft: 8 }}
                />
              </div>
            </div>
          </>
        )}

        {errorMsg && (
          <div
            style={{
              fontSize: 12,
              color: "#fca5a5",
              marginTop: 12,
              whiteSpace: "pre-wrap",
            }}
          >
            {errorMsg}
          </div>
        )}

        <p style={{ ...smallTextStyle, marginTop: 16 }}>
          Backend (FastAPI + Dynamiqs) solves Schrödinger&apos;s equation
          for a d-dimensional qudit with a fixed diagonal Hamiltonian.
          The frontend shows the discrete Wigner function W(q,p; t) as an
          interactive heatmap.
        </p>
      </div>

      {/* Main visualization area */}
      <div style={mainStyle}>
        {currentW ? (
          <WignerHeatmap W={currentW} />
        ) : (
          <div style={{ color: "#9ca3af", fontSize: 14 }}>
            Run a simulation to see W(q,p; t).
          </div>
        )}
      </div>
    </div>
  );
};

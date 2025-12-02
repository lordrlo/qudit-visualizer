// src/App.tsx
import React, { useEffect, useRef, useState } from "react";
import { runSimulation, applyGate } from "./api";
import type {
  SimulationResponse,
  InitialStateType,
  ComplexNumber,
  GateName,
} from "./api";
import { WignerHeatmap } from "./components/WignerHeatmap";

type Mode = "continuous" | "gates";

const containerStyle: React.CSSProperties = {
  display: "flex",
  height: "100vh",
  fontFamily:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
  background: "#020617",
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
  const [mode, setMode] = useState<Mode>("continuous");

  const [d, setD] = useState(3);

  const [initialType, setInitialType] =
    useState<InitialStateType>("basis");
  const [basisIndex, setBasisIndex] = useState(0);

  // Canonical "current state" ψ(q) = psiRe[q] + i psiIm[q]
  const [psiRe, setPsiRe] = useState<number[]>([1, 0, 0]); // |0>
  const [psiIm, setPsiIm] = useState<number[]>([0, 0, 0]);

  // Separate editor arrays for custom amplitudes (what the user types)
  const [customRe, setCustomRe] = useState<number[]>([1, 0, 0]);
  const [customIm, setCustomIm] = useState<number[]>([0, 0, 0]);

  // Trajectory / single-shot result from backend
  const [simData, setSimData] = useState<SimulationResponse | null>(
    null
  );

  const [frame, setFrame] = useState(0); // index in simData.ts
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const lastTimeRef = useRef<number | null>(null);
  const frameIdRef = useRef<number | null>(null);

  const nSteps = simData?.ts.length ?? 0;

  // ---- Helper: preview Wigner for current ψ (single time t=0) ----

  async function previewFromPsi(
    dim: number,
    reArr: number[],
    imArr: number[]
  ) {
    try {
      setErrorMsg(null);

      const psi_custom: ComplexNumber[] = Array.from({ length: dim }).map(
        (_, q) => ({
          re: reArr[q] ?? 0,
          im: imArr[q] ?? 0,
        })
      );

      const res = await runSimulation({
        d: dim,
        hamiltonian: "diagonal_quadratic",
        initial_state: "custom",
        basis_index: 0,
        t_max: 0,
        n_steps: 1,
        psi_custom,
      });

      setSimData(res);
      setFrame(0);
      setPlaying(false);
    } catch (err: any) {
      setErrorMsg(err.message ?? String(err));
      setSimData(null);
    }
  }

  // ---- Initial preview on mount ----

  useEffect(() => {
    previewFromPsi(d, psiRe, psiIm);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---- Keep psiRe/psiIm in sync with trajectory + frame ----

  useEffect(() => {
    if (!simData || !simData.psi || simData.psi.length === 0) return;

    const n = simData.psi.length;
    if (n === 0) return;

    const idx = Math.min(
      Math.max(Math.floor(frame), 0),
      n - 1
    );
    const psi_t = simData.psi[idx];

    if (!psi_t || psi_t.length !== d) return;

    setPsiRe(psi_t.map((c) => c.re));
    setPsiIm(psi_t.map((c) => c.im));
  }, [simData, frame, d]);

  // ---- Preview effect: when ψ changes and there is no trajectory, recompute Wigner ----

  useEffect(() => {
    if (simData) return; // if we have a trajectory, that is the source of truth
    previewFromPsi(d, psiRe, psiIm);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d, psiRe, psiIm, simData]);

  // ---- Animation loop for continuous mode ----

  useEffect(() => {
    if (
      !playing ||
      !simData ||
      mode !== "continuous" ||
      !simData.ts ||
      simData.ts.length <= 1
    ) {
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

        const dt = (dtMs / 1000) * speed;
        const totalTime =
          simData.ts[simData.ts.length - 1] || 1;
        const framesPerSecond =
          (simData.ts.length - 1) / totalTime;
        const dFrame = dt * framesPerSecond;

        setFrame((prev) => {
          if (!simData || !simData.ts) return prev;
          let next = prev + dFrame;
          const maxIndex = simData.ts.length;
          if (next >= maxIndex) next -= maxIndex;
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
  }, [playing, speed, simData, mode]);

  // ---- Dimension change ----

  function handleDimensionChange(nd: number) {
    let newRe: number[] = [];
    let newIm: number[] = [];

    if (initialType === "basis") {
      const idx = Math.min(basisIndex, nd - 1);
      newRe = Array(nd).fill(0);
      newRe[idx] = 1;
      newIm = Array(nd).fill(0);
    } else if (initialType === "equal_superposition") {
      const amp = 1 / Math.sqrt(nd);
      newRe = Array(nd).fill(amp);
      newIm = Array(nd).fill(0);
    } else {
      // "custom" – keep as much as possible from custom editor
      newRe = customRe.slice(0, nd);
      newIm = customIm.slice(0, nd);
      while (newRe.length < nd) newRe.push(0);
      while (newIm.length < nd) newIm.push(0);
    }

    setD(nd);
    setBasisIndex((old) => Math.min(old, nd - 1));
    setPsiRe(newRe);
    setPsiIm(newIm);
    setCustomRe(newRe);
    setCustomIm(newIm);
    setPlaying(false);
    setSimData(null); // old trajectory no longer valid
  }

  // ---- Initial type changes presets ----

  function handleInitialTypeChange(newType: InitialStateType) {
    let newRe = [...psiRe];
    let newIm = [...psiIm];

    if (newType === "basis") {
      newRe = Array(d).fill(0);
      newRe[Math.min(basisIndex, d - 1)] = 1;
      newIm = Array(d).fill(0);
      setCustomRe(newRe);
      setCustomIm(newIm);
    } else if (newType === "equal_superposition") {
      const amp = 1 / Math.sqrt(d);
      newRe = Array(d).fill(amp);
      newIm = Array(d).fill(0);
      setCustomRe(newRe);
      setCustomIm(newIm);
    } else {
      // "custom": start from current ψ
      setCustomRe([...psiRe]);
      setCustomIm([...psiIm]);
    }

    setInitialType(newType);
    setPsiRe(newRe);
    setPsiIm(newIm);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Basis index slider ----

  function handleBasisIndexChange(idx: number) {
    const clamped = Math.min(Math.max(idx, 0), d - 1);
    const newRe = Array(d).fill(0);
    newRe[clamped] = 1;
    const newIm = Array(d).fill(0);

    setBasisIndex(clamped);
    setPsiRe(newRe);
    setPsiIm(newIm);
    setCustomRe(newRe);
    setCustomIm(newIm);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Custom amplitude editing (editor only) ----

  function handlePsiComponentChange(
    q: number,
    part: "re" | "im",
    value: number
  ) {
    const newCustomRe = [...customRe];
    const newCustomIm = [...customIm];

    if (part === "re") newCustomRe[q] = value;
    else newCustomIm[q] = value;

    setCustomRe(newCustomRe);
    setCustomIm(newCustomIm);

    // Canonical ψ is whatever the user typed (possibly unnormalized);
    // the backend + preview will normalize for the Wigner.
    const newPsiRe = [...newCustomRe];
    const newPsiIm = [...newCustomIm];

    setPsiRe(newPsiRe);
    setPsiIm(newPsiIm);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Continuous evolution from current state ----

  async function handleRun() {
    setLoading(true);
    setErrorMsg(null);
    setPlaying(false);
    setFrame(0);
    try {
      const psi_custom: ComplexNumber[] = Array.from({ length: d }).map(
        (_, q) => ({
          re: psiRe[q] ?? 0,
          im: psiIm[q] ?? 0,
        })
      );

      const res = await runSimulation({
        d,
        hamiltonian: "diagonal_quadratic",
        initial_state: "custom",
        basis_index: 0,
        t_max: DEFAULT_T_MAX,
        n_steps: DEFAULT_STEPS,
        psi_custom,
      });

      setSimData(res);
      setFrame(0);
    } catch (err: any) {
      setErrorMsg(err.message ?? String(err));
      setSimData(null);
    } finally {
      setLoading(false);
    }
  }

  // ---- Single-shot gate application ----

  async function handleApplyGate(gate: GateName) {
    try {
      setLoading(true);
      setErrorMsg(null);
      setPlaying(false);

      const psi: ComplexNumber[] = Array.from({ length: d }).map(
        (_, q) => ({
          re: psiRe[q] ?? 0,
          im: psiIm[q] ?? 0,
        })
      );

      const res = await applyGate({
        d,
        gate,
        psi,
      });

      // new current state
      const newRe = res.psi.map((c) => c.re);
      const newIm = res.psi.map((c) => c.im);
      setPsiRe(newRe);
      setPsiIm(newIm);

      // Note: we do NOT touch customRe/customIm here,
      // so the "chooser" keeps the initial specification.

      // show its Wigner as a single-step trajectory
      const fakeSim: SimulationResponse = {
        d: res.d,
        ts: [0],
        W: [res.W],
        psi: [res.psi],
      };
      setSimData(fakeSim);
      setFrame(0);
    } catch (err: any) {
      setErrorMsg(err.message ?? String(err));
    } finally {
      setLoading(false);
    }
  }

  const currentIndex =
    simData && simData.ts.length > 0
      ? Math.min(
          Math.max(Math.floor(frame), 0),
          simData.ts.length - 1
        )
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
      {/* Sidebar */}
      <div style={sidebarStyle}>
        <h1 style={{ fontSize: 20, fontWeight: 600, marginBottom: 12 }}>
          Discrete Wigner Visualizer
        </h1>

        {/* Mode */}
        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Mode</label>
          <select
            value={mode}
            onChange={(e) =>
              setMode(e.target.value as Mode)
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
            <option value="continuous">
              Continuous evolution (Hamiltonian)
            </option>
            <option value="gates">
              Gate mode (X, Y, Z, F, T)
            </option>
          </select>
        </div>

        {/* Dimension */}
        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Dimension d (odd)</label>
          <select
            value={d}
            onChange={(e) =>
              handleDimensionChange(parseInt(e.target.value, 10))
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
            {[3, 5, 7].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </div>

        {/* Initial/current state controls */}
        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Initial / current state</label>
          <select
            value={initialType}
            onChange={(e) =>
              handleInitialTypeChange(e.target.value as InitialStateType)
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
                  handleBasisIndexChange(parseInt(e.target.value, 10))
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
                Enter amplitudes ψ = Σ (a₍q₎ + i b₍q₎) |q⟩.
                The state is normalized on the backend, but these
                fields keep exactly what you type.
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
                    value={customRe[q] ?? 0}
                    onChange={(e) =>
                      handlePsiComponentChange(
                        q,
                        "re",
                        parseFloat(e.target.value || "0")
                      )
                    }
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
                    value={customIm[q] ?? 0}
                    onChange={(e) =>
                      handlePsiComponentChange(
                        q,
                        "im",
                        parseFloat(e.target.value || "0")
                      )
                    }
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

        {/* Current state display */}
        <div style={{ marginBottom: 12 }}>
          <label style={labelStyle}>Current state ψ(q)</label>
          <div
            style={{
              maxHeight: 120,
              overflowY: "auto",
              fontFamily: "monospace",
              fontSize: 11,
              background: "#020617",
              borderRadius: 4,
              border: "1px solid #4b5563",
              padding: 6,
            }}
          >
            {Array.from({ length: d }).map((_, q) => (
              <div key={q}>
                ψ[{q}] = {psiRe[q].toFixed(3)}{" "}
                {psiIm[q] >= 0 ? "+" : "-"} {Math.abs(psiIm[q]).toFixed(3)} i
              </div>
            ))}
          </div>
        </div>

        {/* Continuous mode controls */}
        {mode === "continuous" && (
          <>
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

            {simData && nSteps > 1 && (
              <>
                <div style={{ marginBottom: 12 }}>
                  <label style={labelStyle}>
                    Time t = {currentT.toFixed(2)}
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={nSteps - 1}
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
          </>
        )}

        {/* Gate mode controls */}
        {mode === "gates" && (
          <div style={{ marginBottom: 12 }}>
            <label style={labelStyle}>Gates</label>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
                marginBottom: 4,
              }}
            >
              {(["X", "Y", "Z", "F", "T"] as GateName[]).map((g) => (
                <button
                  key={g}
                  onClick={() => handleApplyGate(g)}
                  style={{
                    ...buttonStyle,
                    background: "#4b5563",
                    color: "#e5e7eb",
                    padding: "4px 10px",
                  }}
                  disabled={loading}
                >
                  {g}
                </button>
              ))}
            </div>
            <div style={smallTextStyle}>
              Each gate is applied to the current state ψ, and W(q,p)
              updates. The custom amplitudes panel keeps your original
              specification.
            </div>
          </div>
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
          The Wigner plot always shows the current state ψ. Continuous
          evolution computes a trajectory ψ(tₙ) and lets you move along it
          in time; gate mode applies unitaries directly to ψ. Custom
          amplitudes are edited in the panel above without being
          renormalized in-place.
        </p>
      </div>

      {/* Main visualization */}
      <div style={mainStyle}>
        {currentW ? (
          <WignerHeatmap W={currentW} />
        ) : (
          <div style={{ color: "#9ca3af", fontSize: 14 }}>
            Adjust the state, run a simulation, or apply a gate to see
            W(q,p).
          </div>
        )}
      </div>
    </div>
  );
};

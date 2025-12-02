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

import { create, all } from "mathjs";

const math = create(all, {});

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

function parseComplexExpression(
  expr: string
): { re: number; im: number } | null {
  let s = expr.trim();
  if (!s) return null;

  // Single convention:
  // - imaginary unit: i
  // - constants: pi
  // - functions: exp, sqrt, sin, cos, ...
  // Small convenience: allow capital I as alias of i
  s = s.replace(/\bI\b/g, "i");

  // Optional convenience: treat e^(...) as exp(...),
  // but the *documented* convention is exp(...)
  s = s.replace(/e\^\s*\(/g, "exp(");

  try {
    const val = math.evaluate(s);

    if (math.isComplex(val)) {
      return { re: Number(val.re), im: Number(val.im) };
    }

    if (typeof val === "number") {
      return { re: Number(val), im: 0 };
    }

    if ((val as any)?.valueOf) {
      const v = (val as any).valueOf();
      if (typeof v === "number") {
        return { re: v, im: 0 };
      }
    }

    return null;
  } catch (e) {
    console.error("Failed to parse complex expression:", expr, "→", s, e);
    return null;
  }
}

// Coherent state |q,p> = D_{q,p} |0> with vacuum = |0>
// Using D_{q,p} ~ Z^p X^q => psi[n] = 0 for n != q, psi[q] = exp(2π i p q / d)
function buildCoherentState(
  d: number,
  q0: number,
  p0: number
): { re: number[]; im: number[] } {
  const re = new Array<number>(d).fill(0);
  const im = new Array<number>(d).fill(0);

  const q = ((q0 % d) + d) % d;
  const p = ((p0 % d) + d) % d;

  const theta = (2 * Math.PI * p * q) / d;
  re[q] = Math.cos(theta);
  im[q] = Math.sin(theta);

  return { re, im };
}

export const App: React.FC = () => {
  const [mode, setMode] = useState<Mode>("continuous");

  const [d, setD] = useState(3);

  const [initialType, setInitialType] =
    useState<InitialStateType>("basis");
  const [basisIndex, setBasisIndex] = useState(0);

  // Coherent-state parameters (q,p)
  const [cohQ, setCohQ] = useState(0);
  const [cohP, setCohP] = useState(0);

  // Canonical "current state" ψ(q) = psiRe[q] + i psiIm[q]
  const [psiRe, setPsiRe] = useState<number[]>([1, 0, 0]); // |0>
  const [psiIm, setPsiIm] = useState<number[]>([0, 0, 0]);

  // Custom amplitudes: internal numeric + expression-as-typed
  const [customRe, setCustomRe] = useState<number[]>([1, 0, 0]);
  const [customIm, setCustomIm] = useState<number[]>([0, 0, 0]);
  const [customExpr, setCustomExpr] = useState<string[]>(["1", "0", "0"]);

  // Custom gate matrix editor: U_ij = gateRe[i][j] + i gateIm[i][j], driven by expressions
  const [gateRe, setGateRe] = useState<number[][]>([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  const [gateIm, setGateIm] = useState<number[][]>([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]);
  const [gateExpr, setGateExpr] = useState<string[][]>([
    ["1", "0", "0"],
    ["0", "1", "0"],
    ["0", "0", "1"],
  ]);

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

  // Keep custom gate matrix sized to d x d (reset to identity on d change)
  useEffect(() => {
    const idRe = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => (i === j ? 1 : 0))
    );
    const idIm = Array.from({ length: d }, () =>
      Array.from({ length: d }, () => 0)
    );
    const expr = Array.from({ length: d }, (_, i) =>
      Array.from({ length: d }, (_, j) => (i === j ? "1" : "0"))
    );
    setGateRe(idRe);
    setGateIm(idIm);
    setGateExpr(expr);
  }, [d]);

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
      !simData.ts.length ||
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
    let newExpr: string[] = [];

    if (initialType === "basis") {
      const idx = Math.min(basisIndex, nd - 1);
      newRe = Array(nd).fill(0);
      newRe[idx] = 1;
      newIm = Array(nd).fill(0);
      newExpr = Array(nd).fill("0");
      newExpr[idx] = "1";
    } else if (initialType === "equal_superposition") {
      const amp = 1 / Math.sqrt(nd);
      newRe = Array(nd).fill(amp);
      newIm = Array(nd).fill(0);
      newExpr = Array(nd).fill(`1/sqrt(${nd})`);
    } else if (initialType === "coherent") {
      const q = Math.min(cohQ, nd - 1);
      const p = Math.min(cohP, nd - 1);
      const coh = buildCoherentState(nd, q, p);
      newRe = coh.re;
      newIm = coh.im;
      newExpr = newRe.map((val, i) => {
        const im = newIm[i];
        if (Math.abs(val) < 1e-12 && Math.abs(im) < 1e-12) return "0";
        if (Math.abs(im) < 1e-12) return `${val}`;
        return `${val}+i*${im}`;
      });
      setCohQ(q);
      setCohP(p);
    } else {
      // "custom" – keep as much as possible from custom editor
      newRe = customRe.slice(0, nd);
      newIm = customIm.slice(0, nd);
      newExpr = customExpr.slice(0, nd);
      while (newRe.length < nd) newRe.push(0);
      while (newIm.length < nd) newIm.push(0);
      while (newExpr.length < nd) newExpr.push("0");
    }

    setD(nd);
    setBasisIndex((old) => Math.min(old, nd - 1));
    setPsiRe(newRe);
    setPsiIm(newIm);
    setCustomRe(newRe);
    setCustomIm(newIm);
    setCustomExpr(newExpr);
    setPlaying(false);
    setSimData(null); // old trajectory no longer valid
  }

  // ---- Initial type changes presets (basis / equal / coherent / custom) ----

  function handleInitialTypeChange(newType: InitialStateType) {
    let newRe: number[] = [];
    let newIm: number[] = [];
    let newExpr: string[] = [];

    if (newType === "basis") {
      newRe = Array(d).fill(0);
      newRe[Math.min(basisIndex, d - 1)] = 1;
      newIm = Array(d).fill(0);
      newExpr = Array(d).fill("0");
      newExpr[Math.min(basisIndex, d - 1)] = "1";
    } else if (newType === "equal_superposition") {
      const amp = 1 / Math.sqrt(d);
      newRe = Array(d).fill(amp);
      newIm = Array(d).fill(0);
      newExpr = Array(d).fill(`1/sqrt(${d})`);
    } else if (newType === "coherent") {
      const q = Math.min(cohQ, d - 1);
      const p = Math.min(cohP, d - 1);
      const coh = buildCoherentState(d, q, p);
      newRe = coh.re;
      newIm = coh.im;
      newExpr = newRe.map((val, i) => {
        const im = newIm[i];
        if (Math.abs(val) < 1e-12 && Math.abs(im) < 1e-12) return "0";
        if (Math.abs(im) < 1e-12) return `${val}`;
        return `${val}+i*${im}`;
      });
    } else {
      // "custom": start from current ψ
      newRe = [...psiRe];
      newIm = [...psiIm];
      newExpr = newRe.map((val, i) => {
        const im = newIm[i];
        if (Math.abs(val) < 1e-12 && Math.abs(im) < 1e-12) return "0";
        if (Math.abs(im) < 1e-12) return `${val}`;
        return `${val}+i*${im}`;
      });
    }

    setInitialType(newType);
    setPsiRe(newRe);
    setPsiIm(newIm);
    setCustomRe(newRe);
    setCustomIm(newIm);
    setCustomExpr(newExpr);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Basis index slider ----

  function handleBasisIndexChange(idx: number) {
    const clamped = Math.min(Math.max(idx, 0), d - 1);
    const newRe = Array(d).fill(0);
    newRe[clamped] = 1;
    const newIm = Array(d).fill(0);
    const newExpr = Array(d).fill("0");
    newExpr[clamped] = "1";

    setBasisIndex(clamped);
    setPsiRe(newRe);
    setPsiIm(newIm);
    setCustomRe(newRe);
    setCustomIm(newIm);
    setCustomExpr(newExpr);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Coherent state parameter changes ----

  function handleCoherentChange(kind: "q" | "p", value: number) {
    const q = kind === "q" ? Math.min(Math.max(value, 0), d - 1) : cohQ;
    const p = kind === "p" ? Math.min(Math.max(value, 0), d - 1) : cohP;

    setCohQ(q);
    setCohP(p);

    const coh = buildCoherentState(d, q, p);
    const newRe = coh.re;
    const newIm = coh.im;
    const newExpr = newRe.map((val, i) => {
      const im = newIm[i];
      if (Math.abs(val) < 1e-12 && Math.abs(im) < 1e-12) return "0";
      if (Math.abs(im) < 1e-12) return `${val}`;
      return `${val}+i*${im}`;
    });

    setPsiRe(newRe);
    setPsiIm(newIm);
    setCustomRe(newRe);
    setCustomIm(newIm);
    setCustomExpr(newExpr);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Custom amplitude expression editing ----

  function handlePsiExpressionChange(q: number, expr: string) {
    const exprArr = [...customExpr];
    exprArr[q] = expr;
    setCustomExpr(exprArr);

    const parsed = parseComplexExpression(expr);
    if (!parsed) return;

    const newCustomRe = [...customRe];
    const newCustomIm = [...customIm];
    newCustomRe[q] = parsed.re;
    newCustomIm[q] = parsed.im;
    setCustomRe(newCustomRe);
    setCustomIm(newCustomIm);

    setPsiRe([...newCustomRe]);
    setPsiIm([...newCustomIm]);
    setPlaying(false);
    setSimData(null);
  }

  // ---- Custom gate matrix expression editing ----

  function handleGateExpressionChange(
    i: number,
    j: number,
    expr: string
  ) {
    setGateExpr((prev) => {
      const m = prev.map((row) => [...row]);
      m[i][j] = expr;
      return m;
    });

    const parsed = parseComplexExpression(expr);
    if (!parsed) return;

    setGateRe((prev) => {
      const m = prev.map((row) => [...row]);
      m[i][j] = parsed.re;
      return m;
    });
    setGateIm((prev) => {
      const m = prev.map((row) => [...row]);
      m[i][j] = parsed.im;
      return m;
    });
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

  // ---- Single-shot gate application (preset & custom) ----

  async function handleApplyGate(
    gate: GateName,
    U?: ComplexNumber[][]
  ) {
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
        U,
      });

      // new current state
      const newRe = res.psi.map((c) => c.re);
      const newIm = res.psi.map((c) => c.im);
      setPsiRe(newRe);
      setPsiIm(newIm);

      // Wigner of resulting state
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

  async function handleApplyCustomGate() {
    const U: ComplexNumber[][] = gateRe.map((row, i) =>
      row.map((re, j) => ({
        re,
        im: gateIm[i]?.[j] ?? 0,
      }))
    );

    await handleApplyGate("custom" as GateName, U);
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

  const presetGates: GateName[] = ["X", "Y", "Z", "F", "T"];

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
              Gate mode (X, Y, Z, F, T, custom)
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
            <option value="coherent">Coherent state |q,p⟩</option>
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

          {initialType === "coherent" && (
            <div style={{ marginTop: 8 }}>
              <div style={{ ...smallTextStyle, marginBottom: 4 }}>
                Coherent states |q,p⟩ = D₍q,p₎ |0⟩, with |0⟩ the q = 0
                computational basis state.
              </div>
              <div style={{ marginBottom: 8 }}>
                <label style={{ ...labelStyle, fontSize: 12 }}>
                  q (position-like): {cohQ}
                </label>
                <input
                  type="range"
                  min={0}
                  max={d - 1}
                  value={cohQ}
                  onChange={(e) =>
                    handleCoherentChange("q", parseInt(e.target.value, 10))
                  }
                  style={rangeStyle}
                />
              </div>
              <div>
                <label style={{ ...labelStyle, fontSize: 12 }}>
                  p (momentum-like): {cohP}
                </label>
                <input
                  type="range"
                  min={0}
                  max={d - 1}
                  value={cohP}
                  onChange={(e) =>
                    handleCoherentChange("p", parseInt(e.target.value, 10))
                  }
                  style={rangeStyle}
                />
              </div>
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
                Specify amplitudes ψ = Σ (a₍q₎ + i b₍q₎) |q⟩ using
                expressions, e.g. <code>exp(i*pi/3)/sqrt(2)</code>,{" "}
                <code>e^(2)/sqrt(3)</code>, or <code>1/2+ i*sqrt(3)/2</code>.
                The state is normalized on the
                backend.
              </div>
              {Array.from({ length: d }).map((_, q) => (
                <div
                  key={q}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 4,
                    marginBottom: 4,
                    flexWrap: "wrap",
                  }}
                >
                  <span style={{ fontSize: 12 }}>|{q}⟩:</span>

                  {/* Expression input */}
                  <input
                    type="text"
                    value={customExpr[q] ?? ""}
                    onChange={(e) =>
                      handlePsiExpressionChange(q, e.target.value)
                    }
                    placeholder="e^(i*pi/3)/sqrt(2)"
                    style={{
                      minWidth: 180,
                      padding: "2px 4px",
                      borderRadius: 4,
                      border: "1px solid #4b5563",
                      background: "#020617",
                      color: "#e5e7eb",
                      fontSize: 12,
                    }}
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
            <label style={labelStyle}>Preset gates</label>
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
                marginBottom: 4,
              }}
            >
              {presetGates.map((g) => (
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
            <div style={{ ...smallTextStyle, marginBottom: 8 }}>
              Each gate is applied to the current state ψ, and W(q,p)
              updates.
            </div>

            {/* Custom gate editor */}
            <div style={{ marginTop: 8 }}>
              <label style={labelStyle}>Custom unitary U (d × d)</label>
              <div style={{ ...smallTextStyle, marginBottom: 4 }}>
                Edit U₍i,j₎ as expressions, e.g.{" "}
                <code>exp(i*pi/3)/sqrt(2)</code>,{" "}
                <code>1/sqrt(2)</code>, or <code>1/3 + i*sqrt(3)/2</code>.
                No unitarity check is enforced; make sure U is unitary if
                you care.
              </div>
              <div
                style={{
                  maxHeight: 160,
                  overflowY: "auto",
                  border: "1px solid #4b5563",
                  borderRadius: 4,
                  padding: 4,
                }}
              >
                {Array.from({ length: d }).map((_, i) => (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      gap: 4,
                      marginBottom: 4,
                      alignItems: "center",
                    }}
                  >
                    <span style={{ fontSize: 10 }}>row {i}</span>
                    {Array.from({ length: d }).map((_, j) => (
                      <div
                        key={j}
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: 2,
                        }}
                      >
                        {/* Expression field */}
                        <input
                          type="text"
                          value={gateExpr[i]?.[j] ?? ""}
                          onChange={(e) =>
                            handleGateExpressionChange(
                              i,
                              j,
                              e.target.value
                            )
                          }
                          placeholder={i === j ? "1" : "0"}
                          style={{
                            width: 90,
                            padding: "1px 3px",
                            borderRadius: 4,
                            border: "1px solid #4b5563",
                            background: "#020617",
                            color: "#e5e7eb",
                            fontSize: 10,
                          }}
                        />
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <button
                onClick={handleApplyCustomGate}
                disabled={loading}
                style={{
                  ...buttonStyle,
                  marginTop: 8,
                  background: "#facc15",
                  color: "#1f2937",
                  opacity: loading ? 0.6 : 1,
                }}
              >
                {loading ? "Applying..." : "Apply custom unitary"}
              </button>
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
          in time; gate mode applies preset unitaries (X, Y, Z, F, T) or a
          custom unitary U directly to ψ. Expressions like{" "}
          <code>e^(i*pi/3)</code>, <code>0.5 + i*sqrt(3)/2</code>, or{" "}
          <code>Exp[I*Pi/3]</code> are supported for both the state and
          the unitary.
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

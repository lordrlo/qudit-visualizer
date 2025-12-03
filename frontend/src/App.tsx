// src/App.tsx
import React, { useEffect, useRef, useState } from "react";
import "./App.css";
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

type EvolutionMode = "gate" | "continuous";
type HamiltonianType = "diagonal_quadratic" | "custom";

const DEFAULT_T_MAX = 10;
const DEFAULT_STEPS = 201;

function parseComplexExpression(
  expr: string
): { re: number; im: number } | null {
  let s = expr.trim();
  if (!s) return null;

  // single convention: i, pi, exp, sqrt, ...
  s = s.replace(/\bI\b/g, "i");
  s = s.replace(/e\^\s*\(/g, "exp("); // e^(...) -> exp(...)

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
      if (typeof v === "number") return { re: v, im: 0 };
    }
    return null;
  } catch (e) {
    console.error("Failed to parse complex expression:", expr, "→", s, e);
    return null;
  }
}

// Coherent state |q,p> = D_{q,p}|0>, with |0> the q=0 basis state.
// Using D_{q,p} ~ Z^p X^q -> localized at q with phase exp(2π i p q / d)
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
  // --- Simulation-setup "config" (offline until Generate is pressed) ---

  const [configD, setConfigD] = useState<number>(3);
  const [configDInput, setConfigDInput] = useState<string>("3");
  const [configInitialType, setConfigInitialType] =
    useState<InitialStateType>("basis");
  const [configBasisIndex, setConfigBasisIndex] = useState<number>(0);
  const [configCohQ, setConfigCohQ] = useState<number>(0);
  const [configCohP, setConfigCohP] = useState<number>(0);
  const [configCustomRe, setConfigCustomRe] = useState<number[]>([1, 0, 0]);
  const [configCustomIm, setConfigCustomIm] = useState<number[]>([0, 0, 0]);
  const [configCustomExpr, setConfigCustomExpr] = useState<string[]>([
    "1",
    "0",
    "0",
  ]);

  // --- "Current" system actually being simulated / visualized ---

  const [currentD, setCurrentD] = useState<number>(3);
  const [psiRe, setPsiRe] = useState<number[]>([1, 0, 0]);
  const [psiIm, setPsiIm] = useState<number[]>([0, 0, 0]);

  const [simData, setSimData] = useState<SimulationResponse | null>(null);
  const [frame, setFrame] = useState<number>(0);
  const [playing, setPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Evolution controls
  const [evoMode, setEvoMode] = useState<EvolutionMode>("gate");
  const [hamiltonianType, setHamiltonianType] =
    useState<HamiltonianType>("diagonal_quadratic");
  const [tMax, setTMax] = useState<number>(DEFAULT_T_MAX);

  // Gate + Hamiltonian matrices (always sized to currentD)
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
  const [customGateMode, setCustomGateMode] = useState<boolean>(false);

  const [HRe, setHRe] = useState<number[][]>([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]);
  const [HIm, setHIm] = useState<number[][]>([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ]);
  const [HExpr, setHExpr] = useState<string[][]>([
    ["0", "0", "0"],
    ["0", "0", "0"],
    ["0", "0", "0"],
  ]);

  const lastTimeRef = useRef<number | null>(null);
  const frameIdRef = useRef<number | null>(null);

  const nSteps = simData?.ts.length ?? 0;

  // --- Resize gate & H matrices whenever currentD changes ---

  useEffect(() => {
    const idRe = Array.from({ length: currentD }, (_, i) =>
      Array.from({ length: currentD }, (_, j) => (i === j ? 1 : 0))
    );
    const idIm = Array.from({ length: currentD }, () =>
      Array.from({ length: currentD }, () => 0)
    );
    const expr = Array.from({ length: currentD }, (_, i) =>
      Array.from({ length: currentD }, (_, j) => (i === j ? "1" : "0"))
    );
    setGateRe(idRe);
    setGateIm(idIm);
    setGateExpr(expr);
  }, [currentD]);

  useEffect(() => {
    const hre = Array.from({ length: currentD }, () =>
      Array.from({ length: currentD }, () => 0)
    );
    const him = Array.from({ length: currentD }, () =>
      Array.from({ length: currentD }, () => 0)
    );
    const hexpr = Array.from({ length: currentD }, () =>
      Array.from({ length: currentD }, () => "0")
    );
    setHRe(hre);
    setHIm(him);
    setHExpr(hexpr);
  }, [currentD]);

  // --- Helper: backend preview from arbitrary ψ (t_max = 0) ---

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

  // --- Initialize with |0> in d=3 ---

  useEffect(() => {
    previewFromPsi(currentD, psiRe, psiIm);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- Keep ψ(q) in sync with current frame of simData ---

  useEffect(() => {
    if (!simData || !simData.psi || simData.psi.length === 0) return;
    const n = simData.psi.length;
    const idx = Math.min(Math.max(Math.floor(frame), 0), n - 1);
    const psi_t = simData.psi[idx];
    if (!psi_t || psi_t.length !== currentD) return;
    setPsiRe(psi_t.map((c) => c.re));
    setPsiIm(psi_t.map((c) => c.im));
  }, [simData, frame, currentD]);

  // --- Animation loop (continuous evolution only) ---

  useEffect(() => {
    if (
      !playing ||
      evoMode !== "continuous" ||
      !simData ||
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
        const totalTime = simData.ts[simData.ts.length - 1] || 1;
        const fps = (simData.ts.length - 1) / totalTime;
        const dFrame = dt * fps;

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
  }, [playing, speed, simData, evoMode]);

  // --- Simulation-setup handlers (CONFIG ONLY) ---

  function recomputeConfigStateFor(
    nd: number,
    initType: InitialStateType,
    basisIdx: number,
    cohQ: number,
    cohP: number,
    currRe: number[],
    currIm: number[],
    currExpr: string[]
  ) {
    let newRe: number[] = [];
    let newIm: number[] = [];
    let newExpr: string[] = [];

    if (initType === "basis") {
      const idx = Math.min(basisIdx, nd - 1);
      newRe = Array(nd).fill(0);
      newRe[idx] = 1;
      newIm = Array(nd).fill(0);
      newExpr = Array(nd).fill("0");
      newExpr[idx] = "1";
      return {
        newRe,
        newIm,
        newExpr,
        newBasis: idx,
        newCohQ: cohQ,
        newCohP: cohP,
      };
    }

    if (initType === "equal_superposition") {
      const amp = 1 / Math.sqrt(nd);
      newRe = Array(nd).fill(amp);
      newIm = Array(nd).fill(0);
      newExpr = Array(nd).fill(`1/sqrt(${nd})`);
      return {
        newRe,
        newIm,
        newExpr,
        newBasis: basisIdx,
        newCohQ: cohQ,
        newCohP: cohP,
      };
    }

    if (initType === "coherent") {
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
      return {
        newRe,
        newIm,
        newExpr,
        newBasis: basisIdx,
        newCohQ: q,
        newCohP: p,
      };
    }

    // custom: keep as much as possible
    newRe = currRe.slice(0, nd);
    newIm = currIm.slice(0, nd);
    newExpr = currExpr.slice(0, nd);
    while (newRe.length < nd) newRe.push(0);
    while (newIm.length < nd) newIm.push(0);
    while (newExpr.length < nd) newExpr.push("0");

    return {
      newRe,
      newIm,
      newExpr,
      newBasis: Math.min(basisIdx, nd - 1),
      newCohQ: Math.min(cohQ, nd - 1),
      newCohP: Math.min(cohP, nd - 1),
    };
  }

   function handleConfigDimensionChangeInput(value: string) {
    // update the text in the box regardless, so typing "1", "10" works
    setConfigDInput(value);

    const nd = parseInt(value, 10);
    if (Number.isNaN(nd) || nd < 2) {
      // don't touch configD yet; wait until it's a valid d ≥ 2
      return;
    }

    if (nd === configD) return;

    const {
      newRe,
      newIm,
      newExpr,
      newBasis,
      newCohQ,
      newCohP,
    } = recomputeConfigStateFor(
      nd,
      configInitialType,
      configBasisIndex,
      configCohQ,
      configCohP,
      configCustomRe,
      configCustomIm,
      configCustomExpr
    );

    setConfigD(nd);
    setConfigBasisIndex(newBasis);
    setConfigCohQ(newCohQ);
    setConfigCohP(newCohP);
    setConfigCustomRe(newRe);
    setConfigCustomIm(newIm);
    setConfigCustomExpr(newExpr);
  }

  function handleConfigInitialTypeChange(newType: InitialStateType) {
    const {
      newRe,
      newIm,
      newExpr,
      newBasis,
      newCohQ,
      newCohP,
    } = recomputeConfigStateFor(
      configD,
      newType,
      configBasisIndex,
      configCohQ,
      configCohP,
      configCustomRe,
      configCustomIm,
      configCustomExpr
    );

    setConfigInitialType(newType);
    setConfigBasisIndex(newBasis);
    setConfigCohQ(newCohQ);
    setConfigCohP(newCohP);
    setConfigCustomRe(newRe);
    setConfigCustomIm(newIm);
    setConfigCustomExpr(newExpr);
  }

  function handleConfigBasisIndexChange(idx: number) {
    const clamped = Math.min(Math.max(idx, 0), configD - 1);
    const newRe = Array(configD).fill(0);
    newRe[clamped] = 1;
    const newIm = Array(configD).fill(0);
    const newExpr = Array(configD).fill("0");
    newExpr[clamped] = "1";

    setConfigBasisIndex(clamped);
    setConfigCustomRe(newRe);
    setConfigCustomIm(newIm);
    setConfigCustomExpr(newExpr);
  }

  function handleConfigCoherentChange(kind: "q" | "p", value: number) {
    const q =
      kind === "q"
        ? Math.min(Math.max(value, 0), configD - 1)
        : configCohQ;
    const p =
      kind === "p"
        ? Math.min(Math.max(value, 0), configD - 1)
        : configCohP;

    const coh = buildCoherentState(configD, q, p);
    const newRe = coh.re;
    const newIm = coh.im;
    const newExpr = newRe.map((val, i) => {
      const im = newIm[i];
      if (Math.abs(val) < 1e-12 && Math.abs(im) < 1e-12) return "0";
      if (Math.abs(im) < 1e-12) return `${val}`;
      return `${val}+i*${im}`;
    });

    setConfigCohQ(q);
    setConfigCohP(p);
    setConfigCustomRe(newRe);
    setConfigCustomIm(newIm);
    setConfigCustomExpr(newExpr);
  }

  function handleConfigPsiExpressionChange(q: number, expr: string) {
    const exprArr = [...configCustomExpr];
    exprArr[q] = expr;
    setConfigCustomExpr(exprArr);

    const parsed = parseComplexExpression(expr);
    if (!parsed) return;

    const newRe = [...configCustomRe];
    const newIm = [...configCustomIm];
    newRe[q] = parsed.re;
    newIm[q] = parsed.im;
    setConfigCustomRe(newRe);
    setConfigCustomIm(newIm);
  }

  // --- "Generate" button: apply config to current system & preview ---

  async function handleGenerate() {
    setLoading(true);
    setErrorMsg(null);
    setPlaying(false);
    setFrame(0);

    const dim = configD; // validated ≥ 2
    const reArr = configCustomRe.slice(0, dim);
    const imArr = configCustomIm.slice(0, dim);

    setCurrentD(dim);
    setPsiRe(reArr);
    setPsiIm(imArr);

    try {
      await previewFromPsi(dim, reArr, imArr);
    } finally {
      setLoading(false);
    }
  }

  // --- Evolution-card handlers (current system) ---

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

  function handleHamiltonianExpressionChange(
    i: number,
    j: number,
    expr: string
  ) {
    setHExpr((prev) => {
      const m = prev.map((row) => [...row]);
      m[i][j] = expr;
      return m;
    });

    const parsed = parseComplexExpression(expr);
    if (!parsed) return;

    setHRe((prev) => {
      const m = prev.map((row) => [...row]);
      m[i][j] = parsed.re;
      return m;
    });
    setHIm((prev) => {
      const m = prev.map((row) => [...row]);
      m[i][j] = parsed.im;
      return m;
    });
  }

  async function handleRunContinuous() {
    setLoading(true);
    setErrorMsg(null);
    setPlaying(false);
    setFrame(0);
    try {
      const dim = currentD;
      const psi_custom: ComplexNumber[] = Array.from({ length: dim }).map(
        (_, q) => ({
          re: psiRe[q] ?? 0,
          im: psiIm[q] ?? 0,
        })
      );
      const tEnd = tMax > 0 ? tMax : DEFAULT_T_MAX;

      let hamiltonian: "diagonal_quadratic" | "custom" =
        "diagonal_quadratic";
      let H_custom: ComplexNumber[][] | undefined;

      if (hamiltonianType === "custom") {
        hamiltonian = "custom";
        H_custom = HRe.map((row, i) =>
          row.map((re, j) => ({
            re,
            im: HIm[i]?.[j] ?? 0,
          }))
        );
      }

      const res = await runSimulation({
        d: dim,
        hamiltonian,
        initial_state: "custom",
        basis_index: 0,
        t_max: tEnd,
        n_steps: DEFAULT_STEPS,
        psi_custom,
        H_custom,
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

  async function handleApplyGate(
    gate: GateName,
    U?: ComplexNumber[][]
  ) {
    setLoading(true);
    setErrorMsg(null);
    setPlaying(false);
    try {
      const dim = currentD;
      const psi: ComplexNumber[] = Array.from({ length: dim }).map(
        (_, q) => ({
          re: psiRe[q] ?? 0,
          im: psiIm[q] ?? 0,
        })
      );

      const res = await applyGate({
        d: dim,
        gate,
        psi,
        U,
      });

      const newRe = res.psi.map((c) => c.re);
      const newIm = res.psi.map((c) => c.im);
      setPsiRe(newRe);
      setPsiIm(newIm);

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

  // --- Derived values for visualization ---

  const currentIndex =
    simData && simData.ts.length > 0
      ? Math.min(Math.max(Math.floor(frame), 0), simData.ts.length - 1)
      : 0;

  const currentW =
    simData && simData.W.length > 0 ? simData.W[currentIndex] : null;

  const currentT =
    simData && simData.ts.length > 0 ? simData.ts[currentIndex] : 0;

  const presetGates: GateName[] = ["X", "Y", "Z", "F", "T"];

  // --- JSX ---

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1 className="app-title">Discrete Wigner Lab</h1>
          <p className="app-subtitle">
            Qudit phase-space visualizer: prepare a state, then evolve it via
            gates or a Hamiltonian.
          </p>
        </div>
        <div className="header-badges">
          <span className="badge">Dynamiqs · FastAPI</span>
          <span className="badge badge-soft">React · TypeScript</span>
        </div>
      </header>

      <div className="app-shell">
        {/* LEFT: setup + evolution */}
        <aside className="sidebar">
          {/* Simulation setup */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Simulation setup</h2>
            </div>
            <div className="panel-body">
              <div className="field">
                <label className="field-label">Dimension d</label>
                <input type="number"
                  min={2}
                  value={configDInput}
                  onChange={(e) => handleConfigDimensionChangeInput(e.target.value)}
                  className="input input-small"
                />
                <p className="field-helper">
                  Choose any integer d ≥ 2 (odd or even). Larger d makes the
                  simulation heavier.
                </p>
              </div>

              <div className="field">
                <label className="field-label">Initial state</label>
                <select
                  value={configInitialType}
                  onChange={(e) =>
                    handleConfigInitialTypeChange(
                      e.target.value as InitialStateType
                    )
                  }
                  className="select"
                >
                  <option value="basis">Computational basis |i⟩</option>
                  <option value="equal_superposition">
                    Equal superposition
                  </option>
                  <option value="coherent">Coherent state |q,p⟩</option>
                  <option value="custom">Custom amplitudes</option>
                </select>
              </div>

              {configInitialType === "basis" && (
                <div className="field">
                  <label className="field-label">
                    i index:{" "}
                    <span className="mono">{configBasisIndex}</span>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={configD - 1}
                    value={configBasisIndex}
                    onChange={(e) =>
                      handleConfigBasisIndexChange(
                        parseInt(e.target.value, 10)
                      )
                    }
                    className="slider"
                  />
                </div>
              )}

              {configInitialType === "coherent" && (
                <div className="coherent-grid">
                  <div className="field">
                    <label className="field-label">
                      q (position-like):{" "}
                      <span className="mono">{configCohQ}</span>
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={configD - 1}
                      value={configCohQ}
                      onChange={(e) =>
                        handleConfigCoherentChange(
                          "q",
                          parseInt(e.target.value, 10)
                        )
                      }
                      className="slider"
                    />
                  </div>
                  <div className="field">
                    <label className="field-label">
                      p (momentum-like):{" "}
                      <span className="mono">{configCohP}</span>
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={configD - 1}
                      value={configCohP}
                      onChange={(e) =>
                        handleConfigCoherentChange(
                          "p",
                          parseInt(e.target.value, 10)
                        )
                      }
                      className="slider"
                    />
                  </div>
                  <p className="field-helper">
                    Coherent states |q,p⟩ = D<sub>q,p</sub>|0⟩ with |0⟩ the
                    q = 0 computational basis state.
                  </p>
                </div>
              )}

              {configInitialType === "custom" && (
                <div className="field">
                  <p className="field-helper">
                    Specify amplitudes ψ = Σ ψ<sub>i</sub> |i⟩ using mathjs expressions, e.g.{" "}
                    <code>exp(i*pi/3)/sqrt(2)</code>,{" "}
                    <code>cos(2pi) + 3i</code>. The state is normalized
                    on the backend when you press{" "}
                    <span className="mono">Generate</span>.
                  </p>
                  <div className="scroll-area">
                    {Array.from({ length: configD }).map((_, q) => (
                      <div className="amp-row" key={q}>
                        <span className="amp-label">|{q}⟩</span>
                        <input
                          type="text"
                          value={configCustomExpr[q] ?? ""}
                          onChange={(e) =>
                            handleConfigPsiExpressionChange(q, e.target.value)
                          }
                          placeholder="exp(i*pi/3)/sqrt(2)"
                          className="input input-compact"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <button
                onClick={handleGenerate}
                disabled={loading}
                className={`btn btn-primary ${
                  loading ? "btn-disabled" : ""
                }`}
              >
                {loading ? "Generating..." : "Generate"}
              </button>
            </div>
          </section>

          {/* Evolution card */}
          <section className="panel">
            <div className="panel-header">
              <h2 className="panel-title">Evolution</h2>
              <button
                className="btn btn-ghost"
                onClick={() => {
                  setEvoMode((m) => (m === "gate" ? "continuous" : "gate"));
                  setPlaying(false);
                }}
              >
                {evoMode === "gate" ? "Continuous" : "Gate"}
              </button>
            </div>
            <div className="panel-body">
              {evoMode === "gate" ? (
                <>
                  <label className="field-label">Gate sequence</label>
                  <div className="gate-grid">
                    {presetGates.map((g) => (
                      <button
                        key={g}
                        className="btn btn-ghost"
                        disabled={loading}
                        onClick={() => handleApplyGate(g)}
                      >
                        {g}
                      </button>
                    ))}
                    <button
                      className={
                        customGateMode
                          ? "btn btn-secondary"
                          : "btn btn-ghost"
                      }
                      onClick={() =>
                        setCustomGateMode((on) => !on)
                      }
                    >
                      Custom
                    </button>
                  </div>
                  <p className="field-helper">
                    Each gate is applied to the current state ψ, and W(q,p)
                    updates.
                  </p>

                  {customGateMode && (
                    <div className="field">
                      <label className="field-label">
                        Custom unitary U (d × d)
                      </label>
                      <p className="field-helper">
                        Edit U(i,j) as expressions, e.g.{" "}
                        <code>exp(i*pi/3)/sqrt(2)</code>,{" "}
                        <code>1/sqrt(2)</code>. No unitarity check is
                        enforced.
                      </p>
                      <div className="scroll-area">
                        {Array.from({ length: currentD }).map((_, i) => (
                          <div className="matrix-row" key={i}>
                            {Array.from({ length: currentD }).map((_, j) => (
                              <input
                                key={j}
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
                                className="input input-compact"
                              />
                            ))}
                          </div>
                        ))}
                      </div>
                      <button
                        onClick={handleApplyCustomGate}
                        disabled={loading}
                        className={`btn btn-warning ${
                          loading ? "btn-disabled" : ""
                        }`}
                      >
                        {loading ? "Applying..." : "Apply"}
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <>
                  <div className="field">
                    <label className="field-label">Hamiltonian</label>
                    <div className="gate-grid">
                      <button
                        className={
                          hamiltonianType === "diagonal_quadratic"
                            ? "btn btn-secondary"
                            : "btn btn-ghost"
                        }
                        onClick={() =>
                          setHamiltonianType("diagonal_quadratic")
                        }
                      >
                        H<sub>ij</sub> = i²/d  δ<sub>ij</sub>
                      </button>
                      <button
                        className={
                          hamiltonianType === "custom"
                            ? "btn btn-secondary"
                            : "btn btn-ghost"
                        }
                        onClick={() =>
                          setHamiltonianType("custom")
                        }
                      >
                        Custom
                      </button>
                    </div>
                  </div>

                  <div className="field">
                    <label className="field-label">Final time tₘₐₓ</label>
                    <input
                      type="number"
                      value={tMax}
                      min={0}
                      step={0.1}
                      onChange={(e) => {
                        const v = parseFloat(e.target.value);
                        setTMax(Number.isFinite(v) ? v : DEFAULT_T_MAX);
                      }}
                      className="input input-small"
                    />
                  </div>

                  {hamiltonianType === "custom" && (
                    <div className="field">
                      <p className="field-helper">
                        Edit H(q,q&apos;) as expressions, e.g.{" "}
                        <code>0</code>, <code>1</code>,{" "}
                        <code>exp(i*pi/3)</code>. No Hermiticity check is
                        enforced; you probably want H = H†.
                      </p>
                      <div className="scroll-area">
                        {Array.from({ length: currentD }).map((_, i) => (
                          <div className="matrix-row" key={i}>
                            {Array.from({ length: currentD }).map((_, j) => (
                              <input
                                key={j}
                                type="text"
                                value={HExpr[i]?.[j] ?? ""}
                                onChange={(e) =>
                                  handleHamiltonianExpressionChange(
                                    i,
                                    j,
                                    e.target.value
                                  )
                                }
                                placeholder="0"
                                className="input input-compact"
                              />
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <button
                    onClick={handleRunContinuous}
                    disabled={loading}
                    className={`btn btn-primary ${
                      loading ? "btn-disabled" : ""
                    }`}
                  >
                    {loading ? "Evolving..." : "Run continuous evolution"}
                  </button>

                  {simData && nSteps > 1 && (
                    <>
                      <div className="field field-inline">
                        <label className="field-label">Time</label>
                        <span className="time-chip">
                          t = {currentT.toFixed(3)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={nSteps - 1}
                        value={currentIndex}
                        onChange={(e) =>
                          setFrame(parseInt(e.target.value, 10))
                        }
                        className="slider"
                      />
                      <div className="playbar">
                        <button
                          onClick={() => setPlaying((p) => !p)}
                          className="btn btn-secondary"
                        >
                          {playing ? "Pause" : "Play"}
                        </button>
                        <div className="playbar-speed">
                          <span>Speed</span>
                          <input
                            type="range"
                            min={0.1}
                            max={4}
                            step={0.1}
                            value={speed}
                            onChange={(e) =>
                              setSpeed(parseFloat(e.target.value))
                            }
                            className="slider slider-thin"
                          />
                        </div>
                      </div>
                    </>
                  )}
                </>
              )}

              {errorMsg && (
                <div className="error-box" style={{ marginTop: 8 }}>
                  {errorMsg}
                </div>
              )}
            </div>
          </section>
        </aside>

        {/* RIGHT: Wigner + current state */}
        <main className="main">
          <div className="visual-panel">
            <div className="visual-header">
              <div>
                <h2 className="visual-title">Wigner heatmap</h2>
                <p className="visual-subtitle">
                  Discrete Wigner function W(q,p) = (1/d) Tr[ρ A<sub>q,p</sub>].
                  Hover over the grid to inspect cell values.
                </p>
              </div>
              <div className="visual-meta">
                <span className="chip">d = {currentD}</span>
                {evoMode === "continuous" ? (
                  <span className="chip chip-outline">
                    t = {currentT.toFixed(3)}
                  </span>
                ) : (
                  <span className="chip chip-outline">Gate mode</span>
                )}
              </div>
            </div>

            <div className="visual-body">
              <div className="visual-layout">
                <div className="visual-heatmap">
                  {currentW ? (
                    <WignerHeatmap W={currentW} />
                  ) : (
                    <div className="visual-placeholder">
                      Press <span className="mono">Generate</span>, apply a
                      gate, or run a continuous evolution to see W(q,p).
                    </div>
                  )}
                </div>
                <div className="visual-current-state">
                  <div className="field">
                    <label className="field-label">Current state ψ(i)</label>
                    <div className="state-box state-box-table">
                      <table className="state-table">
                        <thead>
                          <tr>
                            <th>q</th>
                            <th>Re&nbsp;ψ<sub>i</sub></th>
                            <th>Im&nbsp;ψ<sub>i</sub></th>
                            <th>|ψ<sub>i</sub>|²</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Array.from({ length: currentD }).map((_, q) => {
                            const re = psiRe[q] ?? 0;
                            const im = psiIm[q] ?? 0;
                            const prob = re * re + im * im;
                            return (
                              <tr key={q}>
                                <td className="mono">{q}</td>
                                <td className="mono">{re.toFixed(3)}</td>
                                <td className="mono">{im.toFixed(3)}</td>
                                <td className="mono">{prob.toFixed(3)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

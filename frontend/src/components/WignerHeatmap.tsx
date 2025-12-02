// src/components/WignerHeatmap.tsx
import React, { useEffect, useRef, useState } from "react";

interface Props {
  W: number[][];
}

// simple diverging colormap: negative -> blue, zero -> white, positive -> red
function colorForValue(v: number, vmin: number, vmax: number): string {
  if (vmax === vmin) return "rgb(255,255,255)";
  const maxAbs = Math.max(Math.abs(vmin), Math.abs(vmax)) || 1e-6;
  const x = v / maxAbs; // in [-1,1]

  if (x >= 0) {
    const r = 255;
    const g = Math.round(255 * (1 - x));
    const b = Math.round(255 * (1 - x));
    return `rgb(${r},${g},${b})`;
  } else {
    const y = -x;
    const r = Math.round(255 * (1 - y));
    const g = Math.round(255 * (1 - y));
    const b = 255;
    return `rgb(${r},${g},${b})`;
  }
}

type HoverCell = { q: number; p: number } | null;

export const WignerHeatmap: React.FC<Props> = ({ W }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sizeRef = useRef<number>(0); // actual drawing size in px
  const d = W.length;

  const [hoverCell, setHoverCell] = useState<HoverCell>(null);
  const [hoverValue, setHoverValue] = useState<number | null>(null);

  // redraw heatmap whenever W or d changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parentWidth = canvas.parentElement?.clientWidth ?? 500;
    const size = Math.min(parentWidth, 500);
    canvas.width = size;
    canvas.height = size;
    sizeRef.current = size;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let vmin = Infinity;
    let vmax = -Infinity;
    for (let q = 0; q < d; q++) {
      for (let p = 0; p < d; p++) {
        const v = W[q][p];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }

    const cell = size / d;
    for (let q = 0; q < d; q++) {
      for (let p = 0; p < d; p++) {
        ctx.fillStyle = colorForValue(W[q][p], vmin, vmax);
        // q vertical (bottom-up), p horizontal (left-right)
        ctx.fillRect(p * cell, (d - 1 - q) * cell, cell, cell);
      }
    }

    // grid lines
    ctx.strokeStyle = "rgba(0,0,0,0.3)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= d; i++) {
      const xy = i * cell;
      ctx.beginPath();
      ctx.moveTo(xy, 0);
      ctx.lineTo(xy, size);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, xy);
      ctx.lineTo(size, xy);
      ctx.stroke();
    }
  }, [W, d]);

  // whenever W changes and we have a hovered cell, update its value
  useEffect(() => {
    if (!hoverCell) {
      setHoverValue(null);
      return;
    }
    const { q, p } = hoverCell;
    if (q >= 0 && q < d && p >= 0 && p < d) {
      setHoverValue(W[q][p]);
    } else {
      setHoverValue(null);
    }
  }, [W, d, hoverCell]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const size = sizeRef.current;
    if (!canvas || size === 0) return;

    const rect = canvas.getBoundingClientRect();

    // account for potential CSS scaling
    const scaleX = size / rect.width;
    const scaleY = size / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    const cell = size / d;
    const p = Math.floor(x / cell);
    const qFromTop = Math.floor(y / cell);
    const q = d - 1 - qFromTop; // invert vertical index

    if (p < 0 || p >= d || q < 0 || q >= d) {
      setHoverCell(null);
      return;
    }

    setHoverCell({ q, p });
  };

  const handleMouseLeave = () => {
    setHoverCell(null);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        style={{
          background: "#ffffff",
          borderRadius: "8px",
          boxShadow: "0 4px 10px rgba(0,0,0,0.4)",
          cursor: "crosshair",
        }}
      />
      <div style={{ marginTop: 8, fontSize: 14, color: "#e5e7eb" }}>
        Discrete Wigner function{" "}
        <span style={{ fontFamily: "monospace" }}>
          W(q,p) = (1/d) Tr[ρ A<sub>q,p</sub>]
        </span>
      </div>
      {hoverCell && hoverValue !== null && (
        <div
          style={{
            marginTop: 4,
            fontSize: 13,
            color: "#e5e7eb",
            fontFamily: "monospace",
          }}
        >
          q = {hoverCell.q}, p = {hoverCell.p}, W(q,p) ={" "}
          {hoverValue.toFixed(5)}
        </div>
      )}
    </div>
  );
};

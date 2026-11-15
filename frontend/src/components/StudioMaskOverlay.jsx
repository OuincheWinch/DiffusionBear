import { useCallback, useEffect, useRef, useState, memo } from "react";
import { useI18n } from "../i18n/I18nContext";
import { CanvasProgressOverlay } from "./ResultCanvas";
import "./StudioMaskOverlay.css";

const MIN_BRUSH = 4;
const MAX_BRUSH = 240;

function StudioMaskOverlay({
  referenceImageUrl,
  aspectRatioWidth = 768,
  aspectRatioHeight = 512,
  maskData = null,
  onChangeMask = null,
  onClose = null,
  busy = false,
  progress = null,
  phase = null,
  phaseDetail = null,
  generatingPrompt = null,
  t: propT = null,
}) {
  const { t: i18nT } = useI18n();
  const t = propT || i18nT;
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const canvasBoxRef = useRef(null);
  const paintingRef = useRef(false);
  const lastPointRef = useRef(null);

  const arW = Number(aspectRatioWidth) || 1;
  const arH = Number(aspectRatioHeight) || 1;

  const [naturalSize, setNaturalSize] = useState({ width: arW, height: arH });
  const maxDim = Math.max(naturalSize.width, naturalSize.height);
  const defaultBrushSize = Math.max(8, Math.round(maxDim * 0.05));
  const preset5 = defaultBrushSize;
  const preset10 = Math.max(16, Math.round(maxDim * 0.10));
  const preset20 = Math.max(32, Math.round(maxDim * 0.20));

  const [brushSize, setBrushSize] = useState(defaultBrushSize);
  const [mode, setMode] = useState("paint"); // "paint" | "erase"
  const [hasPaint, setHasPaint] = useState(Boolean(maskData));
  const [maskPercent, setMaskPercent] = useState(0);

  // Dynamic cursor ring & scale state
  const [cursorPos, setCursorPos] = useState({ x: -1000, y: -1000 });
  const [isHovering, setIsHovering] = useState(false);
  const [isPainting, setIsPainting] = useState(false);
  const [boxScale, setBoxScale] = useState(1);

  // Observe canvas box scale relative to bitmap pixels
  useEffect(() => {
    const box = canvasBoxRef.current;
    const canvas = canvasRef.current;
    if (!box || !canvas) return;
    const updateScale = () => {
      if (canvas.width > 0 && box.clientWidth > 0) {
        setBoxScale(box.clientWidth / canvas.width);
      }
    };
    updateScale();
    const ro = new ResizeObserver(updateScale);
    ro.observe(box);
    return () => ro.disconnect();
  }, [naturalSize]);

  // Export binary mask where white (255) marks regions to regenerate and black (0) marks untouched pixels
  const exportBinaryMask = useCallback((canvas) => {
    if (!canvas || !canvas.width || !canvas.height) return null;
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = canvas.width;
    exportCanvas.height = canvas.height;
    const expCtx = exportCanvas.getContext("2d");
    if (!expCtx) return null;

    // Solid black background (untouched)
    expCtx.fillStyle = "#000000";
    expCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);

    const srcCtx = canvas.getContext("2d");
    if (!srcCtx) return null;
    const srcData = srcCtx.getImageData(0, 0, canvas.width, canvas.height);
    const expData = expCtx.getImageData(0, 0, exportCanvas.width, exportCanvas.height);

    const src = srcData.data;
    const out = expData.data;
    let paintedCount = 0;

    for (let i = 0; i < src.length; i += 4) {
      if (src[i + 3] > 16) {
        out[i] = 255;
        out[i + 1] = 255;
        out[i + 2] = 255;
        out[i + 3] = 255;
        paintedCount += 1;
      }
    }

    if (paintedCount === 0) return null;
    expCtx.putImageData(expData, 0, 0);
    return exportCanvas.toDataURL("image/png");
  }, []);

  const updateCoverage = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas || !canvas.width || !canvas.height) return;

    const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let painted = 0;
    let counted = 0;
    const stride = 4 * 4; // Sample every 4th pixel for high performance
    for (let i = 3; i < data.length; i += stride) {
      counted += 1;
      if (data[i] > 16) painted += 1;
    }
    const pct = counted ? (painted / counted) * 100 : 0;
    setMaskPercent(pct);
    setHasPaint(pct > 0);
  }, []);

  // Initialize canvas size when natural image dimensions are resolved
  const initCanvas = useCallback((w, h) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const needResize = canvas.width !== w || canvas.height !== h;
    if (needResize) {
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.clearRect(0, 0, w, h);
    }

    // If maskData was already provided, restore painted mask strokes
    if (maskData) {
      const img = new Image();
      img.onload = () => {
        try {
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          const temp = document.createElement("canvas");
          temp.width = w;
          temp.height = h;
          const tCtx = temp.getContext("2d");
          if (!tCtx) return;
          tCtx.drawImage(img, 0, 0, w, h);
          const imgData = tCtx.getImageData(0, 0, w, h);
          const d = imgData.data;
          for (let i = 0; i < d.length; i += 4) {
            if (d[i] >= 128) {
              d[i] = 239;     // Crimson red
              d[i + 1] = 68;
              d[i + 2] = 68;
              d[i + 3] = 255; // Solid in backing store
            } else {
              d[i + 3] = 0;   // Transparent
            }
          }
          ctx.putImageData(imgData, 0, 0);
          updateCoverage();
        } catch (e) {
          console.warn("[StudioMaskOverlay] Failed to restore mask data:", e);
        }
      };
      img.src = maskData;
    } else if (needResize) {
      setHasPaint(false);
      setMaskPercent(0);
    }
  }, [maskData, updateCoverage]);

  const handleImageLoaded = useCallback((e) => {
    const img = e.target;
    if (!img) return;
    const nw = img.naturalWidth || arW;
    const nh = img.naturalHeight || arH;
    setNaturalSize({ width: nw, height: nh });
    const newDefaultBrush = Math.max(8, Math.round(Math.max(nw, nh) * 0.05));
    setBrushSize((prev) => prev || newDefaultBrush);
    initCanvas(nw, nh);
  }, [arW, arH, initCanvas]);

  // Handle cached image already complete on mount
  useEffect(() => {
    const img = imgRef.current;
    if (img && img.complete && img.naturalWidth > 0) {
      handleImageLoaded({ target: img });
    }
  }, [referenceImageUrl, handleImageLoaded]);

  const pointerToMask = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  }, []);

  const updateCursorPosition = useCallback((e) => {
    const box = canvasBoxRef.current;
    if (!box) return;
    const rect = box.getBoundingClientRect();
    setCursorPos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  }, []);

  const paintAt = useCallback((e) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    const point = pointerToMask(e);
    if (!ctx || !point) return;

    const last = lastPointRef.current;
    const radius = Math.max(2, brushSize * 0.5);

    ctx.globalCompositeOperation = mode === "erase" ? "destination-out" : "source-over";

    ctx.beginPath();
    if (last) {
      const steps = Math.ceil(
        Math.hypot(point.x - last.x, point.y - last.y) / Math.max(1, radius / 3)
      );
      for (let i = 1; i <= steps; i += 1) {
        const x = last.x + ((point.x - last.x) * i) / steps;
        const y = last.y + ((point.y - last.y) * i) / steps;
        ctx.moveTo(x + radius, y);
        ctx.arc(x, y, radius, 0, Math.PI * 2);
      }
    } else {
      ctx.moveTo(point.x + radius, point.y);
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    }

    if (mode === "paint") {
      ctx.fillStyle = "#ef4444";
    }
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
    lastPointRef.current = point;
    setHasPaint(true);
  }, [brushSize, mode, pointerToMask]);

  function handlePointerDown(e) {
    if (busy || e.button !== 0) return;
    try {
      e.target.setPointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture fails
    }
    paintingRef.current = true;
    setIsPainting(true);
    setIsHovering(true);
    lastPointRef.current = null;
    updateCursorPosition(e);
    paintAt(e);
  }

  function handlePointerMove(e) {
    updateCursorPosition(e);
    if (!paintingRef.current || busy) return;
    paintAt(e);
  }

  function handlePointerUp(e) {
    if (!paintingRef.current) return;
    paintingRef.current = false;
    setIsPainting(false);
    lastPointRef.current = null;
    try {
      if (e.target.hasPointerCapture?.(e.pointerId)) {
        e.target.releasePointerCapture(e.pointerId);
      }
    } catch {
      // Ignore
    }
    updateCoverage();
    const dataUrl = exportBinaryMask(canvasRef.current);
    onChangeMask?.(dataUrl);
  }

  function handlePointerEnter(e) {
    setIsHovering(true);
    updateCursorPosition(e);
  }

  function handlePointerLeave() {
    if (!paintingRef.current) {
      setIsHovering(false);
    }
  }

  function clearMask() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setHasPaint(false);
    setMaskPercent(0);
    onChangeMask?.(null);
  }

  const handleClose = useCallback(() => {
    if (canvasRef.current) {
      updateCoverage();
      const dataUrl = exportBinaryMask(canvasRef.current);
      onChangeMask?.(dataUrl);
    }
    onClose?.();
  }, [exportBinaryMask, onChangeMask, onClose, updateCoverage]);

  // Keyboard shortcuts: B (paint), E (erase), [ / ] (brush size), Escape (close)
  useEffect(() => {
    function onKey(e) {
      const target = e.target;
      const tag = target?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || target?.isContentEditable) {
        return;
      }

      if (e.key === "Escape") {
        handleClose();
      } else if (e.key === "b" || e.key === "B") {
        e.preventDefault();
        setMode("paint");
      } else if (e.key === "e" || e.key === "E") {
        e.preventDefault();
        setMode("erase");
      } else if (e.key === "[") {
        e.preventDefault();
        setBrushSize((prev) => {
          const step = prev > 50 ? 10 : prev > 20 ? 5 : 2;
          return Math.max(MIN_BRUSH, prev - step);
        });
      } else if (e.key === "]") {
        e.preventDefault();
        setBrushSize((prev) => {
          const step = prev >= 50 ? 10 : prev >= 20 ? 5 : 2;
          return Math.min(MAX_BRUSH, prev + step);
        });
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleClose]);

  const curW = naturalSize.width || arW;
  const curH = naturalSize.height || arH;
  const imgAspect = `${curW} / ${curH}`;
  const stageMaxWidth = `min(100%, ${(580 * curW) / curH}px)`;
  const displayDiameter = Math.max(4, Math.round(brushSize * boxScale));

  return (
    <div className="studio-mask-overlay">
      {/* Top Floating Apple-grade Control Bar */}
      <div className="studio-mask-top-bar" role="toolbar" aria-label={t("fill.tools")}>
        <div className="studio-mask-tools" role="group" aria-label={t("fill.tools")}>
          <button
            type="button"
            className={`studio-mask-btn ${mode === "paint" ? "active" : ""}`}
            onClick={() => setMode("paint")}
            title={`${t("fill.paint")} (B)`}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M18.375 2.625a2.121 2.121 0 1 1 3 3L12 15l-4 1 1-4Z"/>
            </svg>
            <span>{t("fill.paint")}</span>
            <kbd className="studio-mask-kbd">B</kbd>
          </button>
          <button
            type="button"
            className={`studio-mask-btn ${mode === "erase" ? "active" : ""}`}
            onClick={() => setMode("erase")}
            title={`${t("fill.erase")} (E)`}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21"/><path d="M22 21H7"/><path d="m5 11 9 9"/>
            </svg>
            <span>{t("fill.erase")}</span>
            <kbd className="studio-mask-kbd">E</kbd>
          </button>
        </div>

        <div className="studio-mask-size-control">
          <span className="studio-mask-size-label" title="Shortcuts: [ to decrease, ] to increase">
            {t("fill.brushSize")} :
          </span>
          <div className="studio-mask-presets" role="group" aria-label="Brush size presets">
            <button
              type="button"
              className={`studio-mask-preset-btn ${brushSize === preset5 ? "active" : ""}`}
              onClick={() => setBrushSize(preset5)}
              title="5% brush diameter"
            >
              5%
            </button>
            <button
              type="button"
              className={`studio-mask-preset-btn ${brushSize === preset10 ? "active" : ""}`}
              onClick={() => setBrushSize(preset10)}
              title="10% brush diameter"
            >
              10%
            </button>
            <button
              type="button"
              className={`studio-mask-preset-btn ${brushSize === preset20 ? "active" : ""}`}
              onClick={() => setBrushSize(preset20)}
              title="20% brush diameter"
            >
              20%
            </button>
          </div>
          <input
            type="range"
            min={MIN_BRUSH}
            max={MAX_BRUSH}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            aria-label={t("fill.brushSize")}
            title="Adjust brush size ([ to decrease, ] to increase)"
          />
          <span className="studio-mask-size-value">{brushSize}px</span>
        </div>

        <div className="studio-mask-status">
          {maskPercent > 0 ? (
            <span
              className="studio-mask-coverage-pill"
              role="status"
              title={t("fill.coverage", { percent: maskPercent.toFixed(1) })}
            >
              <span className="studio-mask-pulse-dot" aria-hidden="true" />
              <span>{t("fill.coverage", { percent: maskPercent.toFixed(1) })}</span>
            </span>
          ) : (
            <span className="studio-mask-empty-pill" title="Drag across reference image to paint inpaint mask">
              🎨 {t("params.ref.paintOnReference")}
            </span>
          )}
        </div>

        <div className="studio-mask-actions">
          {hasPaint && (
            <button
              type="button"
              className="studio-mask-btn danger"
              onClick={clearMask}
              title={t("fill.clear")}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
              </svg>
              <span>{t("fill.clear")}</span>
            </button>
          )}
          <button
            type="button"
            className="studio-mask-btn primary"
            onClick={handleClose}
            title={`${t("canvas.mask.doneTitle")} (Esc)`}
          >
            <span>✓ {t("canvas.mask.done")}</span>
            <kbd className="studio-mask-kbd primary">Esc</kbd>
          </button>
        </div>
      </div>

      {/* Main Studio Canvas Viewport with Backdrop Reference Image */}
      <div className="studio-mask-stage">
        <div
          ref={canvasBoxRef}
          className="studio-mask-canvas-box"
          style={{
            aspectRatio: imgAspect,
            maxWidth: stageMaxWidth,
          }}
        >
          <img
            ref={imgRef}
            src={referenceImageUrl}
            alt="Reference backdrop"
            className="studio-mask-backdrop-img"
            draggable={false}
            onLoad={handleImageLoaded}
            onDragStart={(e) => e.preventDefault()}
          />

          <canvas
            ref={canvasRef}
            className={`studio-mask-paint-canvas mode-${mode} ${isHovering ? "cursor-hidden" : ""}`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
            onPointerEnter={handlePointerEnter}
            onPointerLeave={handlePointerLeave}
          />

          {/* Dynamic Brush Ring & Precision Reticle */}
          {isHovering && !busy && (
            <div
              className={`studio-mask-cursor-ring mode-${mode} ${isPainting ? "is-painting" : ""}`}
              style={{
                transform: `translate3d(${cursorPos.x}px, ${cursorPos.y}px, 0) translate(-50%, -50%)`,
                width: `${displayDiameter}px`,
                height: `${displayDiameter}px`,
              }}
              aria-hidden="true"
            >
              <span className="studio-mask-cursor-dot" />
            </div>
          )}

          <CanvasProgressOverlay
            busy={busy}
            progress={progress}
            phase={phase}
            phaseDetail={phaseDetail}
            generatingPrompt={generatingPrompt}
          />
        </div>
      </div>
    </div>
  );
}

export default memo(StudioMaskOverlay);

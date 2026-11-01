import { useState, useRef, useEffect, useCallback, memo } from "react";
import { extractNounId, clampBbox, RESIZE_HANDLES, boxColor } from "../utils/spatialBoxes";
import { CanvasProgressOverlay } from "./ResultCanvas";
import "./SpatialCanvasOverlay.css";

function SpatialCanvasOverlay({
  boxes = [],
  onChangeBoxes,
  activeBoxId,
  onSetActiveBoxId,
  aspectRatioWidth = 1,
  aspectRatioHeight = 1,
  backdropUrl = null,
  referenceImageUrl = null,
  isEditingEnabled = true,
  onUploadBackdropFiles = null,
  onClearReference = null,
  busy = false,
  progress = null,
  phase = null,
  phaseDetail = null,
  generatingPrompt = null,
  onCommitBoxes = null,
  t = (k) => k,
}) {
  const containerRef = useRef(null);
  const hudInputRef = useRef(null);
  const backdropFileInputRef = useRef(null);

  const arW = Number(aspectRatioWidth) || 1;
  const arH = Number(aspectRatioHeight) || 1;
  const maxWidthStyle = `min(100%, ${(600 * arW) / arH}px)`;

  // Interaction state
  const [drawingStart, setDrawingStart] = useState(null); // { x, y } in 0..1000
  const [currentDragRect, setCurrentDragRect] = useState(null); // [ymin, xmin, ymax, xmax]
  const [dragMode, setDragMode] = useState(null); // null | "create" | "move" | handleName
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [showHud, setShowHud] = useState(false);
  const [dragOverCanvas, setDragOverCanvas] = useState(false);
  // Visual-only backdrop mute: hides the reference/result image behind a
  // neutral grey so the box layout reads without the artwork. The image stays
  // attached and is still sent as the generation reference.
  const [backdropMuted, setBackdropMuted] = useState(false);
  // A new backdrop/reference deserves to be seen again: un-mute on swap.
  useEffect(() => {
    setBackdropMuted(false);
  }, [referenceImageUrl, backdropUrl]);

  const drawingStartRef = useRef(null);
  const currentDragRectRef = useRef(null);
  const commitTimeoutRef = useRef(null);

  // Active box object: match by uid OR id
  const activeBox = boxes.find(
    (b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId
  );
  // Index drives the shared per-box palette, so the HUD badge matches the
  // outline color of the box it edits.
  const activeBoxIndex = boxes.findIndex(
    (b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId
  );
  const activeBoxColor = boxColor(activeBoxIndex >= 0 ? activeBoxIndex : 0);

  const [localDesc, setLocalDesc] = useState("");

  // Sync local text when active box switches
  useEffect(() => {
    setLocalDesc(activeBox?.desc || "");
  }, [activeBox?.uid, activeBox?.id]);

  const lastFocusedBoxIdRef = useRef(null);

  // Auto-focus HUD input ONLY once when HUD opens or activeBoxId changes to a different box
  useEffect(() => {
    if (showHud && activeBoxId) {
      if (lastFocusedBoxIdRef.current !== activeBoxId) {
        lastFocusedBoxIdRef.current = activeBoxId;
        const timer = setTimeout(() => {
          hudInputRef.current?.focus();
        }, 50);
        return () => clearTimeout(timer);
      }
    } else if (!showHud) {
      lastFocusedBoxIdRef.current = null;
    }
  }, [showHud, activeBoxId]);

  // Convert DOM client coordinates to 0..1000 coordinates relative to container with edge magnetic snap
  const getNormalizedCoords = useCallback((clientX, clientY, snap = true, refCoord = null) => {
    if (!containerRef.current) return [0, 0];
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return [0, 0];

    let x = Math.round(((clientX - rect.left) / rect.width) * 1000);
    let y = Math.round(((clientY - rect.top) / rect.height) * 1000);

    // Magnetic edge snapping: within 25 units (~2.5% of canvas), lock cleanly to 0 or 1000
    // ONLY snap if refCoord doesn't already sit at that edge (prevents collapsing box to 0 width/height)
    const SNAP_MARGIN = 25;
    if (snap) {
      const nearLeft = x <= SNAP_MARGIN;
      const nearRight = x >= 1000 - SNAP_MARGIN;
      const nearTop = y <= SNAP_MARGIN;
      const nearBottom = y >= 1000 - SNAP_MARGIN;

      if (nearLeft && (!refCoord || refCoord.x > SNAP_MARGIN)) {
        x = 0;
      } else if (nearRight && (!refCoord || refCoord.x < 1000 - SNAP_MARGIN)) {
        x = 1000;
      }

      if (nearTop && (!refCoord || refCoord.y > SNAP_MARGIN)) {
        y = 0;
      } else if (nearBottom && (!refCoord || refCoord.y < 1000 - SNAP_MARGIN)) {
        y = 1000;
      }
    }

    return [Math.max(0, Math.min(1000, y)), Math.max(0, Math.min(1000, x))];
  }, []);

  // Handle pointer down on canvas
  const handlePointerDown = (e) => {
    if (!isEditingEnabled) return;
    if (e.button !== 0) return; // Left click only

    // If clicked on HUD, resize handle, or eye button, ignore canvas pointerdown
    if (e.target.closest(".spatial-box-hud") || e.target.closest(".spatial-box-handle") || e.target.closest(".spatial-canvas-top-bar") || e.target.closest(".spatial-grid-empty-cta") || e.target.closest(".spatial-box-eye")) {
      return;
    }

    const [y, x] = getNormalizedCoords(e.clientX, e.clientY);
    const clickedBox = e.target.closest(".spatial-box");

    if (clickedBox) {
      // Smallest box at the point wins: a full-canvas backdrop box must not
      // swallow clicks aimed at the smaller boxes stacked above it.
      const pointPicks = boxes.filter((b) => {
        const [y1, x1, y2, x2] = b.bbox || [0, 0, 1000, 1000];
        return y >= y1 && x >= x1 && y <= y2 && x <= x2;
      });
      const targetBox = pointPicks.length > 1
        ? pointPicks.reduce((a, b) => {
            const areaA = ((a.bbox || [0, 0, 1000, 1000])[2] - (a.bbox || [0, 0, 1000, 1000])[0])
              * ((a.bbox || [0, 0, 1000, 1000])[3] - (a.bbox || [0, 0, 1000, 1000])[1]);
            const areaB = ((b.bbox || [0, 0, 1000, 1000])[2] - (b.bbox || [0, 0, 1000, 1000])[0])
              * ((b.bbox || [0, 0, 1000, 1000])[3] - (b.bbox || [0, 0, 1000, 1000])[1]);
            return areaB < areaA ? b : a;
          })
        : pointPicks[0];
      if (targetBox) {
        const targetId = targetBox.uid || targetBox.id;
        onSetActiveBoxId?.(targetId);
        setShowHud(true);
        setDragMode("move");
        const b = targetBox.bbox || [0, 0, 1000, 1000];
        setDragOffset({ y: y - b[0], x: x - b[1] });
        try {
          containerRef.current?.setPointerCapture(e.pointerId);
        } catch {
          // Ignored
        }
      }
      return;
    }

    // Otherwise, click on empty canvas starts drawing a new box
    const startPt = { y, x };
    drawingStartRef.current = startPt;
    setDrawingStart(startPt);
    const initialRect = [y, x, y, x];
    currentDragRectRef.current = initialRect;
    setCurrentDragRect(initialRect);
    setDragMode("create");
    setShowHud(false);
    onSetActiveBoxId?.(null);
    try {
      containerRef.current?.setPointerCapture(e.pointerId);
    } catch {
      // Ignored
    }
  };

  // Handle pointer move
  const handlePointerMove = useCallback((e) => {
    if (!dragMode) return;
    const start = drawingStartRef.current || drawingStart;
    const [y, x] = getNormalizedCoords(e.clientX, e.clientY, true, start);

    if (dragMode === "create" && start) {
      const rect = clampBbox(start.y, start.x, y, x);
      currentDragRectRef.current = rect;
      setCurrentDragRect(rect);
    } else if (dragMode === "move" && activeBoxId) {
      const targetBox = boxes.find((b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId);
      if (!targetBox) return;
      const [y1, x1, y2, x2] = targetBox.bbox;
      const h = y2 - y1;
      const w = x2 - x1;
      let newY1 = Math.max(0, Math.min(1000 - h, y - dragOffset.y));
      let newX1 = Math.max(0, Math.min(1000 - w, x - dragOffset.x));

      // Snap flush to edge when moving
      if (newX1 <= 20) newX1 = 0;
      else if (newX1 >= 1000 - w - 20) newX1 = 1000 - w;
      if (newY1 <= 20) newY1 = 0;
      else if (newY1 >= 1000 - h - 20) newY1 = 1000 - h;

      const updated = boxes.map((b) =>
        ((b.uid && b.uid === activeBoxId) || b.id === activeBoxId)
          ? { ...b, bbox: [newY1, newX1, newY1 + h, newX1 + w] }
          : b
      );
      onChangeBoxes?.(updated);
    } else if (RESIZE_HANDLES.includes(dragMode) && activeBoxId) {
      const targetBox = boxes.find((b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId);
      if (!targetBox) return;
      let [y1, x1, y2, x2] = targetBox.bbox;

      if (dragMode.includes("n")) y1 = Math.min(y2 - 8, y);
      if (dragMode.includes("s")) y2 = Math.max(y1 + 8, y);
      if (dragMode.includes("w")) x1 = Math.min(x2 - 8, x);
      if (dragMode.includes("e")) x2 = Math.max(x1 + 8, x);

      const clamped = clampBbox(y1, x1, y2, x2);
      const updated = boxes.map((b) =>
        ((b.uid && b.uid === activeBoxId) || b.id === activeBoxId)
          ? { ...b, bbox: clamped }
          : b
      );
      onChangeBoxes?.(updated);
    }
  }, [activeBoxId, boxes, dragMode, dragOffset.x, dragOffset.y, drawingStart, getNormalizedCoords, onChangeBoxes]);

  // Handle pointer up
  const handlePointerUp = useCallback((e) => {
    if (!dragMode) return;
    try {
      if (containerRef.current && e?.pointerId != null && containerRef.current.hasPointerCapture?.(e.pointerId)) {
        containerRef.current.releasePointerCapture(e.pointerId);
      }
    } catch {
      // Ignored if capture wasn't held
    }

    if (dragMode === "create") {
      const start = drawingStartRef.current || drawingStart;
      let rect = currentDragRectRef.current || currentDragRect;
      if (!rect && e?.clientX != null && e?.clientY != null && start) {
        const [y, x] = getNormalizedCoords(e.clientX, e.clientY, true, start);
        rect = clampBbox(start.y, start.x, y, x);
      }
      if (rect) {
        const [y1, x1, y2, x2] = rect;
        // Keep if box has size >= 8 units on 1000 grid (~0.8% of canvas)
        if (y2 - y1 >= 8 && x2 - x1 >= 8) {
          const newIndex = boxes.length + 1;
          const newUid = `box_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
          const newId = `object_${newIndex}`;
          const newBox = {
            uid: newUid,
            id: newId,
            bbox: [y1, x1, y2, x2],
            desc: "",
            kind: "object",
            action: referenceImageUrl ? "add" : "object",
            lang: "En",
          };
          const nextBoxes = [...boxes, newBox];
          onChangeBoxes?.(nextBoxes);
          onCommitBoxes?.(nextBoxes);
          onSetActiveBoxId?.(newUid);
          setShowHud(true);
        }
      }
    } else if (dragMode === "move" || (dragMode && RESIZE_HANDLES.includes(dragMode))) {
      onCommitBoxes?.(boxes);
    }

    drawingStartRef.current = null;
    currentDragRectRef.current = null;
    setDrawingStart(null);
    setCurrentDragRect(null);
    setDragMode(null);
  }, [boxes, currentDragRect, dragMode, drawingStart, getNormalizedCoords, onChangeBoxes, onCommitBoxes, onSetActiveBoxId, referenceImageUrl]);

  // Global window listeners while dragMode is active so drag release outside canvas or edge gestures always commit
  useEffect(() => {
    if (!dragMode) return;

    const onGlobalPointerMove = (e) => {
      handlePointerMove(e);
    };

    const onGlobalPointerUp = (e) => {
      handlePointerUp(e);
    };

    const onGlobalPointerCancel = (e) => {
      handlePointerUp(e);
    };

    window.addEventListener("pointermove", onGlobalPointerMove, { passive: true });
    window.addEventListener("pointerup", onGlobalPointerUp);
    window.addEventListener("pointercancel", onGlobalPointerCancel);

    return () => {
      window.removeEventListener("pointermove", onGlobalPointerMove);
      window.removeEventListener("pointerup", onGlobalPointerUp);
      window.removeEventListener("pointercancel", onGlobalPointerCancel);
    };
  }, [dragMode, handlePointerMove, handlePointerUp]);

  // Close HUD when generation starts
  useEffect(() => {
    if (busy) {
      setShowHud(false);
    }
  }, [busy]);

  // Clean up commit timeout
  useEffect(() => {
    return () => {
      if (commitTimeoutRef.current) clearTimeout(commitTimeoutRef.current);
    };
  }, []);

  const debounceCommit = useCallback((updatedBoxes) => {
    if (commitTimeoutRef.current) clearTimeout(commitTimeoutRef.current);
    commitTimeoutRef.current = setTimeout(() => {
      onCommitBoxes?.(updatedBoxes);
    }, 250);
  }, [onCommitBoxes]);

  // Resize handle pointer down
  const handleHandlePointerDown = (e, handle) => {
    e.stopPropagation();
    setDragMode(handle);
    try {
      containerRef.current?.setPointerCapture(e.pointerId);
    } catch {
      // Ignored
    }
  };

  // Delete active box
  const handleDeleteActiveBox = useCallback(() => {
    if (!activeBoxId) return;
    const nextBoxes = boxes.filter(
      (b) => (b.uid ? b.uid !== activeBoxId : b.id !== activeBoxId)
    );
    onChangeBoxes?.(nextBoxes);
    onCommitBoxes?.(nextBoxes);
    onSetActiveBoxId?.(null);
    setShowHud(false);
  }, [activeBoxId, boxes, onChangeBoxes, onCommitBoxes, onSetActiveBoxId]);

  // Update field on active box
  const updateActiveBox = useCallback((patch, shouldCommit = false) => {
    if (!activeBoxId) return null;
    let nextActiveId = activeBoxId;
    const nextBoxes = boxes.map((b, idx) => {
      const isMatch = (b.uid && b.uid === activeBoxId) || b.id === activeBoxId;
      if (!isMatch) return b;
      const merged = { ...b, ...patch };
      // Mode change: update ID immediately
      if (patch.kind === "text") {
        merged.id = extractNounId(merged.desc, "text", idx + 1, merged.lang);
      } else if (patch.kind === "object") {
        merged.id = extractNounId(merged.desc, "object", idx + 1, merged.lang);
      }
      if (activeBoxId === b.id && merged.id !== b.id) {
        nextActiveId = merged.id;
      }
      return merged;
    });
    if (nextActiveId !== activeBoxId) {
      onSetActiveBoxId?.(nextActiveId);
    }
    onChangeBoxes?.(nextBoxes);
    if (shouldCommit) {
      if (commitTimeoutRef.current) {
        clearTimeout(commitTimeoutRef.current);
        commitTimeoutRef.current = null;
      }
      onCommitBoxes?.(nextBoxes);
    }
    return nextBoxes;
  }, [activeBoxId, boxes, onChangeBoxes, onCommitBoxes, onSetActiveBoxId]);

  // Drag & drop image files onto canvas
  const handleCanvasDragOver = (e) => {
    if (e.dataTransfer?.types?.includes("Files")) {
      e.preventDefault();
      e.stopPropagation();
      setDragOverCanvas(true);
    }
  };

  const handleCanvasDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOverCanvas(false);
  };

  const handleCanvasDrop = (e) => {
    if (e.dataTransfer?.files?.length && onUploadBackdropFiles) {
      e.preventDefault();
      e.stopPropagation();
      setDragOverCanvas(false);
      onUploadBackdropFiles(e.dataTransfer.files);
    }
  };

  // Keyboard shortcut listener for active box
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!activeBoxId) return;
      if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
      if (e.key === "Backspace" || e.key === "Delete") {
        e.preventDefault();
        handleDeleteActiveBox();
      } else if (e.key === "Escape") {
        setShowHud(false);
        onSetActiveBoxId?.(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [activeBoxId, handleDeleteActiveBox, onSetActiveBoxId]);

  return (
    <div
      className="spatial-canvas-stage"
      style={{
        maxWidth: maxWidthStyle,
      }}
    >
      {/* Backdrop bar lives ABOVE the canvas so it never covers the artwork:
          it sits in the flow between the header and the image. */}
      {/* Top Backdrop Status Bar */}
      <div
        className="spatial-canvas-top-bar"
        style={{ pointerEvents: dragMode ? "none" : "auto" }}
      >
          {referenceImageUrl ? (
            <div className="spatial-backdrop-status">
              <span className="spatial-backdrop-tag">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                {t("canvas.spatial.backdropRef") || "Backdrop: Reference"}
              </span>
              <button
                type="button"
                className="spatial-backdrop-action-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setBackdropMuted((m) => !m);
                }}
                title={backdropMuted
                  ? t("canvas.spatial.showBackdrop") || "Show reference behind the boxes"
                  : t("canvas.spatial.hideBackdrop") || "Hide the reference (keep it attached)"}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>
                {backdropMuted
                  ? (t("canvas.spatial.showBackdrop") || "Show reference")
                  : (t("canvas.spatial.hideBackdrop") || "Hide reference")}
              </button>
              {onClearReference && (
                <button
                  type="button"
                  className="spatial-backdrop-action-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClearReference();
                  }}
                  title={t("canvas.spatial.removeBackdrop") || "Remove Backdrop"}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                  {t("canvas.spatial.removeBackdrop") || "Remove"}
                </button>
              )}
            </div>
          ) : backdropUrl ? (
            <div className="spatial-backdrop-status">
              <span className="spatial-backdrop-tag">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                {t("canvas.spatial.backdropResult") || "Backdrop: Last Result"}
              </span>
              <button
                type="button"
                className="spatial-backdrop-action-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setBackdropMuted((m) => !m);
                }}
                title={backdropMuted
                  ? t("canvas.spatial.showBackdrop") || "Show image behind the boxes"
                  : t("canvas.spatial.hideBackdrop") || "Hide the image (keep it attached)"}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>
                {backdropMuted
                  ? (t("canvas.spatial.showBackdrop") || "Show image")
                  : (t("canvas.spatial.hideBackdrop") || "Hide image")}
              </button>
              {onUploadBackdropFiles && (
                <button
                  type="button"
                  className="spatial-backdrop-action-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    backdropFileInputRef.current?.click();
                  }}
                  title={t("canvas.spatial.addBackdropBtn") || "Add Backdrop Image"}
                >
                  + {t("canvas.spatial.customImage") || "Reference"}
                </button>
              )}
            </div>
          ) : (
            <div className="spatial-backdrop-status">
              <span className="spatial-backdrop-tag blueprint">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
                {t("canvas.spatial.blueprintTag") || "Blueprint Grid"}
              </span>
              {onUploadBackdropFiles && (
                <button
                  type="button"
                  className="spatial-backdrop-action-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    backdropFileInputRef.current?.click();
                  }}
                  title={t("canvas.spatial.addBackdropBtn") || "Add Backdrop Image"}
                >
                  + {t("canvas.spatial.addBackdropBtn") || "Add Backdrop"}
                </button>
              )}
            </div>
          )}
          {boxes.length > 0 && (
            <button
              type="button"
              className="spatial-backdrop-action-btn spatial-clear-top-btn"
              onClick={(e) => {
                e.stopPropagation();
                onChangeBoxes?.([]);
                onSetActiveBoxId?.(null);
                onCommitBoxes?.([]);
              }}
              title={t("generate.spatialBox.clearAll") || "Clear all spatial boxes"}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
              {t("generate.spatialBox.clearAllBtn", { count: boxes.length })}
            </button>
          )}
        </div>
    <div
      className="spatial-canvas-wrapper"
      style={{
        aspectRatio: `${arW} / ${arH}`,
        maxWidth: maxWidthStyle,
      }}
    >
      <input
        ref={backdropFileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,.heic,.HEIC,image/heic,image/heif"
        style={{ display: "none" }}
        onChange={(e) => {
          if (e.target.files?.length && onUploadBackdropFiles) {
            onUploadBackdropFiles(e.target.files);
          }
        }}
      />

      <div
        ref={containerRef}
        className={`spatial-canvas-overlay ${isEditingEnabled ? "editing-enabled" : ""}${dragOverCanvas ? " drag-over" : ""}`}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onDragOver={handleCanvasDragOver}
        onDragLeave={handleCanvasDragLeave}
        onDrop={handleCanvasDrop}
      >
        {/* Layer 0: Backdrop */}
        <div className="spatial-canvas-backdrop">
          {backdropMuted ? (
            <div className="spatial-backdrop-muted" aria-hidden="true" />
          ) : referenceImageUrl ? (
            <img
              src={referenceImageUrl}
              alt="Reference backdrop"
              className="spatial-backdrop-img"
              draggable={false}
              onDragStart={(e) => e.preventDefault()}
            />
          ) : backdropUrl ? (
            <img
              src={backdropUrl}
              alt="Result backdrop"
              className="spatial-backdrop-img"
              draggable={false}
              onDragStart={(e) => e.preventDefault()}
            />
          ) : (
            <div className="spatial-blueprint-grid">
              <div className="guide-line guide-h guide-25" />
              <div className="guide-line guide-h guide-50" />
              <div className="guide-line guide-h guide-75" />
              <div className="guide-line guide-v guide-25" />
              <div className="guide-line guide-v guide-50" />
              <div className="guide-line guide-v guide-75" />
              {boxes.length === 0 && (
                <div className="spatial-grid-empty-cta">
                  <span className="spatial-empty-icon" aria-hidden="true">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>
                  </span>
                  <strong className="spatial-empty-title">Studio Spatial Canvas</strong>
                  <p className="spatial-empty-desc">
                    {t("canvas.spatial.emptyDesc") || "Draw boxes to place objects & typography, or add a background image to edit."}
                  </p>
                  {onUploadBackdropFiles && (
                    <button
                      type="button"
                      className="spatial-add-backdrop-btn"
                      onClick={() => backdropFileInputRef.current?.click()}
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                      {t("canvas.spatial.addBackdropBtn") || "Add Backdrop Image"}
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>


        {/* Drag-over dropzone overlay */}
        {dragOverCanvas && (
          <div className="spatial-dropzone-overlay">
            <div className="spatial-dropzone-badge">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              {t("canvas.spatial.dropHere") || "Drop image here to set as backdrop"}
            </div>
          </div>
        )}

        {/* Layer 1: Boxes */}
        <div className="spatial-canvas-boxes">
          {boxes.map((box, index) => {
            const [y1, x1, y2, x2] = box.bbox || [0, 0, 1000, 1000];
            const isActive = (box.uid && box.uid === activeBoxId) || box.id === activeBoxId;
            const color = boxColor(index);
            const isDraggingThis = isActive && currentDragRect && dragMode !== "create";
            const curYmin = isDraggingThis ? currentDragRect[0] : y1;
            const isNearTop = curYmin < 38;
            const boxStyle = {
              top: `${y1 / 10}%`,
              left: `${x1 / 10}%`,
              height: `${(y2 - y1) / 10}%`,
              width: `${(x2 - x1) / 10}%`,
              "--box-color": color,
            };

            return (
              <div
                key={box.uid || box.id || index}
                data-box-id={box.uid || box.id}
                className={`spatial-box ${isActive ? "active" : ""} ${box.hidden ? "box-hidden" : ""} kind-${box.kind || "object"} action-${box.action || "add"}`}
                style={boxStyle}
              >
                {/* Header Pill */}
                <div className={`spatial-box-header ${isNearTop ? "is-inside" : ""}`}>
                  <span className="spatial-box-badge" style={{ borderColor: color }}>
                    #{index + 1} {box.id}
                  </span>
                  <button
                    type="button"
                    className={`spatial-box-eye ${box.hidden ? "is-hidden" : ""}`}
                    onPointerDown={(e) => e.stopPropagation()}
                    onPointerUp={(e) => e.stopPropagation()}
                    onClick={(e) => {
                      e.stopPropagation();
                      const next = boxes.map((b, bi) => (
                        bi === index ? { ...b, hidden: !b.hidden } : b
                      ));
                      onChangeBoxes?.(next);
                      onCommitBoxes?.(next);
                    }}
                    title={box.hidden
                      ? t("canvas.spatial.showOnCanvas") || "Show box again"
                      : t("canvas.spatial.hideFromCanvas") || "Hide box from canvas (stays in the prompt)"}
                    aria-label={box.hidden
                      ? t("canvas.spatial.showOnCanvas") || "Show box again"
                      : t("canvas.spatial.hideFromCanvas") || "Hide box from canvas (stays in the prompt)"}
                  >
                    {box.hidden ? (
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
                    ) : (
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                    )}
                  </button>
                  {box.action === "remove" && <span className="spatial-tag-remove">✕</span>}
                </div>

                {/* In-box Label Preview */}
                <div className="spatial-box-label" title={box.desc} style={{ color }}>
                  {box.kind === "text" ? `“${box.desc || "..."}”` : box.desc || `[${box.id}]`}
                </div>

                {/* 8 Resize Handles (Active Only) */}
                {isActive && isEditingEnabled && (
                  <div className="spatial-box-handles">
                    {RESIZE_HANDLES.map((handle) => (
                      <div
                        key={handle}
                        className={`spatial-box-handle handle-${handle}`}
                        onPointerDown={(e) => handleHandlePointerDown(e, handle)}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {/* Temporary Drag Rectangle (While Drawing) */}
          {currentDragRect && dragMode === "create" && (
            <div
              className="spatial-box spatial-box-drawing"
              style={{
                top: `${currentDragRect[0] / 10}%`,
                left: `${currentDragRect[1] / 10}%`,
                height: `${(currentDragRect[2] - currentDragRect[0]) / 10}%`,
                width: `${(currentDragRect[3] - currentDragRect[1]) / 10}%`,
                "--box-color": boxColor(boxes.length),
              }}
            >
              <div className={`spatial-box-header ${currentDragRect[0] < 38 ? "is-inside" : ""}`}>
                <span className="spatial-box-badge">+{boxes.length + 1} New Box</span>
              </div>
            </div>
          )}
        </div>

        {/* Layer 2: Floating Inline HUD on Active Box */}
        {showHud && activeBox && (
          <div
            className="spatial-box-hud"
            style={{
              top: `clamp(10px, ${activeBox.bbox[2] / 10 + 2}%, calc(100% - 110px))`,
              left: `clamp(10px, ${activeBox.bbox[1] / 10}%, calc(100% - 320px))`,
            }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            {/* Box name badge — same palette color as the box outline */}
            <div className="spatial-hud-row spatial-hud-badge-row">
              <span
                className="spatial-box-badge spatial-hud-badge"
                style={{ borderColor: activeBoxColor, color: activeBoxColor }}
              >
                #{activeBoxIndex + 1} {activeBox.id}
              </span>
              <button
                type="button"
                className={`spatial-box-eye ${activeBox.hidden ? "is-hidden" : ""}`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={(e) => {
                  e.stopPropagation();
                  const next = boxes.map((b) => (
                    (b.uid && b.uid === activeBoxId) || b.id === activeBoxId
                      ? { ...b, hidden: !b.hidden }
                      : b
                  ));
                  onChangeBoxes?.(next);
                  onCommitBoxes?.(next);
                }}
                title={activeBox.hidden
                  ? (t("canvas.spatial.showOnCanvas") || "Show box again")
                  : (t("canvas.spatial.hideFromCanvas") || "Hide box from canvas (stays in the prompt)")}
                aria-label={activeBox.hidden
                  ? (t("canvas.spatial.showOnCanvas") || "Show box again")
                  : (t("canvas.spatial.hideFromCanvas") || "Hide box from canvas (stays in the prompt)")}
              >
                {activeBox.hidden ? (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
                ) : (
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                )}
              </button>
            </div>

            {/* Mode Pills */}
            <div className="spatial-hud-row spatial-hud-modes">
              <button
                type="button"
                className={`hud-pill ${activeBox.kind !== "text" ? "active" : ""}`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  updateActiveBox({ kind: "object" }, true);
                  setTimeout(() => hudInputRef.current?.focus(), 10);
                }}
                title="Standard object prompt (e.g. hat_1, clock_2)"
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "3px" }}><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
                {t("generate.spatialBox.hudObject") || "Object"}
              </button>
              <button
                type="button"
                className={`hud-pill ${activeBox.kind === "text" ? "active" : ""}`}
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  updateActiveBox({ kind: "text" }, true);
                  setTimeout(() => hudInputRef.current?.focus(), 10);
                }}
                title="Render exact typographic text (e.g. En_Text_1)"
              >
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "3px" }}><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg>
                {t("generate.spatialBox.hudText") || "Text"}
              </button>

              {referenceImageUrl && (
                <>
                  <span className="hud-divider" />
                  <button
                    type="button"
                    className={`hud-pill ${activeBox.action === "add" ? "active" : ""}`}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => updateActiveBox({ action: "add" }, true)}
                    title="Add or replace content inside this region"
                  >
                    + {t("generate.spatialBox.hudAdd") || "Add"}
                  </button>
                  <button
                    type="button"
                    className={`hud-pill ${activeBox.action === "remove" ? "active" : ""}`}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => updateActiveBox({ action: "remove" }, true)}
                    title="Remove/erase this region from reference image"
                  >
                    ✕ {t("generate.spatialBox.hudRemove") || "Remove"}
                  </button>
                </>
              )}
            </div>

            {/* Inline Input Field */}
            <div className="spatial-hud-row spatial-hud-input-row">
              <input
                ref={hudInputRef}
                type="text"
                className="spatial-hud-input"
                value={localDesc}
                placeholder={
                  activeBox.kind === "text"
                    ? (t("generate.spatialBox.hudPlaceholderText") || "Exact text to render here...")
                    : (t("generate.spatialBox.hudPlaceholderObject") || "Describe what goes in this box... (Enter to save)")
                }
                onChange={(e) => {
                  const val = e.target.value;
                  setLocalDesc(val);
                  const updated = updateActiveBox({ desc: val });
                  if (updated) debounceCommit(updated);
                }}
                onBlur={() => {
                  const updatedDesc = localDesc;
                  if (activeBox.kind === "object") {
                    const idx = boxes.findIndex((b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId);
                    const newId = extractNounId(updatedDesc, "object", idx + 1, activeBox.lang);
                    updateActiveBox({ desc: updatedDesc, id: newId }, true);
                  } else {
                    updateActiveBox({ desc: updatedDesc }, true);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    const updatedDesc = localDesc;
                    let newId = activeBox.id;
                    if (activeBox.kind === "object") {
                      const idx = boxes.findIndex((b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId);
                      newId = extractNounId(updatedDesc, "object", idx + 1, activeBox.lang);
                    }
                    updateActiveBox({ desc: updatedDesc, id: newId }, true);
                    setShowHud(false);
                  } else if (e.key === "Escape") {
                    setShowHud(false);
                  }
                }}
              />
              <button
                type="button"
                className="spatial-hud-btn-delete"
                onClick={handleDeleteActiveBox}
                title={t("generate.spatialBox.deleteBox") || "Delete this box"}
                aria-label={t("generate.spatialBox.deleteBox") || "Delete this box"}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
              </button>
              <button
                type="button"
                className="spatial-hud-btn-done"
                onClick={() => {
                  const updatedDesc = localDesc;
                  let newId = activeBox.id;
                  if (activeBox.kind === "object") {
                    const idx = boxes.findIndex((b) => (b.uid && b.uid === activeBoxId) || b.id === activeBoxId);
                    newId = extractNounId(updatedDesc, "object", idx + 1, activeBox.lang);
                  }
                  updateActiveBox({ desc: updatedDesc, id: newId }, true);
                  setShowHud(false);
                }}
                title={t("generate.spatialBox.done") || "Done"}
                aria-label={t("generate.spatialBox.done") || "Done"}
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>
              </button>
            </div>
          </div>
        )}

        {/* Layer 3: Progress Overlay while generating */}
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

export default memo(SpatialCanvasOverlay);

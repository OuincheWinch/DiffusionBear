import { useEffect, useState, memo } from "react";
import { api, downloadImage, fetchImageBlob, imageUrl } from "../api";
import { nativeDragSuppression, bindFullImageDrag, exportImageNatively } from "../utils/dragDrop";
import { useI18n } from "../i18n/I18nContext";
import ImageActions from "./ImageActions";
import { copyText } from "../clipboard";
import SpatialCanvasOverlay from "./SpatialCanvasOverlay";
import StudioMaskOverlay from "./StudioMaskOverlay";

function ResultCanvas({
  currentImage,
  onSetCurrentImage,
  busy,
  progress,
  phase = null,
  phaseDetail = null,
  batchImages = [],
  generatingPrompt = null,
  onSetReferenceImage,
  onReuse,
  onVariation,
  onDeleteImage,
  onFillImage = null,
  canSetReference = false,
  maxReferenceImages = 1,
  supportsBoundingBox = false,
  spatialBoxes = [],
  onSetSpatialBoxes,
  activeBoxId = null,
  onSetActiveBoxId,
  referenceImageUrl = null,
  onUploadRefFiles = null,
  onClearReference = null,
  aspectRatioWidth = 768,
  aspectRatioHeight = 512,
  onCommitBoxes = null,
  isPaintingMask = false,
  onSetIsPaintingMask = null,
  paintedMaskData = null,
  onSetPaintedMaskData = null,
}) {
   const { t } = useI18n();
   const [upscaling, setUpscaling] = useState(null);
  // The prompt is clamped until asked for; a long one otherwise pushes the
  // metadata and the action row off the panel. Mirrors the gallery card.
  const [promptExpanded, setPromptExpanded] = useState(false);
  useEffect(() => { setPromptExpanded(false); }, [currentImage?.id]); // null | "lanczos-2" | "lanczos-4"
   const [copiedField, setCopiedField] = useState(null); // null | "image" | "prompt" | "seed"
    const [downloading, setDownloading] = useState(false);
    const [actionError, setActionError] = useState(null);


  async function handleUpscale(scale = 2) {
    if (!currentImage?.id) return;
    const key = `lanczos-${scale}`;
    setUpscaling(key);
    try {
      const upscaled = await api(`/api/images/${currentImage.id}/upscale`, {
        method: "POST",
        body: JSON.stringify({ scale }),
      });
      onSetCurrentImage?.(upscaled);
    } catch (e) {
      setActionError(e?.message || t("canvas.error.upscaleFailed"));
      console.error("Upscale failed:", e);
    } finally {
      setUpscaling(null);
    }
  }

  async function copyImageToClipboard() {
    if (!currentImage?.id) return;
    try {
      const blob = await fetchImageBlob(currentImage.id);
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type || "image/png"]: blob }),
      ]);
      setCopiedField("image");
      setTimeout(() => setCopiedField(null), 2000);
    } catch {
      // No clipboard for binary data. Opening the file is a real fallback -- the user
      // gets the image -- but say so, because a new tab with no explanation reads as a
      // glitch.
      setActionError(t("canvas.action.copyImageFallback"));
      window.open(imageUrl(currentImage.id), "_blank");
    }
  }

  async function copyPrompt() {
    if (!currentImage?.prompt) return;
    // Shared helper: modern API, then the legacy textarea, then a visible failure.
    // The empty `catch {}` these replaced is why the button did nothing at all from a
    // browser where the app is not a secure context, with no error to show.
    if (!(await copyText(currentImage.prompt))) {
      setActionError(t("canvas.action.copyFailed"));
      return;
    }
    setCopiedField("prompt");
    setTimeout(() => setCopiedField(null), 1500);
  }

   async function copySeed() {
     if (currentImage?.seed == null) return;
     if (!(await copyText(String(currentImage.seed)))) {
       setActionError(t("canvas.action.copyFailed"));
       return;
     }
     setCopiedField("seed");
     setTimeout(() => setCopiedField(null), 1500);
   }

   async function handleDownload() {
     if (!currentImage?.id || downloading) return;
     // Packaged app first: WKWebView cannot download blob: URLs -- it hands
     // the link to macOS, which answers "no application set to open the URL".
     // The shell's save panel is the only working export there, exactly as the
     // gallery card already does. A false return (bridge threw) falls through
     // to the blob path, which is correct in a real browser.
     if (typeof window !== "undefined" && window.webkit?.messageHandlers?.native) {
       if (exportImageNatively(currentImage.id)) return;
     }
     setDownloading(true);
     try {
       const filename = currentImage.file
         || `${currentImage.id}.${currentImage.format || "png"}`;
       await downloadImage(currentImage.id, filename);
       } catch (err) {
         setActionError(err?.message || t("canvas.error.downloadFailed"));
         console.error("Download failed:", err);
       } finally {

       setDownloading(false);
     }
   }

   async function handleDelete() {
     if (!currentImage?.id) return;
     if (!window.confirm(t("gallery.deleteConfirm"))) return;
     try {
       await api(`/api/images/${currentImage.id}`, { method: "DELETE" });
       onDeleteImage?.(currentImage.id);
       window.dispatchEvent(new CustomEvent("mlx:image-deleted", { detail: { id: currentImage.id } }));
       const remaining = (batchImages || []).filter((img) => img.id !== currentImage.id);
       if (remaining.length > 0) {
         onSetCurrentImage?.(remaining[0]);
       } else {
         onSetCurrentImage?.(null);
       }
     } catch (e) {
       setActionError(e?.message || t("gallery.deleteFailed", { message: e?.message || "" }) || "Delete failed");
       console.error("[ResultCanvas] Delete failed:", e);
     }
   }

   return (

    <div className="result-canvas-panel">
      <div className="canvas-header">
        <h3>{t("canvas.heading")}</h3>
        {busy ? (
          /* role="status" + aria-live="polite": a render here runs 30-280s, and
             the step pill is the only place that reports it. Without a live region
             a screen reader user has no way of knowing the app is doing anything.
             "polite" rather than "assertive" because the steps tick every few
             seconds and must not interrupt. */
          <span
            className={`canvas-meta-pill busy-pill ${phase ? `phase-${phase}` : ""}`}
            role="status"
            aria-live="polite"
            aria-busy="true"
          >
            {phase === "downloading"
              ? t("canvas.pill.downloadingModel")
              : phase === "loading_model"
                ? t("canvas.pill.loadingMemory")
                : phase === "compiling"
                  ? t("canvas.pill.compilingShaders")
                  : phase === "saving"
                    ? t("canvas.pill.finalizingImage")
                    : progress?.steps > 0
                      ? t("canvas.pill.step", { step: progress.step, steps: progress.steps })
                      : t("canvas.pill.working")}
          </span>
        ) : currentImage ? (
          <span className="canvas-meta-pill">
            {t("canvas.metaPill", {
              width: currentImage.width,
              height: currentImage.height,
              time: currentImage.generation_time,
            })}
          </span>
        ) : null}
        {referenceImageUrl && !busy && (
          <button
            type="button"
            className={`canvas-mask-toggle-btn ${isPaintingMask ? "active" : ""} ${paintedMaskData ? "has-mask" : ""}`}
            onClick={() => onSetIsPaintingMask?.(!isPaintingMask)}
            title={isPaintingMask ? t("params.ref.stopPainting") : paintedMaskData ? t("params.ref.editMask") : t("params.ref.paintOnReference")}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px" }}>
              <path d="M18.375 2.625a2.121 2.121 0 1 1 3 3L12 15l-4 1 1-4Z"/>
            </svg>
            <span>
              {isPaintingMask
                ? t("params.ref.stopPainting")
                : paintedMaskData
                  ? t("params.ref.editMask")
                  : t("params.ref.paintOnReference")}
            </span>
            {paintedMaskData && !isPaintingMask && (
              <span className="canvas-mask-armed-dot" title="Mask armed" aria-hidden="true" />
            )}
          </button>
        )}
      </div>

      <div className="canvas-viewport">
        {isPaintingMask && referenceImageUrl ? (
          <div className="canvas-image-container mask-overlay-wrapper">
            <StudioMaskOverlay
              referenceImageUrl={referenceImageUrl}
              aspectRatioWidth={aspectRatioWidth}
              aspectRatioHeight={aspectRatioHeight}
              maskData={paintedMaskData}
              onChangeMask={onSetPaintedMaskData}
              onClose={() => onSetIsPaintingMask?.(false)}
              busy={busy}
              progress={progress}
              phase={phase}
              phaseDetail={phaseDetail}
              generatingPrompt={generatingPrompt}
              t={t}
            />
          </div>
        ) : supportsBoundingBox ? (
          <div className="canvas-image-container spatial-container-wrapper">
            <SpatialCanvasOverlay
              boxes={spatialBoxes}
              onChangeBoxes={onSetSpatialBoxes}
              activeBoxId={activeBoxId}
              onSetActiveBoxId={onSetActiveBoxId}
              aspectRatio={`${aspectRatioWidth} / ${aspectRatioHeight}`}
              aspectRatioWidth={aspectRatioWidth}
              aspectRatioHeight={aspectRatioHeight}
              backdropUrl={currentImage ? imageUrl(currentImage.id) : null}
              referenceImageUrl={referenceImageUrl}
              isEditingEnabled={!busy}
              onUploadBackdropFiles={onUploadRefFiles}
              onClearReference={onClearReference}
              busy={busy}
              progress={progress}
              phase={phase}
              phaseDetail={phaseDetail}
              generatingPrompt={generatingPrompt}
              onCommitBoxes={onCommitBoxes}
              t={t}
            />
          </div>
        ) : (referenceImageUrl && (!currentImage || busy)) ? (
          <div className="canvas-image-container">
            <img
              src={referenceImageUrl}
              alt="Reference backdrop"
              className="canvas-image"
              style={nativeDragSuppression()}
            />
            <CanvasProgressOverlay
              busy={busy}
              progress={progress}
              phase={phase}
              phaseDetail={phaseDetail}
              generatingPrompt={generatingPrompt}
            />
          </div>
        ) : currentImage ? (
          <div className="canvas-image-container">
            <img
              key={currentImage.id}
              src={imageUrl(currentImage.id)}
              alt={currentImage.prompt || t("canvas.image.altFallback")}
              className="canvas-image"
              data-mlx-image-id={currentImage.id}
              data-mlx-file-url={currentImage.file_url || ""}
              style={nativeDragSuppression()}
              title={t("canvas.image.dragTitle")}
{...bindFullImageDrag(currentImage)}
              />
                          <CanvasProgressOverlay
              busy={busy}
              progress={progress}
              phase={phase}
              phaseDetail={phaseDetail}
              generatingPrompt={generatingPrompt}
            />
          </div>
        ) : (
          <div className="canvas-empty">
            {busy ? (
              <CanvasProgressOverlay
                busy={busy}
                progress={progress}
                standalone
                phase={phase}
                phaseDetail={phaseDetail}
                generatingPrompt={generatingPrompt}
              />
            ) : (
              <>
                <div className="empty-icon">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                    <circle cx="9" cy="9" r="2" />
                    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                  </svg>
                </div>
                <p className="empty-title">{t("canvas.empty.title")}</p>
                <p className="empty-subtitle">
                  {t("canvas.empty.subtitle")}
                </p>
              </>
            )}
          </div>
        )}
      </div>

      {/* Batch Filmstrip / Gallery */}
      {batchImages && batchImages.length > 0 && (
        <div className="canvas-batch-strip">
          <div className="batch-strip-header">
            <span className="batch-strip-title">
              {t("canvas.batchStrip.title", {
                count: batchImages.length,
                totalSuffix: progress?.batch && progress.batch > batchImages.length
                  ? t("canvas.batchStrip.totalSuffix", { total: progress.batch })
                  : "",
              })}
            </span>
            {busy && progress?.batch && batchImages.length < progress.batch && (
              <span className="batch-strip-status">
                {t("canvas.batchStrip.rendering", { n: batchImages.length + 1 })}
              </span>
            )}
          </div>
          <div className="batch-strip-items">
            {batchImages.map((img, idx) => (
              <div
                key={img.id || idx}
                role="button"
                tabIndex={0}
                className={`batch-strip-thumb ${currentImage?.id === img.id ? "active" : ""}`}
                onClick={() => onSetCurrentImage?.(img)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSetCurrentImage?.(img);
                  }
                }}
                title={t("canvas.batchStrip.thumbTitle", { n: idx + 1, seed: img.seed })}
                {...bindFullImageDrag(img)}
              >
                <img
                  src={imageUrl(img.id, true)}
                  alt={t("canvas.batchStrip.thumbAlt", { n: idx + 1 })}
                  {...bindFullImageDrag(img)}
                />
                <span className="batch-thumb-badge">#{idx + 1}</span>
              </div>
            ))}
            {busy && progress?.batch && batchImages.length < progress.batch && (
              <div className="batch-strip-thumb placeholder" title={t("canvas.batchStrip.renderingImage", { n: batchImages.length + 1 })}>
                <div className="mini-spinner" />
                <span className="batch-thumb-badge">#{batchImages.length + 1}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {currentImage && (
        <div className={`canvas-details ${busy ? "canvas-details-prev" : ""}`}>
          {busy && (
            <div className="canvas-prev-note">
              <span className="canvas-prev-tag">{t("canvas.prevTag")}</span>
              <span className="canvas-prev-hint">{t("canvas.prevHint")}</span>
            </div>
          )}
          {(currentImage.prompt || "").length > 280 && (
            <button
              type="button"
              className="btn-mini"
              onClick={() => setPromptExpanded((v) => !v)}
            >
              {promptExpanded ? t("gallery.promptLess") : t("gallery.promptMore")}
            </button>
          )}
          <p className={`canvas-prompt${promptExpanded ? " expanded" : ""}`}
             title={currentImage.prompt}>
            &ldquo;{currentImage.prompt}&rdquo;
          </p>

          <div className="canvas-badges" role="region" aria-label={t("canvas.badge.model")}>
            <span className="badge">
              <strong>{t("canvas.badge.model")}</strong>
              <span className="badge-val">{currentImage.model?.split("/").pop()}</span>
            </span>
            <span className="badge">
              <strong>{t("canvas.badge.seed")}</strong>
              <span className="badge-val">{currentImage.seed}</span>
            </span>
            <span className="badge">
              <strong>{t("canvas.badge.steps")}</strong>
              <span className="badge-val">{currentImage.steps}</span>
            </span>
            {currentImage.sampler && (
              <span className="badge">
                <strong>{t("canvas.badge.sampler")}</strong>
                <span className="badge-val">{currentImage.sampler}</span>
              </span>
            )}
            {currentImage.guidance != null && (
              <span className="badge">
                <strong>{t("canvas.badge.cfg")}</strong>
                <span className="badge-val">{currentImage.guidance}</span>
              </span>
            )}
            {currentImage.fast_vae && (
              <span className="badge">
                <strong>{t("canvas.badge.vae")}</strong>
                <span className="badge-val">{t("canvas.badge.vaeValue")}</span>
              </span>
            )}
          </div>

          {actionError && <p className="error" role="alert">{actionError}</p>}
          <ImageActions
            onCopyImage={copyImageToClipboard}
            onCopyPrompt={copyPrompt}
            onCopySeed={copySeed}
            onUpscale2={() => handleUpscale(2)}
            onUpscale4={() => handleUpscale(4)}
            onReuse={onReuse ? () => onReuse(currentImage) : undefined}
            onVariation={onVariation ? () => onVariation(currentImage) : undefined}
            onFill={onFillImage && currentImage ? () => onFillImage(currentImage) : undefined}
            onDownload={handleDownload}
            showDelete={true}
            onDelete={handleDelete}
            upscaling={upscaling === "lanczos-2" ? 2 : upscaling === "lanczos-4" ? 4 : null}
            busy={busy}
            copied={copiedField === "image" || copiedField === "prompt" || copiedField === "seed" ? copiedField : null}
            onSetReference={canSetReference ? () => {
              const filename = currentImage.file || (currentImage.format ? `${currentImage.id}.${currentImage.format.toLowerCase()}` : `${currentImage.id}.png`);
              onSetReferenceImage?.({
                path: filename,
                preview: imageUrl(currentImage.id),
              });
            } : undefined}
            referenceTitle={t("canvas.action.useAsRefTitle", { max: maxReferenceImages })}
          />
        </div>
      )}
    </div>
  );
}

export function CanvasProgressOverlay({
  busy,
  progress,
  standalone = false,
  generatingPrompt = null,
  phase = null,
  phaseDetail = null,
}) {
  const { t } = useI18n();
  if (!busy) return null;
  const isProgress = progress && progress.steps > 0;
  const batchTotal = progress?.batch || 1;
  const batchIdx = (progress?.image_index != null ? progress.image_index : progress?.saved_index ? progress.saved_index - 1 : 0) + 1;
  const isBatch = batchTotal > 1;

  let mainText = t("canvas.progress.preparing");
  if (isProgress) {
    if (phase === "saving" || progress.step >= progress.steps) {
      mainText = t("canvas.progress.decodingVae", { elapsed: Math.round(progress.elapsed || 0) });
    } else {
      mainText = t("canvas.progress.step", {
        step: progress.step,
        steps: progress.steps,
        elapsed: Math.round(progress.elapsed || 0),
      });
    }
  } else {
    if (phase === "downloading") {
      mainText = t("canvas.progress.phaseDownloading");
    } else if (phase === "loading_model") {
      mainText = t("canvas.progress.phaseLoadingModel");
    } else if (phase === "compiling") {
      mainText = t("canvas.progress.phaseCompiling");
    } else if (phase === "saving") {
      mainText = t("canvas.progress.phaseSaving");
    } else if (standalone) {
      mainText = t("canvas.progress.phaseStandalone");
    }
  }

  return (
    <div className={standalone ? "canvas-busy-standalone" : "canvas-busy-overlay"}>
      <div className="spinner" />
      {isBatch && (
        <span className="canvas-batch-pill">
          {t("canvas.batchPill", { index: batchIdx, total: batchTotal })}
        </span>
      )}
      <p>{mainText}</p>
      {phaseDetail && !isProgress && (
        <p className="canvas-phase-detail">{phaseDetail}</p>
      )}
      {isProgress && (
        <div className="canvas-progress-track">
          <div
            className="progress-fill"
            style={{
              // scaleX rather than width: animating width re-runs layout every tick.
              transform: `scaleX(${Math.min(100, (progress.step / progress.steps) * 100) / 100})`,
            }}
          />
        </div>
      )}
      {generatingPrompt && (
        <p className="canvas-generating-prompt" title={generatingPrompt}>
          &ldquo;{generatingPrompt}&rdquo;
        </p>
      )}
    </div>
  );
}

export default memo(ResultCanvas);

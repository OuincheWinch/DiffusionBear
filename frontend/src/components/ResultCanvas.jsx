import { useState, memo } from "react";
import { api, downloadImage, fetchImageBlob, imageUrl } from "../api";
import { bindFullImageDrag } from "../utils/dragDrop";
import { useI18n } from "../i18n/I18nContext";

function CanvasProgressOverlay({ busy, progress, standalone = false, generatingPrompt = null, phase = null, phaseDetail = null }) {
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
              width: `${Math.min(100, (progress.step / progress.steps) * 100)}%`,
            }}
          />
        </div>
      )}
      {generatingPrompt && (
        <p className="canvas-busy-prompt" title={generatingPrompt}>
          &ldquo;{generatingPrompt}&rdquo;
        </p>
      )}
    </div>
  );
}

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
   onVariation,
   canSetReference = false,
   maxReferenceImages = 1,
 }) {
   const { t } = useI18n();
   const [upscaling, setUpscaling] = useState(null); // null | "lanczos-2" | "lanczos-4"
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
      window.open(imageUrl(currentImage.id), "_blank");
    }
  }

  async function copyPrompt() {
    if (!currentImage?.prompt) return;
    try {
      await navigator.clipboard.writeText(currentImage.prompt);
      setCopiedField("prompt");
      setTimeout(() => setCopiedField(null), 1500);
    } catch {}
  }

   async function copySeed() {
     if (currentImage?.seed == null) return;
     try {
       await navigator.clipboard.writeText(String(currentImage.seed));
       setCopiedField("seed");
       setTimeout(() => setCopiedField(null), 1500);
     } catch {}
   }

   async function handleDownload() {
     if (!currentImage?.id || downloading) return;
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
      </div>

      <div className="canvas-viewport">
        {currentImage ? (
          <div className="canvas-image-container">
            <img
              key={currentImage.id}
              src={imageUrl(currentImage.id)}
              alt={currentImage.prompt || t("canvas.image.altFallback")}
              className="canvas-image"
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
                <div className="empty-icon">🎨</div>
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
          <p className="canvas-prompt" title={currentImage.prompt}>
            &ldquo;{currentImage.prompt}&rdquo;
          </p>

          <div className="canvas-badges">
            <span className="badge">
              <strong>{t("canvas.badge.model")}</strong> {currentImage.model?.split("/").pop()}
            </span>
            <span className="badge">
              <strong>{t("canvas.badge.seed")}</strong> {currentImage.seed}
            </span>
            <span className="badge">
              <strong>{t("canvas.badge.steps")}</strong> {currentImage.steps}
            </span>
            {currentImage.sampler && (
              <span className="badge">
                <strong>{t("canvas.badge.sampler")}</strong> {currentImage.sampler}
              </span>
            )}
            {currentImage.guidance != null && (
              <span className="badge">
                <strong>{t("canvas.badge.cfg")}</strong> {currentImage.guidance}
              </span>
            )}
            {currentImage.fast_vae && (
              <span className="badge">
                <strong>{t("canvas.badge.vae")}</strong>{t("canvas.badge.vaeValue")}
              </span>
            )}
          </div>

           {actionError && <p className="error" role="alert">{actionError}</p>}
           <div className="canvas-actions">
             <button
               type="button"
               className="action-btn"
               onClick={() => handleUpscale(2)}
               disabled={upscaling != null || busy}
               title={t("canvas.action.upscale2xTitle")}
             >

              {upscaling === "lanczos-2" ? t("canvas.action.upscale2xBusy") : t("canvas.action.upscale2xIdle")}
            </button>
             <button
               type="button"
               className="action-btn"
               onClick={() => handleUpscale(4)}
               disabled={upscaling != null || busy}
               title={t("canvas.action.upscale4xTitle")}
             >

              {upscaling === "lanczos-4" ? t("canvas.action.upscale4xBusy") : t("canvas.action.upscale4xIdle")}
            </button>
             {canSetReference && (
               <button
                 type="button"
                 className="action-btn"
                 onClick={() => {
                   const filename = currentImage.file || (currentImage.format ? `${currentImage.id}.${currentImage.format.toLowerCase()}` : `${currentImage.id}.png`);
                   onSetReferenceImage?.({
                     path: filename,
                     preview: imageUrl(currentImage.id),
                   });
                 }}
                 title={t("canvas.action.useAsRefTitle", { max: maxReferenceImages })}
               >
                 {t("canvas.action.useAsRef")}
               </button>
             )}

             <button
               type="button"
               className="action-btn"
               onClick={() => onVariation?.(currentImage)}
               disabled={busy}
               title={t("canvas.action.variationTitle")}
             >

              {t("canvas.action.variation")}
            </button>
             <button
               type="button"
               className="action-btn"
               onClick={copyImageToClipboard}
               title={t("canvas.action.copyImageTitle")}
             >

              {copiedField === "image" ? t("canvas.action.copyImageDone") : t("canvas.action.copyImageIdle")}
            </button>
             <button type="button" className="action-btn" onClick={copyPrompt} title={t("canvas.action.copyPromptTitle")}>
               {copiedField === "prompt" ? t("canvas.action.copyPromptDone") : t("canvas.action.copyPromptIdle")}
             </button>
             <button type="button" className="action-btn" onClick={copySeed} title={t("canvas.action.copySeedTitle")}>
               {copiedField === "seed" ? t("canvas.action.copySeedDone") : t("canvas.action.copySeedIdle")}
             </button>
             <button
               type="button"
               className="action-btn"
               onClick={handleDownload}
               disabled={downloading}
               title={t("canvas.action.downloadTitle")}
             >
               {downloading ? t("canvas.action.downloadBusy") : t("canvas.action.downloadIdle")}
             </button>


          </div>
        </div>
      )}
    </div>
  );
}

export default memo(ResultCanvas);

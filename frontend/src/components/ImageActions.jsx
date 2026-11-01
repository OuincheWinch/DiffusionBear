import { useI18n } from "../i18n/I18nContext";

/**
 * Cohesive action toolbar for a generated image, shared between the gallery detail card
 * and the studio canvas.
 *
 * Actions are logically organized into two functional rows:
 * - Row 1 (Creative Workflow & Enhancements):
 *   - Workflow group: "Reuse params", "Variation", "Use as reference"
 *   - Upscale group: "Upscale 2x", "Upscale 4x"
 * - Row 2 (Utilities & Finalization):
 *   - Clipboard group: "Copy image", "Copy prompt", "Copy seed", "Finder"
 *   - Finalize group: "Export..." (primary), "Delete" (danger)
 *
 * All emojis are replaced with fine-line vector SVGs adhering to Apple HIG.
 */
export default function ImageActions({
  onCopyImage,
  onCopyPrompt,
  onCopySeed,
  onReveal,
  onUpscale2,
  onUpscale4,
  onReuse,
  onVariation,
  onDownload,
  onDelete,
  onSetReference,
  onFill,
  referenceTitle,
  upscaling = null,
  busy = false,
  disabled = false,
  showDelete = true,
  copied = null,
  doneLabels = {},
  revealed = false,
  isFavorite = false,
  onToggleFavorite = null,
  isBlurred = false,
  onToggleBlur = null,
}) {
  const { t } = useI18n();
  const off = disabled || busy;

  const hasRow1 = Boolean(onReuse || onVariation || onSetReference || onFill || onUpscale2 || onUpscale4);
  const hasRow2 = Boolean(onCopyImage || onCopyPrompt || onCopySeed || onReveal || onDownload || onToggleFavorite || onToggleBlur || (showDelete && onDelete));

  return (
    <div className="image-actions" role="group" aria-label={t("gallery.actionsLabel")}>
      {hasRow1 && (
        <div className="image-actions-row image-actions-primary-row">
          <div className="image-actions-group action-group-workflow">
            {onReuse && (
              <button
                type="button"
                className="action-btn"
                disabled={off}
                onClick={onReuse}
                title={t("gallery.reuseParams")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                  <path d="M21 3v5h-5" />
                  <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                  <path d="M3 21v-5h5" />
                </svg>
                <span>{t("gallery.reuseParams")}</span>
              </button>
            )}
            {onVariation && (
              <button
                type="button"
                className="action-btn"
                disabled={off}
                onClick={onVariation}
                title={t("canvas.action.variationTitle")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polyline points="16 3 21 3 21 8" />
                  <line x1="4" y1="20" x2="21" y2="3" />
                  <polyline points="21 16 21 21 16 21" />
                  <line x1="15" y1="15" x2="21" y2="21" />
                  <line x1="4" y1="4" x2="9" y2="9" />
                </svg>
                <span>{t("canvas.action.variation")}</span>
              </button>
            )}
            {onSetReference && (
              <button
                type="button"
                className="action-btn"
                disabled={off}
                onClick={onSetReference}
                title={referenceTitle}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                  <circle cx="9" cy="9" r="2" />
                  <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
                </svg>
                <span>{t("canvas.action.useAsRef")}</span>
              </button>
            )}
            {onFill && (
              <button
                type="button"
                className="action-btn"
                disabled={off}
                onClick={onFill}
                title={t("fill.cardButton")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="m18 15-6-6" />
                  <path d="m14 2 8 8" />
                  <path d="m11 5 8 8" />
                  <path d="m9 13-5 5v3h3l5-5" />
                </svg>
                <span>{t("fill.cardButton")}</span>
              </button>
            )}
          </div>

          <div className="image-actions-group action-group-upscale">
            {onUpscale2 && (
              <button
                type="button"
                className="action-btn"
                disabled={off || upscaling != null}
                onClick={onUpscale2}
                title={t("gallery.upscale2xTitle")}
              >
                <svg className="action-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                </svg>
                <span>{upscaling === 2 ? t("gallery.upscaling") : t("gallery.upscale2xBtn")}</span>
              </button>
            )}
            {onUpscale4 && (
              <button
                type="button"
                className="action-btn"
                disabled={off || upscaling != null}
                onClick={onUpscale4}
                title={t("gallery.upscale4xTitle")}
              >
                <svg className="action-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                </svg>
                <span>{upscaling === 4 ? t("gallery.upscaling") : t("gallery.upscale4xBtn")}</span>
              </button>
            )}
          </div>
        </div>
      )}

      {hasRow2 && (
        <div className="image-actions-row image-actions-secondary-row">
          <div className="image-actions-group action-group-clipboard">
            {onCopyImage && (
              <button
                type="button"
                className={`action-btn${copied === "image" ? " done" : ""}`}
                disabled={off}
                onClick={onCopyImage}
                title={t("gallery.copyImageFullTitle")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                  <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                </svg>
                <span>{copied === "image" ? (doneLabels.image || t("gallery.copyImageDone")) : t("gallery.copyImageBtn")}</span>
              </button>
            )}
            {onCopyPrompt && (
              <button
                type="button"
                className={`action-btn${copied === "prompt" ? " done" : ""}`}
                disabled={off}
                onClick={onCopyPrompt}
                title={t("canvas.action.copyPromptTitle")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                </svg>
                <span>{copied === "prompt" ? (doneLabels.prompt || t("gallery.promptCopiedDone")) : t("gallery.copyPromptBtn")}</span>
              </button>
            )}
            {onCopySeed && (
              <button
                type="button"
                className={`action-btn${copied === "seed" ? " done" : ""}`}
                disabled={off}
                onClick={onCopySeed}
                title={t("canvas.action.copySeedTitle")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <line x1="4" y1="9" x2="20" y2="9" />
                  <line x1="4" y1="15" x2="20" y2="15" />
                  <line x1="10" y1="3" x2="8" y2="21" />
                  <line x1="16" y1="3" x2="14" y2="21" />
                </svg>
                <span>{copied === "seed" ? (doneLabels.seed || t("gallery.seedCopiedDone")) : t("gallery.copySeedBtn")}</span>
              </button>
            )}
            {onReveal && (
              <button
                type="button"
                className={`action-btn${revealed ? " done" : ""}`}
                disabled={off}
                onClick={onReveal}
                title={t("gallery.revealFullTitle")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                </svg>
                <span>{revealed ? t("gallery.revealedDone") : t("gallery.revealBtn")}</span>
              </button>
            )}
          </div>

          <div className="image-actions-group action-group-finalize">
            {onToggleFavorite && (
              <button
                type="button"
                className={`action-btn${isFavorite ? " active-fav" : ""}`}
                disabled={off}
                onClick={onToggleFavorite}
                title={isFavorite ? t("gallery.favoriteRemove") : t("gallery.favoriteAdd")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill={isFavorite ? "#eab308" : "none"} stroke={isFavorite ? "#ca8a04" : "currentColor"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                </svg>
                <span>{isFavorite ? t("gallery.favoriteRemove") : t("gallery.favoriteAdd")}</span>
              </button>
            )}
            {onToggleBlur && (
              <button
                type="button"
                className={`action-btn${isBlurred ? " active-blur" : ""}`}
                disabled={off}
                onClick={onToggleBlur}
                title={isBlurred ? t("gallery.blurToggleOff") : t("gallery.blurToggleOn")}
              >
                {isBlurred ? (
                  <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                ) : (
                  <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                    <line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                )}
                <span>{isBlurred ? t("gallery.blurToggleOff") : t("gallery.blurToggleOn")}</span>
              </button>
            )}
            {onDownload && (
              <button
                type="button"
                className="action-btn primary"
                disabled={off}
                onClick={onDownload}
                title={t("canvas.action.downloadTitle")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                <span>{t("gallery.exportBtn")}</span>
              </button>
            )}
            {showDelete && onDelete && (
              <button
                type="button"
                className="action-btn danger"
                disabled={off}
                onClick={onDelete}
                title={t("gallery.deleteBtn")}
              >
                <svg className="action-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
                <span>{t("gallery.deleteBtn")}</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
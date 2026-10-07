import { memo } from "react";
import { useI18n } from "../i18n/I18nContext";

function clampRefStrength(v) {
  const n = Math.round((Number(v) + Number.EPSILON) * 100) / 100;
  return Math.min(1, Math.max(0.01, n));
}

function GenerationParams({
  modelInfo,
  width,
  setWidth,
  height,
  setHeight,
  steps,
  setSteps,
  guidance,
  setGuidance,
  seed,
  setSeed,
  batch,
  setBatch,
  negativePrompt,
  setNegativePrompt,
  sampler,
  setSampler,
  cacheInterval,
  setCacheInterval,
   supportsRef,
   supportsMultiRef,
   maxReferenceImages = 1,
   refImages,

  refStrength,
  setRefStrength,
  pickRefImages,
  removeRefImage,
  insertIntoPrompt,
  maxPixels,
  setMaxPixels,
}) {
   const { t } = useI18n();
   // De-duplicated and sorted so the dropdown order is stable regardless of how the
   // registry listed them, and so a registry repeating a name cannot fake a choice.
   const samplerChoices = [...new Set(modelInfo.samplers || [])].sort();

   // Guidance-distilled models (Juggernaut XL Lightning, FLUX.2-klein, ...) clamp
   // the step count in the engine; max_steps comes from /api/models so the slider
   // cannot promise steps the run will never take.
   const stepsCap = Number(modelInfo?.max_steps) > 0 ? Number(modelInfo.max_steps) : 50;
   const distillCapped = Boolean(modelInfo?.max_steps);

  const maxRefImages = Math.max(1, Number(maxReferenceImages) || 1);


  function randomizeSeed() {
    setSeed("");
  }

  function incrementSeed(delta) {
    setSeed((prev) => {
      const cur = prev === "" || isNaN(Number(prev)) ? 0 : Number(prev);
      return Math.max(0, cur + delta);
    });
  }

  return (
    <div className="generation-params-container">
      {/* Steps, seed and batch sit outside the disclosure on purpose.

          These are not "advanced". A generation is tuned by changing the step count, and
          a seed is how you get a different image or come back to one you liked; hiding
          either behind a collapsed panel made the primary path feel lighter only by
          making the primary path worse. They are the three controls people actually
          reach for, so they are visible without a click.

          What stays behind the disclosure is advanced BY NATURE: the negative prompt
          (unsupported on FLUX.2), the sampler (only where there is a real choice), the
          pixel cap and DeepCache. */}
      <div className="param-grid param-grid-primary">

        <label>
          <span className="step-label-header">
            {t("params.steps.label", { steps })}
            {modelInfo?.id === "krea2-turbo" && steps <= 4 && (
              <span className="distill-subtle-tag" title={t("params.steps.distillTitle")}>
                {t("params.steps.distillTag")}
              </span>
            )}
            {distillCapped && !(modelInfo?.id === "krea2-turbo" && steps <= 4) && (
              <span className="distill-subtle-tag" title={t("params.steps.distillCapTitle", { max: stepsCap })}>
                {t("params.steps.distillCapTag", { max: stepsCap })}
              </span>
            )}
          </span>
          <input
            type="range"
            min="1"
            max={stepsCap}
            value={Math.min(Number(steps) || 1, stepsCap)}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
        </label>

        <label>
          <div className="seed-label-row">
            <span>{t("params.seed.label")}</span>
            <div className="seed-quick-actions">
              <button
                type="button"
                className="btn-tiny"
                onClick={randomizeSeed}
                title={t("params.seed.randomTitle")}
              >
                🎲
              </button>
              <button
                type="button"
                className="btn-tiny"
                onClick={() => incrementSeed(1)}
                title={t("params.seed.nextTitle")}
              >
                +1
              </button>
              <button
                type="button"
                className="btn-tiny"
                onClick={() => incrementSeed(1024)}
                title={t("params.seed.nextBatchTitle")}
              >
                +1024
              </button>
            </div>
          </div>
          <input
            type="number"
            placeholder={t("params.seed.placeholder")}
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </label>
        <label>
          {t("params.batch.label")}
          <select
            value={batch}
            onChange={(e) => setBatch(Number(e.target.value))}
          >
            {Array.from({ length: 16 }, (_, i) => i + 1).map((n) => (
              <option key={n} value={n}>
                {t(n > 1 ? "params.batch.optionMany" : "params.batch.optionOne", { n })}
                {n > 1 ? t("params.batch.optionSeedSuffix") : null}
              </option>
            ))}
          </select>
        </label>
      </div>

      <details className="advanced-settings">
        <summary className="advanced-settings-summary">
          <span className="advanced-settings-chevron" aria-hidden="true" />
          {t("params.advanced.label")}
        </summary>
      <div className="advanced-settings-body">
        <div className="param-grid">
          {modelInfo.max_pixels ? (
          <label>
            <span className="step-label-header">
              {t("params.maxPixels.label")}
              <span
                className="distill-subtle-tag"
                 title={t("params.maxPixels.oomGuardTitle")}

              >
                {t("params.maxPixels.oomGuardTag")}
              </span>
            </span>
            <input
              type="number"
              min="65536"
              max={modelInfo.max_pixels}
              step="16384"
              value={Number(maxPixels) || modelInfo.max_pixels}
              onChange={(e) => {
                const v = Number(e.target.value);
                if (!e.target.value) return;
                const clamped = Math.max(65536, Math.min(Number(modelInfo.max_pixels), Math.round(v)));
                setMaxPixels(clamped);
                const ratio = Math.sqrt(clamped / (Number(width) * Number(height)));
                if (Number(width) * Number(height) > clamped) {
                  setWidth(Math.max(256, Math.round(Number(width) * ratio / 16) * 16));
                  setHeight(Math.max(256, Math.round(Number(height) * ratio / 16) * 16));
                }
              }}
            />
          </label>
          ) : null}

        {modelInfo.supports_guidance && (
          <label>
            {t("params.guidance.label")}
            <input
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={guidance}
              onChange={(e) => setGuidance(e.target.value)}
            />
          </label>
        )}
        </div>

      {modelInfo.supports_negative && (
        <label>
          {t("params.negative.label")}
          <textarea
            value={negativePrompt}
            onChange={(e) => setNegativePrompt(e.target.value)}
            placeholder={t("params.negative.placeholder")}
            rows={2}
          />
        </label>
      )}

      {/* Only shown when there is a real choice.

          A select with one option is not a control, it is a readout -- and it invites the
          question "what should I pick here?" for a question the engine has already
          answered. Most models expose no sampler list at all (FLUX.2, Krea 2, Z-Image are
          guidance-distilled and always use their native schedule), so the old truthy check
          hid those correctly. A single-sampler model would have slipped through and shown
          a one-item dropdown, which is why the test is a count and not a presence. */}
      {samplerChoices.length > 1 && (
        <label>
          {t("params.sampler.label")}
          <select value={sampler} onChange={(e) => setSampler(e.target.value)}>
            {samplerChoices.map((sm) => (
              <option key={sm} value={sm}>
                {sm}
              </option>
            ))}
          </select>
        </label>
      )}

      {modelInfo.engine === "sdxl" && (
        <label>
          {t("params.deepCache.label")}
          <select value={cacheInterval} onChange={(e) => setCacheInterval(Number(e.target.value))}>
            <option value={1}>{t("params.deepCache.off")}</option>
            <option value={2}>{t("params.deepCache.level2")}</option>
            <option value={3}>{t("params.deepCache.level3")}</option>
          </select>
        </label>
      )}
      </div>
      </details>

      {supportsRef && (
        <fieldset className={`ref-section${refImages.length ? "" : " is-empty"}`}>
          {refImages.length > 0 && (
            <legend>
              {t("params.ref.legend", { count: refImages.length, max: maxRefImages })}
              {supportsMultiRef && <span className="ref-badge-pill">{t("params.ref.inContextBadge")}</span>}
            </legend>
          )}

          <div className={`ref-gallery-row${refImages.length ? "" : " is-empty"}`}>
            {refImages.map((img, idx) => (
              <div className="ref-card" key={img.id || img.path}>
                <div className="ref-thumb-wrapper">
                   {img.preview ? (
                     <img src={img.preview} alt={t("params.ref.thumbAlt", { n: idx + 1 })} />
                   ) : (
                     <span className="ref-preview-missing" aria-label={t("params.ref.previewUnavailable", { n: idx + 1 })}>
                       {idx + 1}
                     </span>
                   )}

                  <span className="ref-index-badge">{t("params.ref.indexBadge", { n: idx + 1 })}</span>
                  <button
                    type="button"
                    className="ref-remove-btn"
                    onClick={() => removeRefImage(idx)}
                    title={t("params.ref.removeTitle")}
                  >
                    ✕
                  </button>
                </div>
                <button
                  type="button"
                  className="ref-insert-chip"
                  onClick={() => insertIntoPrompt(`Image ${idx + 1}`)}
                  title={t("params.ref.insertTitle", { n: idx + 1 })}
                >
                  {t("params.ref.promptTagBtn")}
                </button>
              </div>
            ))}

            {refImages.length < maxRefImages && (
              <label
                className={`ref-add-card${refImages.length ? "" : " is-slim"}`}
                title={t("params.ref.addCardTitle")}
              >
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp,.heic,.HEIC,image/heic,image/heif"
                  multiple={supportsMultiRef}
                  onChange={pickRefImages}
                  style={{ display: "none" }}
                />
                <span className="ref-add-plus">＋</span>
                <span className="ref-add-text">
                  {refImages.length === 0
                    ? t("params.ref.addTextFirst")
                    : t("params.ref.addTextNext", { n: refImages.length + 1 })}
                </span>
              </label>
            )}
          </div>

          {refImages.length > 0 && !supportsMultiRef && (
            <div className="ref-strength">
              <div className="ref-strength-head">
                <span className="ref-strength-label">{t("params.ref.strengthLabel")}</span>
                <input
                  className="ref-strength-input"
                  type="number"
                  min="0.01"
                  max="1"
                  step="0.01"
                  value={Number(refStrength).toFixed(2)}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isFinite(v)) setRefStrength(clampRefStrength(v));
                  }}
                />
              </div>
              <input
                type="range"
                min="0.01"
                max="1"
                step="0.01"
                value={Number(refStrength)}
                onChange={(e) => setRefStrength(Number(e.target.value))}
              />
              <div className="ref-strength-steps">
                <button
                  type="button"
                  className="btn-mini"
                  title={t("params.ref.decrease01Title")}
                  onClick={() => setRefStrength((v) => clampRefStrength(Number(v) - 0.1))}
                >
                  −0.1
                </button>
                <button
                  type="button"
                  className="btn-mini"
                  title={t("params.ref.decrease001Title")}
                  onClick={() => setRefStrength((v) => clampRefStrength(Number(v) - 0.01))}
                >
                  −0.01
                </button>
                <span className="ref-strength-value">{Number(refStrength).toFixed(2)}</span>
                <button
                  type="button"
                  className="btn-mini"
                  title={t("params.ref.increase001Title")}
                  onClick={() => setRefStrength((v) => clampRefStrength(Number(v) + 0.01))}
                >
                  +0.01
                </button>
                <button
                  type="button"
                  className="btn-mini"
                  title={t("params.ref.increase01Title")}
                  onClick={() => setRefStrength((v) => clampRefStrength(Number(v) + 0.1))}
                >
                  +0.1
                </button>
              </div>
            </div>
          )}
          {supportsMultiRef && refImages.length > 0 && (
            <p className="hint">
              💡 <strong>{t("params.ref.multiRefHintLead")}</strong> {t("params.ref.multiRefHintBody")}
            </p>
          )}
        </fieldset>
      )}
    </div>
  );
}

export default memo(GenerationParams);

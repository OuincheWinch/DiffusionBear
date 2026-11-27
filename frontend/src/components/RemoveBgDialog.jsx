import { useEffect, useRef, useState } from "react";
import Dialog from "./Dialog";
import { useI18n } from "../i18n/I18nContext";

const QUALITY_OPTIONS = [
  { value: 512, labelKey: "fill.rmbgQualityFast" },
  { value: 768, labelKey: "fill.rmbgQualityHq" },
  { value: 0, labelKey: "fill.rmbgQualityNative" },
];

export default function RemoveBgDialog({
  open,
  onClose,
  onConfirm,
  busy = false,
  error = null,
}) {
  const { t } = useI18n();
  const [subject, setSubject] = useState("");
  const [maxSide, setMaxSide] = useState(512);
  const inputRef = useRef(null);

  useEffect(() => {
    if (open) {
      setSubject("");
      setMaxSide(512);
    }
  }, [open]);

  function handleSubmit(e) {
    e.preventDefault();
    if (busy) return;
    onConfirm?.({
      subject: subject.trim() || null,
      maxSide: maxSide > 0 ? maxSide : null,
    });
  }

  return (
    <Dialog
      open={open}
      onClose={busy ? undefined : onClose}
      bodyClassName="modal-body--stack"
      ariaLabel={t("fill.rmbgTitle")}
      initialFocusRef={inputRef}
      closeOnBackdrop={!busy}
    >
      <form onSubmit={handleSubmit}>
        <h3 style={{ margin: "0 0 8px", fontSize: "1.05rem" }}>{t("fill.rmbgTitle")}</h3>
        <p style={{ margin: "0 0 14px", fontSize: "0.88rem", opacity: 0.8, lineHeight: 1.5 }}>
          {t("fill.rmbgDescription")}
        </p>

        <div style={{ marginBottom: "14px" }}>
          <span style={{ display: "block", fontSize: "0.84rem", fontWeight: 600, marginBottom: "6px" }}>
            {t("fill.rmbgQualityLabel")}
          </span>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "6px" }}>
            {QUALITY_OPTIONS.map((opt) => {
              const selected = maxSide === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  disabled={busy}
                  onClick={() => setMaxSide(opt.value)}
                  className={`action-btn${selected ? " primary" : ""}`}
                  style={{
                    justifyContent: "center",
                    padding: "7px 8px",
                    fontSize: "0.82rem",
                  }}
                >
                  {t(opt.labelKey)}
                </button>
              );
            })}
          </div>
        </div>

        <label
          htmlFor="rmbg-subject-input"
          style={{ display: "block", fontSize: "0.84rem", fontWeight: 600, marginBottom: "6px" }}
        >
          {t("fill.rmbgSubjectLabel")}
        </label>
        <input
          id="rmbg-subject-input"
          ref={inputRef}
          type="text"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          disabled={busy}
          placeholder={t("fill.rmbgSubjectPlaceholder")}
          style={{
            width: "100%",
            boxSizing: "border-box",
            padding: "9px 11px",
            borderRadius: "8px",
            border: "1px solid var(--border)",
            background: "var(--surface-2)",
            color: "inherit",
            fontSize: "0.9rem",
            marginBottom: "8px",
          }}
        />
        <p style={{ margin: "0 0 14px", fontSize: "0.78rem", opacity: 0.65 }}>
          {t("fill.rmbgSubjectHint")}
        </p>

        {error && (
          <p style={{ margin: "0 0 12px", fontSize: "0.84rem", color: "var(--danger)" }}>{error}</p>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
          <button
            type="button"
            className="action-btn"
            onClick={onClose}
            disabled={busy}
          >
            {t("fill.cancel")}
          </button>
          <button
            type="submit"
            className="action-btn primary"
            disabled={busy}
          >
            {busy ? t("fill.rmbgRunning") : t("fill.rmbgSubmit")}
          </button>
        </div>
      </form>
    </Dialog>
  );
}

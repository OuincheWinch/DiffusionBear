import { stopRmbg, useRmbgStatus } from "../hooks/useRmbgStatus";
import { useI18n } from "../i18n/I18nContext";

export default function RemoveBgProgress({ active }) {
  const { t } = useI18n();
  const status = useRmbgStatus();

  if (!active && !status?.running) return null;

  const step = Number(status?.step || 0);
  const steps = Number(status?.steps || 4);
  const elapsed = status?.elapsed ? `${Math.round(status.elapsed)}s` : "";
  const detail = status?.detail || t("fill.rmbgDescription");
  const ratio = step > 0 && steps > 0 ? Math.min(1, Math.max(0.08, step / steps)) : 0.12;

  return (
    <div
      role="status"
      aria-live="polite"
      style={{
        marginBottom: "10px",
        padding: "10px 12px",
        borderRadius: "10px",
        border: "1px solid rgba(168, 85, 247, 0.45)",
        background: "rgba(168, 85, 247, 0.10)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "8px", marginBottom: "6px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.86rem", fontWeight: 600 }}>
          <span
            aria-hidden="true"
            style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              background: "#c084fc",
              boxShadow: "0 0 8px #c084fc",
              display: "inline-block",
            }}
          />
          <span>{t("fill.rmbgRunning")}</span>
          {step > 0 && (
            <span style={{ opacity: 0.9, fontFamily: "ui-monospace, monospace", fontSize: "0.8rem" }}>
              {step}/{steps}
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {elapsed && (
            <span style={{ fontSize: "0.78rem", opacity: 0.75, fontFamily: "ui-monospace, monospace" }}>
              {elapsed}
            </span>
          )}
          <button
            type="button"
            onClick={stopRmbg}
            title={t("generate.btn.cancelTitle")}
            style={{
              background: "rgba(239, 68, 68, 0.16)",
              border: "1px solid rgba(239, 68, 68, 0.4)",
              color: "var(--danger)",
              borderRadius: "6px",
              padding: "2px 8px",
              fontSize: "0.75rem",
              fontWeight: 600,
              cursor: "pointer",
            }}
          >
            {t("generate.btn.cancel")}
          </button>
        </div>
      </div>
      <div style={{ fontSize: "0.78rem", opacity: 0.85, marginBottom: "8px" }}>
        {detail}
      </div>
      <div
        style={{
          height: "4px",
          borderRadius: "999px",
          background: "var(--overlay-4)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: "100%",
            background: "linear-gradient(90deg, #a855f7, #38bdf8)",
            transformOrigin: "left center",
            transform: `scaleX(${ratio})`,
            transition: "transform 0.4s ease",
          }}
        />
      </div>
    </div>
  );
}

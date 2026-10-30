import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";

function TokenRow({
  provider,
  label,
  placeholder,
  configured,
  showInput,
  value,
  setValue,
  onToggle,
  onSave,
  saving,
  feedback,
  inputRef,
  attention,
  t,
}) {
  return (
    <div className={`token-manager-row token-manager-${provider}${attention ? " needs-attention" : ""}`}>
      <div className="token-status-group">
        <span className="civitai-token-status">
          {configured ? (
            <svg
              className="token-status-icon"
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              style={{ verticalAlign: "text-bottom", marginRight: "0.35rem", opacity: 0.85 }}
            >
              <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
          ) : (
            <svg
              className="token-status-icon"
              width="13"
              height="13"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
              style={{ verticalAlign: "text-bottom", marginRight: "0.35rem", opacity: 0.45 }}
            >
              <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
              <path d="M7 11V7a5 5 0 0 1 9.9-1" />
            </svg>
          )}
          {label}: {configured ? t("tokens.configured") : t("tokens.notSet")}
        </span>
        <button type="button" className="btn-token-toggle" onClick={onToggle}>
          {showInput
            ? t("app.cancel")
            : configured
              ? t("tokens.editBtn")
              : t("tokens.setBtn", { label })}
        </button>
      </div>
      {showInput && (
        <div className="civitai-token-input-row">
          <input
            ref={inputRef}
            type="password"
            placeholder={placeholder}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            autoComplete="off"
          />
          <button
            type="button"
            className="btn-token-save"
            onClick={onSave}
            disabled={saving}
          >
            {saving
              ? t("tokens.savingBtn")
              : value.trim()
                ? t("tokens.saveBtn", { label })
                : t("tokens.clearBtn")}
          </button>
        </div>
      )}
      {feedback && (
        <p className={`civitai-feedback ${feedback.type}`}>{feedback.text}</p>
      )}
    </div>
  );
}

export default function TokenManager({
  autofocus = null,
  onTokenSaved = null,
  className = "",
}) {
  const { t } = useI18n();
  const [civitaiToken, setCivitaiToken] = useState("");
  const [civitaiConfigured, setCivitaiConfigured] = useState(false);
  const [showCivitai, setShowCivitai] = useState(false);
  const [hfToken, setHfToken] = useState("");
  const [hfConfigured, setHfConfigured] = useState(false);
  const [showHf, setShowHf] = useState(false);
  const [saving, setSaving] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [attention, setAttention] = useState(null);
  const containerRef = useRef(null);
  const civitaiInputRef = useRef(null);
  const hfInputRef = useRef(null);

  useEffect(() => {
    api("/api/civitai/token").then((r) => setCivitaiConfigured(Boolean(r.configured))).catch(() => {});
    api("/api/hf/token").then((r) => setHfConfigured(Boolean(r.configured))).catch(() => {});
  }, []);

  useEffect(() => {
    const provider = typeof autofocus === "string" ? autofocus : autofocus?.provider;
    if (!provider) return;
    if (provider === "civitai") setShowCivitai(true);
    if (provider === "hf") setShowHf(true);
    setAttention(provider);

    const frame = requestAnimationFrame(() => {
      containerRef.current?.scrollIntoView({ block: "center", behavior: "smooth" });
      const input = provider === "civitai" ? civitaiInputRef.current : hfInputRef.current;
      input?.focus({ preventScroll: true });
    });
    const timer = setTimeout(() => setAttention(null), 4000);
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(timer);
    };
  }, [autofocus]);

  async function save(provider, label) {
    setSaving(provider);
    setFeedback(null);
    try {
      const value = provider === "civitai" ? civitaiToken : hfToken;
      const res = await api(provider === "civitai" ? "/api/civitai/token" : "/api/hf/token", {
        method: "POST",
        body: JSON.stringify({ token: value.trim() }),
      });
      if (provider === "civitai") {
        setCivitaiConfigured(Boolean(res.configured));
        setShowCivitai(false);
        setCivitaiToken("");
      } else {
        setHfConfigured(Boolean(res.configured));
        setShowHf(false);
        setHfToken("");
      }
      setFeedback({
        type: "success",
        text: res.configured
          ? t("tokens.savedFeedback", { label })
          : t("tokens.clearedFeedback", { label }),
      });
      onTokenSaved?.({ provider, configured: Boolean(res.configured) });
      setTimeout(() => setFeedback(null), 3500);
    } catch (e) {
      setFeedback({ type: "error", text: t("tokens.saveFailed", { label, error: e.message }) });
    } finally {
      setSaving(null);
    }
  }

  return (
    <div ref={containerRef} className={`token-manager ${className}`}>
      <TokenRow
        provider="civitai"
        label="Civitai"
        placeholder={t("tokens.civitaiPlaceholder")}
        configured={civitaiConfigured}
        showInput={showCivitai}
        value={civitaiToken}
        setValue={setCivitaiToken}
        onToggle={() => setShowCivitai((v) => !v)}
        onSave={() => save("civitai", t("tokens.labelCivitai"))}
        saving={saving === "civitai"}
        feedback={null}
        inputRef={civitaiInputRef}
        attention={attention === "civitai"}
        t={t}
      />
      <TokenRow
        provider="hf"
        label="Hugging Face"
        placeholder={t("tokens.hfPlaceholder")}
        configured={hfConfigured}
        showInput={showHf}
        value={hfToken}
        setValue={setHfToken}
        onToggle={() => setShowHf((v) => !v)}
        onSave={() => save("hf", t("tokens.labelHf"))}
        saving={saving === "hf"}
        feedback={null}
        inputRef={hfInputRef}
        attention={attention === "hf"}
        t={t}
      />
      {feedback && <p className={`civitai-feedback ${feedback.type}`}>{feedback.text}</p>}
    </div>
  );
}

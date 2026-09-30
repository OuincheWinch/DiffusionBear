import { useEffect, useState } from "react";
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
  t,
}) {
  return (
    <div className={`token-manager-row token-manager-${provider}`}>
      <div className="token-status-group">
        <span className="civitai-token-status">
          {configured ? "🔑 " : "🔓 "}{label}:{" "}
          {configured ? t("tokens.configured") : t("tokens.notSet")}
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

  useEffect(() => {
    api("/api/civitai/token").then((r) => setCivitaiConfigured(Boolean(r.configured))).catch(() => {});
    api("/api/hf/token").then((r) => setHfConfigured(Boolean(r.configured))).catch(() => {});
  }, []);

  useEffect(() => {
    if (autofocus) {
      const timer = setTimeout(() => {
        if (autofocus === "civitai") setShowCivitai(true);
        if (autofocus === "hf") setShowHf(true);
      }, 0);
      return () => clearTimeout(timer);
    }
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
    <div className={`token-manager ${className}`}>
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
        t={t}
      />
      {feedback && <p className={`civitai-feedback ${feedback.type}`}>{feedback.text}</p>}
    </div>
  );
}

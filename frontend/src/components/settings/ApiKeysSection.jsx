import { useCallback, useEffect, useState } from "react";
import { API_TOKEN_STORAGE_KEY, api } from "../../api";
import { copyText } from "../../clipboard";
import { useI18n } from "../../i18n/I18nContext";
import Dialog from "../Dialog";
import ApiConnectionGuide from "./ApiConnectionGuide";
import RemoteConnectionSection from "./RemoteConnectionSection";

function masked(value) {
  if (!value?.prefix) return "—";
  return `${value.prefix}…${value.last4 || ""}`;
}

function formatTime(value) {
  if (!value) return "—";
  try {
    return new Date(value * 1000).toLocaleString();
  } catch {
    return "—";
  }
}

export default function ApiKeysSection({ onFeedback }) {
  const { t } = useI18n();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [label, setLabel] = useState("");
  const [creating, setCreating] = useState(false);
  const [reveal, setReveal] = useState(null);
  const [copied, setCopied] = useState(false);
  const [portStatus, setPortStatus] = useState(null);
  const [portDraft, setPortDraft] = useState("8001");
  const [portMessage, setPortMessage] = useState(null);
  const [savingPort, setSavingPort] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api("/api/security/keys"));
      try {
        const status = await api("/api/system/port");
        setPortStatus(status);
        setPortDraft(String(status.configured || 8001));
      } catch {
        setPortStatus(null);
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const hasUsableKey = Boolean(data?.environment?.present || data?.legacy?.present || data?.keys?.length);

  async function createKey(event) {
    event.preventDefault();
    if (creating) return;
    setCreating(true);
    setError(null);
    try {
      const result = await api("/api/security/keys", {
        method: "POST",
        body: JSON.stringify({ label }),
      });
      const first = !hasUsableKey;
      if (first) {
        localStorage.setItem(API_TOKEN_STORAGE_KEY, result.key);
      }
      setReveal({ ...result, autoApplied: first });
      setLabel("");
      setCopied(false);
      await load();
      onFeedback?.({ type: "success", text: t("settings.api.created") });
    } catch (err) {
      const message = String(err.message || err);
      setError(
        message.includes("label")
          ? t("settings.api.duplicateLabel")
          : message || t("settings.api.error"),
      );
    } finally {
      setCreating(false);
    }
  }

  async function toggleApiMode(event) {
    const enabled = event.target.checked;
    setError(null);
    try {
      const result = await api("/api/security/mode", {
        method: "PUT",
        body: JSON.stringify({ enabled }),
      });
      setData((previous) => (previous ? { ...previous, enabled: result.enabled } : previous));
      window.dispatchEvent(new CustomEvent("mlx:api-mode", { detail: { enabled } }));
      onFeedback?.({
        type: "success",
        text: t(enabled ? "settings.api.modeOn" : "settings.api.modeOff"),
      });
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function savePort(event) {
    event.preventDefault();
    const port = Number(portDraft);
    if (!Number.isInteger(port) || port < 1024 || port > 65535) {
      setPortMessage({ type: "error", text: t("settings.api.portInvalid") });
      return;
    }
    setSavingPort(true);
    setPortMessage(null);
    try {
      const status = await api("/api/system/port", {
        method: "PUT",
        body: JSON.stringify({ port }),
      });
      setPortStatus(status);
      setPortMessage({
        type: "success",
        text: status.restart_required
          ? t("settings.api.portRestart", { url: status.next_base_url })
          : t("settings.api.portSaved"),
      });
    } catch (err) {
      setPortMessage({ type: "error", text: err.message || String(err) });
    } finally {
      setSavingPort(false);
    }
  }

  async function copyKey() {
    if (!reveal?.key) return;
    setCopied(await copyText(reveal.key));
  }

  function useHere() {
    if (!reveal?.key) return;
    localStorage.setItem(API_TOKEN_STORAGE_KEY, reveal.key);
    onFeedback?.({ type: "success", text: t("settings.api.useHere") });
  }

  async function revoke(item) {
    const current = data?.current?.id === item.id;
    const warning = current
      ? `${t("settings.api.revokeConfirm")}\n\n${t("settings.api.currentWarning")}`
      : t("settings.api.revokeConfirm");
    if (!window.confirm(warning)) return;
    setError(null);
    try {
      await api(`/api/security/keys/${encodeURIComponent(item.id)}`, {
        method: "DELETE",
      });
      await load();
      onFeedback?.({ type: "success", text: t("settings.api.revoked") });
    } catch (err) {
      setError(
        String(err.message || err).includes("last")
          ? t("settings.api.lastKeyBlocked")
          : err.message || t("settings.api.error"),
      );
    }
  }

  return (
    <section className="params-section">
      {data && (
        <div className={`api-mode-hero${data.enabled ? " is-on" : ""}`}>
          <div className="api-mode-hero__copy">
            <span className="api-mode-hero__eyebrow">{t("settings.api.title")}</span>
            <h2>{t("settings.api.mode")}</h2>
            <p>{t("settings.api.modeHeroDesc")}</p>
            <p className="api-mode-hero__state">
              {data.enabled ? t("settings.api.modeOnHint") : t("settings.api.modeOffHint")}
            </p>
          </div>
          <label className="api-mode-switch">
            <input
              type="checkbox"
              checked={Boolean(data.enabled)}
              onChange={toggleApiMode}
              aria-label={t("settings.api.mode")}
            />
            <span className="api-mode-switch__track" aria-hidden="true">
              <span className="api-mode-switch__knob" />
            </span>
            <span className="api-mode-switch__label">
              {data.enabled ? t("settings.api.on") : t("settings.api.off")}
            </span>
          </label>
        </div>
      )}

      {portStatus && (
        <div className="api-port-card">
          <h4>{t("settings.api.portTitle")}</h4>
          <p className="params-hint">{t("settings.api.portDesc")}</p>
          <form className="api-key-create" onSubmit={savePort}>
            <label className="settings-field">
              <span>{t("settings.api.portLabel")}</span>
              <input
                type="number"
                min={1024}
                max={65535}
                value={portDraft}
                onChange={(e) => setPortDraft(e.target.value)}
              />
            </label>
            <button type="submit" className="btn btn--primary" disabled={savingPort || !data?.enabled}>
              {savingPort ? t("settings.api.portSaving") : t("settings.api.portSave")}
            </button>
          </form>
          <p className="params-hint">
            {t("settings.api.portActive", { active: portStatus.active ?? "?" })}
            {" · "}
            {t("settings.api.portConfigured", { configured: portStatus.configured })}
          </p>
          {portMessage && (
            <p className={portMessage.type === "error" ? "error-text" : "params-hint"} role="status">
              {portMessage.text}
            </p>
          )}
        </div>
      )}

      {!data && error && (
        <div className="api-auth-onboarding" role="note">
          <strong>{t("settings.api.authRequiredTitle")}</strong>
          <p>{t("settings.api.authRequiredBody")}</p>
        </div>
      )}

      <RemoteConnectionSection onFeedback={onFeedback} />
      <ApiConnectionGuide />
      <h3>{t("settings.api.title")}</h3>
      <p className="params-section-desc">{t("settings.api.desc")}</p>

      {error && <p className="error-text" role="alert">{error}</p>}
      {loading && !data ? <p className="hint">{t("app.loading")}</p> : null}

      {data && (
        <>
          <div className="api-key-cards">
            <div className="api-key-card">
              <div className="api-key-card__head">
                <strong>{t("settings.api.environment")}</strong>
                <span className="settings-badge">
                  {data.environment.present ? t("settings.api.readOnly") : t("settings.api.absent")}
                </span>
              </div>
              <code>{data.environment.present ? masked(data.environment) : "—"}</code>
            </div>
            <div className="api-key-card">
              <div className="api-key-card__head">
                <strong>{t("settings.api.legacy")}</strong>
                <span className="settings-badge">
                  {data.legacy.present ? t("settings.api.readOnly") : t("settings.api.absent")}
                </span>
              </div>
              <code>{data.legacy.present ? masked(data.legacy) : "—"}</code>
            </div>
          </div>

          <h4 className="api-keys-subtitle">{t("settings.api.managed")}</h4>
          {!data.keys.length ? (
            <p className="hint">{t("settings.api.empty")}</p>
          ) : (
            <div className="api-key-list">
              {data.keys.map((item) => (
                <div className="api-key-row" key={item.id}>
                  <div>
                    <strong>{item.label}</strong>
                    {data.current?.id === item.id && (
                      <span className="settings-badge">{t("settings.api.current")}</span>
                    )}
                    <code className="api-key-mask">{item.prefix}…{item.last4}</code>
                    <span className="api-key-meta">
                      {t("settings.api.createdAt")}: {formatTime(item.created_at)} ·{" "}
                      {t("settings.api.lastUsed")}: {formatTime(item.last_used_at)}
                    </span>
                  </div>
                  <button type="button" className="btn btn--ghost" onClick={() => revoke(item)}>
                    {t("settings.api.revoke")}
                  </button>
                </div>
              ))}
            </div>
          )}

          <form className="api-key-create" onSubmit={createKey}>
            <label className="settings-field">
              <span>{t("settings.api.label")}</span>
              <input
                type="text"
                maxLength={64}
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder={t("settings.api.labelPlaceholder")}
                required
              />
            </label>
            <button type="submit" className="btn btn--primary" disabled={creating || !label.trim()}>
              {creating ? t("settings.api.creating") : t("settings.api.create")}
            </button>
          </form>
        </>
      )}

      <Dialog
        open={Boolean(reveal)}
        onClose={() => {
          setReveal(null);
          setCopied(false);
        }}
        ariaLabel={t("settings.api.onlyOnce")}
      >
        <h3>{t("settings.api.onlyOnce")}</h3>
        <p className="hint">{t("settings.api.onlyOnceHint")}</p>
        <pre className="api-key-reveal"><code>{reveal?.key}</code></pre>
        <div className="api-key-actions">
          <button type="button" className="btn btn--primary" onClick={copyKey}>
            {copied ? t("settings.api.copied") : t("settings.api.copy")}
          </button>
          <button type="button" className="btn btn--ghost" onClick={useHere}>
            {t("settings.api.useHere")}
          </button>
        </div>
        {reveal?.autoApplied && (
          <p className="params-hint">{t("settings.api.autoApplied")}</p>
        )}
      </Dialog>
    </section>
  );
}

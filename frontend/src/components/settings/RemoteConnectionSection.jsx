import { useMemo, useState } from "react";
import {
  API_BASE,
  API_BASE_STORAGE_KEY,
  API_TOKEN_STORAGE_KEY,
  normalizeApiBase,
} from "../../api";
import { useI18n } from "../../i18n/I18nContext";

function currentLabel(base) {
  return base || "local";
}

export default function RemoteConnectionSection({ onFeedback }) {
  const { t } = useI18n();
  const initialBase = useMemo(() => {
    try {
      return globalThis.localStorage?.getItem(API_BASE_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  }, []);
  const initialKey = useMemo(() => {
    try {
      return globalThis.localStorage?.getItem(API_TOKEN_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  }, []);
  const [baseUrl, setBaseUrl] = useState(initialBase);
  const [apiKey, setApiKey] = useState(initialKey);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState(null);

  async function probe(base, key) {
    const health = await fetch(`${base}/api/version`, { cache: "no-store" });
    if (!health.ok) throw new Error(`HTTP ${health.status}`);
    const headers = key ? { Authorization: `Bearer ${key}` } : {};
    const models = await fetch(`${base}/api/models`, {
      cache: "no-store",
      headers,
    });
    if (models.status === 401) {
      throw new Error(t("settings.api.badKey"));
    }
    if (!models.ok) throw new Error(`HTTP ${models.status}`);
  }

  async function testConnection() {
    setTesting(true);
    setMessage(null);
    const base = normalizeApiBase(baseUrl);
    if (baseUrl.trim() && !base) {
      setMessage({ type: "error", text: t("settings.api.invalidUrl") });
      setTesting(false);
      return;
    }
    try {
      await probe(base, apiKey.trim());
      setMessage({ type: "success", text: t("settings.api.testOk") });
    } catch (err) {
      setMessage({
        type: "error",
        text: `${t("settings.api.testFailed")} ${err?.message || ""}`.trim(),
      });
    } finally {
      setTesting(false);
    }
  }

  async function saveConnection() {
    const base = normalizeApiBase(baseUrl);
    const key = apiKey.trim();
    if (baseUrl.trim() && !base) {
      setMessage({ type: "error", text: t("settings.api.invalidUrl") });
      return;
    }
    if (base && !key) {
      setMessage({ type: "error", text: t("settings.api.keyRequired") });
      return;
    }
    if (base) {
      setTesting(true);
      setMessage(null);
      try {
        await probe(base, key);
      } catch (err) {
        setMessage({
          type: "error",
          text: `${t("settings.api.testFailed")} ${err?.message || ""}`.trim(),
        });
        setTesting(false);
        return;
      }
      setTesting(false);
    }
    try {
      if (base) {
        localStorage.setItem(API_BASE_STORAGE_KEY, base);
      } else {
        localStorage.removeItem(API_BASE_STORAGE_KEY);
      }
      if (key) {
        localStorage.setItem(API_TOKEN_STORAGE_KEY, key);
      } else {
        localStorage.removeItem(API_TOKEN_STORAGE_KEY);
      }
    } catch {
      setMessage({ type: "error", text: t("settings.api.storageError") });
      return;
    }
    onFeedback?.({ type: "success", text: t("settings.api.savedReload") });
    window.location.reload();
  }

  function resetLocal() {
    try {
      localStorage.removeItem(API_BASE_STORAGE_KEY);
      localStorage.removeItem(API_TOKEN_STORAGE_KEY);
    } catch {
      // Ignore storage failures; reload still returns to the served origin.
    }
    window.location.reload();
  }

  return (
    <section className="params-section api-connection">
      <h3>{t("settings.api.connectionTitle")}</h3>
      <p className="params-section-desc">{t("settings.api.connectionDesc")}</p>
      <p className="params-hint">
        {t("settings.api.currentBackend")}: <code>{currentLabel(API_BASE)}</code>
      </p>

      <div className="settings-grid">
        <label className="settings-field">
          <span>{t("settings.api.backendUrl")}</span>
          <input
            type="url"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder={t("settings.api.backendUrlPlaceholder")}
            spellCheck={false}
            autoComplete="url"
          />
        </label>
        <label className="settings-field">
          <span>{t("settings.api.keyField")}</span>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder={t("settings.api.keyFieldPlaceholder")}
            autoComplete="new-password"
            spellCheck={false}
          />
        </label>
      </div>

      {message && (
        <p className={message.type === "error" ? "error-text" : "params-hint"} role="status">
          {message.text}
        </p>
      )}

      <div className="api-key-actions">
        <button type="button" className="btn btn--ghost" onClick={testConnection} disabled={testing}>
          {testing ? t("settings.api.testing") : t("settings.api.test")}
        </button>
        <button type="button" className="btn btn--primary" onClick={saveConnection}>
          {t("settings.api.saveConnection")}
        </button>
        <button type="button" className="btn btn--ghost" onClick={resetLocal}>
          {t("settings.api.resetLocal")}
        </button>
      </div>
      <p className="params-hint">{t("settings.api.connectionHint")}</p>
    </section>
  );
}

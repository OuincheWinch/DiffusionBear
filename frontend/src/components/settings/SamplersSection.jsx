import { useEffect, useState } from "react";
import { api, pickPath } from "../../api";
import { useI18n } from "../../i18n/I18nContext";

// Custom-sampler management: install a .py from a local file (native picker)
// or a URL, list what is declared, remove it. Protected pack entries (ours)
// carry a lock and refuse deletion; the app re-extracts them at every startup,
// so a broken one heals itself.
export default function SamplersSection() {
  const { t } = useI18n();
  const [samplers, setSamplers] = useState([]);
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState(null);

  async function refresh() {
    try {
      const res = await api("/api/samplers");
      setSamplers(res.samplers || []);
    } catch (err) {
      setFeedback({ type: "error", text: String(err?.message || err) });
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  function applyResult(entry) {
    setName("");
    setFeedback({
      type: "success",
      text: t("settings.samplers.installed", { name: entry.name, engines: (entry.engines || []).join(", ") }),
    });
    refresh();
  }

  function reportError(err) {
    setFeedback({ type: "error", text: err?.detail || String(err?.message || err) });
  }

  async function pickLocal() {
    setBusy(true);
    setFeedback(null);
    try {
      const res = await pickPath({
        kind: "file",
        title: t("settings.samplers.pickTitle"),
        extensions: ["py"],
      });
      if (!res?.path) return;
      const body = { path: res.path };
      if (name.trim()) body.name = name.trim();
      applyResult(await api("/api/samplers/install-local", {
        method: "POST",
        body: JSON.stringify(body),
      }));
    } catch (err) {
      reportError(err);
    } finally {
      setBusy(false);
      setTimeout(() => setFeedback(null), 5000);
    }
  }

  async function install() {
    if (!url.trim()) return;
    setBusy(true);
    setFeedback(null);
    try {
      const body = { url: url.trim() };
      if (name.trim()) body.name = name.trim();
      applyResult(await api("/api/samplers/install", {
        method: "POST",
        body: JSON.stringify(body),
      }));
      setUrl("");
    } catch (err) {
      reportError(err);
    } finally {
      setBusy(false);
      setTimeout(() => setFeedback(null), 5000);
    }
  }

  async function remove(name) {
    setFeedback(null);
    try {
      await api(`/api/samplers/${name}`, { method: "DELETE" });
      setFeedback({ type: "success", text: t("settings.samplers.removed", { name }) });
      refresh();
    } catch (err) {
      const detail = err?.detail || String(err?.message || err);
      setFeedback({ type: "error", text: detail });
    } finally {
      setTimeout(() => setFeedback(null), 5000);
    }
  }

  return (
    <div className="samplers-section">
      {feedback && (
        <p className={feedback.type === "error" ? "error" : "hint"} role="alert">
          {feedback.text}
        </p>
      )}
      <p className="params-section-desc">{t("settings.samplers.desc")}</p>
      <div className="samplers-install-row">
        <button type="button" className="samplers-pick" disabled={busy} onClick={pickLocal}>
          {busy ? t("settings.samplers.installing") : t("settings.samplers.pickFile")}
        </button>
        <input
          type="text"
          value={name}
          placeholder={t("settings.samplers.namePlaceholder")}
          onChange={(e) => setName(e.target.value)}
        />
      </div>
      <div className="samplers-install-row">
        <input
          type="url"
          value={url}
          placeholder={t("settings.samplers.urlPlaceholder")}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button type="button" disabled={busy || !url.trim()} onClick={install}>
          {t("settings.samplers.install")}
        </button>
      </div>
      <ul className="samplers-list">
        {samplers.length === 0 && <li className="hint">{t("settings.samplers.empty")}</li>}
        {samplers.map((s) => (
          <li key={s.name} className={!s.intact ? "samplers-broken" : ""}>
            <span className="samplers-name">{s.name}</span>
            <span className="samplers-engines">{(s.engines || []).join(", ")}</span>
            {s.protected ? (
              <span className="samplers-protected" title={t("settings.samplers.protectedTitle")}>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ marginRight: "0.25rem", verticalAlign: "middle" }}><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
                {t("settings.samplers.protected")}
              </span>
            ) : (
              <button type="button" onClick={() => remove(s.name)}>
                {t("settings.samplers.remove")}
              </button>
            )}
            {!s.intact && <em className="hint">{t("settings.samplers.broken")}</em>}
          </li>
        ))}
      </ul>
      <p className="params-hint">{t("settings.samplers.defaultHint")}</p>
      <p className="params-hint">{t("settings.samplers.kreaHint")}</p>
    </div>
  );
}

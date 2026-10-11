import { useState, useEffect } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";

export default function DetectedModelsSection({ onModelsChanged, onFeedback }) {
  const { t } = useI18n();
  const [detected, setDetected] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState(null);

  async function load() {
    setLoading(true);
    setErr(null);
    try {
      const data = await api("/api/hf/models/detected");
      setDetected(data?.items || []);
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  async function deleteModel(name) {
    if (!window.confirm(t("models.detected.confirmDelete", { model: name }))) return;
    setBusy(name);
    setErr(null);
    try {
      await api(`/api/hf/models/${encodeURIComponent(name)}`, { method: "DELETE" });
      onFeedback?.({ type: "success", text: t("models.detected.deleted", { model: name }) });
      await load();
      onModelsChanged?.();
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusy(null);
    }
  }

  async function convertModel(name) {
    setBusy(name);
    setErr(null);
    try {
      const result = await api("/api/hf/models/convert", {
        method: "POST",
        body: JSON.stringify({ install_name: name }),
      });
      onFeedback?.({ type: "success", text: t("models.detected.converted", { model: name }) });
      await load();
      onModelsChanged?.();
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusy(null);
    }
  }

  async function registerModel(item) {
    if (!item.usable_as) return;
    setBusy(item.name);
    setErr(null);
    try {
      await api("/api/hf/models/register", {
        method: "POST",
        body: JSON.stringify({ repo_id: item.repo_id || item.name, model_id: item.usable_as }),
      });
      onFeedback?.({ type: "success", text: t("models.detected.registered", { model: item.usable_as_label }) });
      await load();
      onModelsChanged?.();
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusy(null);
    }
  }

  if (loading) {
    return <div className="settings-row"><span className="hint">{t("models.detected.loading")}</span></div>;
  }

  if (!detected.length) return null;

  return (
    <section className="params-section">
      <h3>{t("models.detected.title")}</h3>
      <p className="params-section-desc">{t("models.detected.desc")}</p>
      {err && <p className="error" role="alert">{err}</p>}
      <div className="models-table">
        <div className="models-table-head">
          <span>{t("settings.models.colModel")}</span>
          <span>{t("models.detected.colArchitecture")}</span>
          <span>{t("settings.models.colOnDisk")}</span>
          <span>{t("models.detected.colRunnableAs")}</span>
          <span>{t("settings.models.colActions")}</span>
        </div>
        {detected.map((item) => (
          <div className="models-table-row" key={item.name}>
            <span className="models-name">
              <strong>{item.name}</strong>
              {item.kind !== "diffusion" && (
                <span className="hf-chip subtle">{t(`models.kind.${item.kind}`)}</span>
              )}
            </span>
            <span className="models-arch">
              <span className="hf-chip">{item.architecture_label}</span>
              {item.quantization && <span className="hf-chip">{item.quantization}</span>}
            </span>
            <span className="models-size">{formatBytes(item.bytes || 0)}</span>
            <span className="models-runnable">
              {item.usable_as ? (
                <span className="settings-badge ok">{item.usable_as_label}</span>
              ) : (
                <span className="hf-chip notrunnable" title={item.usable_as_label || ""}>
                  {t("models.notRunnable")}
                </span>
              )}
            </span>
            <span className="models-actions">
              {item.usable_as && !item.registered ? (
                <button
                  type="button"
                  className="btn-mini"
                  disabled={busy === item.name}
                  onClick={() => registerModel(item)}
                >
                  {t("models.useFor", { model: item.usable_as_label })}
                </button>
              ) : item.registered ? (
                <span className="settings-badge ok">{t("models.alreadyRegistered", { model: item.registered })}</span>
              ) : (
                <span className="hf-chip notrunnable">{t("models.notRunnable")}</span>
              )}
              {item.architecture === "sdxl" && (
                <button
                  type="button"
                  className="btn-mini"
                  disabled={busy === item.name}
                  title={t("models.detected.convertTitle")}
                  onClick={() => convertModel(item.name)}
                >
                  {busy === item.name ? t("models.converting") : t("models.detected.convert")}
                </button>
              )}
              <button
                type="button"
                className="btn-mini btn-danger"
                disabled={busy === item.name}
                title={t("models.detected.deleteTitle")}
                onClick={() => deleteModel(item.name)}
              >
                {busy === item.name ? t("models.deleting") : t("models.detected.delete")}
              </button>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
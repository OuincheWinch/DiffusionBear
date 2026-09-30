import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";

export default function HfCacheSection({ onFeedback }) {
  const { t } = useI18n();
  const [cache, setCache] = useState(null);
  const [loading, setLoading] = useState(true);
  const [clearing, setClearing] = useState(null);
  const [err, setErr] = useState(null);

  async function refresh() {
    setLoading(true);
    try {
      const data = await api("/api/hf/cache");
      setCache(data);
      setErr(data.error || null);
    } catch (e) {
      setErr(e.message || String(e));
      setCache(null);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function clearRepo(repo) {
    if (!window.confirm(t("settings.hfCache.clearConfirm", { repo: repo.repo_id, size: formatBytes(repo.size_bytes) }))) {
      return;
    }
    setClearing(repo.repo_id);
    try {
      const res = await api("/api/hf/cache/clear", {
        method: "POST",
        body: JSON.stringify({ repo_id: repo.repo_id }),
      });
      onFeedback?.({ type: "success", text: t("settings.hfCache.freed", { size: formatBytes(res.freed_bytes) }) });
      await refresh();
    } catch (e) {
      onFeedback?.({ type: "error", text: e.message || String(e) });
    } finally {
      setClearing(null);
    }
  }

  if (loading && !cache) {
    return <div className="settings-row"><span className="hint">{t("settings.hfCache.scanning")}</span></div>;
  }
  if (err && !cache) {
    return <div className="settings-row"><span className="error">{err}</span></div>;
  }

  const repos = cache?.repos || [];
  return (
    <div className="settings-row">
      <div className="settings-inline">
        <span className="settings-badge">{t("settings.hfCache.totalBadge", { size: formatBytes(cache?.total_bytes || 0) })}</span>
        <span className="settings-hint">
          {cache?.exists ? cache.root : t("settings.hfCache.rootMissing", { path: cache?.root })}
        </span>
        <button type="button" className="btn-mini" onClick={refresh} disabled={loading}>
          {t("settings.hfCache.refresh")}
        </button>
      </div>
      {repos.length === 0 ? (
        <p className="params-hint">{t("settings.hfCache.empty")}</p>
      ) : (
        <div className="models-table">
          <div className="models-table-head">
            <span>{t("settings.hfCache.colRepository")}</span>
            <span>{t("settings.hfCache.colSize")}</span>
            <span>{t("settings.hfCache.colDetails")}</span>
            <span>{t("settings.hfCache.colActions")}</span>
          </div>
          {repos.map((r) => (
            <div className="models-table-row" key={r.repo_id}>
              <span className="models-name">
                <strong>{r.repo_id}</strong>
                <span className="models-id">{r.repo_type}</span>
              </span>
              <span className="models-size">{formatBytes(r.size_bytes)}</span>
              <span className="models-state">
                {r.n_revisions !== 1
                  ? t("settings.hfCache.revisionMany", { count: r.n_revisions })
                  : t("settings.hfCache.revisionOne", { count: r.n_revisions })}
                {t("settings.hfCache.files", { count: r.n_files })}
              </span>
              <span className="models-actions">
                <button
                  type="button"
                  className="btn-mini"
                  disabled={clearing === r.repo_id}
                  onClick={() => clearRepo(r)}
                >
                  {clearing === r.repo_id ? t("settings.hfCache.removing") : t("settings.hfCache.clear")}
                </button>
              </span>
            </div>
          ))}
        </div>
      )}
      <p className="params-hint">{t("settings.hfCache.footerHint")}</p>
    </div>
  );
}
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import LicenseBadges from "./LicenseBadges";

/**
 * One table of every model the app can see.
 *
 * This replaces two views. The built-in registry and the on-disk detection list used to
 * be separate sections, which meant a repo downloaded from the browser was in neither by
 * id -- the weights were on disk and the Models tab showed nothing. `GET
 * /api/models/inventory` merges them server-side and tags each row `origin`, so the
 * distinction is made once, from one source of truth, instead of being re-derived in the
 * UI from directory names.
 *
 * **No licence fetch happens here.** Rows render from what the endpoint already carries.
 * A refresh is a button the user presses, because a list render that reached for the
 * network would make every open of this tab depend on Civitai's uptime -- and on startup
 * it would fire a burst of requests nobody asked for.
 */

function formatBytes(n) {
  if (!n) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(value < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

export default function InventoryTable({ models = [], onModelsChanged, onFeedback }) {
  const { t } = useI18n();

  const [items, setItems] = useState([]);
  const [filterMode, setFilterMode] = useState("all");
  const [selectedTargets, setSelectedTargets] = useState({});
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const mounted = useRef(true);
  // Re-arm on every setup, not only clear on cleanup. React 19 StrictMode mounts,
  // tears down, and mounts again: with only a cleanup, the flag goes false on the
  // simulated unmount and never comes back, so every fetch result is discarded and
  // the view shows its loading state forever. Production does not double-invoke, so
  // this failed in dev only -- which is the worst possible place for it.
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const data = await api("/api/models/inventory");
      if (mounted.current) setItems(data?.items || []);
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const refreshLicenses = useCallback(async () => {
    setRefreshing(true);
    setError("");
    try {
      // The MODEL endpoint, not the LoRA one. This table lists models; the LoRA
      // endpoint refreshes a different registry, finds nothing to change, and reports
      // it -- which is why pressing this button used to look like it did nothing.
      //
      // Explicit backfill. Nothing does this on startup: a fetch nobody asked for is
      // not a refresh.
      const result = await api("/api/models/refresh-licenses", { method: "POST" });
      if (!mounted.current) return;
      onFeedback?.({
        type: "success",
        text: t("models.detected.licensesRefreshed", {
          updated: result?.updated ?? 0,
          skipped: result?.skipped ?? 0,
        }),
      });
      await load();
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setRefreshing(false);
    }
  }, [load, onFeedback, t]);

  const attachModel = useCallback(async (item, targetModelId) => {
    if (!targetModelId) return;
    setBusy(item.name);
    setError("");
    try {
      await api("/api/storage/adopt", {
        method: "POST",
        body: JSON.stringify({
          name: item.name,
          model_id: targetModelId,
        }),
      });
      const targetModel = models.find((m) => m.id === targetModelId);
      const targetLabel = targetModel?.label || targetModelId;
      onFeedback?.({
        type: "success",
        text: t("models.detected.attachedSuccess", {
          item: item.label || item.name,
          target: targetLabel,
        }),
      });
      window.dispatchEvent(new CustomEvent("mlx:models-changed"));
      await load();
      onModelsChanged?.();
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, [load, models, onFeedback, onModelsChanged, t]);

  const unlinkModel = useCallback(async (item) => {
    const modelId = item.usable_as || item.name;
    setBusy(item.name);
    setError("");
    try {
      await api(`/api/hf/models/register/${encodeURIComponent(modelId)}`, {
        method: "DELETE",
      });
      onFeedback?.({
        type: "success",
        text: t("models.detected.unlinkedSuccess", {
          model: item.usable_as_label || item.label || item.name,
        }),
      });
      window.dispatchEvent(new CustomEvent("mlx:models-changed"));
      await load();
      onModelsChanged?.();
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, [load, onFeedback, onModelsChanged, t]);

  const deleteModel = useCallback(async (item) => {
    if (!window.confirm(t("models.detected.confirmDelete", { model: item.name }))) return;
    setBusy(item.name);
    setError("");
    try {
      // confirm: true is required by the endpoint. Deleting a model is rmtree on the
      // model store -- no trash, no undo -- so the API will not take the word of a
      // caller it cannot see. Sending no body here is a 422, not a deletion.
      await api(`/api/hf/models/${encodeURIComponent(item.name)}`, {
        method: "DELETE",
        body: JSON.stringify({ confirm: true }),
      });
      onFeedback?.({
        type: "success",
        text: t("models.detected.deleted", { model: item.name }),
      });
      await load();
      onModelsChanged?.();
    } catch (e) {
      if (mounted.current) setError(e.message || String(e));
    } finally {
      if (mounted.current) setBusy(null);
    }
  }, [load, onModelsChanged, onFeedback, t]);

  const detectedCount = items.filter((it) => it.origin === "detected").length;
  const registeredCount = items.filter((it) => it.origin === "registry").length;
  const filteredItems = items.filter((it) => {
    if (filterMode === "detected") return it.origin === "detected";
    if (filterMode === "registered") return it.origin === "registry";
    return true;
  });

  return (
    <section className="params-section inventory-table">
      <div className="inventory-head">
        <div className="inventory-head-text">
          <h3>{t("models.detected.title")}</h3>
          <p className="params-section-desc">{t("models.detected.desc")}</p>
        </div>
        <div className="inventory-head-actions">
          <div className="inventory-filter-pills" role="tablist">
            <button
              type="button"
              className={`inventory-filter-pill${filterMode === "all" ? " active" : ""}`}
              onClick={() => setFilterMode("all")}
            >
              {t("models.detected.filterAll") || "Tous"} ({items.length})
            </button>
            <button
              type="button"
              className={`inventory-filter-pill${filterMode === "detected" ? " active" : ""}`}
              onClick={() => setFilterMode("detected")}
            >
              {t("models.detected.filterDetected") || "Sur disque"} ({detectedCount})
            </button>
            <button
              type="button"
              className={`inventory-filter-pill${filterMode === "registered" ? " active" : ""}`}
              onClick={() => setFilterMode("registered")}
            >
              {t("models.detected.filterRegistered") || "Installés"} ({registeredCount})
            </button>
          </div>
          <button
            type="button" className="btn-mini"
            disabled={refreshing} onClick={refreshLicenses}
          >
            {/* Not "Importing…" -- that label belonged to the Import tab and made a
                licence refresh look like a conversion that had hung. */}
            {refreshing ? t("models.detected.refreshingLicenses") : t("models.detected.refreshLicenses")}
          </button>
          <button type="button" className="btn-mini" disabled={loading} onClick={load}>
            {t("models.retry")}
          </button>
        </div>
      </div>

      {error && (
        <div className="settings-feedback error" role="alert">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-2px", marginRight: "6px" }}><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          {error}
        </div>
      )}

      {loading && items.length === 0 && (
        <p className="import-empty">{t("models.searching")}</p>
      )}
      {!loading && filteredItems.length === 0 && !error && (
        <p className="import-empty">{t("models.noResults")}</p>
      )}

      {filteredItems.length > 0 && (
        // A horizontal scroller, not an ellipsis. Model names here routinely run past
        // 200 characters -- Draw Things names LoRA files after the Civitai prompt, and
        // model folders inherit that -- and a name you cannot read is a name you cannot
        // match to the file you are looking for. Scroll only when it genuinely does not
        // fit; the column is given the width first.
        <div className="inventory-scroll">
        <table className="inventory-grid">
          <thead>
            <tr>
              <th>{t("sources.name")}</th>
              <th>{t("models.detected.colOrigin") || t("sources.title")}</th>
              <th>{t("sources.baseModel")}</th>
              <th>{t("sources.license")}</th>
              <th>{t("settings.models.colActions") || ""}</th>
            </tr>
          </thead>
          <tbody>
            {filteredItems.map((item) => (
              <tr key={`${item.origin}:${item.name}`} className="inventory-row">
                <td>
                  <div className="inventory-name-cell">
                    <span className="import-name">{item.label || item.name}</span>
                    {item.bytes ? (
                      <span className="inventory-size-badge">{formatBytes(item.bytes)}</span>
                    ) : null}
                  </div>
                </td>
                <td>
                  <div className="inventory-origin-cell">
                    <span className={`hf-chip${item.origin === "detected" ? " subtle" : ""}`}>
                      {item.origin === "registry" ? t("sources.installed") : t("sources.detected", { count: 1 })}
                    </span>
                    {item.repo_id && (
                      <a
                        className="license-source-link"
                        href={`https://huggingface.co/${item.repo_id}`}
                        target="_blank" rel="noopener noreferrer"
                      >
                        {t("sources.viewOnSource")}
                      </a>
                    )}
                  </div>
                </td>
                <td>
                  <div className="inventory-arch-cell">
                    <span className="hf-chip">{item.architecture_label || item.architecture || "—"}</span>
                    {item.origin === "detected" && !item.usable_as && item.usable_as_label && (
                      <div className="inventory-desc-sub">{item.usable_as_label}</div>
                    )}
                  </div>
                </td>
                <td>
                  <div className="inventory-license-cell">
                    {item.license ? <LicenseBadges license={item.license} compact /> : <span className="text-muted">—</span>}
                  </div>
                </td>
                <td className="inventory-actions-col">
                  {item.origin === "detected" ? (
                    <div className="orphan-actions-group">
                      {item.registered ? (
                        <>
                          <span className="orphan-registered-badge">
                            <span className="status-dot active" aria-hidden="true" style={{ width: 6, height: 6, borderRadius: "50%", background: "currentColor", display: "inline-block", marginRight: 5, verticalAlign: 1 }} />
                            {t("models.detected.registeredBadge") || "Actif"}
                          </span>
                          <button
                            type="button"
                            className="btn-mini btn-unlink-model"
                            title={t("models.detected.unlinkTitle")}
                            disabled={busy === item.name}
                            onClick={() => unlinkModel(item)}
                          >
                            {t("models.detected.unlinkBtn")}
                          </button>
                        </>
                      ) : (
                        <>
                          <select
                            className="orphan-model-select"
                            value={selectedTargets[item.name] ?? item.usable_as ?? ""}
                            onChange={(e) => {
                              const val = e.target.value;
                              setSelectedTargets((prev) => ({ ...prev, [item.name]: val }));
                            }}
                            disabled={busy === item.name}
                            title={t("models.detected.chooseModelPlaceholder")}
                          >
                            <option value="">{t("models.detected.chooseModelPlaceholder")}</option>
                            {models.map((m) => (
                              <option key={m.id} value={m.id}>
                                {m.label || m.id}
                              </option>
                            ))}
                          </select>
                          <button
                            type="button"
                            className="btn-mini btn-download-weights btn-attach-model"
                            title={t("models.detected.attachTitle")}
                            disabled={busy === item.name || !(selectedTargets[item.name] ?? item.usable_as)}
                            onClick={() => attachModel(item, selectedTargets[item.name] ?? item.usable_as)}
                          >
                            {t("models.detected.attachBtn")}
                          </button>
                        </>
                      )}
                      <button
                        type="button"
                        className="btn-mini btn-danger-subtle"
                        disabled={busy === item.name}
                        onClick={() => deleteModel(item)}
                      >
                        {t("models.detected.delete")}
                      </button>
                    </div>
                  ) : (
                    <span className="text-muted">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
      )}
    </section>
  );
}
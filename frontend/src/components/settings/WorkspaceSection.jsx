import { useCallback, useEffect, useState } from "react";
import { api, pickPath } from "../../api";
import { useI18n } from "../../i18n/I18nContext";

const ROWS = [
  { kind: "data", labelKey: "settings.workspace.dataLabel", descKey: "settings.workspace.dataDesc" },
  { kind: "asset", labelKey: "settings.workspace.assetLabel", descKey: "settings.workspace.assetDesc" },
];

export default function WorkspaceSection({ onFeedback }) {
  const { t } = useI18n();
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setState(await api("/api/system/workspace"));
      setError(null);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function choose(kind) {
    setBusy(kind);
    setError(null);
    try {
      const picked = await pickPath({
        kind: "folder",
        title: t(kind === "data" ? "settings.workspace.pickData" : "settings.workspace.pickAsset"),
      });
      if (picked?.cancelled || !picked?.path) return;
      const next = await api("/api/system/workspace", {
        method: "POST",
        body: JSON.stringify({ kind, path: picked.path }),
      });
      setState(next);
      onFeedback?.({ type: "success", text: t("settings.workspace.updated") });
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(null);
    }
  }

  async function resetToDefault(kind) {
    setBusy(kind);
    setError(null);
    try {
      const next = await api("/api/system/workspace", {
        method: "POST",
        body: JSON.stringify({ kind, path: null }),
      });
      setState(next);
      onFeedback?.({ type: "success", text: t("settings.workspace.resetDone") });
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setBusy(null);
    }
  }

  if (loading && !state) return <p className="muted">{t("app.loading")}</p>;
  if (error && !state) return <p className="error">{error}</p>;
  if (!state) return null;

  return (
    <div className="workspace-settings">
      <p className="workspace-warning">{t("settings.workspace.noCopy")}</p>
      {state.restart_required && (
        <p className="workspace-restart" role="status">{t("settings.workspace.restartRequired")}</p>
      )}
      <div className="workspace-rows">
        {ROWS.map((row) => {
          const selected = state.selected?.[row.kind];
          const active = state.active?.[`${row.kind}_dir`];
          const effective = state.effective?.[row.kind];
          const pending = Boolean(selected && selected !== active) || Boolean(state.restart_required && effective !== active);
          return (
            <div className="workspace-row" key={row.kind}>
              <div className="workspace-row-copy">
                <strong>{t(row.labelKey)}</strong>
                <span className="muted small">{t(row.descKey)}</span>
                <div className="workspace-path-line">
                  <span className="muted small">{t("settings.workspace.current")}</span>
                  <code className="workspace-path" title={active}>{active}</code>
                </div>
                <div className="workspace-path-line">
                  <span className="muted small">{t("settings.workspace.afterRestart")}</span>
                  <code className="workspace-path" title={effective}>{effective}</code>
                  {pending && <span className="workspace-pending">{t("settings.workspace.pending")}</span>}
                </div>
              </div>
              <div className="workspace-actions">
                <button type="button" onClick={() => choose(row.kind)} disabled={busy === row.kind}>
                  {busy === row.kind ? t("app.loading") : t("settings.workspace.choose")}
                </button>
                <button
                  type="button"
                  onClick={() => resetToDefault(row.kind)}
                  disabled={busy === row.kind || !selected}
                >
                  {t("settings.workspace.useDefault")}
                </button>
              </div>
            </div>
          );
        })}
      </div>
      {error && <p className="error">{error}</p>}
    </div>
  );
}

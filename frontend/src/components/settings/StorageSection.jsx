import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";

// Read-only by design. This panel answers "where did my disk go", and it has no
// delete button anywhere on purpose: a report that can destroy things is a report
// people stop trusting. Anything worth removing has to be a decision, not a
// side effect of looking.

// Maps the backend's status codes to i18n keys rather than to literal text: the
// table lives at module level, outside any component, so it cannot call useI18n().
const STATUS_KEY = {
  registered: "settings.storage.statusRegistered",
  internal: "settings.storage.statusInternal",
  unrecognised: "settings.storage.statusUnrecognised",
};

function Row({ label, bytes, detail }) {
  return (
    <tr>
      <td>{label}</td>
      <td className="num">{formatBytes(bytes)}</td>
      <td className="muted">{detail}</td>
    </tr>
  );
}

export default function StorageSection({ onFeedback }) {
  const { t } = useI18n();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  const refresh = useCallback(async (force = false) => {
    setLoading(true);
    try {
      const data = await api(`/api/storage${force ? "?force=true" : ""}`);
      setReport(data);
      setErr(null);
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (loading && !report) return <p className="muted">{t("settings.storage.scanning")}</p>;
  if (err) return <p className="error">{t("settings.storage.scanFailed", { error: err })}</p>;
  if (!report) return null;

  const { totals, models, loras, gallery, uploads, secrets, config, unrecognised, roots } = report;
  const missing = loras.missing_entries || 0;
  const unseen = unrecognised.length;
  const wasted = loras.duplicate_wasted_bytes || 0;
  const dupes = loras.duplicates || [];

  return (
    <div className="settings-section">
      <div className="storage-head">
        <div>
          <strong>{formatBytes(totals.bytes)}</strong> {t("settings.storage.onDisk")}
          <div className="muted small">
            {roots.same
              ? t("settings.storage.storeRoot", { path: roots.data_dir })
              : t("settings.storage.rootsSplit", { models: roots.asset_dir, data: roots.data_dir })}
          </div>
        </div>
        <button type="button" onClick={() => refresh(true)} disabled={loading}>
          {loading ? t("settings.storage.scanning") : t("settings.storage.rescan")}
        </button>
      </div>

      <table className="storage-table">
        <tbody>
          <Row
            label={t("settings.storage.rowModels")}
            bytes={totals.by_category.models}
            detail={t("settings.storage.directories", { count: models.length })}
          />
          <Row
            label={t("settings.storage.rowLoras")}
            bytes={totals.by_category.loras}
            detail={(loras.directories || []).length === 1
              ? t("settings.storage.lorasDetailOne", { files: loras.files })
              : t("settings.storage.lorasDetailMany", { files: loras.files, dirs: (loras.directories || []).length })}
          />
          <Row
            label={t("settings.storage.rowGallery")}
            bytes={totals.by_category.gallery}
            detail={t("settings.storage.galleryDetail", { images: gallery.images, thumbnails: gallery.thumbnails })}
          />
          <Row
            label={t("settings.storage.rowUploads")}
            bytes={totals.by_category.uploads}
            detail={t("settings.storage.uploadsDetail", { files: uploads.files })}
          />
        </tbody>
      </table>

      {wasted > 0 && (
        <p className="warning">
          <strong>{t("settings.storage.dupeLead", { size: formatBytes(wasted) })}</strong>{" "}
          {dupes.length === 1
            ? t("settings.storage.dupeBodyOne", { count: dupes.length })
            : t("settings.storage.dupeBodyMany", { count: dupes.length })}
        </p>
      )}

      {missing > 0 && (
        <p className="warning">
          <strong>
            {missing === 1
              ? t("settings.storage.missingOne", { count: missing })
              : t("settings.storage.missingMany", { count: missing })}
          </strong>{" "}
          {t("settings.storage.missingTail")}{" "}
          <span className="mono">{(loras.missing_names || []).slice(0, 6).join(", ")}</span>
          {(loras.missing_names || []).length > 6 && " …"}
        </p>
      )}

      {unseen > 0 && (
        <p className="warning">
          <strong>
            {unseen === 1
              ? t("settings.storage.unseenOne", { count: unseen })
              : t("settings.storage.unseenMany", { count: unseen })}
          </strong>{" "}
          {t("settings.storage.unseenTail")}
        </p>
      )}

      {wasted > 0 && (
        <details>
          <summary>{t("settings.storage.duplicatedFiles")}</summary>
          <table className="storage-table">
            <thead>
              <tr>
                <th>{t("settings.storage.thFile")}</th>
                <th className="num">{t("settings.storage.thEach")}</th>
                <th className="num">{t("settings.storage.thCopies")}</th>
                <th className="num">{t("settings.storage.thWasted")}</th>
              </tr>
            </thead>
            <tbody>
              {dupes.map((d) => (
                <tr key={d.name + d.copies}>
                  <td title={d.paths.join("\n")}>{d.name}</td>
                  <td className="num">{formatBytes(d.bytes_each)}</td>
                  <td className="num">{d.copies}</td>
                  <td className="num">{formatBytes(d.wasted_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      <details>
        <summary>{t("settings.storage.modelDirectories")}</summary>
        <table className="storage-table">
          <thead>
            <tr>
              <th>{t("settings.storage.thDirectory")}</th>
              <th className="num">{t("settings.storage.thSize")}</th>
              <th className="num">{t("settings.storage.thFiles")}</th>
              <th>{t("settings.storage.thStatus")}</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td title={m.path}>{m.name}</td>
                <td className="num">{formatBytes(m.bytes)}</td>
                <td className="num">{m.files}</td>
                <td className="muted">
                  {STATUS_KEY[m.status] ? t(STATUS_KEY[m.status]) : m.status}
                  {m.purpose ? ` — ${m.purpose}` : ""}
                  {m.label ? ` — ${m.label}` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {unseen > 0 && (
          <p className="muted small">
            {t("settings.storage.unaccountedFor", { names: unrecognised.map((m) => `${m.name} (${formatBytes(m.bytes)})`).join(", ") })}
          </p>
        )}
      </details>

      <details>
        <summary>{t("settings.storage.configCreds")}</summary>
        <table className="storage-table">
          <tbody>
            <Row label="settings.json" bytes={config.settings_bytes} detail={t("settings.storage.detailSettings")} />
            <Row label="loras.json" bytes={config.loras_registry_bytes} detail={t("settings.storage.detailLoras")} />
            {secrets.map((s) => (
              <Row
                key={s.name}
                label={s.name}
                bytes={s.bytes}
                detail={s.present ? t("settings.storage.secretPresent") : t("settings.storage.secretMissing")}
              />
            ))}
          </tbody>
        </table>
        <p className="muted small">{t("settings.storage.credentialNote")}</p>
      </details>
    </div>
  );
}

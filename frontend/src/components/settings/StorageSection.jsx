import { useCallback, useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";

// Read-only by design. This panel answers "where did my disk go", and it has no
// delete button anywhere on purpose: a report that can destroy things is a report
// people stop trusting. Anything worth removing has to be a decision, not a
// side effect of looking.

const STATUS_LABEL = {
  registered: "in use",
  internal: "loaded at runtime",
  unrecognised: "unrecognised",
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

  if (loading && !report) return <p className="muted">Scanning…</p>;
  if (err) return <p className="error">Storage scan failed: {err}</p>;
  if (!report) return null;

  const { totals, models, loras, gallery, uploads, secrets, config, unrecognised, roots } = report;
  const missing = loras.missing_entries || 0;
  const unseen = unrecognised.length;

  return (
    <div className="settings-section">
      <div className="storage-head">
        <div>
          <strong>{formatBytes(totals.bytes)}</strong> on disk
          <div className="muted small">
            {roots.same
              ? `store: ${roots.data_dir}`
              : `models: ${roots.asset_dir} · data: ${roots.data_dir}`}
          </div>
        </div>
        <button type="button" onClick={() => refresh(true)} disabled={loading}>
          {loading ? "Scanning…" : "Rescan"}
        </button>
      </div>

      <table className="storage-table">
        <tbody>
          <Row label="Models" bytes={totals.by_category.models} detail={`${models.length} directories`} />
          <Row label="LoRAs" bytes={totals.by_category.loras} detail={`${loras.files} file(s) on disk`} />
          <Row
            label="Generated images"
            bytes={totals.by_category.gallery}
            detail={`${gallery.images} image(s) + ${gallery.thumbnails} thumbnail(s)`}
          />
          <Row label="Uploads" bytes={totals.by_category.uploads} detail={`${uploads.files} file(s)`} />
        </tbody>
      </table>

      {missing > 0 && (
        <p className="warning">
          <strong>
            {missing} registered LoRA{missing === 1 ? "" : "s"} point at a file that is no longer
            there.
          </strong>{" "}
          They will fail silently if selected. Names:{" "}
          <span className="mono">{(loras.missing_names || []).slice(0, 6).join(", ")}</span>
          {(loras.missing_names || []).length > 6 && " …"}
        </p>
      )}

      {unseen > 0 && (
        <p className="warning">
          <strong>
            {unseen} model director{unseen === 1 ? "y is" : "ies are"} not referenced by anything
            in the app.
          </strong>{" "}
          That is not a verdict — a model placed by hand, or one a different build supports, looks
          the same. Worth a look, not an instruction. Nothing here is ever marked reclaimable.
        </p>
      )}

      <details>
        <summary>Model directories</summary>
        <table className="storage-table">
          <thead>
            <tr>
              <th>Directory</th>
              <th className="num">Size</th>
              <th className="num">Files</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td title={m.path}>{m.name}</td>
                <td className="num">{formatBytes(m.bytes)}</td>
                <td className="num">{m.files}</td>
                <td className="muted">
                  {STATUS_LABEL[m.status] || m.status}
                  {m.purpose ? ` — ${m.purpose}` : ""}
                  {m.label ? ` — ${m.label}` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {unseen > 0 && (
          <p className="muted small">
            Unaccounted for: {unrecognised.map((m) => `${m.name} (${formatBytes(m.bytes)})`).join(", ")}
          </p>
        )}
      </details>

      <details>
        <summary>Configuration and credentials</summary>
        <table className="storage-table">
          <tbody>
            <Row label="settings.json" bytes={config.settings_bytes} detail="preferences, presets" />
            <Row label="loras.json" bytes={config.loras_registry_bytes} detail="LoRA registry" />
            {secrets.map((s) => (
              <Row
                key={s.name}
                label={s.name}
                bytes={s.bytes}
                detail={s.present ? "present on disk" : "not set"}
              />
            ))}
          </tbody>
        </table>
        <p className="muted small">
          Credential files are reported by name and size only. This panel never opens them.
        </p>
      </details>
    </div>
  );
}

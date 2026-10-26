import { useEffect, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import LicenseBadges from "./LicenseBadges";

/**
 * Installed models and LoRAs with their licence blocks, or a source link
 * where no block exists.
 *
 * The fallback order is deliberate: a Civitai page first, because the card
 * carries usage pictograms no registry field reproduces; the Hugging Face
 * repo second. Anything with neither gets the honest unknown-licence label
 * rather than a guessed URL -- LicenseBadges enforces the same rule, and a
 * wrong link next to a model would be relied on. Custom-onboarded models and
 * LoRAs carry hf_repo/civitai_ids, so they get working links with no change
 * here.
 */
function sourceLinkFor(entry) {
  const civitaiId = entry.civitai_model_id;
  if (civitaiId && /^\d+$/.test(String(civitaiId).trim())) {
    return `https://civitai.com/models/${encodeURIComponent(String(civitaiId).trim())}`;
  }
  const repo = entry.repo || entry.hf_repo_id;
  if (repo && typeof repo === "string" && !repo.startsWith("local:") && !/^\s*javascript:/i.test(repo)) {
    return `https://huggingface.co/${encodeURIComponent(repo.trim())}`;
  }
  return null;
}

export default function InstalledLicences() {
  const { t } = useI18n();
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    Promise.all([api("/api/models"), api("/api/loras")])
      .then(([modelList, loraList]) => {
        if (!alive) return;
        setModels((Array.isArray(modelList) ? modelList : []).filter((m) => m.installed));
        setLoras(Array.isArray(loraList) ? loraList : []);
      })
      .catch((e) => {
        // A failed fetch must say so: an empty list would read as "nothing
        // installed", which is the opposite of what happened.
        if (alive) setError(e?.message || String(e));
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="installed-licences">
      {error && (
        <p className="civitai-feedback error" role="alert">
          {error}
        </p>
      )}
      <section className="params-section">
        <h3>{t("licences.installedModels")}</h3>
        {models.length === 0 && !error ? (
          <p className="params-section-desc">{t("licences.installedEmpty")}</p>
        ) : (
          <div className="licence-table-wrap">
            <table className="licence-table">
              <colgroup>
                <col style={{ width: "38%" }} />
                <col style={{ width: "38%" }} />
                <col style={{ width: "24%" }} />
              </colgroup>
              <thead>
                <tr>
                  <th>{t("licences.colModel")}</th>
                  <th>{t("licences.colLicence")}</th>
                  <th>{t("licences.colRepository")}</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => {
                  const url = sourceLinkFor(m);
                  return (
                    <tr key={m.id}>
                      <td>
                        <div className="licence-col-name">
                          <strong>{m.label || m.id}</strong>
                        </div>
                        {m.id && m.label && m.id !== m.label && (
                          <div className="licence-extra">{m.id}</div>
                        )}
                      </td>
                      <td>
                        {m.license ? (
                          <LicenseBadges license={m.license} compact />
                        ) : (
                          <span className="license-badge license-badge-unknown">
                            {t("sources.unknownLicense")}
                          </span>
                        )}
                      </td>
                      <td>
                        {url ? (
                          <a
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="licence-source-link"
                            title={url}
                          >
                            <span>{t("sources.viewOnSource")}</span>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                          </a>
                        ) : (
                          <span className="licence-no-source">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <section className="params-section">
        <h3>{t("licences.installedLoras")}</h3>
        {loras.length === 0 && !error ? (
          <p className="params-section-desc">{t("licences.installedEmpty")}</p>
        ) : (
          <div className="licence-table-wrap">
            <table className="licence-table">
              <colgroup>
                <col style={{ width: "38%" }} />
                <col style={{ width: "38%" }} />
                <col style={{ width: "24%" }} />
              </colgroup>
              <thead>
                <tr>
                  <th>{t("licences.colLora")}</th>
                  <th>{t("licences.colLicence")}</th>
                  <th>{t("licences.colRepository")}</th>
                </tr>
              </thead>
              <tbody>
                {loras.map((l) => {
                  const url = sourceLinkFor(l);
                  return (
                    <tr key={l.name}>
                      <td>
                        <div className="licence-col-name">
                          <strong>{l.name}</strong>
                        </div>
                        {l.trigger_word && (
                          <div className="licence-extra">{l.trigger_word}</div>
                        )}
                      </td>
                      <td>
                        {l.license ? (
                          <LicenseBadges license={l.license} compact />
                        ) : (
                          <span className="license-badge license-badge-unknown">
                            {t("sources.unknownLicense")}
                          </span>
                        )}
                      </td>
                      <td>
                        {url ? (
                          <a
                            href={url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="licence-source-link"
                            title={url}
                          >
                            <span>{t("sources.viewOnSource")}</span>
                            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                          </a>
                        ) : (
                          <span className="licence-no-source">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

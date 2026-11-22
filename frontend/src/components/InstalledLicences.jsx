import { Fragment, useEffect, useState } from "react";
import { api } from "../api";
import { civitaiModelUrl, withCivitaiReferral } from "../civitai";
import { knownLicenceFor, sortByName } from "../data/licences";
import { useI18n } from "../i18n/I18nContext";
import LicenseBadges, { RightsBadge } from "./LicenseBadges";

/**
 * Installed models and LoRAs with their licence blocks, or a source link
 * where no block exists.
 *
 * Built-in models reuse the verified licence data from the App & models tab:
 * the registry reports no licence at all for them, and "unknown" beside a
 * licence we already researched would be a false negative. Anything else
 * falls back to metadata-derived links: a Civitai page first, because the
 * card carries usage pictograms no registry field reproduces; a licence
 * source URL second; the Hugging Face repo last. Anything with none gets the
 * honest unknown label rather than a guessed URL -- a wrong link next to a
 * model would be relied on.
 */
function sourceLinkFor(entry) {
  const civitaiId = entry.civitai_model_id;
  if (civitaiId && /^\d+$/.test(String(civitaiId).trim())) {
    return civitaiModelUrl(civitaiId);
  }
  const licenceUrl = entry.license?.source_url || entry.license?.url;
  if (typeof licenceUrl === "string" && /^https?:\/\//i.test(licenceUrl.trim())) {
    return withCivitaiReferral(licenceUrl.trim());
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

  const sortedModels = sortByName(models, (m) => m.label || m.id);
  const sortedLoras = sortByName(loras, (l) => l.name || l.id);

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
            <table className="licence-table licence-table-models">
              <colgroup>
                <col style={{ width: "26%" }} />
                <col style={{ width: "20%" }} />
                <col style={{ width: "14%" }} />
                <col style={{ width: "14%" }} />
                <col style={{ width: "26%" }} />
              </colgroup>
              <thead>
                <tr>
                  <th>{t("licences.colModel")}</th>
                  <th>{t("licences.colLicence")}</th>
                  <th>{t("licences.colOutputs")}</th>
                  <th>{t("licences.colWeights")}</th>
                  <th>{t("licences.colRepository")}</th>
                </tr>
              </thead>
              <tbody>
                {sortedModels.map((m) => {
                  const known = knownLicenceFor(m);
                  const url = known?.url || sourceLinkFor(m);
                  return (
                    <Fragment key={m.id}>
                    <tr>
                      <td>
                        <div className="licence-col-name">
                          <strong>{m.label || m.id}</strong>
                        </div>
                        {m.id && m.label && m.id !== m.label && (
                          <div className="licence-extra">{m.id}</div>
                        )}
                      </td>
                      <td>
                        {known ? (
                          <span className={`license-badge license-badge-name${known.caution ? " licence-caution-badge" : ""}`}>
                            {known.license}
                          </span>
                        ) : m.license ? (
                          <LicenseBadges license={m.license} compact />
                        ) : (
                          <span className="license-badge license-badge-unknown">
                            {t("sources.unknownLicense")}
                          </span>
                        )}
                      </td>
                      <td>
                        {known ? (
                          <RightsBadge value={known.outputs} />
                        ) : (
                          <span className="licence-no-source">—</span>
                        )}
                      </td>
                      <td>
                        {known ? (
                          <RightsBadge value={known.weights} />
                        ) : (
                          <span className="licence-no-source">—</span>
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
                    {known?.conditionKey && (
                      <tr className="licence-condition-row">
                        <td colSpan={5}>{t(known.conditionKey)}</td>
                      </tr>
                    )}
                    </Fragment>
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
                {sortedLoras.map((l) => {
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

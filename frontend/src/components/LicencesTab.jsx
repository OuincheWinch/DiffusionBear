import { useEffect, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import {
  GITHUB_REPO_URL,
  GITHUB_LICENSE_URL,
  AUTHOR_WEBSITE,
  APP_WEBSITE,
  AI_CREDITS,
  LICENCE_SECTIONS,
} from "../data/licences";
import InstalledLicences from "./InstalledLicences";
import { APP_VERSION_LABEL } from "../version";
import logo from "../assets/logo.png";

function ExtLink({ href, children, className, title }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
      title={title || href}
    >
      {children}
    </a>
  );
}

export default function LicencesTab() {
  const { t } = useI18n();
  const [apiVersion, setApiVersion] = useState(null);
  const [licenceTab, setLicenceTab] = useState("app");

  useEffect(() => {
    let alive = true;
    api("/api/version")
      .then((v) => {
        if (alive) setApiVersion(v?.version ? v.version : null);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="parameters-tab licences-tab">
      <div className="params-subtabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={licenceTab === "app"}
          className={`params-subtab${licenceTab === "app" ? " active" : ""}`}
          onClick={() => setLicenceTab("app")}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
          {t("licences.tabAppModels")}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={licenceTab === "installed"}
          className={`params-subtab${licenceTab === "installed" ? " active" : ""}`}
          onClick={() => setLicenceTab("installed")}
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
          {t("licences.tabInstalled")}
        </button>
      </div>

      {licenceTab === "installed" ? (
        <InstalledLicences />
      ) : (
      <>
      <section className="params-section">
        <h3>{t("licences.title")} — DiffusionBear</h3>
        <p className="params-section-desc">
          {t("licences.intro")} {t("licences.introSecond")}
        </p>
      </section>

      <section className="params-section">
        <h3>{t("licences.thisProject")}</h3>
        <div className="licence-project-card">
          <div className="licence-project-info">
            <div className="licence-project-header">
              <strong>DiffusionBear</strong>
              <span className="version-badge">{APP_VERSION_LABEL}</span>
              <span className="license-badge license-badge-name">MIT</span>
            </div>
            <div className="licence-project-row">
              <span>
                {t("licences.releasedUnder")}{" "}
                <ExtLink href={GITHUB_LICENSE_URL} className="licence-source-link">
                  <span>{t("licences.mitLicense")}</span>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </ExtLink>
                {" · "}
                {t("licences.copyright", { year: 2026 })}{" "}
                <ExtLink href={AUTHOR_WEBSITE}>Ouinche</ExtLink>
              </span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", flexWrap: "wrap" }}>
                <span>{t("licences.website") || "Website"}:</span>
                <ExtLink href={APP_WEBSITE} className="licence-source-link">
                  <span>www.DiffusionBear.com</span>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </ExtLink>
                {" · "}
                <span>{t("licences.source")}</span>
                <ExtLink href={GITHUB_REPO_URL} className="licence-source-link">
                  <span>github.com/OuincheWinch/DiffusionBear</span>
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                </ExtLink>
              </span>
            </div>
            <div className="licence-project-row" style={{ marginTop: "0.2rem" }}>
              <span>
                <strong>{t("licences.aiAuthored")}</strong> —{" "}
                {t("licences.aiCredits", { credits: AI_CREDITS.join(", ") })}
              </span>
            </div>
            <p className="licence-hint" style={{ margin: "0.25rem 0 0" }}>
              {t("licences.mitScopeBefore")} <em>{t("licences.mitScopeEm")}</em>
              {t("licences.mitScopeAfter")}
            </p>
          </div>
          <div className="licence-project-logo-wrap">
            <img
              src={logo}
              alt={t("app.logoAlt") || "DiffusionBear"}
              className="licence-project-logo"
            />
          </div>
        </div>
      </section>

      {LICENCE_SECTIONS.map((section) => (
        <section className="params-section" key={section.id}>
          <h3>{section.titleKey ? t(section.titleKey) : section.title}</h3>
          {(section.noteKey || section.note) && (
            <p className="params-section-desc">
              {section.noteKey ? t(section.noteKey) : section.note}
            </p>
          )}
          <div className="licence-table-wrap">
            <table className="licence-table">
              <colgroup>
                <col style={{ width: "38%" }} />
                <col style={{ width: "38%" }} />
                <col style={{ width: "24%" }} />
              </colgroup>
              <thead>
                <tr>
                  <th>{t("licences.colPackage")}</th>
                  <th>{t("licences.colLicence")}</th>
                  <th>{t("licences.colRepository")}</th>
                </tr>
              </thead>
              <tbody>
                {section.packages.map((pkg) => {
                  const cleanUrl = pkg.url.replace("https://", "").replace(/\/$/, "");
                  return (
                    <tr key={pkg.name}>
                      <td>
                        <div className="licence-col-name">
                          <strong>{pkg.name}</strong>
                        </div>
                        {pkg.extra && <div className="licence-extra">{pkg.extra}</div>}
                      </td>
                      <td>
                        <span className={`license-badge license-badge-name${pkg.caution ? " licence-caution-badge" : ""}`}>
                          {pkg.license || t("sources.unknownLicense")}
                        </span>
                      </td>
                      <td>
                        <ExtLink href={pkg.url} className="licence-source-link">
                          <span className="licence-source-label">{cleanUrl}</span>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                        </ExtLink>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      <section className="params-section">
        <h3>{t("licences.versions")}</h3>
        <p className="params-section-desc">
          {t("licences.frontendVersion", { version: APP_VERSION_LABEL })}
          {apiVersion && t("licences.backendApi", { version: apiVersion })}
          {apiVersion && (
            <>
              {" · "}
              <ExtLink href={GITHUB_REPO_URL}>{t("licences.repoLink")}</ExtLink>
            </>
          )}
        </p>
      </section>
      </>
      )}
    </div>
  );
}
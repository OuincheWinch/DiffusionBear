import { useEffect, useState } from "react";
import { api } from "../api";
import { useI18n } from "../i18n/I18nContext";
import {
  GITHUB_REPO_URL,
  GITHUB_LICENSE_URL,
  AUTHOR_WEBSITE,
  AI_CREDITS,
  LICENCE_SECTIONS,
} from "../data/licences";
import { APP_VERSION_LABEL } from "../version";

function ExtLink({ href, children }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}

export default function LicencesTab() {
  const { t } = useI18n();
  const [apiVersion, setApiVersion] = useState(null);

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
      <section className="params-section">
        <h3>⚖ {t("licences.title")} — DiffusionBear</h3>
        <p className="params-section-desc">
          {t("licences.intro")} {t("licences.introSecond")}
        </p>
      </section>

      <section className="params-section">
        <h3>{t("licences.thisProject")}</h3>
        <p>
          <strong>DiffusionBear</strong> — {APP_VERSION_LABEL} —{" "}
          {t("licences.releasedUnder")}{" "}
          <ExtLink href={GITHUB_LICENSE_URL}>{t("licences.mitLicense")}</ExtLink>,{" "}
          {t("licences.copyright", { year: 2026 })}{" "}
          <ExtLink href={AUTHOR_WEBSITE}>Ouinche</ExtLink>. {t("licences.source")}{" "}
          <ExtLink href={GITHUB_REPO_URL}>github.com/OuincheWinch/DiffusionBear</ExtLink>.
        </p>
        <p>
          <strong>{t("licences.aiAuthored")}</strong> —{" "}
          {t("licences.aiCredits", { credits: AI_CREDITS.join(", ") })}
        </p>
        <p className="licence-hint">
          {t("licences.mitScopeBefore")} <em>{t("licences.mitScopeEm")}</em>
          {t("licences.mitScopeAfter")}
        </p>
      </section>

      {LICENCE_SECTIONS.map((section) => (
        <section className="params-section" key={section.id}>
          <h3>{section.titleKey ? t(section.titleKey) : section.title}</h3>
          {(section.noteKey || section.note) && (
            <p className="params-section-desc">
              {section.noteKey ? t(section.noteKey) : section.note}
            </p>
          )}
          <table className="licence-table">
            <thead>
              <tr>
                <th>{t("licences.colPackage")}</th>
                <th>{t("licences.colLicence")}</th>
                <th>{t("licences.colRepository")}</th>
              </tr>
            </thead>
            <tbody>
              {section.packages.map((pkg) => (
                <tr key={pkg.name}>
                  <td>
                    {pkg.name}
                    {pkg.extra && <div className="licence-extra">{pkg.extra}</div>}
                  </td>
                  <td className={pkg.caution ? "licence-caution" : ""}>{pkg.license}</td>
                  <td>
                    <ExtLink href={pkg.url}>
                      {pkg.url.replace("https://", "").replace(/\/$/, "")}
                    </ExtLink>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
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
    </div>
  );
}
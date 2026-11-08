import { useI18n } from "../i18n/I18nContext";
import LicenseBadges, { RightsBadge } from "./LicenseBadges";

/**
 * The commercial/non-commercial summary shown before a download starts.
 *
 * It must never invent a right for a model the app has not verified. Built-in
 * models reuse the researched table; everything else shows "Check source" for
 * both questions and links to the source card.
 */
export default function LicenceSummary({ summary, sourceLabel = "" }) {
  const { t } = useI18n();
  if (!summary) return null;

  const { known, license, name, url, outputs, weights, conditionKey } = summary;

  return (
    <div className="licence-accept-summary">
      <div className="licence-accept-name">
        {name ? (
          <span className="license-badge license-badge-name">{name}</span>
        ) : (
          <span className="license-badge license-badge-unknown">
            {t("licences.acceptUnverified")}
          </span>
        )}
      </div>

      <div className="licence-accept-rights">
        <div className="licence-accept-right">
          <span>{t("licences.colOutputs")}</span>
          <RightsBadge value={outputs} />
        </div>
        <div className="licence-accept-right">
          <span>{t("licences.colWeights")}</span>
          <RightsBadge value={weights} />
        </div>
      </div>

      {license && <LicenseBadges license={license} compact />}

      {conditionKey && <p className="licence-hint">{t(conditionKey)}</p>}
      {!known && (
        <p className="licence-hint licence-accept-unknown">
          {t("licences.acceptUnknown")}
        </p>
      )}

      {sourceLabel && <p className="licence-accept-source">{sourceLabel}</p>}

      {url && (
        <a
          className="licence-source-link"
          href={url}
          target="_blank"
          rel="noopener noreferrer"
        >
          {t("licences.acceptOpenLicence")}
        </a>
      )}

      <p className="licence-hint licence-accept-legal">{t("licences.acceptNotLegal")}</p>
    </div>
  );
}

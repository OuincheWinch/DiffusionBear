import { useI18n } from "../i18n/I18nContext";

// What a licence block is allowed to say, and what it is never allowed to say.
//
// The single rule this component exists to enforce: **never render a licence name that
// the source did not give us.** Hugging Face reports the literal string "other" for
// FLUX.1-dev, whose real terms live in the model's README card. Civitai returns no
// name at all, only four permission flags. So when `identifier` is null or "other" we
// say the source did not state one and link to it. We do not write "Unknown" as though
// that were a finding, and we never scrape the card or infer a name from the repo id.
//
// A wrong licence shown next to a model is worse than none shown, because it gets
// relied on. That asymmetry is the whole reason for the "unknown" wording being
// `sources.unknownLicense` -- a statement about the source -- rather than a bare
// "unknown" that reads as a verdict about the work.

const PERMISSION_BADGES = [
  // Civitai's field is `allowCommercialUse`; older payloads use `commercial_use`, and
  // either may be a bool or a list of allowed uses. All three mean "permitted", and
  // that is the only thing this badge claims.
  { keys: ["allowCommercialUse", "commercial_use", "commercialUse"], label: "sources.commercialUse" },
  { keys: ["allowDerivatives", "derivatives"], label: "sources.derivatives" },
  // Inverted: `no_credit` true means the creator REQUIRES you to credit them. The
  // permission is "no credit required", so a true here means the opposite of the
  // others, and rendering it naively would tell the user credit is forbidden.
  { keys: ["no_credit", "allowNoCredit", "allowNoCreditRequired"], label: "sources.creditRequired", invert: true },
  { keys: ["allowDifferentLicense", "different_license"], label: "sources.differentLicense" },
];

function permissionValue(permissions, keys) {
  for (const key of keys) {
    if (key in permissions && permissions[key] !== null && permissions[key] !== undefined) {
      return permissions[key];
    }
  }
  return undefined;
}

// A badge is shown only when the value is present and truthy. An absent flag means the
// source did not express that permission, which is not the same as denying it -- so we
// show nothing rather than a greyed-out "no". The permission block as a whole is
// covered by `sources.noPermissions` when it is present but empty.
function isGranted(value) {
  if (value === true) return true;
  if (Array.isArray(value)) return value.length > 0;
  return false;
}

export function LicenseBadges({ license, compact = false }) {
  const { t } = useI18n();

  // No licence block at all: the source was never asked, or has none. Render nothing.
  // An empty container is the honest answer; a placeholder would imply a check
  // happened and found nothing.
  if (!license) return null;

  const identifier = license.identifier;
  const permissions = license.permissions && typeof license.permissions === "object"
    ? license.permissions
    : null;

  // "other" is Hugging Face telling us the name lives elsewhere. It is not a licence
  // and must never be rendered as one.
  const named = typeof identifier === "string" && identifier !== "" && identifier !== "other";

  const badges = [];
  if (permissions) {
    for (const badge of PERMISSION_BADGES) {
      const raw = permissionValue(permissions, badge.keys);
      if (raw === undefined) continue;
      const granted = badge.invert ? raw === true : isGranted(raw);
      if (!granted) continue;
      badges.push(
        <span className="license-badge" key={badge.label}>
          {t(badge.label)}
        </span>,
      );
    }
  }

  const sourceUrl = license.source_url || license.url;

  return (
    <div className={`license-badges${compact ? " compact" : ""}`}>
      {named ? (
        <span className="license-badge license-badge-name" title={identifier}>
          {identifier}
        </span>
      ) : (
        <span className="license-badge license-badge-unknown">{t("sources.unknownLicense")}</span>
      )}

      {badges}

      {/* A permissions object that produced no badge means the creator expressed
          permissions we could not map, or expressed none. Say so rather than leaving a
          bare unknown badge that looks like the licence itself is missing. */}
      {permissions && badges.length === 0 && (
        <span className="license-badge license-badge-muted">{t("sources.noPermissions")}</span>
      )}

      {sourceUrl && (
        <a
          className="license-source-link"
          href={sourceUrl}
          target="_blank"
          rel="noopener noreferrer"
        >
          {t("sources.viewOnSource")}
        </a>
      )}
    </div>
  );
}

export default LicenseBadges;
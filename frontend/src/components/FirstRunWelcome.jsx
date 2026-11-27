import { useI18n } from "../i18n/I18nContext";
import { defaultGenerationSeconds, modelMemoryNeedGb, ramFit } from "../utils/ramFit";
import ModelInstaller from "./ModelInstaller";
import RamFitBadge from "./RamFitBadge";

// Small, fast models first: a first image in seconds matters more than peak quality.
const PREFERRED = [
  "z-image-turbo",
  "flux2-klein-4b",
  "dreamlite-mobile",
  "anima-turbo",
  "flux3-klein",
  "juggernaut-xl-lightning",
  "flux1-schnell",
];

function rank(model) {
  const index = PREFERRED.indexOf(model.id);
  return index === -1 ? PREFERRED.length : index;
}

export function pickStarterModels(models, ramGb, count = 3) {
  return models
    .filter((m) => !m.is_utility && m.installed === false && m.download_repo)
    .filter((m) => ramFit(m, ramGb) !== "over")
    .sort((a, b) => rank(a) - rank(b) || (modelMemoryNeedGb(a) || 99) - (modelMemoryNeedGb(b) || 99))
    .slice(0, count);
}

function openModelsSection(section) {
  window.dispatchEvent(
    new CustomEvent("db-menu-action", { detail: { action: "tab", tab: "models", section } }),
  );
}

export default function FirstRunWelcome({ models, ramGb, onInstalled }) {
  const { t } = useI18n();
  const picks = pickStarterModels(models, ramGb);

  return (
    <div className="first-run">
      <p className="first-run-title">{t("canvas.welcome.title")}</p>
      <p className="first-run-subtitle">
        {ramGb
          ? t("canvas.welcome.subtitleRam", { ram: Math.round(ramGb) })
          : t("canvas.welcome.subtitle")}
      </p>
      <ul className="first-run-picks">
        {picks.map((m) => {
          const seconds = defaultGenerationSeconds(m);
          return (
            <li key={m.id} className="first-run-pick">
              <div className="first-run-pick-head">
                <span className="first-run-pick-name" title={m.label}>{m.label}</span>
                <RamFitBadge model={m} ramGb={ramGb} />
              </div>
              <span className="first-run-pick-meta">
                {seconds
                  ? t("canvas.welcome.pickMeta", { size: m.size_gb, seconds })
                  : t("canvas.welcome.pickMetaSize", { size: m.size_gb })}
              </span>
              <ModelInstaller modelInfo={m} onInstalled={() => onInstalled?.(m.id)} compact />
            </li>
          );
        })}
      </ul>
      <div className="first-run-links">
        <button type="button" className="first-run-link" onClick={() => openModelsSection("browse")}>
          {t("canvas.welcome.allModels")}
        </button>
        <button type="button" className="first-run-link" onClick={() => openModelsSection("import")}>
          {t("canvas.welcome.fromDiffusionBee")}
        </button>
      </div>
    </div>
  );
}

import { useI18n } from "../i18n/I18nContext";
import { modelMemoryNeedGb, ramFit } from "../utils/ramFit";

export default function RamFitBadge({ model, ramGb }) {
  const { t } = useI18n();
  const fit = ramFit(model, ramGb);
  if (!fit) return null;
  const ram = Math.round(ramGb);
  return (
    <span
      className={`ram-fit-badge is-${fit}`}
      title={t("models.ramFit.needTitle", { need: modelMemoryNeedGb(model) })}
    >
      {t(`models.ramFit.${fit}`, { ram })}
    </span>
  );
}

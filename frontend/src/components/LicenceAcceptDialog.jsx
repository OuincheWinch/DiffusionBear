import { useId, useState } from "react";
import Dialog from "./Dialog";
import LicenceSummary from "./LicenceSummary";
import { useI18n } from "../i18n/I18nContext";

function LicenceAcceptForm({ titleId, modelName, summary, sourceLabel, onAccept, onCancel }) {
  const { t } = useI18n();
  const [checked, setChecked] = useState(false);

  return (
    <>
      <h3 id={titleId}>
        {t("licences.acceptTitle", { model: modelName || t("licences.acceptThisModel") })}
      </h3>
      <p className="licence-accept-intro">{t("licences.acceptIntro")}</p>

      <LicenceSummary summary={summary} sourceLabel={sourceLabel} />

      <label className="licence-accept-check">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => setChecked(event.target.checked)}
        />
        <span>{t("licences.acceptCheckbox")}</span>
      </label>

      <div className="licence-accept-actions">
        <button type="button" className="btn-mini" onClick={onCancel}>
          {t("app.cancel")}
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={!checked}
          onClick={onAccept}
        >
          {t("licences.acceptButton")}
        </button>
      </div>
    </>
  );
}

export default function LicenceAcceptDialog({
  open,
  modelName,
  summary,
  sourceLabel = "",
  onAccept,
  onCancel,
}) {
  const titleId = useId();

  return (
    <Dialog
      open={open}
      onClose={onCancel}
      className="modal licence-accept-overlay"
      bodyClassName="modal-body licence-accept-dialog"
      ariaLabelledBy={titleId}
      closeOnBackdrop={false}
    >
      {open && (
        <LicenceAcceptForm
          titleId={titleId}
          modelName={modelName}
          summary={summary}
          sourceLabel={sourceLabel}
          onAccept={onAccept}
          onCancel={onCancel}
        />
      )}
    </Dialog>
  );
}

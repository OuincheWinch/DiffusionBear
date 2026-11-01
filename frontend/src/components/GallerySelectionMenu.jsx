import { useEffect, useRef } from "react";
import { useI18n } from "../i18n/I18nContext";

export default function GallerySelectionMenu({ x, y, count, onSave, onDelete, onClose }) {
  const { t } = useI18n();
  const ref = useRef(null);

  useEffect(() => {
    const onPointerDown = (event) => {
      if (!ref.current?.contains(event.target)) onClose();
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") onClose();
    };
    const onScroll = () => onClose();
    window.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("scroll", onScroll, true);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("scroll", onScroll, true);
    };
  }, [onClose]);

  const left = Math.max(8, Math.min(x, window.innerWidth - 236));
  const top = Math.max(8, Math.min(y, window.innerHeight - 116));

  return (
    <div
      ref={ref}
      className="gallery-selection-menu"
      style={{ left, top }}
      role="menu"
      aria-label={t("gallery.selectionActions")}
    >
      <button type="button" role="menuitem" className="selection-menu-item" onClick={onSave}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        <span>{t("gallery.saveSelected", { count })}</span>
      </button>
      <button type="button" role="menuitem" className="selection-menu-item danger" onClick={onDelete}>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <polyline points="3 6 5 6 21 6" />
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
        </svg>
        <span>{t("gallery.deleteSelected", { count })}</span>
      </button>
    </div>
  );
}

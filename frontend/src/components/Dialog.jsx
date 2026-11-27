import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";

const focusableSelector = [
  "button:not([disabled])",
  "[href]",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[contenteditable='true']",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

export default function Dialog({
  open,
  onClose,
  children,
  className = "modal",
  bodyClassName = "modal-body",
  ariaLabel,
  ariaLabelledBy,
  ariaDescribedBy,
  closeOnBackdrop = true,
  initialFocusRef,
}) {
  const dialogRef = useRef(null);
  const onCloseRef = useRef(onClose);
  const generatedLabelId = useId();

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open || typeof document === "undefined") return undefined;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    let stopped = false;
    document.body.style.overflow = "hidden";

    const focusFirst = () => {
      if (stopped) return;
      const target = initialFocusRef?.current || dialogRef.current?.querySelector(focusableSelector);
      (target || dialogRef.current)?.focus?.();
    };

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current?.();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(dialogRef.current.querySelectorAll(focusableSelector));
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    const frame = typeof requestAnimationFrame === "function" ? requestAnimationFrame(focusFirst) : null;
    const timer = frame == null ? window.setTimeout(focusFirst, 0) : null;

    return () => {
      stopped = true;
      if (frame != null) cancelAnimationFrame(frame);
      if (timer != null) window.clearTimeout(timer);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus();
    };
  }, [initialFocusRef, open]);

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className={className}
      onMouseDown={(event) => {
        if (closeOnBackdrop && event.target === event.currentTarget) onCloseRef.current?.();
      }}
    >
      <div
        ref={dialogRef}
        className={bodyClassName}
        role="dialog"
        aria-modal="true"
        aria-label={ariaLabelledBy ? undefined : ariaLabel || "Dialog"}
        aria-labelledby={ariaLabelledBy}
        aria-describedby={ariaDescribedBy}
        tabIndex={-1}
        id={ariaLabelledBy ? undefined : generatedLabelId}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}

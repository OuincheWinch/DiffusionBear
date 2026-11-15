import { useLayoutEffect, useRef, useState } from "react";

const DEFAULT_PAGE_SIZE = 24;
const MAX_PAGE_SIZE = 100;
const DEFAULT_CARD_MIN = 180;
const DEFAULT_GAP = 12;

function readPositivePx(styles, name, fallback) {
  const value = Number.parseFloat(styles.getPropertyValue(name));
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

export function useGalleryGrid(active = true) {
  const galleryRef = useRef(null);
  const gridRef = useRef(null);
  const [layout, setLayout] = useState({ columns: 1, pageSize: DEFAULT_PAGE_SIZE });

  useLayoutEffect(() => {
    if (!active) return undefined;
    const gallery = galleryRef.current;
    if (!gallery) return undefined;

    let frame = 0;

    const measure = () => {
      frame = 0;
      const width = gallery.clientWidth;
      if (width <= 0) return;

      const styles = getComputedStyle(gallery);
      const cardMin = readPositivePx(styles, "--gallery-card-min", DEFAULT_CARD_MIN);
      const gap = readPositivePx(styles, "--gallery-gap", DEFAULT_GAP);
      const columns = Math.max(1, Math.floor((width + gap) / (cardMin + gap)));

      const grid = gridRef.current;
      const firstCard = grid?.firstElementChild;
      const rowHeight = firstCard ? firstCard.getBoundingClientRect().height + gap : 0;
      const top = (grid ?? gallery).getBoundingClientRect().top;
      const availableHeight = Math.max(0, window.innerHeight - Math.max(0, top) - gap);
      const measuredRows = rowHeight > 0 ? Math.ceil(availableHeight / rowHeight) : 3;
      const maxRows = Math.max(1, Math.floor(MAX_PAGE_SIZE / columns));
      const rows = Math.max(1, Math.min(measuredRows, maxRows));
      const pageSize = columns * rows;

      setLayout((previous) => (
        previous.columns === columns && previous.pageSize === pageSize
          ? previous
          : { columns, pageSize }
      ));
    };

    const schedule = () => {
      if (frame) return;
      frame = requestAnimationFrame(measure);
    };

    let observer = null;
    if (typeof ResizeObserver !== "undefined") {
      observer = new ResizeObserver(schedule);
      observer.observe(gallery);
    }
    window.addEventListener("resize", schedule);
    schedule();

    return () => {
      if (frame) cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener("resize", schedule);
    };
  }, [active]);

  return {
    galleryRef,
    gridRef,
    columns: layout.columns,
    pageSize: layout.pageSize,
  };
}

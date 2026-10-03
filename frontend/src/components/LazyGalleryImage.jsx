import { useState } from "react";
import { imageUrl } from "../api";
import { bindFullImageDrag, nativeDragSuppression } from "../utils/dragDrop";

export default function LazyGalleryImage({ item, alt }) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [thumbError, setThumbError] = useState(false);

  const src = thumbError ? imageUrl(item.id, false) : imageUrl(item.id, true);

  return (
    <div className="lazy-gallery-wrapper">
      {!isLoaded && <div className="lazy-gallery-skeleton" />}
      <img
        loading="lazy"
        data-mlx-image-id={item.id}
        data-mlx-file-url={item.file_url || ""}
        src={src}
        alt={alt || item.prompt}
        className={`gallery-thumb ${isLoaded ? "loaded" : ""}`}
        style={nativeDragSuppression()}
        onLoad={() => setIsLoaded(true)}
        onError={() => {
          if (!thumbError) {
            setThumbError(true);
          }
        }}
        {...bindFullImageDrag(item)}
      />
    </div>
  );
}

/**
 * The actual drag source: a transparent link over the thumbnail whose href is the image's
 * real file on disk.
 *
 * Why an anchor and not the <img>: macOS builds a web drag's pasteboard from the DOM
 * element being dragged, so an <img> dragging itself offers its last-loaded URL — an
 * http://127.0.0.1:8001 address, which Finder saves as a .webloc link stub pointing at a
 * server that stops with the app. An anchor contributes its href instead, and the href is
 * in the DOM from first render, so there is no press-time race to lose.
 *
 * Why not handle this natively in the shell: tried, and it cannot work. WKWebView hit-tests
 * to an internal content subview, so mouseDown/mouseDragged overrides on a WKWebView
 * subclass never fire for clicks on the page. That produced no drag at all.
 *
 * The overlay is transparent and does not capture the click: WebKit only treats a link as
 * draggable once the pointer moves, and onClick is suppressed so the cell still opens.
 */

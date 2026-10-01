import { useState } from "react";
import { imageUrl } from "../api";
import { bindFullImageDrag } from "../utils/dragDrop";

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

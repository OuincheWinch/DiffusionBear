import { boxColor } from "../utils/spatialBoxes";
import { galleryBoxes } from "../utils/galleryBoxes";
import "./GallerySpatialBoxOverlay.css";

export default function GallerySpatialBoxOverlay({ item }) {
  const boxes = galleryBoxes(item);
  if (!boxes.length) return null;
  return (
    <div className="gallery-spatial-overlay" aria-hidden="true">
      {boxes.map((box, index) => {
        const [ymin, xmin, ymax, xmax] = box.bbox;
        const color = boxColor(index);
        return (
          <div
            key={box.key}
            className={`gallery-spatial-box ${box.kind === "text" ? "is-text" : ""}`}
            style={{
              left: `${xmin / 10}%`,
              top: `${ymin / 10}%`,
              width: `${Math.max(0.5, (xmax - xmin) / 10)}%`,
              height: `${Math.max(0.5, (ymax - ymin) / 10)}%`,
              borderColor: color,
              "--box-color": color,
            }}
          >
            <span className="gallery-spatial-label" title={box.desc || box.id}>
              <span className="gallery-spatial-index">{index + 1}</span>
              {box.desc || box.id}
            </span>
          </div>
        );
      })}
    </div>
  );
}

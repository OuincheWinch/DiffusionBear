import { API_BASE, withMediaToken } from "../api";
import { useI18n } from "../i18n/I18nContext";

const SERVED_EXTENSIONS = /\.(png|jpe?g|webp|heic|heif)$/i;

export function assetUrl(value) {
  if (typeof value !== "string") return null;
  const raw = value.trim();
  if (!raw) return null;
  if (/^(https?:|data:|blob:)/i.test(raw)) return raw;
  if (raw.startsWith("/api/")) return withMediaToken(`${API_BASE}${raw}`);

  let name = raw;
  if (raw.startsWith("file://")) {
    try {
      name = decodeURIComponent(new URL(raw).pathname);
    } catch {
      return null;
    }
  }

  name = name.replace(/\\/g, "/").split("/").pop();
  if (!name) return null;
  if (!SERVED_EXTENSIONS.test(name)) return null;
  return withMediaToken(`${API_BASE}/api/uploads/${encodeURIComponent(name)}`);
}

export function resolveMaskAssetUrl(item) {
  if (!item) return null;
  for (const cand of [item.mask_file, item.mask_image, item.mask_path]) {
    const url = assetUrl(cand);
    if (url) return url;
  }
  return null;
}

function uniqueValues(values) {
  const seen = new Set();
  return values.filter((value) => {
    if (typeof value !== "string") return false;
    const trimmed = value.trim();
    if (!trimmed || seen.has(trimmed)) return false;
    seen.add(trimmed);
    return true;
  });
}

export default function GalleryInputs({ item }) {
  const { t } = useI18n();
  if (!item) return null;

  const references = uniqueValues([
    ...(Array.isArray(item.reference_images) ? item.reference_images : []),
    item.reference_image,
    item.init_image,
    item.reference_image_path,
  ]);
  const masks = uniqueValues([
    item.mask_file,
    item.mask_image,
    item.mask_path,
  ]);

  const seen = new Set();
  const assets = [];
  const push = (value, label) => {
    const url = assetUrl(value);
    if (!url || seen.has(url)) return;
    seen.add(url);
    const index = assets.filter((asset) => asset.kind === label).length + 1;
    assets.push({
      url,
      kind: label,
      label: index > 1 ? `${label} ${index}` : label,
    });
  };

  references.forEach((value) => push(value, t("gallery.inputSource")));
  masks.forEach((value) => push(value, t("gallery.inputMask")));

  if (!assets.length) return null;

  return (
    <section className="detail-inputs" aria-labelledby={`gallery-inputs-${item.id}`}>
      <h4 className="detail-inputs-title" id={`gallery-inputs-${item.id}`}>
        {t("gallery.inputsTitle")}
      </h4>
      <div className="detail-inputs-grid">
        {assets.map((asset) => (
          <a
            key={asset.url}
            className="detail-input-thumb"
            href={asset.url}
            target="_blank"
            rel="noreferrer"
            title={asset.label}
            aria-label={asset.label}
          >
            <img src={asset.url} alt={asset.label} loading="lazy" />
            <span>{asset.label}</span>
          </a>
        ))}
      </div>
    </section>
  );
}

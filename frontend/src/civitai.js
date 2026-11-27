export const CIVITAI_ORIGIN = "https://civitai.com";
export const CIVITAI_REF_CODE = "88C8VEBA";

const CIVITAI_HOST = /(^|\.)civitai\.(com|red|green)$/i;

export function withCivitaiReferral(value) {
  if (!/^https?:\/\//i.test(value)) return value;
  try {
    const url = new URL(value, CIVITAI_ORIGIN);
    if (!CIVITAI_HOST.test(url.hostname)) return value;
    url.protocol = "https:";
    url.hostname = "civitai.com";
    url.port = "";
    url.searchParams.set("ref_code", CIVITAI_REF_CODE);
    return url.toString();
  } catch {
    return value;
  }
}

export function civitaiModelUrl(modelId, versionId) {
  const id = String(modelId || "").trim();
  const url = new URL(`/models/${encodeURIComponent(id)}`, CIVITAI_ORIGIN);
  const version = String(versionId || "").trim();
  if (version) url.searchParams.set("modelVersionId", version);
  return withCivitaiReferral(url.toString());
}

export const CIVITAI_HOME_URL = withCivitaiReferral(CIVITAI_ORIGIN);

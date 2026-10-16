/**
 * Resolves the underlying base architecture key ("sdxl", "flux2", "krea2", "z-image")
 * from a model's ID or metadata.
 */
export function getModelBase(m, modelId) {
  const id = (modelId || m?.id || "").toLowerCase();
  const format = (m?.lora_format || "").toLowerCase();
  if (id.includes("flux") || format.includes("flux")) return "flux2";
  if (id.includes("sdxl") || id.includes("juggernaut") || format.includes("sdxl") || m?.engine === "sdxl") return "sdxl";
  if (id.includes("krea") || format.includes("krea")) return "krea2";
  if (id.includes("z-image") || id.includes("zit") || id.includes("zimage") || format.includes("z-image") || format.includes("zimage")) return "z-image";
  return null;
}

/**
 * Matches an active LoRA object against entries in the LoRA registry using
 * file path, name, or filename stem.
 */
export function findLoraEntry(lora, registry) {
  if (!lora || !registry || !registry.length) return null;
  const lPath = (lora.path || "").toLowerCase();
  const lName = (lora.name || "").toLowerCase();
  const lFilename = lPath ? lPath.split("/").pop().replace(/\.safetensors$/, "").toLowerCase() : "";

  return (
    registry.find((r) => {
      const rPath = (r.path || "").toLowerCase();
      const rName = (r.name || "").toLowerCase();
      if (rPath && lPath && rPath === lPath) return true;
      if (rName && lName && rName === lName) return true;
      if (rName && lFilename && rName === lFilename) return true;
      if (rPath && lPath && rPath.split("/").pop().toLowerCase() === lPath.split("/").pop().toLowerCase()) return true;
      if (rName && lPath && (lPath.endsWith(`/${rName}.safetensors`) || lPath.endsWith(rName))) return true;
      return false;
    }) || null
  );
}

/**
 * Checks if a LoRA object or registry entry is the Krea 2 Turbo 4-step distillation adapter.
 */
export function isKreaDistillLora(lora) {
  if (!lora) return false;
  const path = (lora.path || "").toLowerCase();
  const name = (lora.name || "").toLowerCase();
  const base = (lora.base_model || "").toLowerCase();
  return Boolean(
    lora.autoDistill ||
    path.includes("krea2_turbo_4step") ||
    (base === "krea2" && name.includes("distill")) ||
    (name.includes("krea") && name.includes("distill"))
  );
}

/**
 * True when a LoRA filename/name is itself a step-distillation adapter
 * (SDXL-Lightning, Hyper-SD, LCM, ...). Mirrors the backend guard in
 * generator._distill_lora_active: stacking two of them fights over the same
 * timestep schedule, so applying a preset that carries one must first clear
 * another that is already active.
 */
const DISTILL_LORA_RE = /lightning|hyper[-_ ]?sd|distill|lcm/i;

export function isDistillLoraName(name) {
  return DISTILL_LORA_RE.test(String(name || ""));
}

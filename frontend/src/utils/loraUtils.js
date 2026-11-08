/**
 * Resolves the underlying base architecture key ("sdxl", "flux2", "krea2", "z-image", "qwen", "moss")
 * from a model's ID or metadata.
 */
export function getModelBase(m, modelId) {
  const id = (modelId || m?.id || "").toLowerCase();
  const format = (m?.lora_format || "").toLowerCase();
  if (id.includes("moss") || format.includes("moss")) return "moss";
  if (id.includes("qwen") || format.includes("qwen")) return "qwen";
  if (id.includes("flux") || format.includes("flux")) return "flux2";
  if (id.includes("sdxl") || id.includes("juggernaut") || format.includes("sdxl") || m?.engine === "sdxl") return "sdxl";
  if (id.includes("krea") || format.includes("krea")) return "krea2";
  if (id.includes("fibo") || format.includes("fibo")) return "fibo2";
  if (id.includes("z-image") || id.includes("zit") || id.includes("zimage") || format.includes("z-image") || format.includes("zimage")) return "z-image";
  return null;
}

/** True for the Qwen-Image Edit variant, which shares the Qwen base key. */
export function isQwenEditModel(m, modelId) {
  const id = (modelId || m?.id || "").toLowerCase();
  return id.includes("qwen-image-edit");
}

/**
 * True when a LoRA entry is a MOSS-Image step-distillation adapter (the
 * 8-step DMD / dataspace adapter). Such an adapter has to be sampled with its
 * own CFG-free dataspace schedule at 8 steps, so the generation form rewrites
 * steps/guidance when one is active.
 */
export function isMossDistillLora(lora) {
  if (!lora) return false;
  const path = (lora.path || "").toLowerCase();
  const name = (lora.name || "").toLowerCase();
  const base = (lora.base_model || "").toLowerCase();
  return Boolean(
    lora.autoDistill ||
    base === "moss" && /dmd|distill|dataspace|8[-_ ]?steps/.test(`${name} ${path}`),
    /moss.*(dmd|distill)/.test(path),
  );
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
    path.includes("krea2_turbo_3step") ||
    path.includes("krea2_turbo_4step") ||
    (base === "krea2" && (name.includes("distill") || name.includes("3step") || name.includes("4step"))) ||
    (name.includes("krea") && (name.includes("distill") || name.includes("3step") || name.includes("4step")))
  );
}

/**
 * Checks if a LoRA object or registry entry is the Qwen-Image 2.1 2-step DMD2 distillation adapter.
 */
export function isQwenDistillLora(lora) {
  if (!lora) return false;
  const path = (lora.path || "").toLowerCase();
  const name = (lora.name || "").toLowerCase();
  const base = (lora.base_model || "").toLowerCase();
  return Boolean(
    path.includes("qwen-image-2.1-dmd2") ||
    name.includes("qwen-image-2.1-dmd2") ||
    (base === "qwen" && /dmd2?|2[-_ ]?steps?/.test(`${name} ${path}`)) ||
    (name.includes("qwen") && /dmd2?|2[-_ ]?steps?/.test(`${name} ${path}`))
  );
}

/**
 * True when a LoRA filename/name is itself a step-distillation adapter
 * (SDXL-Lightning, Hyper-SD, LCM, DMD2, ...). Mirrors the backend guard in
 * generator._distill_lora_active: stacking two of them fights over the same
 * timestep schedule, so applying a preset that carries one must first clear
 * another that is already active.
 */
const DISTILL_LORA_RE = /lightning|hyper[-_ ]?sd|distill|lcm|dmd2?/i;

export function isDistillLoraName(name) {
  return DISTILL_LORA_RE.test(String(name || ""));
}

/** Filters registry rows by the text a user typed into the LoRA search field. */
export function filterLoras(loras, query) {
  const rows = Array.isArray(loras) ? loras : [];
  const needle = String(query ?? "").trim().toLowerCase();
  if (!needle) return rows;
  return rows.filter((lora) =>
    [lora?.name, lora?.civitai_model_name, lora?.path].some((value) =>
      String(value ?? "").toLowerCase().includes(needle),
    ),
  );
}

/**
 * Reuse the image's LoRAs when it has any; otherwise keep the current selection.
 * The model-compatibility guard still filters anything the target model cannot use.
 */
export function reuseLoras(sourceLoras, currentLoras) {
  if (Array.isArray(sourceLoras) && sourceLoras.length > 0) {
    return sourceLoras;
  }
  return Array.isArray(currentLoras) ? currentLoras : [];
}

/** Stores the active selection against its architecture base. */
export function rememberLoras(memory, base, loras) {
  if (!base || !Array.isArray(loras)) return memory;
  return { ...memory, [base]: loras };
}

/** Returns the remembered selection, or null when there is nothing to restore. */
export function recallLoras(memory, base) {
  if (!base) return null;
  const saved = memory?.[base];
  return Array.isArray(saved) && saved.length > 0 ? saved : null;
}

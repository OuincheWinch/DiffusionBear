import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, API_BASE } from "../api";
import Dialog from "./Dialog";
import GenerationStack from "./GenerationStack";
import ResultCanvas from "./ResultCanvas";
import UniversalDownloader from "./UniversalDownloader";
import LoraManagerDrawer from "./LoraManagerDrawer";
import ModelInstaller from "./ModelInstaller";
import GenerationParams from "./GenerationParams";
import SizeSelector from "./SizeSelector";
import { findLoraEntry, getModelBase, isKreaDistillLora } from "../utils/loraUtils";
import { normalizeRequest, referenceItems, resolveRequestModel } from "../utils/requestUtils";
import { useGenerationJob } from "../hooks/useGenerationJob";
import { useModelConfig } from "../hooks/useModelConfig";
import { useLoraPanel } from "../hooks/useLoraPanel";
import { useSettings } from "../hooks/useSettings";
import { useI18n } from "../i18n/I18nContext";

function getNextSeed(currentSeed) {
  if (currentSeed != null && currentSeed !== "" && !isNaN(Number(currentSeed))) {
    return Number(currentSeed) + 1;
  }
  return Math.floor(Date.now() % 1000000);
}

function generationLoraPayload(loras) {
  return (Array.isArray(loras) ? loras : [])
    .filter((lora) => typeof lora?.path === "string" && lora.path.trim())
    .map((lora) => {
      const scale = Number(lora.scale ?? 1);
      return { path: lora.path, scale: Number.isFinite(scale) ? scale : 1 };
    });
}

function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function modelNeedsConfirmation(modelInfo) {
  return Boolean(
    modelInfo?.experimental
      || modelInfo?.requires_confirmation
      || modelInfo?.confirmation_required
      || /experimental/i.test(String(modelInfo?.label || "")),
  );
}

function enhancerEngineFor(modelInfo, model) {
  return modelInfo?.id
    || modelInfo?.enhancer_engine
    || modelInfo?.prompt_enhancer_engine
    || modelInfo?.enhancer_key
    || modelInfo?.ecosystem
    || model
    || "";
}

function kreaDistillUpdater(steps, registry) {
  if (steps > 4) {
    return (prev) => prev.filter((l) => !(isKreaDistillLora(l) && l.autoDistill));
  }
  const regEntry = (registry || []).find(
    (l) => l.base_model === "krea2" && isKreaDistillLora(l),
  );
  if (!regEntry?.path) return (prev) => prev;
  return (prev) => {
    if (prev.some(isKreaDistillLora)) return prev;
    return [...prev, {
      path: regEntry.path,
      scale: 1.0,
      name: regEntry.name,
      base_model: regEntry.base_model || "krea2",
      autoDistill: true,
    }];
  };
}

// Swatches carry the i18n key rather than the label: the table lives at module
// level, outside the component, so it cannot call useI18n(). `hex` stays literal
// because it is the value being inserted into the prompt.
const HEX_PALETTE = [
  { key: "generate.palette.neonPink", hex: "#FF3366" },
  { key: "generate.palette.cyberCyan", hex: "#00E5FF" },
  { key: "generate.palette.cyberGold", hex: "#FFD700" },
  { key: "generate.palette.electricViolet", hex: "#7928CA" },
  { key: "generate.palette.neonGreen", hex: "#00FF66" },
  { key: "generate.palette.tangerine", hex: "#FF5500" },
  { key: "generate.palette.pureWhite", hex: "#FFFFFF" },
  { key: "generate.palette.matteBlack", hex: "#111111" },
  { key: "generate.palette.royalBlue", hex: "#2E5BFF" },
  { key: "generate.palette.lavender", hex: "#E056FD" },
  { key: "generate.palette.crimson", hex: "#FF0033" },
  { key: "generate.palette.emerald", hex: "#00B894" },
];

export default function GenerateForm({ onGenerated, initialParams, onModelChange, onImageSaved }) {
  const { t } = useI18n();
  const {
    jobId,
    status,
    jobPhase,
    jobPhaseDetail,
    progress,
    error,
    setError,
    generatingPrompt,
    currentResult,
    setCurrentResult,
    batchResults,
    submittedParams,
    setSubmittedParams,
    showSwitchDialog,
    setShowSwitchDialog,
    activateJob,
    cancelJob,
  } = useGenerationJob({ onGenerated, onImageSaved });

  const {
    models,
    model,
    setModel,
    modelInfo,
    loading: modelsLoading,
    error: modelsError,
    refreshModels,
  } = useModelConfig({ onModelChange });

  const { settings: appSettings } = useSettings();

  // Single source of truth for the app header title: keep it in sync with the
  // model that is actually selected (covers model changes from "Reuse params"
  // in the Browser tab, which updates `model` without going through switchModel).
  useEffect(() => {
    if (modelInfo?.label) onModelChange?.(modelInfo.label);
  }, [modelInfo?.label, onModelChange]);

  const {
    loras,
    setLoras,
    loraRegistry,
    setLoraRegistry,
    refreshLoraRegistry,
    newLora,
    setNewLora,
    savingLora,
    uploadProgress,
    activeTriggerWords,
    uploadLoraFile,
    saveNewLora,
  } = useLoraPanel({ onError: setError });

  const [prompt, setPrompt] = useState("");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(4);
  const [guidance, setGuidance] = useState(1.0);
  const [seed, setSeed] = useState("");
  const [batch, setBatch] = useState(1);
  const [negativePrompt, setNegativePrompt] = useState("");
  const [sampler, setSampler] = useState("euler_trailing");
  const [cacheInterval, setCacheInterval] = useState(1);
  const [dragOver, setDragOver] = useState(false);

  const promptRef = useRef(null);
  const [refImages, setRefImages] = useState([]); // [{id, path, preview, name}]
  const [refStrength, setRefStrength] = useState(0.6);
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [customHex, setCustomHex] = useState("#FF3366");
  const [outputFormat, setOutputFormat] = useState("png");
  const [stealthMode, setStealthMode] = useState(false);
  const [fastVae, setFastVae] = useState(true);
  const [maxPixels, setMaxPixels] = useState(null);
  const [enhancing, setEnhancing] = useState(false);
  const [enhanceJson, setEnhanceJson] = useState(false);
  const [enhanceFeedback, setEnhanceFeedback] = useState(null);
  const [showEnhanceWarning, setShowEnhanceWarning] = useState(false);
  const enhanceAbortRef = useRef(null);
  const enhanceRequestRef = useRef(0);
  const hydrationKeyRef = useRef(null);
  const pendingRequestRef = useRef(null);
  const initialModelDefaultsAppliedRef = useRef(false);

  useEffect(() => () => {
    enhanceAbortRef.current?.abort();
    enhanceRequestRef.current += 1;
  }, []);



  const hydrateRequest = useCallback((request) => {
    const source = normalizeRequest(request);
    const requestedModel = source.model;
    const matched = resolveRequestModel(source, models);
    const target = matched || (!requestedModel ? models.find((item) => item.id === model) : null);
    const targetInfo = target || modelInfo;
    const supportsReference = Boolean(targetInfo?.supports_ref || targetInfo?.supports_multi_reference);
    const targetMaxReferences = Math.max(1, Number(targetInfo?.max_reference_images) || 1);
    const nextSampler = source.sampler && targetInfo?.samplers?.includes(source.sampler)
      ? source.sampler
      : targetInfo?.default_sampler || targetInfo?.samplers?.[0] || "";

    setPrompt(source.prompt);
    setWidth(Number(source.width) || 1024);
    setHeight(Number(source.height) || 1024);
    setSteps(Number(source.steps) || targetInfo?.default_steps || 4);
    setGuidance(source.guidance == null ? targetInfo?.default_guidance ?? 1 : Number(source.guidance));
    setSeed(source.seed ?? "");
    setBatch(Math.max(1, Number(source.batch) || 1));
    setNegativePrompt(source.negative_prompt || "");
    setSampler(nextSampler);
    setCacheInterval(Math.max(1, Number(source.cache_interval) || 1));
    setLoras(Array.isArray(source.loras) ? source.loras : []);
    setOutputFormat(source.output_format || "png");
    setStealthMode(Boolean(source.stealth));
    setFastVae(targetInfo?.supports_fast_vae
      ? (source.fast_vae == null ? Boolean(targetInfo?.default_fast_vae ?? true) : Boolean(source.fast_vae))
      : false);
    setMaxPixels(
      source.max_pixels != null && (!targetInfo?.max_pixels || source.max_pixels <= targetInfo.max_pixels)
        ? Number(source.max_pixels)
        : targetInfo?.max_pixels || null,
    );
    setRefStrength(source.reference_strength == null ? 0.6 : Number(source.reference_strength));
    setRefImages(supportsReference ? referenceItems(source).slice(0, targetMaxReferences) : []);
    if (target) setModel(target.id);
    if (source.id) setCurrentResult(source);
    setEnhanceFeedback(null);
    if (requestedModel && !matched) {
      setError(t("generate.error.modelNotInRegistry", { model: requestedModel }));
    }
    return matched;
  }, [model, modelInfo, models, setError, setModel, t]);

  useEffect(() => {
    if (!initialParams || !models.length) return;
    const key = initialParams.key ?? initialParams.id ?? initialParams;
    if (hydrationKeyRef.current === key) return;
    hydrationKeyRef.current = key;
    hydrateRequest(initialParams);
  }, [hydrateRequest, initialParams, models.length]);

  useEffect(() => {
    if (!models.length || !pendingRequestRef.current) return;
    const request = pendingRequestRef.current;
    pendingRequestRef.current = null;
    hydrateRequest(request);
  }, [hydrateRequest, models.length]);

  // Apply saved global defaults to a freshly-mounted (empty) form once.
  const defaultsApplied = useRef(false);
  useEffect(() => {
    if (defaultsApplied.current || initialParams) return;
    if (!appSettings) return;
    defaultsApplied.current = true;
    setOutputFormat(appSettings.default_output_format || "png");
    setStealthMode(Boolean(appSettings.default_stealth));
    setFastVae(appSettings.default_fast_vae != null ? Boolean(appSettings.default_fast_vae) : true);
    if (appSettings.default_sampler) setSampler(appSettings.default_sampler);
  }, [appSettings, initialParams]);

  useEffect(() => {
    if (initialModelDefaultsAppliedRef.current || initialParams || !modelInfo?.id || pendingRequestRef.current) return;
    initialModelDefaultsAppliedRef.current = true;
    const preset = modelInfo.presets?.[0];
    setSteps(Number(modelInfo.default_steps ?? preset?.steps ?? 4));
    setGuidance(Number(modelInfo.default_guidance ?? 1));
    setCacheInterval(Number(modelInfo.default_cache_interval ?? 1));
    setFastVae(modelInfo.supports_fast_vae ? Boolean(modelInfo.default_fast_vae ?? true) : false);
    setSampler(
      modelInfo.samplers?.length
        ? modelInfo.default_sampler && modelInfo.samplers.includes(modelInfo.default_sampler)
          ? modelInfo.default_sampler
          : modelInfo.samplers[0]
        : "",
    );
    let nextWidth = Number(modelInfo.default_width) || Number(preset?.width) || width;
    let nextHeight = Number(modelInfo.default_height) || Number(preset?.height) || height;
    if (modelInfo.max_pixels && nextWidth * nextHeight > modelInfo.max_pixels) {
      const scale = Math.sqrt(modelInfo.max_pixels / (nextWidth * nextHeight));
      nextWidth = Math.max(256, Math.round((nextWidth * scale) / 16) * 16);
      nextHeight = Math.max(256, Math.round((nextHeight * scale) / 16) * 16);
    }
    setWidth(nextWidth);
    setHeight(nextHeight);
    setMaxPixels(modelInfo.max_pixels ?? null);
  }, [height, initialParams, modelInfo, width]);

  // Reactive Guard: Ensure active LoRAs are strictly compatible with the current model
  useEffect(() => {
    if (!loraRegistry || !loraRegistry.length) return;
    const currentBase = getModelBase(modelInfo, model);
    const supportsLoras = Boolean(modelInfo?.supports_loras);

    setLoras((currentLoras) => {
      if (!supportsLoras || !currentBase) {
        return currentLoras.length > 0 ? [] : currentLoras;
      }
      const filtered = currentLoras.filter((l) => {
        const entry = findLoraEntry(l, loraRegistry);
        const lBase = l.base_model || entry?.base_model;
        if (!lBase) return true;
        return lBase === currentBase;
      });
      return filtered.length !== currentLoras.length ? filtered : currentLoras;
    });
  }, [model, modelInfo, loraRegistry]);

  function toggleTriggerWord(word) {
    if (!word) return;
    const regex = new RegExp(`(^|\\s|,)${escapeRegExp(word)}($|\\s|,)`, "i");
    if (regex.test(prompt)) {
      const updated = prompt
        .replace(regex, " ")
        .replace(/,\s*,/g, ",")
        .replace(/\s{2,}/g, " ")
        .trim();
      setPrompt(updated);
    } else {
      const trimmed = prompt.trim();
      const updated = trimmed
        ? `${trimmed.replace(/,+$/, "")}, ${word}`
        : word;
      setPrompt(updated);
    }
  }

  const switchModel = useCallback((id) => {
    const nextModel = models.find((item) => item.id === id);
    if (!nextModel) {
      setError(t("generate.error.modelNotInRegistry", { model: id }));
      return;
    }
    setModel(id);
    onModelChange?.(nextModel.label);
    const firstPreset = nextModel.presets?.[0];
    const nextSteps = Number(nextModel.default_steps ?? firstPreset?.steps ?? 4);
    setSteps(nextSteps);
    setMaxPixels(nextModel.max_pixels ?? null);
    const defaultWidth = Number(nextModel.default_width);
    const defaultHeight = Number(nextModel.default_height);
    if (defaultWidth && defaultHeight) {
      setWidth(defaultWidth);
      setHeight(defaultHeight);
    } else if (nextModel.max_pixels && width * height > nextModel.max_pixels) {
      const scale = Math.sqrt(nextModel.max_pixels / (width * height));
      setWidth(Math.max(256, Math.round(width * scale / 16) * 16));
      setHeight(Math.max(256, Math.round(height * scale / 16) * 16));
    } else if (firstPreset && (!nextModel.max_pixels || width * height > nextModel.max_pixels)) {
      setWidth(firstPreset.width);
      setHeight(firstPreset.height);
    }
    if (nextModel.default_guidance != null) setGuidance(nextModel.default_guidance);
    if (nextModel.default_cache_interval != null) setCacheInterval(nextModel.default_cache_interval);
    if (nextModel.default_fast_vae != null) setFastVae(Boolean(nextModel.default_fast_vae));
    else if (!nextModel.supports_fast_vae) setFastVae(false);
    if (nextModel.samplers?.length) {
      setSampler(nextModel.default_sampler && nextModel.samplers.includes(nextModel.default_sampler)
        ? nextModel.default_sampler
        : nextModel.samplers[0]);
    } else {
      setSampler("");
    }
    const targetBase = getModelBase(nextModel, id);
    if (!nextModel.supports_loras || !targetBase) {
      setLoras([]);
    } else {
      setLoras((current) => current.filter((lora) => {
        const entry = findLoraEntry(lora, loraRegistry);
        return (entry?.base_model || lora.base_model) === targetBase;
      }));
    }
    setEnhanceFeedback(null);
  }, [height, loraRegistry, models, onModelChange, setError, setModel, t, width]);

  useEffect(() => {
    function handleLoadPrompt(event) {
      const request = event.detail;
      if (!request) return;
      if (!models.length) {
        pendingRequestRef.current = request;
        return;
      }
      hydrateRequest(request);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
    window.addEventListener("mlx:load-prompt", handleLoadPrompt);
    return () => window.removeEventListener("mlx:load-prompt", handleLoadPrompt);
  }, [hydrateRequest, models.length]);

  useEffect(() => {
    if (getModelBase(modelInfo, model) !== "krea2") return;
    setLoras(kreaDistillUpdater(steps, loraRegistry));
  }, [loraRegistry, model, modelInfo, steps]);

  function fmt(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m > 0
      ? t("params.duration.minutesSeconds", { m, s: sec })
      : t("params.duration.secondsOnly", { s: sec });
  }

  const supportsMultiRef = Boolean(modelInfo.supports_multi_reference);
  const maxRefImages = Math.max(1, Number(modelInfo.max_reference_images) || 1);
  const supportsRef = supportsMultiRef || Boolean(modelInfo.supports_ref);
  const enhancerAvailable = modelInfo.supports_prompt_enhancer !== false;
  const modelConfirmationRequired = modelNeedsConfirmation(modelInfo);
  const currentParams = () =>
    JSON.stringify({
      prompt,
      model,
      width: Number(width),
      height: Number(height),
      steps: Number(steps),
      guidance: modelInfo.supports_guidance ? Number(guidance) : null,
      seed: seed === "" ? null : Number(seed),
      quantization: 4,
      batch: Number(batch),
      negative_prompt: modelInfo.supports_negative ? negativePrompt : "",
      sampler: modelInfo.samplers?.length ? sampler : undefined,
      cache_interval: modelInfo.engine === "sdxl" ? Number(cacheInterval) : 1,
      loras: (modelInfo.supports_loras ? generationLoraPayload([...loras].sort((x, y) => x.path.localeCompare(y.path))) : []).slice(0, 16),
      reference_images: supportsRef ? refImages.map((img) => img.path) : [],
      reference_strength: supportsRef && refImages.length > 0 && !supportsMultiRef ? Number(refStrength) : undefined,
      output_format: outputFormat,
      stealth: stealthMode,
      fast_vae: modelInfo.supports_fast_vae ? fastVae : false,
      max_pixels: maxPixels && maxPixels < (modelInfo.max_pixels ?? Infinity) ? Number(maxPixels) : null,
    });

  async function runEnhancePrompt() {
    if (!prompt.trim() || enhancing) return;
    if (!enhancerAvailable) {
      setError(t("generate.error.enhancerUnavailable"));
      return;
    }
    const requestId = ++enhanceRequestRef.current;
    const controller = new AbortController();
    enhanceAbortRef.current = controller;
    setEnhancing(true);
    setEnhanceFeedback(null);
    setError(null);
    try {
      const res = await api("/api/prompt/enhance", {
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify({
          prompt,
          model: enhancerEngineFor(modelInfo, model),
          loras: generationLoraPayload(loras),
          format: enhanceJson ? "json" : "text",
        }),
      });
      if (requestId !== enhanceRequestRef.current) return;
      if (res?.cancelled) {
        setEnhanceFeedback({ type: "info", text: t("generate.enhanceFeedback.cancelled") });
      } else if (res?.enhanced) {
        setPrompt(res.enhanced);
        setEnhanceFeedback({
          type: res.parse_error ? "info" : "success",
          text: res.parse_error
            ? t("generate.enhanceFeedback.proseFallback")
            : t("generate.enhanceFeedback.enhancedFor", { engine: res.engine || modelInfo.label }),
        });
      } else {
        throw new Error(t("generate.error.enhancerNoPrompt"));
      }
    } catch (err) {
      if (requestId !== enhanceRequestRef.current || err?.name === "AbortError") return;
      const message = t("generate.error.enhancerFailed", { message: err.message || err });
      setError(message);
      setEnhanceFeedback({ type: "error", text: message });
    } finally {
      if (requestId === enhanceRequestRef.current) {
        if (enhanceAbortRef.current === controller) enhanceAbortRef.current = null;
        setEnhancing(false);
      }
    }
  }

  function handleEnhancePrompt() {
    if (!prompt.trim() || enhancing) return;
    if (!enhancerAvailable) {
      setError(t("generate.error.enhancerUnavailable"));
      return;
    }
    if (busy) {
      setShowEnhanceWarning(true);
      return;
    }
    runEnhancePrompt();
  }

  function cancelEnhancePrompt() {
    enhanceRequestRef.current += 1;
    enhanceAbortRef.current?.abort();
    enhanceAbortRef.current = null;
    setEnhancing(false);
    setEnhanceFeedback({ type: "info", text: t("generate.enhanceFeedback.cancelled") });
  }

  function removeRefImage(index) {
    setRefImages((prev) => {
      const target = prev[index];
      if (target?.preview && target.preview.startsWith("blob:")) {
        URL.revokeObjectURL(target.preview);
      }
      return prev.filter((_, i) => i !== index);
    });
  }

  function addRefImage(item) {
    if (!supportsRef) {
      setError(t("generate.error.refNotSupported", { model: modelInfo.label }));
      return;
    }
    setRefImages((previous) => {
      if (previous.length >= maxRefImages) {
        setError(t(
          maxRefImages === 1 ? "generate.error.maxReferencesOne" : "generate.error.maxReferencesMany",
          { max: maxRefImages }
        ));
        return previous;
      }
      if (previous.some((image) => image.path === item.path)) return previous;
      return [...previous, item];
    });
  }

  const isImageFile = (f) =>
    Boolean((f.type && f.type.startsWith("image/")) || /\.(heic|heif|png|jpe?g|webp)$/i.test(f.name || ""));

  async function uploadImageFiles(files) {
    if (!supportsRef) {
      setError(t("generate.error.refNotSupported", { model: modelInfo.label }));
      return;
    }
    const list = Array.from(files).filter(isImageFile);
    if (!list.length) return;
    const remainingSlots = Math.max(0, maxRefImages - refImages.length);
    if (remainingSlots === 0) {
      setError(t(
        maxRefImages === 1 ? "generate.error.maxReferencesOne" : "generate.error.maxReferencesMany",
        { max: maxRefImages }
      ));
      return;
    }
    const toUpload = list.slice(0, remainingSlots);
    for (const file of toUpload) {
      const form = new FormData();
      form.append("file", file);
      try {
        const res = await api("/api/uploads", { method: "POST", body: form });
        const isHeic = /\.(heic|heif)$/i.test(file.name || "");
        // Use res.url (served as PNG by backend) for HEIC and when available so all browsers render it
        const previewUrl = res.url ? `${API_BASE}${res.url}` : isHeic ? "" : URL.createObjectURL(file);
        setRefImages((previous) => {
          if (previous.length >= maxRefImages) {
            setError(t(
              maxRefImages === 1 ? "generate.error.maxReferencesOne" : "generate.error.maxReferencesMany",
              { max: maxRefImages }
            ));
            return previous;
          }
          return [
            ...previous,
            {
              id: crypto.randomUUID(),
              path: res.path,
              preview: previewUrl,
              name: file.name,
            },
          ];
        });
      } catch (err) {
        setError(t("generate.error.refUploadFailed", { message: err.message || err }));
      }
    }
  }

  async function pickRefImages(e) {
    if (e.target.files?.length) {
      await uploadImageFiles(e.target.files);
      e.target.value = "";
    }
  }

  function firstDragValue(dt, type) {
    try {
      return (dt?.getData?.(type) || "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean)[0] || "";
    } catch {
      return "";
    }
  }

  /** Attach an image that is already in our own gallery -- no upload, no copy. */
  function attachGalleryImage(imageId) {
    const clean = String(imageId || "").trim();
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(clean)) return false;
    if (!supportsRef) {
      setError(t("generate.error.refNotSupported", { model: modelInfo.label }));
      return false;
    }
    if (refImages.length >= maxRefImages) {
      setError(t(
        maxRefImages === 1 ? "generate.error.maxReferencesOne" : "generate.error.maxReferencesMany",
        { max: maxRefImages }
      ));
      return false;
    }
    // A bare filename: the backend resolves it against the gallery directory, so this
    // works without knowing where the data directory lives.
    const name = `${clean}.png`;
    setRefImages((previous) => [
      ...previous,
      {
        id: crypto.randomUUID(),
        path: name,
        preview: `${API_BASE}/api/images/${clean}/file`,
        name,
      },
    ]);
    return true;
  }

  // The native shell owns a real NSView drop destination, because WKWebView's HTML5
  // drag-and-drop is unreliable for files. When a drop lands there the shell calls
  // this global with absolute paths; there is no way to reach React state from
  // outside, so it has to be a function we own.
  useEffect(() => {
    const acceptDragOver = (event) => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    };
    const onNativeDrop = (paths) => {
      const list = Array.isArray(paths) ? paths : [paths];
      list.filter(Boolean).forEach((value) => importDroppedPath(value));
    };
    window.__mlxDropPaths = onNativeDrop;
    // A drop is only accepted where dragover is cancelled, and any element between
    // here and the textarea can silently reject it and fall back to inserting text.
    document.addEventListener("dragover", acceptDragOver);
    return () => {
      document.removeEventListener("dragover", acceptDragOver);
      if (window.__mlxDropPaths === onNativeDrop) delete window.__mlxDropPaths;
    };
  });

  async function importDroppedPath(path) {
    // A real local file path from Finder. WKWebView cannot hand the page a File, so
    // this is the channel that case arrives on.
    const trimmed = decodeURIComponent(String(path || "").trim().replace(/^file:\/\//, ""));
    if (!trimmed.startsWith("/")) return false;
    try {
      const res = await api("/api/import-path", { method: "POST", body: JSON.stringify({ path: trimmed }) });
      if (trimmed.toLowerCase().endsWith(".safetensors")) {
        setError("");
        return true;
      }
      if (res?.name) {
        const previewUrl = res.url ? `${API_BASE}${res.url}` : "";
        setRefImages((previous) => {
          if (previous.length >= maxRefImages) {
            setError(t(
              maxRefImages === 1 ? "generate.error.maxReferencesOne" : "generate.error.maxReferencesMany",
              { max: maxRefImages }
            ));
            return previous;
          }
          return [...previous, { id: crypto.randomUUID(), path: res.path, preview: previewUrl, name: res.name }];
        });
        return true;
      }
      return false;
    } catch (err) {
      setError(t("generate.error.importFailed", { path: trimmed, message: err.message || err }));
      return false;
    }
  }

  async function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const dt = e.dataTransfer;
    const files = Array.from(dt?.files || []);
    if (files.length) {
      const loraFile = files.find((f) => f.name.endsWith(".safetensors"));
      if (loraFile) {
        await uploadLoraFile(loraFile);
        return;
      }
      const imgFiles = files.filter(isImageFile);
      if (imgFiles.length > 0) {
        await uploadImageFiles(imgFiles);
      }
      return;
    }

    // No File objects. In a real browser this branch never runs; in WKWebView it is
    // the only way anything arrives, and it arrives as a string. Try, in order:
    //   1. our own image id, set by the gallery's drag handler
    //   2. one of our own /api/images/<id>/file URLs
    //   3. a local filesystem path (Finder)
    //   4. give up with a clear message rather than silently pasting a URL
    const internalId = firstDragValue(dt, "application/x-mlx-image-id");
    if (internalId && attachGalleryImage(internalId)) return;

    const raw = firstDragValue(dt, "text/uri-list") || firstDragValue(dt, "text/plain");
    const decoded = decodeURIComponent(raw.replace(/^file:\/\//, ""));
    const own = decoded.match(/\/api\/images\/([A-Za-z0-9][A-Za-z0-9_-]{0,127})\/file/);
    if (own && attachGalleryImage(own[1])) return;

    if (decoded.startsWith("/") && (await importDroppedPath(decoded))) return;

    if (decoded) {
      setError(
        decoded.startsWith("http")
          ? t("generate.error.dropWebUrl", { url: decoded.slice(0, 80) })
          : t("generate.error.dropNothingUsable"),
      );
    }
  }

  function insertIntoPrompt(text) {
    const el = promptRef.current;
    if (!el) {
      setPrompt((prev) => (prev ? `${prev.trim()} ${text}` : text));
      return;
    }
    const start = el.selectionStart ?? el.value.length;
    const end = el.selectionEnd ?? el.value.length;
    const before = el.value.substring(0, start);
    const after = el.value.substring(end);
    const padBefore = before.length > 0 && !before.endsWith(" ") ? " " : "";
    const padAfter = after.length > 0 && !after.startsWith(" ") ? " " : "";
    const newPrompt = `${before}${padBefore}${text}${padAfter}${after}`;
    setPrompt(newPrompt);
    setTimeout(() => {
      el.focus();
      const newPos = start + padBefore.length + text.length;
      el.setSelectionRange(newPos, newPos);
    }, 0);
  }

  const detectedColors = useMemo(() => {
    const matches = prompt.match(/#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b/g);
    return matches ? [...new Set(matches.map((c) => c.toUpperCase()))] : [];
  }, [prompt]);

  function removeColorFromPrompt(hex) {
    const regex = new RegExp(`\\s*${hex}\\b`, "gi");
    setPrompt((prev) => prev.replace(regex, "").replace(/\s{2,}/g, " ").trim());
  }

  function getContrastColor(hex) {
    const clean = hex.replace("#", "");
    const full = clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean;
    const r = parseInt(full.substring(0, 2), 16) || 0;
    const g = parseInt(full.substring(2, 4), 16) || 0;
    const b = parseInt(full.substring(4, 6), 16) || 0;
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return yiq >= 128 ? "#111" : "#fff";
  }

  function modelLaunchBlocked() {
    if (!modelConfirmationRequired) return false;
    const message = modelInfo.confirmation_message
      || t("generate.confirm.experimentalModel", { model: modelInfo.label });
    return !window.confirm(message);
  }

  async function postGenerate(isQueued = false) {
    if (!model || modelsLoading || modelsError) {
      setError(modelsError || t("generate.error.selectModelFirst"));
      return;
    }
    const paramsStr = currentParams();
    let parsedPrompt = prompt;
    try {
      const parsed = JSON.parse(paramsStr);
      if (parsed.prompt) parsedPrompt = parsed.prompt;
    } catch {}

    const res = await api("/api/generate", {
      method: "POST",
      body: paramsStr,
    });
    const { job_id } = res;
    if (!isQueued) {
      activateJob(job_id, parsedPrompt);
    }
    return job_id;
  }

  async function handleVariation(meta) {
    if (!meta) return;
    if (modelLaunchBlocked()) return;
    const newSeed = getNextSeed(meta.seed ?? seed);
    setSeed(newSeed);
    try {
      const payload = {
        ...JSON.parse(currentParams()),
        prompt: meta.prompt || prompt,
        seed: newSeed,
      };
      setSubmittedParams(JSON.stringify(payload));
      const res = await api("/api/generate", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const { job_id } = res;
      activateJob(job_id, payload.prompt);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    if (busy) {
      if (dirty) setShowSwitchDialog(true);
      return;
    }
    if (modelLaunchBlocked()) return;
    try {
      setSubmittedParams(currentParams());
      await postGenerate(false);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function queueNext() {
    setShowSwitchDialog(false);
    if (modelLaunchBlocked()) return;
    setSubmittedParams(currentParams());
    try {
      await postGenerate(true);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function stopAndSwitch() {
    setShowSwitchDialog(false);
    try {
      await cancelJob();
    } catch {}
    setSubmittedParams(currentParams());
    try {
      await postGenerate(false);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  function switchToLoraModel(baseModel, loraEntry) {
    const targetModel = models.find((m) => getModelBase(m, m.id) === baseModel);
    if (targetModel) {
      switchModel(targetModel.id);
      if (loraEntry?.path) {
        setTimeout(() => {
          setLoras([{ path: loraEntry.path, scale: loraEntry.scale ?? 1.0, name: loraEntry.name, base_model: loraEntry.base_model }]);
        }, 50);
      }
    }
  }

  const busy = jobId !== null;
  const engineLoading =
    busy && ["downloading", "loading_model", "compiling", "preparing"].includes(jobPhase);
  const engineBase = getModelBase(modelInfo, model);
  const compatibleLoras = engineBase && modelInfo.supports_loras
    ? loraRegistry.filter((r) => r.base_model === engineBase)
    : [];
  const dirty =
    busy &&
    submittedParams !== null &&
    currentParams() !== submittedParams;

  return (
    <div className="studio-layout">
      <div className="studio-form-pane">
        <form
          className={`generate-form${dragOver ? " drag-active" : ""}`}
          onSubmit={submit}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <div className="prompt-toolbar">
            <div className="prompt-toolbar-left">
              {activeTriggerWords.length > 0 && (
                <div className="trigger-chips-bar">
                  <span className="trigger-chips-label">{t("generate.triggers.label")}</span>
                  <div className="trigger-chips-list">
                    {activeTriggerWords.map((tw) => {
                      const inPrompt = new RegExp(`(^|\\s|,)${escapeRegExp(tw)}($|\\s|,)`, "i").test(prompt);
                      return (
                        <button
                          key={tw}
                          type="button"
                          className={`trigger-chip${inPrompt ? " active" : ""}`}
                          onClick={() => toggleTriggerWord(tw)}
                          title={t(inPrompt ? "generate.triggers.titleRemove" : "generate.triggers.titleAdd")}
                        >
                          {inPrompt ? `✓ ${tw}` : `+ ${tw}`}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
              {refImages.length > 0 && (
                <div className="ref-quick-chips">
                  <span className="ref-chips-label">{t("generate.refTags.label")}</span>
                  {refImages.map((_, i) => (
                    <button
                      key={i}
                      type="button"
                      className="ref-tag-chip"
                      onClick={() => insertIntoPrompt(`Image ${i + 1}`)}
                      title={t("generate.refTags.titleInsert", { n: i + 1 })}
                    >
                      {t("generate.refTags.add", { n: i + 1 })}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="prompt-toolbar-right">
              <button
                type="button"
                className="color-tool-btn"
                 onClick={handleEnhancePrompt}
                 disabled={enhancing || !prompt.trim() || !enhancerAvailable}
                 title={t(enhancerAvailable
                   ? "generate.enhance.titleAvailable"
                   : "generate.error.enhancerUnavailable")}

              >
                {t(enhancing ? "generate.enhance.btnBusy" : "generate.enhance.btnIdle")}
              </button>
              {enhancing && (
                <button
                  type="button"
                  className="color-tool-btn enhance-cancel-btn"
                  onClick={cancelEnhancePrompt}
                  title={t("generate.enhance.cancelTitle")}
                >
                  {t("generate.enhance.cancelBtn")}
                </button>
              )}
              <button
                type="button"
                className={`color-tool-btn${enhanceJson ? " active" : ""}`}
                onClick={() => setEnhanceJson((prev) => !prev)}
                title={t("generate.enhance.jsonTitle")}
              >
                {t("generate.enhance.jsonBtn")}
              </button>
              <button
                type="button"
                className={`color-tool-btn${showColorPicker ? " active" : ""}`}
                onClick={() => setShowColorPicker((prev) => !prev)}
                title={t("generate.color.title")}
              >
                {t("generate.color.btn", { hex: customHex.slice(1) })}
              </button>
            </div>
          </div>

          {showColorPicker && (
            <div className="color-popover">
              <div className="color-popover-header">
                 <span className="color-popover-title">
                   {t("generate.color.popoverTitle", { model: modelInfo.label })}
                 </span>

                <button
                  type="button"
                  className="btn-mini"
                  onClick={() => setShowColorPicker(false)}
                >
                  ✕
                </button>
              </div>
              <div className="color-swatch-grid">
                {HEX_PALETTE.map((c) => (
                  <button
                    key={c.hex}
                    type="button"
                    className="color-swatch-btn"
                    style={{ backgroundColor: c.hex, color: getContrastColor(c.hex) }}
                    onClick={() => insertIntoPrompt(c.hex)}
                    title={t("generate.color.swatchTitle", { name: t(c.key), hex: c.hex })}
                  >
                    {t(c.key)}
                  </button>
                ))}
              </div>
              <div className="custom-color-row">
                <input
                  type="color"
                  value={customHex}
                  onChange={(e) => setCustomHex(e.target.value.toUpperCase())}
                  title={t("generate.color.pickCustom")}
                />
                <input
                  type="text"
                  value={customHex}
                  onChange={(e) => setCustomHex(e.target.value.toUpperCase())}
                  placeholder={t("generate.color.hexPlaceholder")}
                  maxLength={7}
                  className="hex-input"
                />
                <button
                  type="button"
                  className="btn-mini btn-color-insert"
                  onClick={() => insertIntoPrompt(customHex)}
                >
                  {t("generate.color.insert", { hex: customHex.replace("#", "") })}
                </button>
              </div>
            </div>
          )}

          <textarea
            ref={promptRef}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            // The prompt box is the deepest drop target, and a drop is only accepted
            // if `dragover` is cancelled *there*. Cancelling it on the form alone was
            // not enough: WebKit treated the textarea as an editable and inserted the
            // dragged URL as literal text instead of firing a drop event at all.
            onDragOver={(e) => {
              e.preventDefault();
              if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
            }}
            onDrop={(e) => e.preventDefault()}
            placeholder={t("generate.prompt.placeholder")}
            rows={4}
            required
          />

          {detectedColors.length > 0 && (
            <div className="active-hex-bar">
              <span className="active-hex-label">{t("generate.hexBar.label")}</span>
              <div className="active-hex-list">
                {detectedColors.map((hex) => (
                   <button
                     key={hex}
                     type="button"
                     className="active-hex-badge"
                     style={{ backgroundColor: hex, color: getContrastColor(hex) }}
                     title={t("generate.hexBar.removeTitle", { hex })}
                     aria-label={t("generate.hexBar.removeTitle", { hex })}
                     onClick={() => removeColorFromPrompt(hex)}
                   >
                     {hex} <span className="badge-remove">✕</span>
                   </button>

                ))}
              </div>
            </div>
          )}
          <div className="preset-row">
            <div className="model-select-group">
              <select
                className="model-select"
                value={model}
                onChange={(e) => switchModel(e.target.value)}
              >
                 {modelsLoading && <option value="">{t("generate.model.loading")}</option>}
                 {models.map((m) => (
                   <option key={m.id} value={m.id} className={m.installed ? "" : "model-option-offline"}>
                     {m.installed === false ? "⬇ " : ""}{m.label}
                   </option>
                 ))}

              </select>
                {modelsError && (
                  <p className="error" role="alert">
                    {modelsError}{" "}
                    <button type="button" className="btn-mini" onClick={refreshModels}>
                      {t("app.retry")}
                    </button>
                  </p>
                )}
               {modelInfo?.civitai_version_id && (

                <a
                  href={`https://civitai.red/models/${modelInfo.civitai_model_id || ""}?modelVersionId=${modelInfo.civitai_version_id}&ref_code=88C8VEBA`}
                  target="_blank"
                  rel="noreferrer"
                  className="civitai-badge model-civitai-badge"
                  title={t("generate.civitai.title", {
                    modelId: modelInfo.civitai_model_id || "",
                    versionId: modelInfo.civitai_version_id,
                  })}
                >
                  {t("generate.civitai.badge", { versionId: modelInfo.civitai_version_id })}
                </a>
              )}
            </div>
            {modelInfo?.installed === false && (
              <ModelInstaller
                modelInfo={modelInfo}
                onInstalled={refreshModels}
              />
            )}
          </div>
      <div className="generate-btn-row generate-btn-row-primary">
        <button
          type="submit"
          className="generate-btn"
           disabled={modelsLoading || Boolean(modelsError) || !model || (busy && !dirty) || modelInfo?.installed === false}

          title={
            modelInfo?.installed === false
              ? t("generate.btn.downloadFirstTitle")
              : undefined
          }
        >
          {busy
            ? dirty
              ? t("generate.btn.queueWithNewParams")
              : status === "queued"
                ? t("generate.btn.queued")
                : jobPhase === "downloading"
                  ? t("generate.btn.downloadingModel")
                  : jobPhase === "loading_model"
                    ? t("generate.btn.loadingMemory")
                    : jobPhase === "compiling"
                      ? t("generate.btn.compilingShaders")
                      : jobPhase === "saving"
                        ? t("generate.btn.finalizingImage")
                        : progress && progress.steps > 0
                          ? t("generate.btn.stepProgress", { step: progress.step, steps: progress.steps })
                          : t("generate.btn.generating")
            : t("generate.btn.generate")}
        </button>
        {busy && (
          <button
            type="button"
            className="cancel-action-btn"
            title={t("generate.btn.cancelTitle")}
            onClick={async () => {
              await cancelJob();
            }}
          >
            {t("generate.btn.cancel")}
          </button>
        )}
      </div>

          <SizeSelector
            width={width}
            setWidth={setWidth}
            height={height}
            setHeight={setHeight}
            maxPixels={maxPixels}
            modelInfo={modelInfo}
          />

          <GenerationParams
            modelInfo={modelInfo}
            width={width}
            setWidth={setWidth}
            height={height}
            setHeight={setHeight}
            steps={steps}
            setSteps={setSteps}
            guidance={guidance}
            setGuidance={setGuidance}
            seed={seed}
            setSeed={setSeed}
            batch={batch}
            setBatch={setBatch}
            negativePrompt={negativePrompt}
            setNegativePrompt={setNegativePrompt}
            sampler={sampler}
            setSampler={setSampler}
            cacheInterval={cacheInterval}
            setCacheInterval={setCacheInterval}
             supportsRef={supportsRef}
             supportsMultiRef={supportsMultiRef}
             maxReferenceImages={maxRefImages}
             refImages={refImages}

            refStrength={refStrength}
            setRefStrength={setRefStrength}
            pickRefImages={pickRefImages}
            removeRefImage={removeRefImage}
            insertIntoPrompt={insertIntoPrompt}
            maxPixels={maxPixels}
            setMaxPixels={setMaxPixels}
          />
      {modelInfo.supports_loras && modelInfo.lora_format ? (
        <details
          className={`advanced-settings lora-disclosure${loras.length ? " has-active" : ""}`}
          open={loras.length > 0}
        >
          <summary className="advanced-settings-summary">
            <span className="advanced-settings-chevron" aria-hidden="true" />
            {t("lora.disclosureLabel")}
            {loras.length > 0 && (
              <span className="lora-disclosure-count">{loras.length}</span>
            )}
          </summary>
          <div className="advanced-settings-body lora-disclosure-body">
        <LoraManagerDrawer
          loras={loras}
          setLoras={setLoras}
            loraRegistry={loraRegistry}
            setLoraRegistry={setLoraRegistry}
            onRegistryRefresh={refreshLoraRegistry}
           modelInfo={modelInfo}

          engineBase={engineBase}
          compatibleLoras={compatibleLoras}
          onSwitchToLoraModel={switchToLoraModel}
          onFeedback={(fb) => {
            if (fb?.type === "error") setError(fb.text);
          }}
        >
          <details className="lora-add">
            <summary>{t("generate.lora.summary")}</summary>
            <UniversalDownloader
               engineBase={engineBase}
               onLoraDownloaded={refreshLoraRegistry}

              onSwitchModel={(base) => switchToLoraModel(base)}
            />
            <div className="lora-divider"><span>{t("generate.lora.orRegisterLocal")}</span></div>
            <div className="lora-add-row">
              <input
                placeholder={t("generate.lora.namePlaceholder")}
                value={newLora.name}
                onChange={(e) => setNewLora({ ...newLora, name: e.target.value })}
              />
              <input
                placeholder={t("generate.lora.pathPlaceholder")}
                value={newLora.path}
                onChange={(e) => setNewLora({ ...newLora, path: e.target.value })}
              />
              <button type="button" onClick={saveNewLora} disabled={savingLora}>
                {t(savingLora ? "generate.lora.saveBtnBusy" : "generate.lora.saveBtnIdle")}
              </button>
            </div>
            <div className="lora-add-row">
              <label className="file-label">
                {t("generate.lora.uploadLabel")}
                <input
                  type="file"
                  accept=".safetensors"
                  onChange={(e) => uploadLoraFile(e.target.files[0])}
                  disabled={savingLora}
                />
              </label>
            </div>
            {uploadProgress && <p className="hint">{uploadProgress}</p>}
            <p className="hint">
              {t("generate.lora.formatHint", { format: modelInfo.lora_format })}
            </p>
          </details>
        </LoraManagerDrawer>
          </div>
        </details>
      ) : (
        <p className="hint">{t("generate.lora.unavailableHint", { model: modelInfo.label })}</p>
      )}

      {busy && progress && progress.steps > 0 && (
        <div className="progress-box">
          <p className="hint">
            {progress.batch > 1 && (
              <>{t("generate.progress.batchSaved", {
                index: progress.saved_index ?? Math.floor(progress.step / progress.steps) + 1,
                batch: progress.batch,
              })} </>
            )}
            {t("generate.progress.stepElapsed", { step: progress.step, steps: progress.steps })}{" "}
            {fmt(progress.elapsed)}
            {progress.eta_seconds != null && (
              <>{t("generate.progress.eta", { eta: fmt(progress.eta_seconds) })}</>
            )}
          </p>
        </div>
      )}
      {busy && !progress && (
        <p className="hint">
          {jobPhaseDetail ||
            (jobPhase === "downloading"
              ? t("generate.progress.phaseDownloading")
              : jobPhase === "loading_model"
                ? t("generate.progress.phaseLoadingMemory")
                : jobPhase === "compiling"
                  ? t("generate.progress.phaseCompiling")
                  : t("generate.progress.phasePreparing"))}
        </p>
      )}
      <GenerationStack />
       {error && (
         <p className="error" role="alert">
           {error}
         </p>
       )}
       {enhanceFeedback && (
         <p
           className={`civitai-feedback ${enhanceFeedback.type}`}
           role={enhanceFeedback.type === "error" ? "alert" : "status"}
         >
           {enhanceFeedback.text}
         </p>
       )}

       </form>

      </div>

      <Dialog
        open={showSwitchDialog}
        onClose={() => setShowSwitchDialog(false)}
        bodyClassName="modal-body switch-dialog"
        ariaLabelledBy="switch-dialog-title"
        ariaDescribedBy="switch-dialog-description"
      >
        <h3 id="switch-dialog-title">{t("generate.switchDialog.title")}</h3>
        <p id="switch-dialog-description">
          {t("generate.switchDialog.body")}
        </p>
        <div className="detail-actions">
          <button type="button" onClick={queueNext}>{t("generate.switchDialog.queueBtn")}</button>
          <button type="button" onClick={stopAndSwitch}>{t("generate.switchDialog.switchBtn")}</button>
          <button type="button" onClick={() => setShowSwitchDialog(false)}>
            {t("generate.switchDialog.cancelBtn")}
          </button>
        </div>
      </Dialog>

      <Dialog
        open={showEnhanceWarning}
        onClose={() => setShowEnhanceWarning(false)}
        bodyClassName="modal-body switch-dialog"
        ariaLabelledBy="enhance-warning-title"
        ariaDescribedBy="enhance-warning-description"
      >
        <h3 id="enhance-warning-title">
          {t(engineLoading ? "generate.enhanceWarning.titleLoading" : "generate.enhanceWarning.titleGenerating")}
        </h3>
        <p id="enhance-warning-description">
          {t(engineLoading
            ? "generate.enhanceWarning.leadLoading"
            : "generate.enhanceWarning.leadGenerating")}{" "}
          {t("generate.enhanceWarning.body")}
        </p>
        <div className="detail-actions">
          <button
            type="button"
            onClick={() => {
              setShowEnhanceWarning(false);
              runEnhancePrompt();
            }}
          >
            {t("generate.enhanceWarning.confirmBtn")}
          </button>
          <button type="button" onClick={() => setShowEnhanceWarning(false)}>
            {t("generate.enhanceWarning.cancelBtn")}
          </button>
        </div>
      </Dialog>

      <div className="studio-canvas-pane">
        <ResultCanvas
          currentImage={currentResult}
          onSetCurrentImage={setCurrentResult}
          busy={busy}
          progress={progress}
          phase={jobPhase}
          phaseDetail={jobPhaseDetail}
          batchImages={batchResults}
          generatingPrompt={busy ? (generatingPrompt || prompt) : null}
           canSetReference={supportsRef}
           maxReferenceImages={maxRefImages}
           onSetReferenceImage={(ref) => {
             addRefImage({
               id: crypto.randomUUID(),
               path: ref.path,
               preview: ref.preview,
               name: ref.path.split("/").pop(),
             });
           }}
           onVariation={handleVariation}

        />
      </div>
    </div>
  );
}

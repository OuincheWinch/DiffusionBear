import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, API_BASE, loraCompatibility, pickPath } from "../api";
import Dialog from "./Dialog";
import GenerationStack from "./GenerationStack";
import ResultCanvas from "./ResultCanvas";
import UniversalDownloader from "./UniversalDownloader";
import LoraManagerDrawer from "./LoraManagerDrawer";
import ModelInstaller from "./ModelInstaller";
import PersistentDownloadProgress from "./PersistentDownloadProgress";
import GenerationParams from "./GenerationParams";
import SizeSelector from "./SizeSelector";
import { findLoraEntry, getModelBase, isKreaDistillLora, isMossDistillLora } from "../utils/loraUtils";
import { normalizeRequest, referenceItems, resolveRequestModel } from "../utils/requestUtils";
import { useGenerationJob } from "../hooks/useGenerationJob";
import { useModelConfig } from "../hooks/useModelConfig";
import { useLoraPanel } from "../hooks/useLoraPanel";
import { useSettings } from "../hooks/useSettings";
import { useI18n } from "../i18n/I18nContext";
import { composeBoxesIntoPrompt, parseBoxesFromPromptText, boxColor, extractNounId } from "../utils/spatialBoxes";

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
  // Labels may still say "experimental" for honesty; only an explicit backend
  // flag may interrupt a launch. Qwen-Image 2.1 is now a supported engine and
  // its label carried the word, which made every launch show a modal.
  return Boolean(
    modelInfo?.experimental
      || modelInfo?.requires_confirmation
      || modelInfo?.confirmation_required,
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

const SPATIAL_BOX_PRESETS = [
  { key: "center", label: "Center", coords: "[250, 250, 750, 750]" },
  { key: "left", label: "Left", coords: "[100, 50, 900, 500]" },
  { key: "right", label: "Right", coords: "[100, 500, 900, 950]" },
  { key: "top", label: "Top", coords: "[50, 100, 500, 900]" },
  { key: "bottom", label: "Bottom", coords: "[500, 100, 950, 900]" },
];

function parseBoundingBoxes(text) {
  if (!text || typeof text !== "string") return [];
  const regex = /\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]/g;
  const boxes = [];
  let match;
  while ((match = regex.exec(text)) !== null) {
    const ymin = parseInt(match[1], 10);
    const xmin = parseInt(match[2], 10);
    const ymax = parseInt(match[3], 10);
    const xmax = parseInt(match[4], 10);
    if (
      ymin >= 0 && ymin <= 1000 &&
      xmin >= 0 && xmin <= 1000 &&
      ymax >= 0 && ymax <= 1000 &&
      xmax >= 0 && xmax <= 1000 &&
      ymin <= ymax && xmin <= xmax
    ) {
      boxes.push([ymin, xmin, ymax, xmax]);
    }
  }
  return boxes.slice(0, 10);
}

export default function GenerateForm({ onGenerated, initialParams, onModelChange, onImageSaved, onImageDeleted, pendingReference, onPendingReferenceConsumed, onReferenceLimit, activeTab, onGenerationStatusChange }) {
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
    runningModelId,
    currentResult,
    setCurrentResult,
    batchResults,
    setBatchResults,
    submittedParams,
    setSubmittedParams,
    showSwitchDialog,
    setShowSwitchDialog,
    activateJob,
    cancelJob,
  } = useGenerationJob({ onGenerated, onImageSaved });

  // A cancel that fails means the job is still running. The dialog closes either way,
  // so without this the user was told their generation had stopped when it had not.
  const [switchError, setSwitchError] = useState("");
  const [activeDownloads, setActiveDownloads] = useState([]);

  const {
    models,
    model,
    setModel,
    modelInfo,
    loading: modelsLoading,
    error: modelsError,
    refreshModels,
  } = useModelConfig({ onModelChange });

  const { settings: appSettings, refresh: refreshAppSettings } = useSettings();

  // Short tag for the Enhance button ("1.5B" / "0.5B"): product designations,
  // not translated prose -- same class as model labels from the backend.
  // Unknown or missing values mean qwen15, matching the backend default.
  const enhancerModelTag = appSettings?.enhancer_model === "qwen05" ? "0.5B" : "1.5B";

  // Single source of truth for the app header title: keep it in sync with the
  // model that is actually selected (covers model changes from "Reuse params"
  // in the Browser tab, which updates `model` without going through switchModel).
  useEffect(() => {
    if (modelInfo?.label) onModelChange?.(modelInfo.label, modelInfo.installed !== false);
  }, [modelInfo?.label, modelInfo?.installed, onModelChange]);

  // Broadcast generation status to App header for persistent cross-tab indicator
  useEffect(() => {
    const isGenerating = status === "generating" || status === "queued";
    const runningModel = runningModelId ? models.find((m) => m.id === runningModelId) : null;
    const runningModelLabel = runningModel?.label || runningModelId || modelInfo?.label;
    const runningModelRunnable = runningModel ? runningModel.installed !== false : (modelInfo?.installed !== false);

    onGenerationStatusChange?.({
      isGenerating,
      status,
      phase: jobPhase,
      progress,
      runningModelId,
      runningModelLabel,
      runningModelRunnable,
    });
  }, [status, jobPhase, progress, runningModelId, models, modelInfo, onGenerationStatusChange]);

  // Settings (including the enhancer backend choice) can change in the
  // Parameters tab while this form stays mounted. Refresh on return so the
  // Enhance button tag never shows a stale model.
  useEffect(() => {
    if (activeTab === "generate") refreshAppSettings?.();
  }, [activeTab, refreshAppSettings]);
  // A pending reference arriving from the browser card is appended through the
  // same addRefImage every other entry point uses, so model support, duplicate
  // filtering and the over-limit dialog behave identically everywhere.
  useEffect(() => {
    if (pendingReference == null) return;
    addRefImage({
      id: crypto.randomUUID(),
      path: pendingReference.path,
      preview: pendingReference.preview,
      name: String(pendingReference.path || "").split("/").pop(),
    });
    onPendingReferenceConsumed?.();
    // addRefImage is a per-render closure; depending on it would refire.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingReference]);

  // Names of selected LoRAs the current model cannot accept, with the reason.
  const [loraIncompatible, setLoraIncompatible] = useState({});
  const loraCheckRef = useRef(0);

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

  const handleBrowseLora = async () => {
    try {
      const res = await pickPath({
        kind: "file",
        title: t("generate.lora.pathPlaceholder"),
        extensions: ["safetensors", "bin"],
      });
      if (!res.cancelled && res.path) {
        const filename = res.path.split("/").pop() || "";
        const guessedName = filename.replace(/\.(safetensors|bin)$/i, "");
        setNewLora((prev) => ({
          ...prev,
          path: res.path,
          name: prev.name || guessedName,
        }));
      }
    } catch (e) {
      console.error(e);
    }
  };

  const [prompt, setPrompt] = useState("");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(4);
  const [guidance, setGuidance] = useState(1.0);
  const [seed, setSeed] = useState("");
  const [batch, setBatch] = useState(1);
  const [negativePrompt, setNegativePrompt] = useState("");
  const [sampler, setSampler] = useState("euler_trailing");
  const [mossDistill, setMossDistill] = useState("auto");
  const [cacheInterval, setCacheInterval] = useState(1);
  const [clipSkip, setClipSkip] = useState(2);
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
  const [preEnhancePrompt, setPreEnhancePrompt] = useState(null);
  const [showEnhanceWarning, setShowEnhanceWarning] = useState(false);
  /* Item waiting on the over-limit dialog: Yes replaces the last reference,
     No drops this item. Null means no dialog. */
  const [refLimitItem, setRefLimitItem] = useState(null);
  const enhanceAbortRef = useRef(null);
  const enhanceRequestRef = useRef(0);
  const hydrationKeyRef = useRef(null);
  const pendingRequestRef = useRef(null);
  const initialModelDefaultsAppliedRef = useRef(false);
  const [spatialBoxes, setSpatialBoxes] = useState([]);
  const [activeBoxId, setActiveBoxId] = useState(null);

  const isMossLayoutModel = () =>
    Boolean(modelInfo && getModelBase(modelInfo, model) === "moss");

  const handleCommitSpatialBoxes = useCallback((nextBoxes) => {
    setSpatialBoxes(nextBoxes);
    // MOSS composes its layout on the backend from spatial_boxes; prompt
    // mutation here would only bury the user's text under re-parsed tokens.
    if (modelInfo?.supports_bounding_box && getModelBase(modelInfo, model) !== "moss") {
      setPrompt((prevPrompt) => composeBoxesIntoPrompt(prevPrompt, nextBoxes, refImages.length > 0));
    }
  }, [model, modelInfo?.supports_bounding_box, refImages.length]);

  const handleClearSpatialBoxes = useCallback(() => {
    setSpatialBoxes([]);
    setActiveBoxId(null);
    if (modelInfo?.supports_bounding_box && getModelBase(modelInfo, model) !== "moss") {
      setPrompt((prevPrompt) => composeBoxesIntoPrompt(prevPrompt, [], refImages.length > 0));
    }
  }, [model, modelInfo?.supports_bounding_box, refImages.length]);

  function handleAddSpatialBox(bbox) {
    if (spatialBoxes.length >= 10) return;
    const newIdx = spatialBoxes.length + 1;
    const newUid = `box_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const newId = `object_${newIdx}`;
    const newBox = {
      uid: newUid,
      id: newId,
      bbox: bbox || [250, 250, 750, 750],
      desc: "",
      kind: "object",
      action: refImages.length > 0 ? "add" : "object",
      lang: "En",
    };
    const nextBoxes = [...spatialBoxes, newBox];
    setSpatialBoxes(nextBoxes);
    setActiveBoxId(newUid);
    if (modelInfo?.supports_bounding_box && getModelBase(modelInfo, model) !== "moss") {
      setPrompt((prevPrompt) => composeBoxesIntoPrompt(prevPrompt, nextBoxes, refImages.length > 0));
    }
  }

  function handleComposeBoxesIntoPrompt() {
    if (!spatialBoxes.length) return;
    setPrompt((prevPrompt) => composeBoxesIntoPrompt(prevPrompt, spatialBoxes, refImages.length > 0));
  }

  const handlePromptChange = useCallback((e) => {
    const nextVal = e.target.value;
    setPrompt(nextVal);
    if (!modelInfo?.supports_bounding_box) return;
    // MOSS: boxes live only in state (backend builds the layout JSON), so a
    // prose edit must never parse tokens back out of the text nor wipe the
    // drawn boxes. This is what made "moving a box wipes the text" and the
    // reuse-then-click path drop every box.
    if (isMossLayoutModel()) return;

    if (!nextVal.trim()) {
      if (spatialBoxes.length > 0) {
        setSpatialBoxes([]);
        setActiveBoxId(null);
      }
      return;
    }

    const parsed = parseBoxesFromPromptText(nextVal, { width, height });
    if (parsed && parsed.length > 0) {
      const isDifferent =
        parsed.length !== spatialBoxes.length ||
        parsed.some((pb, idx) => {
          const sb = spatialBoxes[idx];
          if (!sb) return true;
          return (
            pb.id !== sb.id ||
            pb.desc !== sb.desc ||
            pb.kind !== sb.kind ||
            pb.action !== sb.action ||
            pb.bbox[0] !== sb.bbox[0] ||
            pb.bbox[1] !== sb.bbox[1] ||
            pb.bbox[2] !== sb.bbox[2] ||
            pb.bbox[3] !== sb.bbox[3]
          );
        });

      if (isDifferent) {
        const merged = parsed.map((pb, idx) => {
          const match = spatialBoxes.find((b) => b.id === pb.id) || spatialBoxes[idx];
          return {
            ...pb,
            uid: match?.uid || pb.uid,
          };
        });
        setSpatialBoxes(merged);
        if (!merged.some((b) => b.uid === activeBoxId || b.id === activeBoxId)) {
          setActiveBoxId(merged[0]?.uid || merged[0]?.id || null);
        }
      }
    } else if (spatialBoxes.length > 0 && !nextVal.includes("[") && prompt.includes("[")) {
      setSpatialBoxes([]);
      setActiveBoxId(null);
    }
  }, [model, modelInfo?.supports_bounding_box, prompt, spatialBoxes, activeBoxId, width, height]);

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

    // A MOSS sidecar prompt is the final layout JSON the engine consumed,
    // not prose. Reusing those params must put the caption back in the
    // textarea, not the JSON blob: the boxes ride in spatial_boxes and the
    // backend rebuilds the layout from this caption plus those boxes.
    // Otherwise the whole old layout enters compose as the new caption (the
    // "reuse then click then generate ignores the boxes" report).
    const hydratedBase = getModelBase(targetInfo, matched?.id || requestedModel || model);
    if (hydratedBase === "moss" && source.prompt && source.prompt.trim().startsWith("{")) {
      let caption = null;
      try {
        caption = JSON.parse(source.prompt.trim())?.summary?.caption ?? null;
      } catch {
        /* partial JSON while typing: keep the raw text */
      }
      setPrompt(caption || source.prompt);
    } else {
      setPrompt(source.prompt);
    }
    setWidth(Number(source.width) || 1024);
    setHeight(Number(source.height) || 1024);
    setSteps(Number(source.steps) || targetInfo?.default_steps || 4);
    setGuidance(source.guidance == null ? targetInfo?.default_guidance ?? 1 : Number(source.guidance));
    setSeed(source.seed ?? "");
    setBatch(Math.max(1, Number(source.batch) || 1));
    setNegativePrompt(source.negative_prompt || "");
    setSampler(nextSampler);
    const validMossModes = ["auto", "stock", "dmd4", "dmd8"];
    setMossDistill(validMossModes.includes(source.moss_distill) ? source.moss_distill : "auto");
    setCacheInterval(Math.max(1, Number(source.cache_interval) || 1));
    setClipSkip(Math.max(1, Math.min(12, Number(source.clip_skip ?? source.clipSkip ?? targetInfo?.default_clip_skip ?? 2))));
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
    if (Array.isArray(source.spatial_boxes)) {
      const sanitized = source.spatial_boxes.map((b, idx) => ({
        ...b,
        id: b.id
          || extractNounId(b.desc, b.kind === "text" ? "text" : "object", idx + 1, b.lang || "En")
          || `object_${idx + 1}`,
        uid: b.uid || `box_${Date.now()}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
      }));
      setSpatialBoxes(sanitized);
      if (sanitized.length > 0) setActiveBoxId(sanitized[0].uid || sanitized[0].id);
    } else if (Array.isArray(source.bounding_boxes)) {
      const restored = source.bounding_boxes.map((b, idx) => ({
        uid: `box_${Date.now()}_${idx}_${Math.random().toString(36).slice(2, 6)}`,
        id: `object_${idx + 1}`,
        bbox: b,
        desc: "",
        kind: "object",
        action: "object",
        lang: "En",
      }));
      setSpatialBoxes(restored);
      if (restored.length > 0) setActiveBoxId(restored[0].uid || restored[0].id);
    } else {
      // No stored boxes: recover them from the prompt itself so "Reuse params"
      // redraws the studio canvas for older generations whose sidecar predates
      // spatial_boxes (BFL JSON, MOSS layout JSON, or legacy bracket pairs).
      const parsedBoxes = targetInfo?.supports_bounding_box
        ? parseBoxesFromPromptText(source.prompt || "", { width: source.width, height: source.height })
        : null;
      if (parsedBoxes && parsedBoxes.length > 0) {
        const parsed = parsedBoxes.map((b) => ({
          ...b,
          uid: b.uid || `box_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        }));
        setSpatialBoxes(parsed);
        setActiveBoxId(parsed[0].uid || parsed[0].id);
      } else {
        setSpatialBoxes([]);
        setActiveBoxId(null);
      }
    }
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
    setClipSkip(Number(modelInfo.default_clip_skip ?? 2));
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

  // Synchronize when model defaults or global settings are saved in Parameters tab
  useEffect(() => {
    const handleDefaultsUpdate = () => {
      if (!modelInfo?.id || pendingRequestRef.current) return;
      const customOverrides = appSettings?.model_defaults?.[modelInfo.id];
      const targetW = Number(customOverrides?.width ?? modelInfo.default_width);
      const targetH = Number(customOverrides?.height ?? modelInfo.default_height);
      if (targetW && targetH) {
        setWidth(targetW);
        setHeight(targetH);
      }
      if (customOverrides?.steps != null) setSteps(Number(customOverrides.steps));
      if (customOverrides?.guidance != null) setGuidance(Number(customOverrides.guidance));
      if (customOverrides?.cache_interval != null) setCacheInterval(Number(customOverrides.cache_interval));
      if (customOverrides?.sampler) setSampler(customOverrides.sampler);
    };
    window.addEventListener("mlx:settings-changed", handleDefaultsUpdate);
    window.addEventListener("mlx:models-changed", handleDefaultsUpdate);
    return () => {
      window.removeEventListener("mlx:settings-changed", handleDefaultsUpdate);
      window.removeEventListener("mlx:models-changed", handleDefaultsUpdate);
    };
  }, [modelInfo?.id, modelInfo?.default_width, modelInfo?.default_height, appSettings]);

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
    onModelChange?.(nextModel.label, nextModel.installed !== false);
    const firstPreset = nextModel.presets?.[0];
    const nextSteps = Number(nextModel.default_steps ?? firstPreset?.steps ?? 4);
    setSteps(nextSteps);
    setMaxPixels(nextModel.max_pixels ?? null);
    // Saved per-model defaults must win when the user has not overridden the size.
    // "Untouched" means the current size still equals the previous model's default:
    // comparing against it is what lets a user keep a deliberately chosen custom
    // size across a model switch while a fresh pick of the new model gets its own
    // saved width/height. The guard used to test `!width || !height`, but both
    // states start at 1024, so the saved defaults were never applied.
    const defW = Number(nextModel.default_width) || Number(firstPreset?.width) || 1024;
    const defH = Number(nextModel.default_height) || Number(firstPreset?.height) || 1024;
    const prevModel = models.find((item) => item.id === model);
    const prevPreset = prevModel?.presets?.[0];
    const prevDefW = Number(prevModel?.default_width) || Number(prevPreset?.width);
    const prevDefH = Number(prevModel?.default_height) || Number(prevPreset?.height);
    const hasCustomDims = Boolean(nextModel.has_custom_dimensions || appSettings?.model_defaults?.[id]?.width);
    const dimsUntouched =
      hasCustomDims || !width || !height || (Number(width) === prevDefW && Number(height) === prevDefH);
    if (dimsUntouched) {
      setWidth(defW);
      setHeight(defH);
    } else if (nextModel.max_pixels && width * height > nextModel.max_pixels) {
      const scale = Math.sqrt(nextModel.max_pixels / (width * height));
      setWidth(Math.max(256, Math.round((width * scale) / 16) * 16));
      setHeight(Math.max(256, Math.round((height * scale) / 16) * 16));
    }
    if (nextModel.default_guidance != null) setGuidance(nextModel.default_guidance);
    if (nextModel.default_cache_interval != null) setCacheInterval(nextModel.default_cache_interval);
    if (nextModel.default_clip_skip != null) setClipSkip(nextModel.default_clip_skip);
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
    if (!nextModel.supports_ref && !nextModel.supports_multi_reference) {
      setRefImages((current) => {
        current.forEach((img) => {
          if (img?.preview?.startsWith("blob:")) URL.revokeObjectURL(img.preview);
        });
        return [];
      });
    } else if (!nextModel.supports_multi_reference) {
      setRefImages((current) => {
        current.slice(1).forEach((img) => {
          if (img?.preview?.startsWith("blob:")) URL.revokeObjectURL(img.preview);
        });
        return current.slice(0, 1);
      });
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

  // A MOSS step-distillation adapter (8-step DMD/dataspace) only reproduces its
  // reference output on its own CFG-free schedule at 8 steps. When one is active
  // rewrite steps/guidance so the form, the progress readout and the engine agree.
  useEffect(() => {
    if (getModelBase(modelInfo, model) !== "moss") return;
    if (!loras.some(isMossDistillLora)) return;
    if (steps !== 8) setSteps(8);
    if (guidance !== 1) setGuidance(1);
  }, [guidance, loras, model, modelInfo, steps]);
  // Forced MOSS distillation: the segmented control mirrors what the engine
  // will actually run (a DMD adapter samples CFG-free on its own code count),
  // and resets to Auto whenever the model is not MOSS-SFT.
  useEffect(() => {
    if (!modelInfo || getModelBase(modelInfo, model) !== "moss" || modelInfo.id !== "moss-image-sft") {
      if (mossDistill !== "auto") setMossDistill("auto");
    }
  }, [model, modelInfo, mossDistill]);
  // No step/guidance snapping: an explicit mode wins whatever the slider says.
  // The engine clamps the run to the adapter's own code count (moss_engine
  // min(steps, codes)), Stock keeps the requested trajectory untouched.

  // A guidance-distilled model clamps the step count in the engine (Juggernaut
  // XL Lightning to 12); mirror it here so the slider, the progress readout and
  // the run agree, including for a hydrated request or a persisted
  // model_defaults.steps that asked for more.
  const stepsCap = Number(modelInfo?.max_steps) > 0 ? Number(modelInfo.max_steps) : null;
  useEffect(() => {
    if (stepsCap && Number(steps) > stepsCap) setSteps(stepsCap);
  }, [steps, stepsCap, setSteps]);

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
  // The browser card's reference tooltip needs the current model's limit, which
  // lives here with the model state. Reported up so App can hand it down. This
  // effect MUST sit below the definition above: its dependency array evaluates
  // during render, and placing it earlier throws a TDZ ReferenceError that
  // takes down the whole tab behind the recovery boundary.
  useEffect(() => {
    onReferenceLimit?.(maxRefImages);
  }, [maxRefImages, onReferenceLimit]);
  const enhancerAvailable = modelInfo.supports_prompt_enhancer !== false;
  const modelConfirmationRequired = modelNeedsConfirmation(modelInfo);
  const currentParams = () => {
    const rawBoxes = modelInfo?.supports_bounding_box ? parseBoundingBoxes(prompt) : [];
    const activeSpatialBoxes = modelInfo?.supports_bounding_box && spatialBoxes.length > 0 ? spatialBoxes : undefined;
    const effectiveBoxes = activeSpatialBoxes ? activeSpatialBoxes.map((b) => b.bbox) : (rawBoxes.length > 0 ? rawBoxes : undefined);
    return JSON.stringify({
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
      moss_distill: getModelBase(modelInfo, model) === "moss" && modelInfo.id === "moss-image-sft" ? mossDistill : "auto",
      cache_interval: modelInfo.engine === "sdxl" ? Number(cacheInterval) : 1,
      clip_skip: modelInfo.engine === "sdxl" ? Number(clipSkip) : undefined,
      loras: (modelInfo.supports_loras ? generationLoraPayload([...loras].sort((x, y) => x.path.localeCompare(y.path))) : []).slice(0, 16),
      reference_images: supportsRef ? refImages.map((img) => img.path) : [],
      reference_strength: supportsRef && refImages.length > 0 && !supportsMultiRef ? Number(refStrength) : undefined,
      output_format: outputFormat,
      stealth: stealthMode,
      fast_vae: modelInfo.supports_fast_vae ? fastVae : false,
      max_pixels: maxPixels && maxPixels < (modelInfo.max_pixels ?? Infinity) ? Number(maxPixels) : null,
      bounding_boxes: effectiveBoxes,
      spatial_boxes: activeSpatialBoxes,
    });
  };

  async function runEnhancePrompt() {
    if (!prompt.trim() || enhancing) return;
    // Snapshot the pre-enhancement prompt. The enhancer overwrites the box in place, so
    // once it has run there is no way back to what the user typed -- which is why the
    // output has to be good before it lands. Kept so it can always be put back.
    const originalPrompt = prompt;
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
      } else if (res?.rejected) {
        // The backend refused its own output -- it returned its instructions, or JSON
        // it could not parse. Falling through to the generic "no prompt" error would
        // hide the reason; shipping `res.enhanced` would be empty anyway. The prompt
        // box is deliberately left as the user wrote it.
        setEnhanceFeedback({
          type: "error",
          text: t("generate.enhanceFeedback.rejected", {
            message: res.message || res.reason || "",
          }),
        });
      } else if (res?.enhanced) {
        setPreEnhancePrompt(originalPrompt);
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

  function revertEnhancedPrompt() {
    if (preEnhancePrompt == null) return;
    setPrompt(preEnhancePrompt);
    setPreEnhancePrompt(null);
    setEnhanceFeedback({ type: "info", text: t("generate.enhanceFeedback.reverted") });
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
    if (refImages.some((image) => image.path === item.path)) return;
    if (refImages.length >= maxRefImages) {
      // Full: ask instead of erroring. Yes replaces the last slot with the
      // new image, No leaves everything as it was. Both buttons (canvas and
      // browser card) arrive here, so the choice is identical everywhere.
      setRefLimitItem(item);
      return;
    }
    setRefImages((previous) => {
      if (previous.length >= maxRefImages) return previous;
      if (previous.some((image) => image.path === item.path)) return previous;
      return [...previous, item];
    });
  }

  function confirmRefLimitReplace() {
    if (refLimitItem == null) return;
    setRefImages((previous) => {
      const evicted = previous[previous.length - 1];
      if (evicted?.preview && evicted.preview.startsWith("blob:")) {
        URL.revokeObjectURL(evicted.preview);
      }
      if (previous.some((image) => image.path === refLimitItem.path)) return previous;
      return [...previous.slice(0, Math.max(0, maxRefImages - 1)), refLimitItem];
    });
    setRefLimitItem(null);
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

  // A LoRA that does not fit the model is refused here rather than discovered by the
  // loader ninety seconds in, with a message about "44 of the 144 layers". Setting an
  // error and returning false is the same shape as modelLaunchBlocked, so every caller
  // gets the refusal for free instead of only the form submit checking.
  function loraLaunchBlocked() {
    const names = Object.keys(loraIncompatible);
    if (!names.length) return false;
    setError(t("generate.error.loraIncompatible", {
      count: names.length,
      names: names.join(", "),
    }));
    return true;
  }

  // Re-checked whenever the selection or the model changes, because the verdict depends
  // on BOTH: the same adapter is fine on one model and fatal on another. The request id
  // is what keeps a slow answer for an old selection from overwriting a newer one.
  useEffect(() => {
    if (!modelInfo?.supports_loras || !loras.length) {
      setLoraIncompatible({});
      return;
    }
    const requestId = ++loraCheckRef.current;
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(
        loras.map(async (entry) => {
          try {
            const verdict = await loraCompatibility(entry.name, model);
            return [entry.name, verdict?.compatible === false ? verdict.message : null];
          } catch {
            // An unreachable check must not block a generation that may well be fine;
            // the loader remains the authority and will say so if it is not.
            return [entry.name, null];
          }
        }),
      );
      if (cancelled || requestId !== loraCheckRef.current) return;
      const bad = {};
      for (const [name, message] of entries) {
        if (message) bad[name] = message;
      }
      setLoraIncompatible(bad);
    })();
    return () => { cancelled = true; };
  }, [loras, model, modelInfo?.supports_loras]);

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
    } catch {
      // Deliberate, not a swallowed failure: the params blob is optional, and text
      // that is not JSON is a prompt, which is exactly what we fall back to.
      parsedPrompt = prompt;
    }

    const res = await api("/api/generate", {
      method: "POST",
      body: paramsStr,
    });
    const { job_id } = res;
    if (!isQueued) {
      const displayPrompt = parsedPrompt.trim()
        || (spatialBoxes.length > 0 ? spatialBoxes.map((b) => b.desc || b.id).join(", ") : prompt);
      activateJob(job_id, displayPrompt, model);
    }
    return job_id;
  }

  async function handleVariation(meta) {
    if (!meta) return;
    if (loraLaunchBlocked()) return;
    if (modelLaunchBlocked()) return;
    const newSeed = getNextSeed(meta.seed ?? seed);
    setSeed(newSeed);
    try {
      const variationPrompt = meta.prompt || prompt;
      const varBoxes = modelInfo?.supports_bounding_box ? parseBoundingBoxes(variationPrompt) : [];
      const payload = {
        ...JSON.parse(currentParams()),
        prompt: variationPrompt,
        seed: newSeed,
        bounding_boxes: varBoxes.length > 0 ? varBoxes : undefined,
      };
      setSubmittedParams(JSON.stringify(payload));
      const res = await api("/api/generate", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      const { job_id } = res;
      activateJob(job_id, payload.prompt, payload.model || model);
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
    if (loraLaunchBlocked()) return;
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
    if (loraLaunchBlocked()) return;
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
    } catch (err) {
      // The dialog is closing either way, but the user was told their job is stopping.
      // If it is not, say so rather than switching silently to an engine that is still
      // saturating the GPU.
      setSwitchError(err?.message || String(err));
      console.error("[DiffusionBear] cancel before switch failed:", err);
    }
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
          {/* Errors live at the TOP of the form, not the bottom.

              They used to render as a quiet 0.85rem paragraph after <GenerationStack />,
              i.e. below the prompt, the canvas controls and the advanced panels. By the
              time a generation failed you had to scroll to the end of a long form to find
              out why -- and a raw MLX message like "[METAL] Command buffer execution
              failed: Insufficient Memory (00000008:kiOGPU...)" reads as a stray log line
              rather than as the reason nothing happened.

              Sticky, because the failure almost always arrives after a long wait during
              which the user has scrolled away from the top of the form. */}
          {error && (
            <div className="error-banner" role="alert" aria-live="assertive">
              <div className="error-banner-head">
                <svg className="error-banner-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <strong className="error-banner-title">{t("generate.error.title")}</strong>
                <button
                  type="button"
                  className="error-banner-dismiss"
                  onClick={() => setError(null)}
                  title={t("app.dismiss")}
                  aria-label={t("app.dismiss")}
                >
                  ✕
                </button>
              </div>
              <p className="error-banner-body">{error}</p>
            </div>
          )}

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
              {modelInfo?.supports_bounding_box && (
                <div className="spatial-box-bar ref-quick-chips">
                  <span className="spatial-box-label ref-chips-label">{t("generate.spatialBox.label") || "Spatial Box"}</span>
                  <button
                    type="button"
                    className="spatial-box-chip ref-tag-chip"
                    onClick={() => handleAddSpatialBox([250, 250, 750, 750])}
                    title={t("generate.spatialBox.titleAdd") || "Center [250, 250, 750, 750]"}
                    aria-label="[+] Box"
                  >
                    [+] Box
                  </button>
                  {SPATIAL_BOX_PRESETS.map((p) => (
                    <button
                      key={p.key}
                      type="button"
                      className="spatial-box-chip ref-tag-chip"
                      onClick={() => {
                        try {
                          const coords = JSON.parse(p.coords);
                          handleAddSpatialBox(coords);
                        } catch {
                          insertIntoPrompt(`${p.coords} `);
                        }
                      }}
                      title={`${t(`generate.spatialBox.presets.${p.key}`) || p.label} ${p.coords}`}
                      aria-label={`${t(`generate.spatialBox.presets.${p.key}`) || p.label} ${p.coords}`}
                    >
                      {t(`generate.spatialBox.presets.${p.key}`) || p.label} {p.coords}
                    </button>
                  ))}
                  {spatialBoxes.length > 0 && (
                    <>
                      <button
                        type="button"
                        className="spatial-box-chip ref-tag-chip compose-boxes-btn"
                        onClick={handleComposeBoxesIntoPrompt}
                        title={t("generate.spatialBox.composeTitle") || "Compose Studio Canvas boxes into prompt text"}
                        aria-label={t("generate.spatialBox.composeTitle") || "Compose Studio Canvas boxes into prompt text"}
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                        {t("generate.spatialBox.composeBtn") || "Compose"}
                      </button>
                      <button
                        type="button"
                        className="spatial-box-chip ref-tag-chip clear-boxes-btn explicit"
                        onClick={handleClearSpatialBoxes}
                        title={t("generate.spatialBox.clearAll") || "Clear all spatial boxes"}
                        aria-label={t("generate.spatialBox.clearAll") || "Clear all spatial boxes"}
                      >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" style={{ verticalAlign: "-1px", marginRight: "4px" }}><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        {t("generate.spatialBox.clearAllBtn", { count: spatialBoxes.length })}
                      </button>
                    </>
                  )}
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
                {enhancing ? (
                  <svg className="spin-icon" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" strokeDasharray="32" strokeDashoffset="12"/>
                  </svg>
                ) : (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/>
                  </svg>
                )}
                {t(enhancing ? "generate.enhance.btnBusy" : "generate.enhance.btnIdle", { model: enhancerModelTag })}
              </button>
              {enhancing && (
                <button
                  type="button"
                  className="color-tool-btn enhance-cancel-btn"
                  onClick={cancelEnhancePrompt}
                  title={t("generate.enhance.cancelTitle")}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
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
                <span
                  className="color-swatch-disc"
                  style={{ backgroundColor: customHex }}
                  aria-hidden="true"
                />
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
                  aria-label={t("app.close") || "Close"}
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
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
            onChange={handlePromptChange}
            // The prompt box is the deepest drop target, and a drop is only accepted
            // if `dragover` is cancelled *there*. Cancelling it on the form alone was
            // not enough: WebKit treated the textarea as an editable and inserted the
            // dragged URL as literal text instead of firing a drop event at all.
            onDragOver={(e) => {
              e.preventDefault();
              if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
            }}
            onDrop={(e) => e.preventDefault()}
            placeholder={
              spatialBoxes.length > 0
                ? (t("generate.prompt.placeholderSpatialBoxes") || "Optional: Global prompt (Studio Canvas boxes will be auto-composed)...")
                : t("generate.prompt.placeholder")
            }
            // 12 rows: the prompt is the primary input of the form and the
            // spatial layout with auto-composed boxes needs the height.
            rows={12}
            required={spatialBoxes.length === 0}
          />

          {/* Box ↔ prompt association without a second text layer: one colored
              chip per box (same palette as its canvas outline), click focuses
              the box in the studio canvas. */}
          {spatialBoxes.length > 0 && (
            <div className="prompt-box-legend" role="list" aria-label={t("generate.prompt.boxLegend") || "Box colors"}>
              {spatialBoxes.map((b, i) => (
                <button
                  key={b.uid || b.id || i}
                  type="button"
                  role="listitem"
                  className="prompt-box-chip"
                  style={{ "--hl-color": boxColor(i) }}
                  onClick={() => setActiveBoxId(b.uid || b.id)}
                  title={b.desc || ""}
                >
                  <span className="prompt-box-chip-swatch" aria-hidden="true" />
                  #{i + 1} {b.id}
                </button>
              ))}
            </div>
          )}

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
                value={model || ""}
                onChange={(e) => switchModel(e.target.value)}
              >
                 {modelsLoading && <option value="">{t("generate.model.loading")}</option>}
                 {models.map((m) => {
                   const statusText = m.installed === false
                     ? (t("models.detected.notInstalled") || "non installé")
                     : (t("app.ready") || "prêt");
                   return (
                     <option
                       key={m.id}
                       value={m.id}
                       className={m.installed ? "model-option-ready" : "model-option-offline"}
                     >
                       {m.label} ({statusText})
                     </option>
                   );
                 })}

              </select>
              <span className={`model-status-pill ${modelInfo?.installed === false ? "uninstalled" : "ready"}`}>
                <span className="status-dot" aria-hidden="true" />
                {modelInfo?.installed === false
                  ? (t("app.weightsRequired") || "Weights required")
                  : (t("app.ready") || "Ready")}
              </span>
              {(modelInfo?.has_custom_dimensions || Boolean(appSettings?.model_defaults?.[modelInfo?.id]?.width || appSettings?.model_defaults?.[modelInfo?.id]?.steps)) && (
                <span className="model-custom-pill" title={t("settings.defaults.customBadge") || "Custom"}>
                  {t("settings.defaults.customBadge") || "Custom"}
                </span>
              )}
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
            {modelInfo?.installed === false && !activeDownloads.some((t) => t.model_id === modelInfo.id) && (
              <ModelInstaller
                modelInfo={modelInfo}
                onInstalled={refreshModels}
              />
            )}
          </div>
          <PersistentDownloadProgress
            models={models}
            onInstalled={refreshModels}
            onActiveTasksChange={setActiveDownloads}
          />
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
            mossDistill={mossDistill}
            setMossDistill={setMossDistill}
            cacheInterval={cacheInterval}
            setCacheInterval={setCacheInterval}
            clipSkip={clipSkip}
            setClipSkip={setClipSkip}
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
            {Object.keys(loraIncompatible).length > 0 && (
              <div className="lora-compat-warning" role="alert">
                <strong className="lora-compat-title">
                  {t(Object.keys(loraIncompatible).length === 1
                    ? "generate.error.loraIncompatibleTitleOne"
                    : "generate.error.loraIncompatibleTitleMany", {
                    count: Object.keys(loraIncompatible).length,
                  })}
                </strong>
                <ul className="lora-compat-list">
                  {Object.entries(loraIncompatible).map(([name, message]) => (
                    <li key={name}>
                      <span className="lora-compat-name">{name}</span>
                      <span className="lora-compat-detail">{message}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
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
              <div className="path-picker-row lora-path-picker">
                <input
                  placeholder={t("generate.lora.pathPlaceholder")}
                  value={newLora.path}
                  onChange={(e) => setNewLora({ ...newLora, path: e.target.value })}
                />
                <button
                  type="button"
                  className="btn-mini path-picker-btn"
                  onClick={handleBrowseLora}
                  title={t("app.browse")}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                  {t("app.browse")}
                </button>
              </div>
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
       {enhanceFeedback && (
         <p
           className={`civitai-feedback ${enhanceFeedback.type}`}
           role={enhanceFeedback.type === "error" ? "alert" : "status"}
         >
           {enhanceFeedback.text}
         </p>
       )}
       {preEnhancePrompt != null && preEnhancePrompt !== prompt && (
         <div className="enhance-revert">
           <span className="enhance-revert-label">
             {t("generate.enhanceFeedback.originalLabel")}
           </span>
           <span className="enhance-revert-original">{preEnhancePrompt}</span>
           <button
             type="button"
             className="action-btn"
             onClick={revertEnhancedPrompt}
           >
             {t("generate.enhanceFeedback.revert")}
           </button>
         </div>
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
        {switchError && (
          <p className="error" role="alert">
            {switchError}
          </p>
        )}
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
        open={refLimitItem != null}
        onClose={() => setRefLimitItem(null)}
        bodyClassName="modal-body switch-dialog"
        ariaLabelledBy="ref-limit-title"
        ariaDescribedBy="ref-limit-description"
      >
        <h3 id="ref-limit-title">{t("generate.refLimit.title", { max: maxRefImages })}</h3>
        <p id="ref-limit-description">
          {t("generate.refLimit.body", { max: maxRefImages })}
        </p>
        <div className="detail-actions">
          <button type="button" onClick={confirmRefLimitReplace}>{t("generate.refLimit.yesBtn")}</button>
          <button type="button" onClick={() => setRefLimitItem(null)}>
            {t("generate.refLimit.noBtn")}
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
          generatingPrompt={busy
            ? (generatingPrompt
              || (prompt.length > 160 ? `${prompt.slice(0, 160)}…` : prompt))
            : null}
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
           onReuse={hydrateRequest}
           onVariation={handleVariation}
           onDeleteImage={(deletedId) => {
             setBatchResults?.((prev) => (Array.isArray(prev) ? prev.filter((img) => img.id !== deletedId) : []));
             onImageDeleted?.(deletedId);
           }}
           supportsBoundingBox={Boolean(modelInfo?.supports_bounding_box)}
           spatialBoxes={spatialBoxes}
           onSetSpatialBoxes={setSpatialBoxes}
           activeBoxId={activeBoxId}
           onSetActiveBoxId={setActiveBoxId}
           onCommitBoxes={handleCommitSpatialBoxes}
           referenceImageUrl={refImages?.[0]?.preview || (refImages?.[0]?.path ? `${API_BASE}/api/images/${refImages[0].path}/file` : null)}
           onUploadRefFiles={uploadImageFiles}
           onClearReference={() => removeRefImage(0)}
           aspectRatioWidth={Number(width) || 768}
           aspectRatioHeight={Number(height) || 512}
        />
      </div>
    </div>
  );
}

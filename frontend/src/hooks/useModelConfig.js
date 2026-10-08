import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";

const EMPTY_MODEL_INFO = {
  id: "",
  label: "Loading models…",
  default_steps: 4,
  default_guidance: 1,
  supports_guidance: false,
  supports_negative: false,
  supports_loras: false,
  supports_ref: false,
  supports_multi_reference: false,
  max_reference_images: 1,
  supports_fast_vae: false,
  lora_format: "",
  samplers: [],
  presets: [],
};

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function normalizeModel(raw) {
  const capabilities = {
    ...(raw.capabilities || {}),
    ...(raw.effective_capabilities || {}),
  };
  const supportsLoras = Boolean(firstDefined(
    capabilities.supports_loras,
    raw.supports_loras,
    Boolean(raw.lora_format),
  ));
  const supportsMultiReference = Boolean(firstDefined(
    capabilities.supports_multi_reference,
    raw.supports_multi_reference,
    false,
  ));
  const supportsReference = Boolean(firstDefined(
    capabilities.supports_ref,
    raw.supports_ref,
    supportsMultiReference,
  ));
  const maxReferenceImages = Math.max(
    1,
    Number(firstDefined(
      capabilities.max_reference_images,
      raw.max_reference_images,
      1,
    )) || 1,
  );
  const presets = Array.isArray(firstDefined(capabilities.presets, raw.presets)) ? firstDefined(capabilities.presets, raw.presets) : [];

  return {
    ...raw,
    ...capabilities,
    supports_loras: supportsLoras,
    supports_multi_reference: supportsMultiReference,
    supports_ref: supportsReference,
    max_reference_images: maxReferenceImages,
    presets,
    samplers: Array.isArray(firstDefined(capabilities.samplers, raw.samplers)) ? firstDefined(capabilities.samplers, raw.samplers) : [],
  };
}

export function useModelConfig({ onModelChange }) {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const modelRef = useRef(null);
  const onModelChangeRef = useRef(onModelChange);

  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  useEffect(() => {
    onModelChangeRef.current = onModelChange;
  }, [onModelChange]);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const response = await api("/api/models");
      const list = Array.isArray(response) ? response : response?.models || response?.items || [];
      const normalized = list.filter(Boolean).map(normalizeModel);
      const responseDefault = Array.isArray(response) ? null : response?.default_model || response?.default_model_id;
      setModels(normalized);
      setError(null);

      let selected = null;
      setModel((previous) => {
        if (previous && normalized.some((item) => item.id === previous)) {
          selected = normalized.find((item) => item.id === previous);
          return previous;
        }
        selected = normalized.find((item) => item.id === responseDefault)
          || normalized.find((item) => item.default || item.is_default)
          || normalized.find((item) => item.installed !== false)
          || normalized[0]
          || null;
        return selected?.id || null;
      });
      modelRef.current = selected?.id || null;
      if (selected?.label) onModelChangeRef.current?.(selected.label);
      return normalized;
    } catch (err) {
      setError(err.message || String(err));
      return [];
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const modelInfo = useMemo(
    () => models.find((item) => item.id === model) || EMPTY_MODEL_INFO,
    [model, models],
  );

  return {
    models,
    model,
    setModel,
    modelInfo,
    loading,
    error,
    refreshModels: loadModels,
  };
}

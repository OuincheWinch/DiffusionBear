import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { findLoraEntry } from "../utils/loraUtils";
import { useI18n } from "../i18n/I18nContext";

export function useLoraPanel({ onError }) {
  // A hook is used inside a hook, so the translator comes from the provider rather
  // than the module-level fallback, which would ignore a live language change.
  const { t } = useI18n();
  const [loras, setLoras] = useState([]);
  const [loraRegistry, setLoraRegistry] = useState([]);
  const [newLora, setNewLora] = useState({ name: "", path: "" });
  const [savingLora, setSavingLora] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(null);
  const registryRequestRef = useRef(0);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const refreshLoraRegistry = useCallback(async () => {
    const requestId = ++registryRequestRef.current;
    try {
      const list = await api("/api/loras");
      if (requestId !== registryRequestRef.current) return null;
      const normalized = Array.isArray(list) ? list : [];
      setLoraRegistry(normalized);
      return normalized;
    } catch (err) {
      if (requestId === registryRequestRef.current) onErrorRef.current?.(err.message || String(err));
      return null;
    }
  }, []);

  useEffect(() => {
    refreshLoraRegistry();
    return () => {
      registryRequestRef.current += 1;
    };
  }, [refreshLoraRegistry]);

  const activeTriggerWords = useMemo(() => {
    const words = new Set();
    for (const lora of loras) {
      const entry = findLoraEntry(lora, loraRegistry);
      if (entry?.triggers) {
        for (const trigger of entry.triggers) {
          if (typeof trigger === "string" && trigger.trim()) words.add(trigger.trim());
        }
      }
    }
    return Array.from(words);
  }, [loras, loraRegistry]);

  async function uploadLoraFile(file) {
    if (!file) return;
    setSavingLora(true);
    setUploadProgress(t("generate.lora.uploading"));
    try {
      const form = new FormData();
      form.append("file", file);
      const name = newLora.name.trim() || file.name.replace(/\.safetensors$/, "");
      const entry = await api(`/api/loras/upload?name=${encodeURIComponent(name)}`, {
        method: "POST",
        body: form,
      });
      setLoraRegistry((previous) => [
        ...previous.filter((item) => item.name !== entry.name && item.path !== entry.path),
        entry,
      ]);
      await refreshLoraRegistry();
      setNewLora({ name: "", path: "" });
      setUploadProgress(t("generate.lora.registered", { name: entry.name }));
    } catch (err) {
      onErrorRef.current?.(`LoRA upload failed: ${err.message || err}`);
      setUploadProgress(null);
    } finally {
      setSavingLora(false);
    }
  }

  async function saveNewLora() {
    if (!newLora.name.trim() || !newLora.path.trim()) return;
    setSavingLora(true);
    try {
      const entry = await api("/api/loras", {
        method: "POST",
        body: JSON.stringify({
          name: newLora.name.trim(),
          path: newLora.path.trim(),
        }),
      });
      setLoraRegistry((previous) => [
        ...previous.filter((item) => item.name !== entry.name && item.path !== entry.path),
        entry,
      ]);
      await refreshLoraRegistry();
      setNewLora({ name: "", path: "" });
    } catch (err) {
      onErrorRef.current?.(err.message || String(err));
    } finally {
      setSavingLora(false);
    }
  }

  return {
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
  };
}

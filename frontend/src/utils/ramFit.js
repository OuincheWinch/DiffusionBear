import { useEffect, useState } from "react";
import { api } from "../api";

let memoryRequest = null;

function fetchMachineMemoryGb() {
  if (!memoryRequest) {
    memoryRequest = api("/api/engine/status")
      .then((status) => Number(status?.meta?.memory_gb) || null)
      .catch(() => {
        memoryRequest = null;
        return null;
      });
  }
  return memoryRequest;
}

export function useMachineMemoryGb() {
  const [ramGb, setRamGb] = useState(null);
  useEffect(() => {
    let alive = true;
    fetchMachineMemoryGb().then((gb) => {
      if (alive) setRamGb(gb);
    });
    return () => {
      alive = false;
    };
  }, []);
  return ramGb;
}

// Peak memory a model needs: the measured resident figure when the manifest has
// one, otherwise its weights on disk.
export function modelMemoryNeedGb(model) {
  const resident = Number(model?.resident_gb);
  if (resident > 0) return resident;
  const size = Number(model?.size_gb);
  return size > 0 ? size : null;
}

// "good" leaves room for macOS and other apps, "tight" runs but may swap,
// "over" exceeds unified memory.
export function ramFit(model, ramGb) {
  const need = modelMemoryNeedGb(model);
  if (!need || !ramGb) return null;
  if (need <= ramGb * 0.7) return "good";
  if (need <= ramGb) return "tight";
  return "over";
}

const REFERENCE_PIXELS = 512 * 768;

export function defaultGenerationSeconds(model) {
  const perStep = Number(model?.per_step_seconds);
  const steps = Number(model?.default_steps);
  if (!(perStep > 0) || !(steps > 0)) return null;
  const pixels = (Number(model.default_width) || 512) * (Number(model.default_height) || 768);
  return Math.round(perStep * steps * Math.max(0.25, pixels / REFERENCE_PIXELS));
}

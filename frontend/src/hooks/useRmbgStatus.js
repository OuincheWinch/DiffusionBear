import { useEffect, useState } from "react";
import { api } from "../api";

const _listeners = new Set();
let _sharedStatus = null;
let _pollTimer = null;

function _emit(status) {
  _sharedStatus = status;
  for (const fn of _listeners) {
    fn(status);
  }
}

async function _pollOnce() {
  try {
    const res = await api("/api/images/remove-background/status");
    _emit(res);
    if (!res?.running && _pollTimer !== null) {
      clearInterval(_pollTimer);
      _pollTimer = null;
    }
  } catch {
    // Ignore transient network errors while subprocess starts
  }
}

export function notifyRmbgStarted() {
  _emit({
    running: true,
    phase: "preparing",
    detail: "Starting Bria RMBG-3.0...",
    step: 0,
    steps: 4,
    elapsed: 0,
  });
  if (_pollTimer === null) {
    _pollTimer = setInterval(_pollOnce, 800);
  }
}

export function notifyRmbgFinished() {
  if (_pollTimer !== null) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
  _emit({ running: false, phase: null, detail: null, step: 0, steps: 4, elapsed: 0 });
}

export async function stopRmbg() {
  try {
    await api("/api/images/remove-background/cancel", { method: "POST" });
  } finally {
    notifyRmbgFinished();
  }
}

export function useRmbgStatus() {
  const [status, setStatus] = useState(() => _sharedStatus);

  useEffect(() => {
    _listeners.add(setStatus);
    if (_sharedStatus === null) {
      _pollOnce();
    }
    return () => {
      _listeners.delete(setStatus);
    };
  }, []);

  return status;
}

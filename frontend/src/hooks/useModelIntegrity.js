import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

// Single shared snapshot + poller across App, GenerateForm, and ModelsSection
// so startup or user-triggered verifications sync everywhere with 1× polling.
let _sharedIntegrity = null;
let _wasRunning = false;
let _pollTimer = null;
const _subscribers = new Set();

function _notifySubscribers(snapshot, completedNow) {
  for (const fn of _subscribers) {
    fn(snapshot, completedNow);
  }
}

function _publishSnapshot(data) {
  const completedNow = Boolean(_wasRunning && !data?.running);
  _wasRunning = Boolean(data?.running);
  _sharedIntegrity = data;
  _notifySubscribers(data, completedNow);
}

async function _pollOnce() {
  try {
    const data = await api("/api/models/integrity");
    _publishSnapshot(data);
    if (data?.running && _subscribers.size > 0) {
      if (_pollTimer) clearTimeout(_pollTimer);
      _pollTimer = setTimeout(_pollOnce, 1000);
    } else {
      _pollTimer = null;
    }
    return data;
  } catch {
    _pollTimer = null;
    return null;
  }
}

function _ensurePolling(snapshot) {
  if (snapshot?.running && !_pollTimer && _subscribers.size > 0) {
    _pollTimer = setTimeout(_pollOnce, 1000);
  }
}

export function useModelIntegrity({ onVerificationComplete } = {}) {
  const [integrity, setIntegrity] = useState(_sharedIntegrity);
  const [repairing, setRepairing] = useState(null);
  const [actionError, setActionError] = useState(null);
  const onCompleteRef = useRef(onVerificationComplete);
  onCompleteRef.current = onVerificationComplete;

  useEffect(() => {
    const listener = (snapshot, completedNow) => {
      setIntegrity(snapshot);
      if (completedNow) {
        onCompleteRef.current?.(snapshot);
      }
    };
    _subscribers.add(listener);
    if (_sharedIntegrity !== null) {
      setIntegrity(_sharedIntegrity);
    } else {
      _pollOnce();
    }
    _ensurePolling(_sharedIntegrity);

    return () => {
      _subscribers.delete(listener);
      if (_subscribers.size === 0 && _pollTimer) {
        clearTimeout(_pollTimer);
        _pollTimer = null;
      }
    };
  }, []);

  const fetchIntegrity = useCallback(() => _pollOnce(), []);

  const verify = useCallback(async (modelId = null) => {
    setActionError(null);
    try {
      const res = await api("/api/models/integrity/verify", {
        method: "POST",
        body: JSON.stringify(modelId ? { model_id: modelId } : {}),
      });
      if (res?.snapshot) {
        _publishSnapshot(res.snapshot);
        _ensurePolling(res.snapshot);
      }
      return res;
    } catch (e) {
      setActionError(e.message || String(e));
      return null;
    }
  }, []);

  const repairModel = useCallback(async (modelId, mode = "partial") => {
    if (!modelId) return null;
    setActionError(null);
    setRepairing({ modelId, mode });
    try {
      const res = await api(`/api/models/${encodeURIComponent(modelId)}/repair`, {
        method: "POST",
        body: JSON.stringify({ mode, license_accepted: true }),
      });
      await _pollOnce();
      if (res?.status === "already_ok") {
        onCompleteRef.current?.(_sharedIntegrity);
      }
      return res;
    } catch (e) {
      setActionError(e.message || String(e));
      return null;
    } finally {
      setRepairing(null);
    }
  }, []);

  return {
    integrity,
    repairing,
    actionError,
    refresh: fetchIntegrity,
    verifyAll: () => verify(null),
    verifyOne: (modelId) => verify(modelId),
    repairModel,
  };
}

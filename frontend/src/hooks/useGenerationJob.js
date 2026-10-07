import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";

const ACTIVE_STATUSES = new Set(["generating", "queued"]);
const POLL_DELAY_MS = 500;
const IDLE_DELAY_MS = 1500;
// Backoff used while the browser window is in the background. A generation can
// run for minutes, and a 500ms poll is ~800 requests per run competing for the
// same CPU that MLX needs. Polling never stops, so a job started from another
// tab is still discovered; returning to the window triggers an immediate poll.
const HIDDEN_POLL_DELAY_MS = 5000;
const HIDDEN_IDLE_DELAY_MS = 10000;

function isWindowHidden() {
  return typeof document !== "undefined" && document.hidden;
}

function isAbortError(error) {
  return error?.name === "AbortError";
}

function findNextJob(list, currentId) {
  return (Array.isArray(list) ? list : []).find(
    (job) => job.id !== currentId && ACTIVE_STATUSES.has(job.status),
  );
}

export function useGenerationJob({ onGenerated, onImageSaved }) {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [jobPhase, setJobPhase] = useState(null);
  const [jobPhaseDetail, setJobPhaseDetail] = useState(null);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [generatingPrompt, setGeneratingPrompt] = useState(null);
  const [currentResult, setCurrentResult] = useState(null);
  const [batchResults, setBatchResults] = useState([]);
  const [submittedParams, setSubmittedParams] = useState(null);
  const [showSwitchDialog, setShowSwitchDialog] = useState(false);

  const lastSavedRef = useRef(null);
  const lastAutoShownIdRef = useRef(null);
  const savedIdsRef = useRef(new Set());
  const jobIdRef = useRef(null);
  const sequenceRef = useRef(0);
  const pollAbortRef = useRef(null);
  const onGeneratedRef = useRef(onGenerated);
  const onImageSavedRef = useRef(onImageSaved);
  const errorSourceRef = useRef(null);

  const setActionError = useCallback((value) => {
    errorSourceRef.current = "action";
    setError(value);
  }, []);

  const setPollingError = useCallback((value) => {
    errorSourceRef.current = "poll";
    setError(value);
  }, []);

  const clearPollingError = useCallback(() => {
    if (errorSourceRef.current !== "poll") return;
    errorSourceRef.current = null;
    setError(null);
  }, []);

  useEffect(() => {
    onGeneratedRef.current = onGenerated;
    onImageSavedRef.current = onImageSaved;
  }, [onGenerated, onImageSaved]);

  function invalidatePolling() {
    sequenceRef.current += 1;
    pollAbortRef.current?.abort();
    pollAbortRef.current = null;
  }

  function clearTracking() {
    invalidatePolling();
    jobIdRef.current = null;
    setJobId(null);
    setStatus(null);
    setJobPhase(null);
    setJobPhaseDetail(null);
    setProgress(null);
    setGeneratingPrompt(null);
  }

  function trackJob(job, { resetBatch = false } = {}) {
    if (!job?.id) return;
    const changed = jobIdRef.current !== job.id;
    if (changed) {
      invalidatePolling();
      jobIdRef.current = job.id;
      lastSavedRef.current = null;
      lastAutoShownIdRef.current = null;
      savedIdsRef.current = new Set();
      if (resetBatch) setBatchResults(Array.isArray(job.partial_results) ? job.partial_results : []);
    }
    setJobId(job.id);
    setStatus(job.status || "queued");
    setJobPhase(job.phase || null);
    setJobPhaseDetail(job.phase_detail || null);
    setProgress(job.progress || null);
    setGeneratingPrompt(job.request?.prompt ?? job.prompt ?? null);
    if (Array.isArray(job.partial_results) && (!changed || resetBatch)) {
      setBatchResults(job.partial_results);
    }
  }

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    Promise.allSettled([
      api("/api/gallery?limit=1", { signal: controller.signal }),
      api("/api/jobs?limit=10", { signal: controller.signal }),
    ]).then(([galleryResult, jobsResult]) => {
      if (!active) return;
      if (galleryResult.status === "fulfilled") {
        const first = galleryResult.value?.items?.[0];
        if (first) setCurrentResult((previous) => previous ?? first);
      }
      if (jobsResult.status === "fulfilled") {
        const activeJob = (jobsResult.value || []).find((job) => ACTIVE_STATUSES.has(job.status));
        if (activeJob) trackJob(activeJob, { resetBatch: true });
      }
    });
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  async function completeJob(job, sequence, signal) {
    if (sequence !== sequenceRef.current || jobIdRef.current !== job.id) return;
    const finalResults = Array.isArray(job.results)
      ? job.results
      : Array.isArray(job.partial_results)
        ? job.partial_results
        : [];

    if (finalResults.length > 0) {
      setBatchResults(finalResults);
      for (const result of finalResults) {
        if (result?.id && !savedIdsRef.current.has(result.id)) {
          savedIdsRef.current.add(result.id);
          lastSavedRef.current = result.id;
          onImageSavedRef.current?.(result.id, result.seed, finalResults.length);
        }
      }
      const last = finalResults[finalResults.length - 1];
      if (last) {
        lastAutoShownIdRef.current = last.id;
        setCurrentResult(last);
        onGeneratedRef.current?.(last);
      }
    } else if (job.result) {
      if (!savedIdsRef.current.has(job.result.id)) {
        savedIdsRef.current.add(job.result.id);
        lastSavedRef.current = job.result.id;
        onImageSavedRef.current?.(job.result.id, job.result.seed, 1);
      }
      lastAutoShownIdRef.current = job.result.id;
      setCurrentResult(job.result);
      onGeneratedRef.current?.(job.result);
      setBatchResults([job.result]);
    } else if (job.status === "done") {
       setActionError(job.error || "Generation completed without producing an image.");
    }

    let nextJob = null;
    try {
      const activeList = await api("/api/jobs?limit=10", { signal });
      if (sequence !== sequenceRef.current || jobIdRef.current !== job.id) return;
      nextJob = findNextJob(activeList, job.id);
    } catch (err) {
      if (!isAbortError(err) && sequence === sequenceRef.current) {
         setActionError(err.message || String(err));
      }
    }

    if (nextJob) {
       setActionError(null);
       trackJob(nextJob, { resetBatch: true });
      return;
    }
    clearTracking();
  }

  useEffect(() => {
    if (!jobId) return undefined;
    const sequence = sequenceRef.current;
    const controller = new AbortController();
    let timer = null;
    let stopped = false;
    pollAbortRef.current?.abort();
    pollAbortRef.current = controller;

    const isCurrent = () => !stopped && sequence === sequenceRef.current && jobIdRef.current === jobId;
    const schedule = (delay = POLL_DELAY_MS) => {
      if (!isCurrent()) return;
      const base = delay === POLL_DELAY_MS && isWindowHidden() ? HIDDEN_POLL_DELAY_MS : delay;
      timer = window.setTimeout(poll, base);
    };

    async function poll() {
      if (!isCurrent()) return;
      try {
        const job = await api(`/api/jobs/${jobId}`, { signal: controller.signal });
        if (!isCurrent() || !job) return;
        setStatus(job.status);
        setJobPhase(job.phase || null);
        setJobPhaseDetail(job.phase_detail || null);
         clearPollingError();
         const actualPrompt = job.request?.prompt ?? job.prompt;
        if (actualPrompt) setGeneratingPrompt(actualPrompt);
        if (job.progress) {
          setProgress(job.progress);
          const savedId = job.progress.saved_id;
          if (savedId && !savedIdsRef.current.has(savedId)) {
            savedIdsRef.current.add(savedId);
            lastSavedRef.current = savedId;
            onImageSavedRef.current?.(savedId, job.progress.saved_index, job.progress.batch);
          }
        }
        if (Array.isArray(job.partial_results)) setBatchResults(job.partial_results);

        if (job.last_result?.id && job.last_result.id !== lastAutoShownIdRef.current) {
          lastAutoShownIdRef.current = job.last_result.id;
          setCurrentResult(job.last_result);
          onGeneratedRef.current?.(job.last_result);
        } else if (
          job.progress?.saved_id
          && job.progress.saved_id !== lastAutoShownIdRef.current
          && !job.last_result
        ) {
          const savedId = job.progress.saved_id;
          lastAutoShownIdRef.current = savedId;
          api(`/api/images/${savedId}`, { signal: controller.signal })
            .then((image) => {
              if (!isCurrent() || !image) return;
              setCurrentResult(image);
              onGeneratedRef.current?.(image);
              setBatchResults((previous) => {
                if (previous.some((item) => item.id === image.id)) return previous;
                return [...previous, image];
              });
            })
            .catch((err) => {
               if (!isAbortError(err) && isCurrent()) setPollingError(err.message || String(err));
            });
        }

        if (["done", "error", "cancelled"].includes(job.status)) {
          if (job.status !== "done") {
             setActionError(job.error || (job.status === "cancelled" ? "Generation cancelled" : "Generation failed"));
            const partials = Array.isArray(job.partial_results) ? job.partial_results : [];
            if (partials.length > 0) {
              setBatchResults(partials);
              const last = partials[partials.length - 1];
              if (last?.id && last.id !== lastAutoShownIdRef.current) {
                lastAutoShownIdRef.current = last.id;
                setCurrentResult(last);
                onGeneratedRef.current?.(last);
              }
            }
          }
          await completeJob(job, sequence, controller.signal);
          return;
        }
        schedule();
      } catch (err) {
        if (isAbortError(err) || !isCurrent()) return;
        if (err?.status === 404 || err?.status === 410) {
           setActionError("The generation job is no longer available.");
          clearTracking();
          return;
        }
          setPollingError(err.message || String(err));
         schedule(1000);
      }
    }

    schedule(0);
    function handleVisibility() {
      if (!isWindowHidden() && isCurrent() && timer != null) {
        window.clearTimeout(timer);
        schedule(0);
      }
    }
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", handleVisibility);
      if (timer != null) window.clearTimeout(timer);
      if (pollAbortRef.current === controller) pollAbortRef.current = null;
      controller.abort();
    };
  }, [jobId]);

  useEffect(() => {
    function handleCancelAll() {
      clearTracking();
    }

    function handleCancelJob(event) {
      if (event.detail?.id && event.detail.id === jobIdRef.current) clearTracking();
    }

    window.addEventListener("mlx:cancel-all", handleCancelAll);
    window.addEventListener("mlx:cancel-job", handleCancelJob);
    return () => {
      window.removeEventListener("mlx:cancel-all", handleCancelAll);
      window.removeEventListener("mlx:cancel-job", handleCancelJob);
    };
  }, []);

  useEffect(() => {
    if (jobId) return undefined;
    const controller = new AbortController();
    let timer = null;
    let stopped = false;
    const poll = async () => {
      if (stopped) return;
      try {
        const list = await api("/api/jobs?limit=10", { signal: controller.signal });
        if (stopped) return;
        const activeJob = (Array.isArray(list) ? list : []).find((job) => ACTIVE_STATUSES.has(job.status));
        if (activeJob) {
          trackJob(activeJob, { resetBatch: true });
          return;
        }
      } catch (err) {
         if (!isAbortError(err) && !stopped) setPollingError(err.message || String(err));
      }
      if (!stopped) timer = window.setTimeout(poll, isWindowHidden() ? HIDDEN_IDLE_DELAY_MS : IDLE_DELAY_MS);
    };
    timer = window.setTimeout(poll, 0);
    function handleVisibility() {
      if (!stopped && !isWindowHidden() && timer != null) {
        window.clearTimeout(timer);
        timer = window.setTimeout(poll, 0);
      }
    }
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", handleVisibility);
      if (timer != null) window.clearTimeout(timer);
      controller.abort();
    };
  }, [jobId]);

  function activateJob(newJobId, promptText) {
    if (!newJobId) return;
    invalidatePolling();
    jobIdRef.current = newJobId;
    lastSavedRef.current = null;
    lastAutoShownIdRef.current = null;
    savedIdsRef.current = new Set();
    setBatchResults([]);
    setStatus("queued");
    setJobPhase("loading_model");
    setJobPhaseDetail("Queued · Waiting for engine...");
    setJobId(newJobId);
    setGeneratingPrompt(promptText || null);
     setProgress(null);
     setActionError(null);
   }

   async function cancelJob() {
    const currentId = jobIdRef.current;
    clearTracking();
    if (!currentId) return;
    try {
      await api(`/api/jobs/${currentId}/cancel`, { method: "POST" });
    } catch {}
  }

  return {
    jobId,
    status,
    jobPhase,
    jobPhaseDetail,
    progress,
    error,
     setError: setActionError,
     generatingPrompt,
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
    lastSavedRef,
    lastAutoShownIdRef,
  };
}

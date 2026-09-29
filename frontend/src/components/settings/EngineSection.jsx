import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";

function fmtSeconds(s) {
  if (s == null || isNaN(s)) return "—";
  if (s <= 0) return "now";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

function gauge(label, value, title) {
  return (
    <span className="engine-chip" title={title}>
      <em>{label}</em> {value}
    </span>
  );
}

function normalizeNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function TuneInput({ label, unit, value, onChange, step, max = 128, title }) {
  return (
    <label className="engine-tune" title={title}>
      <span>{label}</span>
      <div className="engine-tune-input">
        <input
          type="number"
          min="0"
          max={max}
          step={step ?? "any"}
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <small>{unit}</small>
      </div>
    </label>
  );
}

export default function EngineSection() {
  const [status, setStatus] = useState(null);
  const [err, setErr] = useState(null);
  const [showStderr, setShowStderr] = useState(false);
  const [draft, setDraft] = useState(() => ({
    wiredGb: "",
    kreaGb: "",
    mfluxIdle: "",
    sdxlIdle: "",
    qwenIdle: "",
  }));
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState(null);

  const syncDraftFromStatus = (s) => {
    setDraft({
      wiredGb: String(s.wired?.generic_limit_gb ?? ""),
      kreaGb: String(s.wired?.krea_limit_gb ?? ""),
      mfluxIdle: String(s.mflux?.idle_kill_s ?? ""),
      sdxlIdle: String(s.sdxl?.idle_kill_s ?? ""),
      qwenIdle: String(s.qwen?.idle_kill_s ?? ""),
    });
  };

  useEffect(() => {
    let alive = true;
    let refreshTimer = null;
    async function poll() {
      try {
        const s = await api("/api/engine/status");
        if (alive) {
          setStatus(s);
          setErr(null);
          if (!dirty) syncDraftFromStatus(s);
        }
      } catch (e) {
        if (alive) setErr(e.message || String(e));
      }
      if (!alive) return;
      // The storage counters are TTL-cached server-side; polling this panel
      // every 5s is only worth it while the window is actually being watched.
      const hidden = typeof document !== "undefined" && document.hidden;
      refreshTimer = setTimeout(poll, hidden ? 30000 : 5000);
    }
    function handleVisibility() {
      if (alive && typeof document !== "undefined" && !document.hidden) {
        clearTimeout(refreshTimer);
        refreshTimer = setTimeout(poll, 0);
      }
    }
    refreshTimer = setTimeout(poll, 0);
    document.addEventListener("visibilitychange", handleVisibility);
    return () => {
      alive = false;
      document.removeEventListener("visibilitychange", handleVisibility);
      clearTimeout(refreshTimer);
    };
  }, [dirty]);

  async function saveConfig(e) {
    e.preventDefault();
    const payload = {};
    const wiredGb = normalizeNum(draft.wiredGb);
    const kreaGb = normalizeNum(draft.kreaGb);
    const mfluxIdle = normalizeNum(draft.mfluxIdle);
    const sdxlIdle = normalizeNum(draft.sdxlIdle);
    const qwenIdle = normalizeNum(draft.qwenIdle);
    if (wiredGb == null || kreaGb == null || mfluxIdle == null || sdxlIdle == null || qwenIdle == null) {
      setErr("All tuning values must be numbers (0 disables).");
      return;
    }
    if (wiredGb < 0 || kreaGb < 0 || mfluxIdle < 0 || sdxlIdle < 0 || qwenIdle < 0) {
      setErr("Values must be ≥ 0.");
      return;
    }
    payload.memory_wired_limit_gb = wiredGb;
    payload.memory_krea_wired_limit_gb = kreaGb;
    payload.idle_kill_s_mflux = Math.round(mfluxIdle);
    payload.idle_kill_s_sdxl = Math.round(sdxlIdle);
    payload.idle_kill_s_qwen = Math.round(qwenIdle);
    setSaving(true);
    setErr(null);
    try {
      const s = await api("/api/engine/config", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setStatus(s);
      setDirty(false);
      setSavedAt(Date.now());
      setTimeout(() => setSavedAt(null), 2600);
    } catch (errMsg) {
      setErr(errMsg.message || String(errMsg));
    } finally {
      setSaving(false);
    }
  }

  if (err) {
    return (
      <div className="settings-row">
        <span className="error">{err}</span>
      </div>
    );
  }
  if (!status) {
    return <div className="settings-row"><span className="hint">Loading engine status…</span></div>;
  }

  const metal = status.metal || {};
  const wired = status.wired || {};
  const mflux = status.mflux || {};
  const sdxl = status.sdxl || {};
  const storage = status.storage || {};

  return (
    <div className="settings-row">
      <div className="engine-grid">
        <div className="engine-card">
          <header>
            <strong>🖥 Metal / GPU</strong>
          </header>
          <div className="engine-chips">
            {metal.name && gauge("GPU", metal.name)}
            {gauge("Memory", metal.memory_size ? formatBytes(metal.memory_size) : "—")}
            {metal.recommended_max_working_set_size
              ? gauge("Max working set", formatBytes(metal.recommended_max_working_set_size))
              : null}
          </div>
          <div className="engine-chips">
            {gauge("Peak memory", metal.peak_memory != null ? formatBytes(metal.peak_memory) : "—", "Highest wired memory reached this session")}
            {gauge("Cache", metal.cache_memory != null ? formatBytes(metal.cache_memory) : "—", "mlx allocator cache")}
          </div>
          <div className="engine-chips">
            {gauge("Wired limit", wired.generic_budget_bytes ? formatBytes(wired.generic_budget_bytes) : "unbounded", `MLX_WIRED_LIMIT_GB=${wired.generic_limit_gb || 0}`)}
            {gauge("krea2 wired", wired.krea_budget_bytes ? formatBytes(wired.krea_budget_bytes) : "unbounded", `MLX_KREA_WIRED_LIMIT_GB=${wired.krea_limit_gb || 0}`)}
          </div>
        </div>

        <div className="engine-card">
          <header>
            <strong>⚡ mflux pipelines</strong>
            <span className={`engine-dot ${mflux.resident ? "on" : "off"}`} />
          </header>
          <p className="engine-resident">
            {mflux.resident
              ? <>
                  <strong>{mflux.model || "model"}</strong> resident in unified memory
                  {mflux.watchdog_armed && mflux.seconds_until_release != null && (
                    <span className="engine-watchdog">
                      auto-release in {fmtSeconds(mflux.seconds_until_release)}
                    </span>
                  )}
                </>
              : "No pipeline resident (cold reload on next generation)"}
          </p>
          <div className="engine-chips">
            {gauge("Idle policy", `${mflux.idle_kill_s ?? 300}s`)}
            {gauge("Prompt cache", String(mflux.prompt_cache_size ?? 0))}
            {gauge("Watchdog", mflux.watchdog_armed ? "armed" : "cold")}
          </div>
        </div>

        <div className="engine-card">
          <header>
            <strong>🧵 SDXL daemon</strong>
            <span className={`engine-dot ${sdxl.resident ? "on" : "off"}`} />
          </header>
          <p className="engine-resident">
            {sdxl.resident
              ? <>
                  <strong>{sdxl.model || "engine"}</strong> alive
                  {sdxl.watchdog_armed && sdxl.seconds_until_release != null && (
                    <span className="engine-watchdog">
                      auto-release in {fmtSeconds(sdxl.seconds_until_release)}
                    </span>
                  )}
                </>
              : "Daemon not running (spawned lazily)"}
          </p>
          <div className="engine-chips">
            {gauge("Idle policy", `${sdxl.idle_kill_s ?? 300}s`)}
            {gauge("Watchdog", sdxl.watchdog_armed ? "armed" : "cold")}
          </div>
          {sdxl.stderr_tail && (
            <>
              <button
                type="button"
                className="btn-mini"
                onClick={() => setShowStderr((v) => !v)}
              >
                {showStderr ? "Hide daemon log" : "Daemon log tail"}
              </button>
              {showStderr && <pre className="engine-stderr">{sdxl.stderr_tail}</pre>}
            </>
          )}
        </div>

        <div className="engine-card">
          <header>
            <strong>⚙️ Runtime tuning</strong>
          </header>
          <p className="engine-tune-note">
            Saved values take effect on the next generation (or the next idle
            rearm) without restarting anything.
          </p>
          <form className="engine-tune-grid" onSubmit={saveConfig}>
            <TuneInput
              label="Wired limit"
              unit="GB"
              value={draft.wiredGb}
              onChange={(v) => { setDraft((d) => ({ ...d, wiredGb: v })); setDirty(true); }}
              step="0.5"
              title="Metal allocator wired limit for FLUX.2 / SDXL (MLX_WIRED_LIMIT_GB). 0 = unbounded."
            />
            <TuneInput
              label="krea2 wired"
              unit="GB"
              value={draft.kreaGb}
              onChange={(v) => { setDraft((d) => ({ ...d, kreaGb: v })); setDirty(true); }}
              step="0.5"
              title="krea2 (13B q4) wired budget (MLX_KREA_WIRED_LIMIT_GB). 0 = unbounded."
            />
            <TuneInput
              label="mflux idle"
              unit="s"
              value={draft.mfluxIdle}
              onChange={(v) => { setDraft((d) => ({ ...d, mfluxIdle: v })); setDirty(true); }}
              step="5"
              max={86400}
              title="Idle seconds before the mflux pipeline is auto-released. 0 = keep resident."
            />
            <TuneInput
              label="sdxl idle"
              unit="s"
              value={draft.sdxlIdle}
              onChange={(v) => { setDraft((d) => ({ ...d, sdxlIdle: v })); setDirty(true); }}
              step="5"
              max={86400}
              title="Idle seconds before the SDXL daemon is killed. 0 = keep alive."
            />
            <TuneInput
              label="qwen idle"
              unit="s"
              value={draft.qwenIdle}
              onChange={(v) => { setDraft((d) => ({ ...d, qwenIdle: v })); setDirty(true); }}
              step="5"
              max={86400}
              title="Idle seconds before the Qwen daemon is killed. 0 = keep alive."
            />
            <div className="engine-tune-actions">
              <button type="submit" className="btn-mini" disabled={saving || !dirty}>
                {saving ? "Saving…" : "Save"}
              </button>
              {savedAt && <span className="engine-saved">Saved ✓</span>}
              <span className="hint">0 = disabled / never auto-release</span>
            </div>
          </form>
        </div>

        <div className="engine-card">
          <header>
            <strong>💾 Storage</strong>
          </header>
          <div className="engine-chips">
            {gauge("Generated", `${storage.image_count ?? 0} images`)}
            {gauge("Size", storage.image_bytes != null ? formatBytes(storage.image_bytes) : "—")}
            {gauge("TAEF", (status.taef || []).length ? status.taef.join(", ") : "none loaded")}
          </div>
          <div className="engine-chips">
            {gauge("Settings file", storage.settings_exists ? "present" : "absent", storage.settings_file)}
          </div>
        </div>
      </div>
    </div>
  );
}
import { useEffect, useState } from "react";
import { api } from "../../api";
import { formatBytes } from "../../utils/formatBytes";
import { useI18n } from "../../i18n/I18nContext";

// `t` is passed in rather than looked up here: this is a module-level helper,
// outside any component, so it has no access to useI18n(). The duration
// formats are the ones params.js already defines for the rest of the UI.
function fmtSeconds(s, t) {
  if (s == null || isNaN(s)) return "—";
  if (s <= 0) return t("settings.engine.now");
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0
    ? t("params.duration.minutesSeconds", { m, s: sec })
    : t("params.duration.secondsOnly", { s: sec });
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

function TuneInput({ label, unit, value, onChange, step, max = 128, title, placeholder }) {
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
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
        <small>{unit}</small>
      </div>
    </label>
  );
}

export default function EngineSection() {
  const { t } = useI18n();
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
      // Empty when auto. Showing the derived figure here would make it look typed,
      // and saving would then persist a machine-specific constant -- the bug this
      // whole change exists to remove.
      wiredGb: s.wired?.generic_auto ? "" : String(s.wired?.generic_limit_gb ?? ""),
      kreaGb: s.wired?.krea_auto ? "" : String(s.wired?.krea_limit_gb ?? ""),
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
    // The idle timers have no derived default, so they must be numbers. The two wired
    // budgets do: an empty field means "auto", i.e. derive from this Mac, and null is the
    // settings schema's existing sentinel for that. Requiring a number here is what
    // stopped a machine-derived budget from ever being saved.
    if (mfluxIdle == null || sdxlIdle == null || qwenIdle == null) {
      setErr(t("settings.engine.tuningNumbersError"));
      return;
    }
    if ((wiredGb != null && wiredGb < 0) || (kreaGb != null && kreaGb < 0)
        || mfluxIdle < 0 || sdxlIdle < 0 || qwenIdle < 0) {
      setErr(t("settings.engine.tuningRangeError"));
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
    return <div className="settings-row"><span className="hint">{t("settings.engine.loading")}</span></div>;
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
            <strong>{t("settings.engine.metalCard")}</strong>
          </header>
          <div className="engine-chips">
            {metal.name && gauge(t("settings.engine.gaugeGpu"), metal.name)}
            {gauge(t("settings.engine.gaugeMemory"), metal.memory_size ? formatBytes(metal.memory_size) : "—")}
            {metal.recommended_max_working_set_size
              ? gauge(t("settings.engine.gaugeMaxWorkingSet"), formatBytes(metal.recommended_max_working_set_size))
              : null}
          </div>
          <div className="engine-chips">
            {gauge(t("settings.engine.gaugePeakMemory"), metal.peak_memory != null ? formatBytes(metal.peak_memory) : "—", t("settings.engine.peakMemoryTitle"))}
            {gauge(t("settings.engine.gaugeCache"), metal.cache_memory != null ? formatBytes(metal.cache_memory) : "—", t("settings.engine.cacheTitle"))}
          </div>
          <div className="engine-chips">
            {gauge(t("settings.engine.gaugeWiredLimit"), wired.generic_budget_bytes ? formatBytes(wired.generic_budget_bytes) : t("settings.engine.unbounded"), `MLX_WIRED_LIMIT_GB=${wired.generic_limit_gb || 0}`)}
            {gauge(t("settings.engine.gaugeKreaWired"), wired.krea_budget_bytes ? formatBytes(wired.krea_budget_bytes) : t("settings.engine.unbounded"), `MLX_KREA_WIRED_LIMIT_GB=${wired.krea_limit_gb || 0}`)}
          </div>
        </div>

        <div className="engine-card">
          <header>
            <strong>{t("settings.engine.mfluxCard")}</strong>
            <span className={`engine-dot ${mflux.resident ? "on" : "off"}`} />
          </header>
          <p className="engine-resident">
            {mflux.resident
              ? <>
                  <strong>{mflux.model || t("settings.engine.modelFallback")}</strong>{" "}
                  {t("settings.engine.residentInMemory")}
                  {mflux.watchdog_armed && mflux.seconds_until_release != null && (
                    <span className="engine-watchdog">
                      {t("settings.engine.autoReleaseIn", { time: fmtSeconds(mflux.seconds_until_release, t) })}
                    </span>
                  )}
                </>
              : t("settings.engine.noPipelineResident")}
          </p>
          <div className="engine-chips">
            {gauge(t("settings.engine.gaugeIdlePolicy"), `${mflux.idle_kill_s ?? 300}s`)}
            {gauge(t("settings.engine.gaugePromptCache"), String(mflux.prompt_cache_size ?? 0))}
            {gauge(t("settings.engine.gaugeWatchdog"), mflux.watchdog_armed ? t("settings.engine.armed") : t("settings.engine.cold"))}
          </div>
        </div>

        <div className="engine-card">
          <header>
            <strong>{t("settings.engine.sdxlCard")}</strong>
            <span className={`engine-dot ${sdxl.resident ? "on" : "off"}`} />
          </header>
          <p className="engine-resident">
            {sdxl.resident
              ? <>
                  <strong>{sdxl.model || t("settings.engine.engineFallback")}</strong>{" "}
                  {t("settings.engine.alive")}
                  {sdxl.watchdog_armed && sdxl.seconds_until_release != null && (
                    <span className="engine-watchdog">
                      {t("settings.engine.autoReleaseIn", { time: fmtSeconds(sdxl.seconds_until_release, t) })}
                    </span>
                  )}
                </>
              : t("settings.engine.daemonNotRunning")}
          </p>
          <div className="engine-chips">
            {gauge(t("settings.engine.gaugeIdlePolicy"), `${sdxl.idle_kill_s ?? 300}s`)}
            {gauge(t("settings.engine.gaugeWatchdog"), sdxl.watchdog_armed ? t("settings.engine.armed") : t("settings.engine.cold"))}
          </div>
          {sdxl.stderr_tail && (
            <>
              <button
                type="button"
                className="btn-mini"
                onClick={() => setShowStderr((v) => !v)}
              >
                {showStderr ? t("settings.engine.hideDaemonLog") : t("settings.engine.showDaemonLog")}
              </button>
              {showStderr && <pre className="engine-stderr">{sdxl.stderr_tail}</pre>}
            </>
          )}
        </div>

        <div className="engine-card">
          <header>
            <strong>{t("settings.engine.tuningCard")}</strong>
          </header>
          <p className="engine-tune-note">{t("settings.engine.tuningNote")}</p>
          <form className="engine-tune-grid" onSubmit={saveConfig}>
            <div className="engine-tune-cell">
              <TuneInput
                label={t("settings.engine.tuneWired")}
                unit="GB"
                value={draft.wiredGb}
                onChange={(v) => { setDraft((d) => ({ ...d, wiredGb: v })); setDirty(true); }}
                step="0.5"
                placeholder={String(wired.generic_derived_gb ?? "")}
                title={t("settings.engine.tuneWiredTitle")}
              />
              {draft.wiredGb === "" && (
                <span className="engine-tune-auto">
                  {t("settings.engine.autoDerived", { gb: wired.generic_derived_gb ?? 0 })}
                </span>
              )}
            </div>
            <div className="engine-tune-cell">
              <TuneInput
                label={t("settings.engine.tuneKrea")}
                unit="GB"
                value={draft.kreaGb}
                onChange={(v) => { setDraft((d) => ({ ...d, kreaGb: v })); setDirty(true); }}
                step="0.5"
                placeholder={String(wired.krea_derived_gb ?? "")}
                title={t("settings.engine.tuneKreaTitle")}
              />
              {draft.kreaGb === "" && (
                <span className="engine-tune-auto">
                  {t("settings.engine.autoDerived", { gb: wired.krea_derived_gb ?? 0 })}
                </span>
              )}
            </div>
            <TuneInput
              label={t("settings.engine.tuneMfluxIdle")}
              unit="s"
              value={draft.mfluxIdle}
              onChange={(v) => { setDraft((d) => ({ ...d, mfluxIdle: v })); setDirty(true); }}
              step="5"
              max={86400}
              title={t("settings.engine.tuneMfluxIdleTitle")}
            />
            <TuneInput
              label={t("settings.engine.tuneSdxlIdle")}
              unit="s"
              value={draft.sdxlIdle}
              onChange={(v) => { setDraft((d) => ({ ...d, sdxlIdle: v })); setDirty(true); }}
              step="5"
              max={86400}
              title={t("settings.engine.tuneSdxlIdleTitle")}
            />
            <TuneInput
              label={t("settings.engine.tuneQwenIdle")}
              unit="s"
              value={draft.qwenIdle}
              onChange={(v) => { setDraft((d) => ({ ...d, qwenIdle: v })); setDirty(true); }}
              step="5"
              max={86400}
              title={t("settings.engine.tuneQwenIdleTitle")}
            />
            <div className="engine-tune-actions">
              <button type="submit" className="btn-mini" disabled={saving || !dirty}>
                {saving ? t("settings.engine.saving") : t("settings.engine.save")}
              </button>
              {savedAt && <span className="engine-saved">{t("settings.engine.saved")}</span>}
              <span className="hint">{t("settings.engine.zeroHint")}</span>
            </div>
          </form>
        </div>

        <div className="engine-card">
          <header>
            <strong>{t("settings.engine.storageCard")}</strong>
          </header>
          <div className="engine-chips">
            {gauge(t("settings.engine.gaugeGenerated"), t("settings.engine.imagesCount", { count: storage.image_count ?? 0 }))}
            {gauge(t("settings.engine.gaugeSize"), storage.image_bytes != null ? formatBytes(storage.image_bytes) : "—")}
            {gauge(t("settings.engine.gaugeTaef"), (status.taef || []).length ? status.taef.join(", ") : t("settings.engine.noneLoaded"))}
          </div>
          <div className="engine-chips">
            {gauge(t("settings.engine.gaugeSettingsFile"), storage.settings_exists ? t("settings.engine.present") : t("settings.engine.absent"), storage.settings_file)}
          </div>
        </div>
      </div>
    </div>
  );
}
import { useEffect, useState } from "react";
import GenerateForm from "./components/GenerateForm";
import Gallery from "./components/Gallery";
import ParametersTab from "./components/ParametersTab";
import ModelsTab from "./components/ModelsTab";
import { useI18n } from "./i18n/I18nContext";
import { useTheme } from "./theme";
import { api, isClientMode } from "./api";
import { APP_TITLE, APP_VERSION_LABEL } from "./version";
import { useModelIntegrity } from "./hooks/useModelIntegrity";
import { useRmbgStatus } from "./hooks/useRmbgStatus";
import logo from "./assets/logo.png";
import "./App.css";

// Declared as data rather than as hand-written buttons so the bar cannot drift out of sync
// with the view switch below -- the tab list and the render conditions are one list now.
const TABS = [
  { id: "generate", labelKey: "app.generate" },
  { id: "browser", labelKey: "app.navBrowser" },
  { id: "models", labelKey: "app.navModels" },
  { id: "params", labelKey: "app.navParams" },
];

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
const SHORTCUT_MOD = IS_MAC ? "\u2318" : "Ctrl+";

export default function App() {
  const { t } = useI18n();
  const { resolved, toggle } = useTheme();
  const rmbgStatus = useRmbgStatus();
  const [tab, setTab] = useState("generate");
  const [paramsSubtab, setParamsSubtab] = useState("prefs");
  const [refreshKey, setRefreshKey] = useState(0);
  const [newImage, setNewImage] = useState(null);
  const [initialParams, setInitialParams] = useState(undefined);
  const [modelLabel, setModelLabel] = useState("FLUX.2-klein 4B");
  const [modelRunnable, setModelRunnable] = useState(true);
  const [apiMode, setApiMode] = useState(false);
  const [clientMode] = useState(isClientMode);
  const { integrity } = useModelIntegrity();
  const [generationJob, setGenerationJob] = useState({
    isGenerating: false,
    status: null,
    phase: null,
    progress: null,
    runningModelId: null,
    runningModelLabel: null,
    runningModelRunnable: true,
  });

  function handleModelChange(label, isRunnable = true) {
    if (label) setModelLabel(label);
    setModelRunnable(Boolean(isRunnable));
  }
  /* A browser-card image waiting to become a reference. GenerateForm consumes
     it through addRefImage (model check, duplicates, over-limit dialog) and
     reports back, so App never touches reference logic itself. */
  const [pendingReference, setPendingReference] = useState(null);
  const [referenceLimit, setReferenceLimit] = useState(1);

  useEffect(() => {
    document.title = APP_TITLE;
  }, []);

  useEffect(() => {
    let alive = true;
    api("/api/version")
      .then((value) => {
        if (alive) setApiMode(Boolean(value?.api_mode));
      })
      .catch(() => {});
    const onMode = (event) => setApiMode(Boolean(event.detail?.enabled));
    window.addEventListener("mlx:api-mode", onMode);
    return () => {
      alive = false;
      window.removeEventListener("mlx:api-mode", onMode);
    };
  }, []);

  function openLicences() {
    setTab("params");
    setParamsSubtab("licences");
  }

  useEffect(() => {
    const validTabs = new Set(TABS.map((entry) => entry.id));
    const validSections = new Set(["prefs", "enhancer", "licences", "api"]);
    function handleMenuAction(event) {
      const detail = event?.detail;
      if (!detail || typeof detail !== "object") return;
      const { action, tab: nextTab, section } = detail;
      if (action === "settings") {
        setTab("params");
        setParamsSubtab(validSections.has(section) ? section : "prefs");
      } else if (action === "licences") {
        openLicences();
      } else if (action === "tab" && validTabs.has(nextTab)) {
        setTab(nextTab);
        if (validSections.has(section)) {
          setParamsSubtab(section);
        }
      }
    }
    window.addEventListener("db-menu-action", handleMenuAction);
    return () => window.removeEventListener("db-menu-action", handleMenuAction);
  }, []);

  useEffect(() => {
    function onKeyDown(event) {
      if (!(event.metaKey || event.ctrlKey) || event.altKey) return;
      // Physical key, so AZERTY (where the number row types & é " ' unshifted) still works.
      const [, pad, digit] = /^(Digit|Numpad)([1-9])$/.exec(event.code || "") || [];
      if (pad === "Numpad" && event.key !== digit) return; // Num Lock off: End/Down/PgDn/Left
      const index = Number(digit) - 1;
      if (!Number.isInteger(index) || index < 0 || index >= TABS.length) return;
      event.preventDefault();
      setTab(TABS[index].id);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function handleGenerated(imageMeta) {
    setRefreshKey((k) => k + 1);
    if (imageMeta) {
      setNewImage(imageMeta);
    }
  }

  function handleImageSaved() {
    setRefreshKey((k) => k + 1);
  }

  return (
    <div className="app">
      <header>
        <div className="brand">
          <button
            type="button"
            className="app-logo-btn"
            onClick={openLicences}
            title={t("app.logoTitle") || "DiffusionBear — Licences"}
            aria-label={t("app.logoTitle") || "DiffusionBear — Licences"}
          >
            <img className="app-logo" src={logo} alt={t("app.logoAlt")} />
          </button>
          <div className="brand-text">
            <h1>DiffusionBear</h1>
            <span className="version-badge">{APP_VERSION_LABEL}</span>
            {apiMode && <span className="mode-badge mode-badge--api">API</span>}
            {clientMode && <span className="mode-badge mode-badge--client">CLIENT</span>}
          </div>
        </div>
        <nav className="studio-nav-segmented">
          {TABS.map((entry, index) => (
            <button
              key={entry.id}
              title={`${t(entry.labelKey)} (${SHORTCUT_MOD}${index + 1})`}
              aria-keyshortcuts={`${IS_MAC ? "Meta" : "Control"}+${index + 1}`}
              className={`studio-nav-btn ${tab === entry.id ? "active" : ""}`}
              aria-current={tab === entry.id ? "page" : undefined}
              onClick={() => setTab(entry.id)}
            >
              {entry.id === "params" && (
                <svg
                  className="studio-nav-icon"
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <circle cx="12" cy="12" r="3" />
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
              )}
              {t(entry.labelKey)}
            </button>
          ))}
        </nav>

        <div className="header-actions">
          {(() => {
            const isRmbgRunning = Boolean(rmbgStatus?.running);
            const isGenerating = Boolean(generationJob.isGenerating) || isRmbgRunning;
            const activeModelLabel = isRmbgRunning
              ? "Bria RMBG-3.0"
              : (generationJob.isGenerating && generationJob.runningModelLabel)
                ? generationJob.runningModelLabel
                : modelLabel;
            const isRunnable = isGenerating
              ? Boolean(generationJob.runningModelRunnable ?? true)
              : modelRunnable;

            return (
              <button
                type="button"
                className={`header-status-capsule ${isGenerating ? "generating" : isRunnable ? "runnable" : "unrunnable"}`}
                onClick={() => setTab("generate")}
                title={isRmbgRunning ? t("fill.rmbgRunning") : isGenerating ? t("app.generationInProgress") : `${activeModelLabel} · ${isRunnable ? t("app.ready") : t("app.weightsRequired")}`}
                aria-label={isRmbgRunning ? t("fill.rmbgRunning") : isGenerating ? t("app.generationInProgress") : `${activeModelLabel} · MLX`}
              >
                <span
                  className={`status-dot ${isGenerating ? "generating pulsing" : isRunnable ? "runnable" : "unrunnable"}`}
                  aria-hidden="true"
                />
                <span className="status-model-name">{activeModelLabel}</span>
                {isRmbgRunning ? (
                  <span className="status-progress-tag">
                    {rmbgStatus?.step > 0
                      ? t("canvas.pill.step", { step: rmbgStatus.step, steps: rmbgStatus.steps || 4 })
                      : t("fill.rmbgRunning")}
                  </span>
                ) : isGenerating && (
                  <span className="status-progress-tag">
                    {generationJob.phase === "downloading"
                      ? t("canvas.pill.downloadingModel")
                      : generationJob.phase === "loading_model"
                        ? t("canvas.pill.loadingMemory")
                        : generationJob.phase === "compiling"
                          ? t("canvas.pill.compilingShaders")
                          : generationJob.phase === "saving"
                            ? t("canvas.pill.finalizingImage")
                            : (generationJob.progress?.steps > 0
                                ? t("canvas.pill.step", { step: generationJob.progress.step, steps: generationJob.progress.steps })
                                : t("canvas.pill.working"))}
                  </span>
                )}
                <span className="status-badge">MLX · Apple Silicon</span>
              </button>
            );
          })()}

          <button
            type="button"
            className="theme-toggle"
            onClick={toggle}
            aria-pressed={resolved === "light"}
            aria-label={t(resolved === "light" ? "app.themeToDark" : "app.themeToLight")}
            title={t(resolved === "light" ? "app.themeToDark" : "app.themeToLight")}
          >
            <span aria-hidden="true" className="theme-toggle-icon">
              {resolved === "light" ? (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
              ) : (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
              )}
            </span>
          </button>
        </div>
      </header>

      <main>
        {integrity?.running && (
          <div className="integrity-global-banner" role="status" aria-live="polite">
            <span className="integrity-spinner" aria-hidden="true" />
            <span>
              {t("models.integrity.hashingBanner", {
                version: integrity.app_version || APP_VERSION_LABEL,
                model: integrity.current_label || "...",
                done: Math.min((integrity.completed_models || 0) + 1, integrity.total_models || 1),
                total: integrity.total_models || 1,
              })}
            </span>
          </div>
        )}
        <div style={{ display: tab === "generate" ? "block" : "none" }}>
          <GenerateForm
            onGenerated={handleGenerated}
            initialParams={initialParams}
            onModelChange={handleModelChange}
            onGenerationStatusChange={setGenerationJob}
            onImageSaved={handleImageSaved}
            onImageDeleted={() => setRefreshKey((k) => k + 1)}
            pendingReference={pendingReference}
            onPendingReferenceConsumed={() => setPendingReference(null)}
            onReferenceLimit={setReferenceLimit}
            activeTab={tab}
          />
        </div>
        <div style={{ display: tab === "browser" ? "block" : "none" }}>
          <Gallery
            refreshKey={refreshKey}
            newImage={newImage}
            activeTab={tab}
            onReuse={(meta) => {
              setInitialParams({ ...meta, key: Date.now() });
              setTab("generate");
            }}
            maxReferenceImages={referenceLimit}
            onSetReference={(ref) => {
              setPendingReference({ ...ref, key: Date.now() });
              setTab("generate");
            }}
            onImageDeleted={() => setRefreshKey((k) => k + 1)}
          />
        </div>
        <div style={{ display: tab === "models" ? "block" : "none" }}>
          <ModelsTab onNavigate={setTab} />
        </div>
        <div style={{ display: tab === "params" ? "block" : "none" }}>
          <ParametersTab
            onNavigate={setTab}
            subtab={paramsSubtab}
            onSubtabChange={setParamsSubtab}
          />
        </div>
      </main>
    </div>
  );
}

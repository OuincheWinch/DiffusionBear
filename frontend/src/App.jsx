import { useEffect, useState } from "react";
import GenerateForm from "./components/GenerateForm";
import Gallery from "./components/Gallery";
import ParametersTab from "./components/ParametersTab";
import ModelsTab from "./components/ModelsTab";
import { useI18n } from "./i18n/I18nContext";
import { useTheme } from "./theme";
import { APP_TITLE, APP_VERSION_LABEL } from "./version";
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

export default function App() {
  const { t } = useI18n();
  const { resolved, toggle } = useTheme();
  const [tab, setTab] = useState("generate");
  const [paramsSubtab, setParamsSubtab] = useState("prefs");
  const [refreshKey, setRefreshKey] = useState(0);
  const [newImage, setNewImage] = useState(null);
  const [initialParams, setInitialParams] = useState(undefined);
  const [modelLabel, setModelLabel] = useState("FLUX.2-klein 4B");
  const [modelRunnable, setModelRunnable] = useState(true);
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
            onClick={() => {
              setTab("params");
              setParamsSubtab("licences");
            }}
            title={t("app.logoTitle") || "DiffusionBear — Licences"}
            aria-label={t("app.logoTitle") || "DiffusionBear — Licences"}
          >
            <img className="app-logo" src={logo} alt={t("app.logoAlt")} />
          </button>
          <div className="brand-text">
            <h1>DiffusionBear</h1>
            <span className="version-badge">{APP_VERSION_LABEL}</span>
          </div>
        </div>
        <nav className="studio-nav-segmented">
          {TABS.map((entry) => (
            <button
              key={entry.id}
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
            const isGenerating = Boolean(generationJob.isGenerating);
            const activeModelLabel = (isGenerating && generationJob.runningModelLabel)
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
                title={isGenerating ? t("app.generationInProgress") : `${activeModelLabel} · ${isRunnable ? t("app.ready") : t("app.weightsRequired")}`}
                aria-label={isGenerating ? t("app.generationInProgress") : `${activeModelLabel} · MLX`}
              >
                <span
                  className={`status-dot ${isGenerating ? "generating pulsing" : isRunnable ? "runnable" : "unrunnable"}`}
                  aria-hidden="true"
                />
                <span className="status-model-name">{activeModelLabel}</span>
                {isGenerating && (
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

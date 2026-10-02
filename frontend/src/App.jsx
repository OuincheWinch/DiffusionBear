import { useEffect, useState } from "react";
import GenerateForm from "./components/GenerateForm";
import Gallery from "./components/Gallery";
import ParametersTab from "./components/ParametersTab";
import ModelsTab from "./components/ModelsTab";
import { useI18n } from "./i18n/I18nContext";
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
  const [tab, setTab] = useState("generate");
  const [refreshKey, setRefreshKey] = useState(0);
  const [newImage, setNewImage] = useState(null);
  const [initialParams, setInitialParams] = useState(undefined);
  const [modelLabel, setModelLabel] = useState("FLUX.2-klein 4B");

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
          <img className="app-logo" src={logo} alt={t("app.logoAlt")} />
          <div>
            <h1>DiffusionBear</h1>
            <span className="version-badge">{APP_VERSION_LABEL}</span>
          </div>
        </div>
        <nav>
          {TABS.map((entry) => (
            <button
              key={entry.id}
              className={tab === entry.id ? "active" : ""}
              aria-current={tab === entry.id ? "page" : undefined}
              onClick={() => setTab(entry.id)}
            >
              {t(entry.labelKey)}
            </button>
          ))}
        </nav>
      </header>

      <main>
        <div style={{ display: tab === "generate" ? "block" : "none" }}>
          <h2 className="active-model-title">{modelLabel} · MLX</h2>
          <GenerateForm
            onGenerated={handleGenerated}
            initialParams={initialParams}
            onModelChange={setModelLabel}
            onImageSaved={handleImageSaved}
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
          />
        </div>
        <div style={{ display: tab === "models" ? "block" : "none" }}>
          <ModelsTab onNavigate={setTab} />
        </div>
        <div style={{ display: tab === "params" ? "block" : "none" }}>
          <ParametersTab onNavigate={setTab} />
        </div>
      </main>
    </div>
  );
}

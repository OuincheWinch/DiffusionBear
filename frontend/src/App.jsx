import { useEffect, useState } from "react";
import GenerateForm from "./components/GenerateForm";
import Gallery from "./components/Gallery";
import ParametersTab from "./components/ParametersTab";
import LicencesTab from "./components/LicencesTab";
import { useI18n } from "./i18n/I18nContext";
import { APP_TITLE, APP_VERSION_LABEL } from "./version";
import logo from "./assets/logo.png";
import "./App.css";

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
          <button
            className={tab === "generate" ? "active" : ""}
            onClick={() => setTab("generate")}
          >
            {t("app.generate")}
          </button>
          <button
            className={tab === "browser" ? "active" : ""}
            onClick={() => setTab("browser")}
          >
            {t("app.navBrowser")}
          </button>
          <button
            className={tab === "params" ? "active" : ""}
            onClick={() => setTab("params")}
          >
            {t("app.navParams")}
          </button>
          <button
            className={tab === "licences" ? "active" : ""}
            onClick={() => setTab("licences")}
          >
            ⚖ {t("licences.title")}
          </button>
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
        <div style={{ display: tab === "params" ? "block" : "none" }}>
          <ParametersTab onNavigate={setTab} />
        </div>
        <div style={{ display: tab === "licences" ? "block" : "none" }}>
          <LicencesTab />
        </div>
      </main>
    </div>
  );
}

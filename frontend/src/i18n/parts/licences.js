// LicencesTab.jsx. The component was written in English, so the literal goes in
// `en` and fr/de/it are translations of it -- the opposite of strings.js, whose
// French originals came first.
//
// Deliberately NOT translated: every row of the third-party tables. The package
// names, SPDX identifiers ("Apache-2.0", "MIT-CMU", "Black Forest Labs --
// Non-Commercial") and repository URLs live in src/data/licences.js and are
// quotations of upstream terms; renaming them would misstate what the upstream
// licence actually says. Only the app's own chrome around them is here.
export const licencesStrings = {
  "licences.sectionBackend": {
    fr: "Backend — environnement MLX / mflux (Python)",
    en: "Backend — MLX / mflux runtime (Python)",
    de: "Backend — MLX-/mflux-Laufzeitumgebung (Python)",
    it: "Backend — runtime MLX / mflux (Python)",
  },
  "licences.sectionFrontend": {
    fr: "Interface web — application React / Vite",
    en: "Frontend — React / Vite SPA",
    de: "Weboberfläche — React-/Vite-SPA",
    it: "Interfaccia web — SPA React / Vite",
  },
  "licences.sectionSdxl": {
    fr: "Moteur SDXL — Juggernaut XL Lightning (Python, environnement sans torch)",
    en: "SDXL engine — Juggernaut XL Lightning (Python, torch-free runtime)",
    de: "SDXL-Engine — Juggernaut XL Lightning (Python, Laufzeitumgebung ohne torch)",
    it: "Motore SDXL — Juggernaut XL Lightning (Python, runtime senza torch)",
  },
  "licences.sectionModels": {
    fr: "Poids des modèles IA",
    en: "Models and weights",
    de: "Modelle und Gewichte",
    it: "Modelli e pesi",
  },
  "licences.noteBackend": {
    fr: "Cœur du moteur de génération pour FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo et le moteur expérimental Qwen-Image 2.1.",
    en: "Core generation stack for FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo and the Qwen-Image 2.1 experimental engine.",
    de: "Kern-Generierungsstack für FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo und die experimentelle Qwen-Image-2.1-Engine.",
    it: "Stack di generazione principale per FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo e il motore sperimentale Qwen-Image 2.1.",
  },
  "licences.noteSdxl": {
    fr: "Les outils de conversion ne sont fournis que dans venv-sdxl ; le moteur déployé n'utilise pas torch.",
    en: "Conversion tools ship in venv-sdxl only; the deployed engine runtime is torch-free.",
    de: "Die Konvertierungswerkzeuge sind nur in venv-sdxl enthalten; die ausgelieferte Engine-Laufzeitumgebung kommt ohne torch aus.",
    it: "Gli strumenti di conversione sono presenti solo in venv-sdxl; il runtime del motore distribuito non usa torch.",
  },
  "licences.noteFrontend": {
    fr: "Interface web. Les mêmes dépendances que toute application Vite moderne.",
    en: "Browser UI. Same packages as any modern Vite app.",
    de: "Weboberfläche. Dieselben Pakete wie jede moderne Vite-App.",
    it: "Interfaccia web. Gli stessi pacchetti di qualsiasi app Vite moderna.",
  },
  "licences.noteWeights": {
    fr: "Les fichiers de poids sont téléchargés depuis Hugging Face / les fiches de modèle lors de la première utilisation — ils ne sont ni inclus ni redistribués par DiffusionBear. Chacun porte ses propres conditions.",
    en: "Weight files are downloaded on first use from Hugging Face / model cards — they are NOT bundled with, nor redistributed by, DiffusionBear. Each carries its own terms.",
    de: "Gewichtsdateien werden bei der ersten Verwendung von Hugging Face / den Modellkarten heruntergeladen — sie sind weder in DiffusionBear enthalten noch werden sie von ihm weiterverbreitet. Jede Datei hat eigene Bedingungen.",
    it: "I file di peso vengono scaricati alla prima uso da Hugging Face / dalle schede dei modelli — non sono inclusi né ridistribuiti da DiffusionBear. Ciascuno ha le proprie condizioni.",
  },

  // The <h3> and the header nav button both compose this key with "⚖" and
  // "— DiffusionBear", so the word is never spelled twice.
  "licences.introSecond": {
    fr: "Chaque paquet tiers ci-dessous conserve sa propre licence ; rien ici n'accorde ni ne révoque ces conditions.",
    en: "Every third-party package below keeps its own licence; nothing here grants or revokes those terms.",
    de: "Jedes unten aufgeführte Paket von Dritten behält seine eigene Lizenz; durch nichts hier werden diese Bedingungen gewährt oder widerrufen.",
    it: "Ogni pacchetto di terze parti elencato qui sotto mantiene la propria licenza; nulla di quanto scritto qui concede o revoca quei termini.",
  },
  "licences.thisProject": {
    fr: "Ce projet",
    en: "This project",
    de: "Dieses Projekt",
    it: "Questo progetto",
  },
  // "released under the ___" + link + ", Copyright ..." -- the article lives in
  // the connector so each language gets the one its grammar needs.
  "licences.releasedUnder": {
    fr: "est distribué sous la",
    en: "is released under the",
    de: "wird unter der",
    it: "è distribuito con la",
  },
  "licences.mitLicense": {
    fr: "licence MIT",
    en: "MIT License",
    de: "MIT-Lizenz",
    it: "licenza MIT",
  },
  // International legal notice: identical in all four languages by convention.
  "licences.copyright": {
    fr: "Copyright © {year}",
    en: "Copyright © {year}",
    de: "Copyright © {year}",
    it: "Copyright © {year}",
  },
  "licences.aiAuthored": {
    fr: "En grande partie écrit par une IA",
    en: "Heavily coded by AI",
    de: "Überwiegend von KI geschrieben",
    it: "Scritto in gran parte dall'IA",
  },
  "licences.aiCredits": {
    fr: "assisté par {credits} aux côtés de son auteur humain. Relu et mesuré à la main.",
    en: "assisted by {credits} together with its human author. Reviewed and benchmarked by hand.",
    de: "unterstützt von {credits} zusammen mit seinem menschlichen Autor. Von Hand geprüft und getestet.",
    it: "assistito da {credits} insieme al suo autore umano. Riletto e verificato a mano.",
  },
  // Split in three around the <em> run: splitting keeps the emphasis and lets each
  // language order the sentence the way it needs to (French needs "que" inside).
  "licences.mitScopeBefore": {
    fr: "La licence MIT ne couvre",
    en: "The MIT licence covers",
    de: "Die MIT-Lizenz deckt",
    it: "La licenza MIT copre",
  },
  "licences.mitScopeEm": {
    fr: "que le code source de ce projet",
    en: "this project's source code only",
    de: "nur den Quellcode dieses Projekts",
    it: "solo il codice sorgente di questo progetto",
  },
  "licences.mitScopeAfter": {
    fr: ". Les poids des modèles ne sont pas couverts par elle — voir le tableau des poids ci-dessous.",
    en: ". The model weights are not covered by it — see the weights table below.",
    de: ". Die Modellgewichte sind davon nicht abgedeckt – siehe Gewichtstabelle unten.",
    it: ". I pesi dei modelli non sono coperti — vedi la tabella dei pesi qui sotto.",
  },
  "licences.colPackage": {
    fr: "Paquet",
    en: "Package",
    de: "Paket",
    it: "Pacchetto",
  },
  "licences.colLicence": {
    fr: "Licence",
    en: "Licence",
    de: "Lizenz",
    it: "Licenza",
  },
  "licences.colRepository": {
    fr: "Dépôt",
    en: "Repository",
    de: "Repository",
    it: "Repository",
  },
  "licences.versions": {
    fr: "Versions",
    en: "Versions",
    de: "Versionen",
    it: "Versioni",
  },
  "licences.frontendVersion": {
    fr: "Interface {version}",
    en: "Frontend {version}",
    de: "Frontend {version}",
    it: "Interfaccia {version}",
  },
  "licences.backendApi": {
    fr: " · API backend {version}",
    en: " · Backend API {version}",
    de: " · Backend-API {version}",
    it: " · API backend {version}",
  },
  "licences.repoLink": {
    fr: "dépôt",
    en: "repo",
    de: "Repo",
    it: "repo",
  },
};

export default licencesStrings;

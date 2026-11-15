// App.jsx -- the header only. Written in English, so the literal is the `en`
// value.
//
// Reused from strings.js rather than redefined: app.generate (the Generate tab)
// and licences.title (the Licences tab). The <h1> and document.title stay the
// literal brand name "DiffusionBear": it is a proper noun, not a sentence.
export const appStrings = {
  "app.themeToLight": {
    fr: "Passer en thème clair",
    en: "Switch to light theme",
    de: "Zum hellen Design wechseln",
    it: "Passa al tema chiaro",
  },
  "app.themeToDark": {
    fr: "Passer en thème sombre",
    en: "Switch to dark theme",
    de: "Zum dunklen Design wechseln",
    it: "Passa al tema scuro",
  },
  // Brand name, so identical in every language -- but screen readers do read it
  // out, so it is a key like any other alt text.
  "app.logoAlt": {
    fr: "DiffusionBear",
    en: "DiffusionBear",
    de: "DiffusionBear",
    it: "DiffusionBear",
  },
  "app.navBrowser": {
    fr: "Navigateur",
    en: "Browser",
    de: "Browser",
    it: "Browser",
  },
  "app.navModels": {
    fr: "Modèles",
    en: "Models",
    de: "Modelle",
    it: "Modelli",
  },
  "app.navParams": {
    fr: "Paramètres",
    en: "Parameters",
    de: "Parameter",
    it: "Parametri",
  },
  "app.browse": {
    fr: "Parcourir…",
    en: "Browse…",
    de: "Durchsuchen…",
    it: "Sfoglia…",
  },
  "app.generationInProgress": {
    fr: "Génération en cours · Cliquer pour afficher le studio",
    en: "Generation in progress · Click to show studio",
    de: "Generierung läuft · Klicken, um das Studio anzuzeigen",
    it: "Generazione in corso · Clicca per visualizzare lo studio",
  },
  "app.logoTitle": {
    fr: "DiffusionBear — Ouvrir les licences",
    en: "DiffusionBear — Open licences",
    de: "DiffusionBear — Lizenzen öffnen",
    it: "DiffusionBear — Apri le licenze",
  },
  "app.ready": {
    fr: "Prêt",
    en: "Ready",
    de: "Bereit",
    it: "Pronto",
  },
  "app.weightsRequired": {
    fr: "Poids requis",
    en: "Weights required",
    de: "Gewichte erforderlich",
    it: "Pesi richiesti",
  },
};

export default appStrings;

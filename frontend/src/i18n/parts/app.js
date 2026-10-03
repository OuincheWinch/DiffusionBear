// App.jsx -- the header only. Written in English, so the literal is the `en`
// value.
//
// Reused from strings.js rather than redefined: app.generate (the Generate tab)
// and licences.title (the Licences tab). The <h1> and document.title stay the
// literal brand name "DiffusionBear": it is a proper noun, not a sentence.
export const appStrings = {
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
    fr: "🧠 Modèles",
    en: "🧠 Models",
    de: "🧠 Modelle",
    it: "🧠 Modelli",
  },
  "app.navParams": {
    fr: "⚙️ Paramètres",
    en: "⚙️ Parameters",
    de: "⚙️ Parameter",
    it: "⚙️ Parametri",
  },
};

export default appStrings;

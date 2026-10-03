// UniversalDownloader.jsx. French is the reference language, as in strings.js;
// the component was still written in English, so the literal is the `en` value.
export const downloaderStrings = {
  "downloader.starting": {
    fr: "Début du téléchargement de « {model} » ({base})…",
    en: "Starting download of \"{model}\" ({base})…",
    de: "Download von \"{model}\" wird gestartet ({base})…",
    it: "Avvio del download di «{model}» ({base})…",
  },
  "downloader.failed": {
    fr: "Échec du téléchargement : {error}",
    en: "Download failed: {error}",
    de: "Download fehlgeschlagen: {error}",
    it: "Download non riuscito: {error}",
  },
  "downloader.cancelFailed": {
    fr: "Échec de l'annulation : {error}",
    en: "Failed to cancel: {error}",
    de: "Abbrechen fehlgeschlagen: {error}",
    it: "Annullamento non riuscito: {error}",
  },
  "downloader.tagDirectUrl": {
    fr: "URL DIRECTE",
    en: "DIRECT URL",
    de: "DIREKTE URL",
    it: "URL DIRETTO",
  },
  "downloader.tagDirect": {
    fr: "DIRECT",
    en: "DIRECT",
    de: "DIREKT",
    it: "DIRETTO",
  },
  "downloader.title": {
    fr: "Téléchargeur de LoRA universel",
    en: "Universal LoRA Downloader",
    de: "Universeller LoRA-Downloader",
    it: "Downloader LoRA universale",
  },
  "downloader.detectedHf": {
    fr: "🤗 Hugging Face détecté",
    en: "🤗 Hugging Face detected",
    de: "🤗 Hugging Face erkannt",
    it: "🤗 Hugging Face rilevato",
  },
  "downloader.detectedCivitai": {
    fr: "⚡ Civitai détecté",
    en: "⚡ Civitai detected",
    de: "⚡ Civitai erkannt",
    it: "⚡ Civitai rilevato",
  },
  "downloader.detectedDirect": {
    fr: "🔗 Lien direct détecté",
    en: "🔗 Direct Link detected",
    de: "🔗 Direktlink erkannt",
    it: "🔗 Link diretto rilevato",
  },
  "downloader.hint": {
    fr: "Collez l'URL ou l'ID d'un modèle Civitai, l'ID de dépôt ou l'URL d'un dépôt Hugging Face, ou n'importe quel lien direct vers un fichier .safetensors.",
    en: "Paste a Civitai model URL or ID, a Hugging Face repo ID or URL, or any direct link to a .safetensors file.",
    de: "Fügen Sie eine Civitai-Modell-URL oder -ID, eine Hugging-Face-Repo-ID oder -URL oder einen beliebigen Direktlink zu einer .safetensors-Datei ein.",
    it: "Incolla l'URL o l'ID di un modello Civitai, l'ID del repository o l'URL di un repository Hugging Face, oppure un link diretto a un file .safetensors.",
  },
  "downloader.urlPlaceholder": {
    fr: "URL ou ID Civitai, dépôt ou URL Hugging Face, ou lien direct .safetensors…",
    en: "Civitai URL or ID, Hugging Face repo or URL, or direct .safetensors link...",
    de: "Civitai-URL oder -ID, Hugging-Face-Repo oder -URL oder direkter .safetensors-Link …",
    it: "URL o ID Civitai, repository o URL Hugging Face, oppure link diretto .safetensors…",
  },
  "downloader.startingBtn": {
    fr: "Démarrage…",
    en: "Starting…",
    de: "Wird gestartet…",
    it: "Avvio…",
  },
  "downloader.downloadBtn": {
    fr: "⚡ Télécharger et enregistrer",
    en: "⚡ Download & Register",
    de: "⚡ Herunterladen & registrieren",
    it: "⚡ Scarica e registra",
  },
  "downloader.namePlaceholder": {
    fr: "Nom personnalisé (facultatif)",
    en: "Custom name (optional)",
    de: "Eigener Name (optional)",
    it: "Nome personalizzato (facoltativo)",
  },
  "downloader.triggersPlaceholder": {
    fr: "Mots déclencheurs : séparés par des virgules (facultatif)",
    en: "Trigger words: comma, separated (optional)",
    de: "Trigger-Wörter: kommagetrennt (optional)",
    it: "Parole trigger: separate da virgole (facoltativo)",
  },
  "downloader.autoDetect": {
    fr: "Détection automatique de l'architecture",
    en: "Auto-Detect Architecture",
    de: "Architektur automatisch erkennen",
    it: "Rilevamento automatico dell'architettura",
  },
  "downloader.cancelTitle": {
    fr: "Annuler le téléchargement",
    en: "Cancel download",
    de: "Download abbrechen",
    it: "Annulla il download",
  },
  "downloader.cancelBtn": {
    fr: "✕ Annuler",
    en: "✕ Cancel",
    de: "✕ Abbrechen",
    it: "✕ Annulla",
  },
  "downloader.ready": { fr: "✓ Prêt", en: "✓ Ready", de: "✓ Fertig", it: "✓ Pronto" },
  "downloader.error": { fr: "✖ Erreur", en: "✖ Error", de: "✖ Fehler", it: "✖ Errore" },
  "downloader.cancelled": {
    fr: "Annulé",
    en: "Cancelled",
    de: "Abgebrochen",
    it: "Annullato",
  },
  "downloader.authRequiredHf": {
    fr: "Ce modèle nécessite une authentification sur Hugging Face.",
    en: "This model requires authentication on Hugging Face.",
    de: "Dieses Modell erfordert eine Anmeldung bei Hugging Face.",
    it: "Questo modello richiede l'autenticazione su Hugging Face.",
  },
  "downloader.authRequiredCivitai": {
    fr: "Ce modèle nécessite une authentification sur Civitai.",
    en: "This model requires authentication on Civitai.",
    de: "Dieses Modell erfordert eine Anmeldung bei Civitai.",
    it: "Questo modello richiede l'autenticazione su Civitai.",
  },
  "downloader.configureHfToken": {
    fr: "🔑 Configurer le jeton HF",
    en: "🔑 Configure HF Token",
    de: "🔑 HF-Token einrichten",
    it: "🔑 Configura il token HF",
  },
  "downloader.configureApiKey": {
    fr: "🔑 Configurer la clé d'API",
    en: "🔑 Configure API Key",
    de: "🔑 API-Schlüssel einrichten",
    it: "🔑 Configura la chiave API",
  },
  "downloader.installedFor": {
    fr: "Installé pour {model}.",
    en: "Installed for {model}.",
    de: "Für {model} installiert.",
    it: "Installato per {model}.",
  },
  "downloader.switchTo": {
    fr: "⚡ Passer à {model}",
    en: "⚡ Switch to {model}",
    de: "⚡ Zu {model} wechseln",
    it: "⚡ Passa a {model}",
  },
};

export default downloaderStrings;

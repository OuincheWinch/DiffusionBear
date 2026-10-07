// ModelInstaller.jsx. Written in English, so the literal is the `en` value and
// fr/de/it are translations of it.
//
// Reused from strings.js rather than redefined: app.retry ("Retry"), app.cancel
// ("Cancel"). `task.status_text` and the "MB" / "MB/s" units come from the
// backend or are unit symbols -- not translated.
export const installerStrings = {
  "installer.tagWeights": { fr: "poids", en: "weights", de: "Gewichte", it: "pesi" },
  "installer.statusInstalled": {
    fr: "✓ Installé",
    en: "✓ Installed",
    de: "✓ Installiert",
    it: "✓ Installato",
  },
  "installer.statusFailed": {
    fr: "✕ Échec du téléchargement",
    en: "✕ Download failed",
    de: "✕ Download fehlgeschlagen",
    it: "✕ Download non riuscito",
  },
  "installer.statusCancelled": {
    fr: "Téléchargement annulé",
    en: "Download cancelled",
    de: "Download abgebrochen",
    it: "Download annullato",
  },
  "installer.statusDownloading": {
    fr: "⬇ Téléchargement…",
    en: "⬇ Downloading…",
    de: "⬇ Wird geladen…",
    it: "⬇ Download…",
  },
  "installer.statusNotInstalled": {
    fr: "⬇ Non installé — téléchargez d'abord les poids",
    en: "⬇ Not installed — download weights first",
    de: "⬇ Nicht installiert – Gewichte zuerst herunterladen",
    it: "⬇ Non installato — scarica prima i pesi",
  },
  "installer.startingBtn": {
    fr: "Démarrage…",
    en: "Starting…",
    de: "Wird gestartet…",
    it: "Avvio…",
  },
  "installer.downloadBtn": {
    fr: "Télécharger",
    en: "Download",
    de: "Herunterladen",
    it: "Scarica",
  },
  "installer.retryBtn": {
    fr: "Reprendre",
    en: "Retry",
    de: "Erneut versuchen",
    it: "Riprova",
  },
  "installer.localBtn": {
    fr: "📁 Local…",
    en: "📁 Local…",
    de: "📁 Lokal…",
    it: "📁 Locale…",
  },
  "installer.localBtnTitle": {
    fr: "Installer depuis un dossier déjà présent sur le disque (cache HF ou local)",
    en: "Install from a folder already on disk (HF cache or local)",
    de: "Aus einem bereits auf der Festplatte vorhandenen Ordner installieren (HF-Cache oder lokal)",
    it: "Installa da una cartella già presente sul disco (cache HF o locale)",
  },
  "installer.pathRequired": {
    fr: "Choisissez un dépôt en cache ou saisissez le chemin d'un dossier",
    en: "Choose a cached repo or enter a folder path",
    de: "Wählen Sie ein zwischengespeichertes Repository oder geben Sie einen Ordnerpfad ein",
    it: "Scegli un repository in cache o inserisci il percorso di una cartella",
  },
  "installer.localHint": {
    fr: "Installer depuis une copie déjà présente sur votre disque — rien n'est téléchargé ni copié ; le modèle se charge directement depuis le dossier.",
    en: "Install from a copy already on your disk — nothing is downloaded or copied; the model loads directly from the folder.",
    de: "Aus einer bereits auf Ihrer Festplatte vorhandenen Kopie installieren — es wird nichts heruntergeladen oder kopiert; das Modell lädt direkt aus dem Ordner.",
    it: "Installa da una copia già presente sul tuo disco — non viene scaricato né copiato nulla; il modello si carica direttamente dalla cartella.",
  },
  "installer.scanningCache": {
    fr: "Analyse du cache Hugging Face…",
    en: "Scanning Hugging Face cache…",
    de: "Hugging-Face-Cache wird durchsucht…",
    it: "Analisi della cache Hugging Face…",
  },
  "installer.pickCachedRepo": {
    fr: "— choisir un dépôt Hugging Face en cache —",
    en: "— pick a cached Hugging Face repo —",
    de: "— zwischengespeichertes Hugging-Face-Repository wählen —",
    it: "— scegli un repository Hugging Face in cache —",
  },
  "installer.noCachedRepo": {
    fr: "— aucun dépôt en cache trouvé —",
    en: "— no cached repos found —",
    de: "— keine zwischengespeicherten Repositories gefunden —",
    it: "— nessun repository in cache trovato —",
  },
  // The path shape after the slash is a literal a user types, so it stays
  // ASCII in every language; only the prose around it moves.
  "installer.pathPlaceholder": {
    fr: "…ou /chemin/vers/le/dossier/modele (ou un dossier de cache HF models--org--name)",
    en: "…or /path/to/model/folder (or an HF cache models--org--name dir)",
    de: "…oder /pfad/zum/modellordner (oder ein HF-Cache-Verzeichnis models--org--name)",
    it: "…oppure /percorso/della/cartella/modello (oppure una dir della cache HF models--org--name)",
  },
  "installer.installingBtn": {
    fr: "Installation…",
    en: "Installing…",
    de: "Wird installiert…",
    it: "Installazione…",
  },
  "installer.installLocalBtn": {
    fr: "Installer en local",
    en: "Install from local",
    de: "Lokal installieren",
    it: "Installa da locale",
  },
};

export default installerStrings;

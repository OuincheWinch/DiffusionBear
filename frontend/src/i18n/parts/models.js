// Strings for the Models tab and its Hugging Face browser.
//
// French is the reference language: this catalogue was authored fr-first, then translated
// into the other three. The HF browser is the first surface here that shows user-authored
// text (repo names) next to our own, so the wording is deliberately terse -- chip labels and
// button text have to fit next to things like "FLUX.2-Klein-4B-4bit" without wrapping.

/**
 * @type {Record<string, {fr: string, en: string, de: string, it: string}>}
 */
export const modelsStrings = {
  "models.browserTitle": {
    fr: "Navigateur de modèles et téléchargement",
    en: "Model Browser & Downloader",
    de: "Model-Browser & Downloader",
    it: "Browser e downloader dei modelli",
  },
  "models.browserDesc": {
    fr: "Recherchez des modèles quantifiés sur Hugging Face et installez-les directement ici.",
    en: "Search Hugging Face for quantised models and install them straight from here.",
    de: "Durchsuchen Sie Hugging Face nach quantisierten Modellen und installieren Sie sie direkt hier.",
    it: "Cerca su Hugging Face i modelli quantizzati e installali direttamente qui.",
  },
  "models.searchPlaceholder": {
    fr: "Rechercher un modèle…",
    en: "Search models…",
    de: "Modelle suchen…",
    it: "Cerca modelli…",
  },
  "models.searchLabel": {
    fr: "Rechercher un modèle",
    en: "Search models",
    de: "Modelle durchsuchen",
    it: "Cerca modelli",
  },
  "models.filterOrg": {
    fr: "Organisation",
    en: "Organisation",
    de: "Organisation",
    it: "Organizzazione",
  },
  "models.filterArchitecture": {
    fr: "Architecture",
    en: "Architecture",
    de: "Architektur",
    it: "Architettura",
  },
  "models.filterQuantization": {
    fr: "Quantification",
    en: "Quantisation",
    de: "Quantisierung",
    it: "Quantizzazione",
  },
  "models.filterType": {
    fr: "Type",
    en: "Type",
    de: "Typ",
    it: "Tipo",
  },
  "models.filterAll": {
    fr: "Tous",
    en: "All",
    de: "Alle",
    it: "Tutti",
  },
  "models.sortBy": {
    fr: "Trier par",
    en: "Sort by",
    de: "Sortieren nach",
    it: "Ordina per",
  },
  "models.sort.downloads": {
    fr: "Téléchargements",
    en: "Downloads",
    de: "Downloads",
    it: "Download",
  },
  "models.sort.likes": {
    fr: "J'aime",
    en: "Likes",
    de: "Likes",
    it: "Mi piace",
  },
  "models.sort.lastModified": {
    fr: "Récemment modifiés",
    en: "Recently updated",
    de: "Zuletzt aktualisiert",
    it: "Aggiornati di recente",
  },
  "models.kind.diffusion": {
    fr: "Générateur",
    en: "Generator",
    de: "Generator",
    it: "Generatore",
  },
  "models.kind.lora": {
    fr: "LoRA",
    en: "LoRA",
    de: "LoRA",
    it: "LoRA",
  },
  "models.kind.upscaler": {
    fr: "Upscaler",
    en: "Upscaler",
    de: "Upscaler",
    it: "Upscaler",
  },
  "models.alphaTag": {
    fr: "Alpha natif",
    en: "Native alpha",
    de: "Natives Alpha",
    it: "Alpha nativo",
  },
  "models.downloads": {
    fr: "téléch.",
    en: "downloads",
    de: "Downloads",
    it: "download",
  },
  "models.searching": {
    fr: "Recherche en cours…",
    en: "Searching…",
    de: "Wird gesucht…",
    it: "Ricerca in corso…",
  },
  "models.resultsCount": {
    fr: "{count} modèle(s)",
    en: "{count} model(s)",
    de: "{count} Modell(e)",
    it: "{count} modello/i",
  },
  "models.noResults": {
    fr: "Aucun modèle ne correspond à ces filtres.",
    en: "No models match these filters.",
    de: "Keine Modelle entsprechen diesen Filtern.",
    it: "Nessun modello corrisponde a questi filtri.",
  },
  "models.download": {
    fr: "Télécharger",
    en: "Download",
    de: "Herunterladen",
    it: "Scarica",
  },
  "models.downloading": {
    fr: "Téléchargement…",
    en: "Downloading…",
    de: "Wird heruntergeladen…",
    it: "Download in corso…",
  },
  "models.preparing": {
    fr: "Préparation…",
    en: "Preparing…",
    de: "Wird vorbereitet…",
    it: "Preparazione…",
  },
  "models.cancel": {
    fr: "Annuler",
    en: "Cancel",
    de: "Abbrechen",
    it: "Annulla",
  },
  "models.installed": {
    fr: "Installé",
    en: "Installed",
    de: "Installiert",
    it: "Installato",
  },
  "models.installedBadge": {
    fr: "INSTALLÉ",
    en: "INSTALLED",
    de: "INSTALLIERT",
    it: "INSTALLATO",
  },
  "models.alreadyHave": {
    fr: "Vous avez déjà {model}",
    en: "You already have {model}",
    de: "{model} ist bereits vorhanden",
    it: "Hai già {model}",
  },
  "models.civitaiSort.downloads": {
    fr: "Plus téléchargés",
    en: "Most downloaded",
    de: "Meist heruntergeladen",
    it: "Più scaricati",
  },
  "models.civitaiSort.rated": {
    fr: "Mieux notés",
    en: "Highest rated",
    de: "Bestbewertet",
    it: "Più votati",
  },
  "models.civitaiSort.newest": {
    fr: "Nouveautés",
    en: "Newest",
    de: "Neueste",
    it: "Più recenti",
  },
  "models.civitaiSort.liked": {
    fr: "Plus aimés",
    en: "Most liked",
    de: "Meistgeliked",
    it: "Più apprezzati",
  },
  "models.notRunnable": {
    fr: "Non exécutable ici",
    en: "Not runnable here",
    de: "Hier nicht ausführbar",
    it: "Non eseguibile qui",
  },
  "models.useFor": {
    fr: "Utiliser pour {model}",
    en: "Use for {model}",
    de: "Für {model} verwenden",
    it: "Usa per {model}",
  },
  "models.stopUsing": {
    fr: "Cesser d'utiliser {model}",
    en: "Stop using {model}",
    de: "{model} nicht mehr verwenden",
    it: "Smetti di usare {model}",
  },
  "models.openOnHf": {
    fr: "Voir sur Hugging Face",
    en: "View on Hugging Face",
    de: "Auf Hugging Face ansehen",
    it: "Vedi su Hugging Face",
  },
};

export default modelsStrings;
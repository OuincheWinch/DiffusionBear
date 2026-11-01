// SourceBrowser.jsx and LicenseBadges.jsx: where a model came from, and what its
// creator allows you to do with it.
//
// The permission strings are deliberately phrased as what the CREATOR ALLOWS, not as
// what the model restricts. "Derivatives allowed" reads as permission; "derivatives
// disallowed" would read as a warning the app has no standing to issue -- the creator
// set the term, and the honest thing next to a badge is the term itself.
//
// "sources.license" is the singular form. licences.title ("Licences") is the page
// heading; a badge sitting on one model is one licence.

export const sourcesStrings = {
  "sources.actionsLabel": {
    fr: "Actions sur l'image",
    en: "Image actions",
    de: "Bildaktionen",
    it: "Azioni sull'immagine",
  },
  "sources.importOnImportTab": {
    fr: "À importer dans l'onglet Import",
    en: "Import from the Import tab",
    de: "Über den Tab „Importieren“ importieren",
    it: "Importa dalla scheda Importa",
  },
  "sources.scale": {
    fr: "Intensité",
    en: "Scale",
    de: "Stärke",
    it: "Intensità",
  },
  "sources.installed": {
    fr: "Installés",
    en: "Installed",
    de: "Installiert",
    it: "Installati",
  },
  "sources.importTab": {
    fr: "Importer",
    en: "Import",
    de: "Importieren",
    it: "Importa",
  },
  "sources.title": {
    fr: "Parcourir les modèles",
    en: "Browse models",
    de: "Modelle durchsuchen",
    it: "Sfoglia i modelli",
  },
  "sources.searchPlaceholder.huggingface": {
    fr: "Rechercher sur Hugging Face",
    en: "Search on Hugging Face",
    de: "Auf Hugging Face suchen",
    it: "Cerca su Hugging Face",
  },
  "sources.searchPlaceholder.civitai": {
    fr: "Rechercher sur Civitai",
    en: "Search on Civitai",
    de: "Auf Civitai suchen",
    it: "Cerca su Civitai",
  },
  "sources.searchPlaceholder.drawthings": {
    fr: "Filtrer par nom",
    en: "Filter by name",
    de: "Nach Namen filtern",
    it: "Filtra per nome",
  },
  "sources.searchPlaceholder.diffusionbee": {
    fr: "Filtrer par nom",
    en: "Filter by name",
    de: "Nach Namen filtern",
    it: "Filtra per nome",
  },
  "sources.import": {
    fr: "Importer",
    en: "Import",
    de: "Importieren",
    it: "Importa",
  },
  "sources.importing": {
    fr: "Import en cours…",
    en: "Importing…",
    de: "Import läuft…",
    it: "Importazione in corso…",
  },
  "sources.imported": {
    fr: "Importé",
    en: "Imported",
    de: "Importiert",
    it: "Importato",
  },
  "sources.baseModel": {
    fr: "Modèle de base",
    en: "Base model",
    de: "Basismodell",
    it: "Modello di base",
  },
  "sources.detected": {
    fr: "{count} fichiers détectés",
    en: "{count} files detected",
    de: "{count} Dateien erkannt",
    it: "{count} file rilevati",
  },
  "sources.fluxNoModelWarning": {
    fr: "Cette application n'a aucun modèle FLUX.1 : une LoRA FLUX ne pourra pas servir à générer.",
    en: "This app has no FLUX.1 model, so a FLUX LoRA cannot be used to generate.",
    de: "Diese App hat kein FLUX.1-Modell, daher kann eine FLUX-LoRA nicht zum Generieren verwendet werden.",
    it: "Questa app non ha modelli FLUX.1, quindi una LoRA FLUX non può essere usata per generare.",
  },
  "sources.skippedTensors": {
    fr: "{count} tenseurs ignorés : aucun module cible ne leur correspond.",
    en: "{count} tensors skipped: no target module matches them.",
    de: "{count} Tensoren übersprungen: kein Zielmodul passt dazu.",
    it: "{count} tensori ignorati: nessun modulo di destinazione corrisponde.",
  },
  "sources.matchCivitai": {
    fr: "Chercher sur Civitai",
    en: "Search on Civitai",
    de: "Auf Civitai suchen",
    it: "Cerca su Civitai",
  },
  "sources.matchByUrl": {
    fr: "Ou collez l'URL de la version",
    en: "Or paste the version URL",
    de: "Oder Versions-URL einfügen",
    it: "Oppure incolla l'URL della versione",
  },
  "sources.noMatch": {
    fr: "Aucune correspondance trouvée",
    en: "No match found",
    de: "Keine Übereinstimmung gefunden",
    it: "Nessuna corrispondenza trovata",
  },
  "sources.modelsComingSoon": {
    fr: "Modèles complets : bientôt",
    en: "Full models: coming soon",
    de: "Vollständige Modelle: demnächst",
    it: "Modelli completi: presto",
  },
  "sources.review": {
    fr: "Vérifier avant d'importer",
    en: "Review before importing",
    de: "Vor dem Import prüfen",
    it: "Controlla prima di importare",
  },
  "sources.name": {
    fr: "Nom",
    en: "Name",
    de: "Name",
    it: "Nome",
  },
  "sources.trigger": {
    fr: "Mot d'activation",
    en: "Trigger word",
    de: "Triggerwort",
    it: "Parola chiave",
  },
  "sources.confirm": {
    fr: "Confirmer l'import",
    en: "Confirm import",
    de: "Import bestätigen",
    it: "Conferma importazione",
  },
  "sources.cancel": {
    fr: "Annuler",
    en: "Cancel",
    de: "Abbrechen",
    it: "Annulla",
  },
  "sources.overwriteRefused": {
    fr: "Un fichier porte déjà ce nom. Rien n'a été écrasé.",
    en: "A file with that name already exists. Nothing was overwritten.",
    de: "Eine Datei mit diesem Namen existiert bereits. Es wurde nichts überschrieben.",
    it: "Esiste già un file con quel nome. Nulla è stato sovrascritto.",
  },
  "sources.license": {
    fr: "Licence",
    en: "License",
    de: "Lizenz",
    it: "Licenza",
  },
  "sources.commercialUse": {
    fr: "Usage commercial",
    en: "Commercial use",
    de: "Kommerzielle Nutzung",
    it: "Uso commerciale",
  },
  "sources.derivatives": {
    fr: "Dérivés autorisés",
    en: "Derivatives allowed",
    de: "Ableitungen erlaubt",
    it: "Derivati consentiti",
  },
  "sources.creditRequired": {
    fr: "Crédit obligatoire",
    en: "Credit required",
    de: "Namensnennung erforderlich",
    it: "Credito obbligatorio",
  },
  "sources.differentLicense": {
    fr: "Licence différente autorisée",
    en: "Different license allowed",
    de: "Andere Lizenz erlaubt",
    it: "Licenza diversa consentita",
  },
  "sources.viewOnSource": {
    fr: "Voir sur la source",
    en: "View on source",
    de: "Auf der Quelle ansehen",
    it: "Vedi alla fonte",
  },
  // Deliberately says the SOURCE did not say. It does not guess a licence, and it does
  // not read as though the model is unlicensed -- which is a claim nobody here can make.
  "sources.unknownLicense": {
    fr: "Licence non indiquée par la source",
    en: "License not stated by the source",
    de: "Lizenz von der Quelle nicht angegeben",
    it: "Licenza non indicata dalla fonte",
  },
  "sources.noPermissions": {
    fr: "Permissions d'usage non indiquées par la source",
    en: "Usage permissions not stated by the source",
    de: "Nutzungsrechte von der Quelle nicht angegeben",
    it: "Permessi d'uso non indicati dalla fonte",
  },
  "sources.fetchFailed": {
    fr: "Impossible de consulter la licence",
    en: "Could not check the license",
    de: "Lizenz konnte nicht geprüft werden",
    it: "Impossibile consultare la licenza",
  },
};

export default sourcesStrings;
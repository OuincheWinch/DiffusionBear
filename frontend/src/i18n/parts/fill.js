// FillBrush.jsx. English-first like downloader.js and settings.js.
export const fillStrings = {
  "fill.title": {
    fr: "Remplissage génératif",
    en: "Generative fill",
    de: "Generatives Füllen",
    it: "Riempimento generativo",
  },
  "fill.canvasLabel": {
    fr: "Zone à régénérer",
    en: "Region to regenerate",
    de: "Zu regenerierender Bereich",
    it: "Area da rigenerare",
  },
  "fill.prompt": { fr: "Prompt", en: "Prompt", de: "Prompt", it: "Prompt" },
  "fill.promptPlaceholder": {
    fr: "Décrivez ce qui doit apparaître dans la zone peinte…",
    en: "Describe what should appear in the painted area…",
    de: "Beschreiben Sie, was im gemalten Bereich erscheinen soll…",
    it: "Descrivi cosa deve apparire nell'area dipinta…",
  },
  "fill.tools": { fr: "Outils", en: "Tools", de: "Werkzeuge", it: "Strumenti" },
  "fill.paint": { fr: "Peindre", en: "Paint", de: "Malen", it: "Dipingi" },
  "fill.erase": { fr: "Gommer", en: "Erase", de: "Radieren", it: "Cancella" },
  "fill.clear": { fr: "Tout effacer", en: "Clear", de: "Löschen", it: "Svuota" },
  "fill.brushSize": { fr: "Taille du pinceau", en: "Brush size", de: "Pinselgröße", it: "Dimensione pennello" },
  "fill.engine": { fr: "Moteur", en: "Engine", de: "Engine", it: "Motore" },
  "fill.coverage": {
    fr: "{percent} % de l\'image sélectionnée",
    en: "{percent}% of the image selected",
    de: "{percent} % des Bildes ausgewählt",
    it: "{percent}% dell\'immagine selezionata",
  },
  "fill.hint": {
    fr: "Peignez la zone à régénérer, puis décrivez ce que vous voulez à la place.",
    en: "Paint the region to regenerate, then describe what you want there.",
    de: "Malen Sie den zu regenerierenden Bereich und beschreiben Sie dann, was dort sein soll.",
    it: "Dipingi l\'area da rigenerare, poi descrivi cosa vuoi che ci sia.",
  },
  "fill.hintLargeRegion": {
    fr: "Zone large : le résultat peut diverger davantage du reste de l\'image, car le moteur régénère tout puis recompose.",
    en: "Large region: the result may diverge more from the rest of the image, because the engine regenerates everything and then recomposites.",
    de: "Großer Bereich: Das Ergebnis kann stärker vom Rest des Bildes abweichen, da die Engine alles neu erzeugt und dann zusammensetzt.",
    it: "Area grande: il risultato può divergere di più dal resto dell\'immagine, perché il motore rigenera tutto e poi ricompone.",
  },
  "fill.submit": {
    fr: "Générer dans la zone",
    en: "Fill this area",
    de: "Bereich füllen",
    it: "Riempi quest\'area",
  },
  "fill.working": {
    fr: "Génération…",
    en: "Generating…",
    de: "Wird erzeugt…",
    it: "Generazione…",
  },
  "fill.errorInvalid": {
    fr: "La requête a été refusée. Vérifiez que la zone peinte correspond à l\'image.",
    en: "The request was refused. Check that the painted area matches the image.",
    de: "Die Anfrage wurde abgelehnt. Prüfen Sie, ob der gemalte Bereich zum Bild passt.",
    it: "La richiesta è stata rifiutata. Verifica che l\'area dipinta corrisponda all\'immagine.",
  },
  "fill.errorEngines": {
    fr: "Impossible de charger la liste des moteurs. Réessayez.",
    en: "Could not load the engine list. Try again.",
    de: "Die Engine-Liste konnte nicht geladen werden. Erneut versuchen.",
    it: "Impossibile caricare l\'elenco dei motori. Riprova.",
  },
"fill.errorFailed": {
      fr: "Le remplissage a échoué. Réessayez.",
      en: "The fill failed. Try again.",
      de: "Das Füllen ist fehlgeschlagen. Erneut versuchen.",
      it: "Il riempimento non è riuscito. Riprova.",
    },
    "fill.cancel": {
      fr: "Annuler",
      en: "Cancel",
      de: "Abbrechen",
      it: "Annulla",
    },
    "fill.cancelled": {
      fr: "Remplissage annulé.",
      en: "Fill cancelled.",
      de: "Füllen abgebrochen.",
      it: "Riempimento annullato.",
    },
    "fill.progressEta": {
      fr: "encore ~{seconds} s",
      en: "~{seconds}s left",
      de: "noch ~{seconds} s",
      it: "~{seconds} s rimanenti",
    },
  "fill.doneBanner": {
    fr: "Remplissage terminé — l\'image a été ajoutée à la galerie.",
    en: "Fill complete — the image has been added to the gallery.",
    de: "Füllen abgeschlossen — das Bild wurde zur Galerie hinzugefügt.",
    it: "Riempimento completato — l\'immagine è stata aggiunta alla galleria.",
  },
  "fill.cardButton": {
    fr: "Remplir une zone",
    en: "Fill a region",
    de: "Bereich füllen",
    it: "Riempi un\'area",
  },
};

export default fillStrings;

// GenerationStack.jsx
export const stackStrings = {
  "stack.status.queued": { fr: "⏳ En file", en: "⏳ Queued", de: "⏳ In Warteschlange", it: "⏳ In coda" },
  "stack.status.generating": { fr: "⚙ Génération", en: "⚙ Generating", de: "⚙ Generierung", it: "⚙ Generazione" },
  "stack.status.done": { fr: "✓ Terminé", en: "✓ Done", de: "✓ Fertig", it: "✓ Completato" },
  "stack.status.error": { fr: "✗ Erreur", en: "✗ Error", de: "✗ Fehler", it: "✗ Errore" },
  "stack.status.cancelled": { fr: "Annulé", en: "Cancelled", de: "Abgebrochen", it: "Annullato" },

  "stack.heading": {
    fr: "File de génération ({count})",
    en: "Generation stack ({count})",
    de: "Generierungsstapel ({count})",
    it: "Coda di generazione ({count})",
  },
  "stack.killCurrentBtn": {
    fr: "✕ Interrompre l'actuelle",
    en: "✕ Kill current",
    de: "✕ Aktuelle abbrechen",
    it: "✕ Interrompi l'attuale",
  },
  "stack.emptyQueueBtn": {
    fr: "⌫ Vider la file",
    en: "⌫ Empty queue",
    de: "⌫ Warteschlange leeren",
    it: "⌫ Svuota la coda",
  },
  "stack.emptyQueueTitle": {
    fr: "Retirer tous les travaux en attente de la file",
    en: "Remove every waiting job from the queue",
    de: "Alle wartenden Jobs aus der Warteschlange entfernen",
    it: "Rimuovi dalla coda tutti i lavori in attesa",
  },
  "stack.recoveryTriggerTitle": {
    fr: "Consulter et réinsérer les prompts annulés ou interrompus",
    en: "Review and re-queue cancelled or interrupted prompts",
    de: "Abgebrochene Prompts ansehen und erneut einreihen",
    it: "Rivedi e reinserisci i prompt annullati o interrotti",
  },
  "stack.recoveryTriggerOne": {
    fr: "↺ {count} récupérable",
    en: "↺ {count} recoverable",
    de: "↺ {count} wiederherstellbar",
    it: "↺ {count} recuperabile",
  },
  "stack.recoveryTriggerMany": {
    fr: "↺ {count} récupérables",
    en: "↺ {count} recoverable",
    de: "↺ {count} wiederherstellbar",
    it: "↺ {count} recuperabili",
  },
  "stack.crashAlertOne": {
    fr: "⚡ {count} génération interrompue lors de la dernière session",
    en: "⚡ {count} generation interrupted during the last session",
    de: "⚡ {count} Generation in der letzten Sitzung unterbrochen",
    it: "⚡ {count} generazione interrotta durante l'ultima sessione",
  },
  "stack.crashAlertMany": {
    fr: "⚡ {count} générations interrompues lors de la dernière session",
    en: "⚡ {count} generations interrupted during the last session",
    de: "⚡ {count} Generationen in der letzten Sitzung unterbrochen",
    it: "⚡ {count} generazioni interrotte durante l'ultima sessione",
  },
  "stack.crashAlertBtn": {
    fr: "Voir & Restaurer",
    en: "View & Restore",
    de: "Ansehen & Wiederherstellen",
    it: "Vedi e Ripristina",
  },

  "stack.phase.downloading": { fr: "📥 Téléchargement", en: "📥 Downloading", de: "📥 Wird geladen", it: "📥 Download" },
  "stack.phase.loadingMemory": { fr: "🧠 Chargement mémoire", en: "🧠 Loading memory", de: "🧠 Lädt in den Speicher", it: "🧠 Caricamento in memoria" },
  "stack.phase.compiling": { fr: "⚡ Compilation", en: "⚡ Compiling", de: "⚡ Kompiliert", it: "⚡ Compilazione" },
  "stack.phase.finalizing": { fr: "🎨 Finalisation", en: "🎨 Finalizing", de: "🎨 Finalisierung", it: "🎨 Finalizzazione" },

  "stack.job.cancelTitle": { fr: "Annuler ce travail", en: "Cancel this job", de: "Diesen Job abbrechen", it: "Annulla questo lavoro" },
  "stack.job.completedIn": {
    fr: "Terminé en {time} s ({steps} étapes",
    en: "Completed in {time}s ({steps} steps",
    de: "Fertig in {time} s ({steps} Schritte",
    it: "Completato in {time} s ({steps} passaggi",
  },
  "stack.job.completedBatch": {
    fr: " · {batch} images)",
    en: " · {batch} images)",
    de: " · {batch} Bilder)",
    it: " · {batch} immagini)",
  },
  "stack.job.decoding": {
    fr: "Décodage de l'image (VAE) · {elapsed} s écoulées",
    en: "Decoding image (VAE) · {elapsed}s elapsed",
    de: "Bild wird dekodiert (VAE) · {elapsed} s vergangen",
    it: "Decodifica immagine (VAE) · {elapsed} s trascorsi",
  },
  "stack.job.stepElapsed": {
    fr: "Étape {step}/{steps} · {elapsed} s écoulées",
    en: "Step {step}/{steps} · {elapsed}s elapsed",
    de: "Schritt {step}/{steps} · {elapsed} s vergangen",
    it: "Passo {step}/{steps} · {elapsed} s trascorsi",
  },
  "stack.job.eta": { fr: " · reste ~{eta} s", en: " · ETA ~{eta}s", de: " · ca. {eta} s", it: " · ~{eta} s rimanenti" },
  "stack.job.imageProgress": {
    fr: " · image {index}/{batch}",
    en: " · image {index}/{batch}",
    de: " · Bild {index}/{batch}",
    it: " · immagine {index}/{batch}",
  },
  "stack.job.waitingInQueue": { fr: "En attente dans la file", en: "Waiting in queue", de: "Wartet in der Schlange", it: "In attesa nella coda" },
  "stack.job.waitingBatch": {
    fr: " · lot de {batch} images",
    en: " · batch of {batch} images",
    de: " · Stapel von {batch} Bildern",
    it: " · lotto di {batch} immagini",
  },
  "stack.job.phaseDownloading": {
    fr: "Téléchargement des poids…",
    en: "Downloading weights…",
    de: "Gewichte werden geladen…",
    it: "Download dei pesi…",
  },
  "stack.job.phaseLoadingWeights": {
    fr: "Chargement des poids en mémoire unifiée…",
    en: "Loading model weights into unified memory…",
    de: "Gewichte werden in den vereinheitlichten Speicher geladen…",
    it: "Caricamento dei pesi in memoria unificata…",
  },
};

export default stackStrings;

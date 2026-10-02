// French is the reference language, not English. The UI was written in French and
// reviewed in French, so those strings are the ones the author actually intended.
// English is a translation of them, not the other way round.
//
// Shape: { key: { fr, en, de, it } }. Every key must exist in all four languages;
// test_i18n.py fails the build if one is missing or empty, because a missing key
// used to surface in the UI as a raw identifier like "gallery.empty".

/**
 * Per-component catalogues live in ./parts/*.js and are merged in below. Splitting
 * them out means two people (or two agents) can translate different components
 * without touching the same file, which is what made this tractable: the whole
 * surface is ~600 strings and a single file was a merge conflict magnet.
 *
 * @type {Record<string, {fr: string, en: string, de: string, it: string}>}
 */
export const STRINGS = {
  // ---------------------------------------------------------------- app shell
  "app.generate": {
    fr: "Générer",
    en: "Generate",
    de: "Erzeugen",
    it: "Genera",
  },
  "app.cancel": {
    fr: "Annuler",
    en: "Cancel",
    de: "Abbrechen",
    it: "Annulla",
  },
  "app.close": {
    fr: "Fermer",
    en: "Close",
    de: "Schließen",
    it: "Chiudi",
  },
  "app.copy": {
    fr: "Copier",
    en: "Copy",
    de: "Kopieren",
    it: "Copia",
  },
  "app.copied": {
    fr: "Copié",
    en: "Copied",
    de: "Kopiert",
    it: "Copiato",
  },
  "app.retry": {
    fr: "Réessayer",
    en: "Retry",
    de: "Erneut versuchen",
    it: "Riprova",
  },
  "app.dismiss": {
    fr: "Ignorer",
    en: "Dismiss",
    de: "Ausblenden",
    it: "Ignora",
  },

  // ------------------------------------------------------------- language picker
  "language.title": {
    fr: "Langue de l'interface",
    en: "Interface language",
    de: "Oberflächensprache",
    it: "Lingua dell'interfaccia",
  },
  "language.auto": {
    fr: "Suivre le système",
    en: "Match system",
    de: "Systemeinstellung folgen",
    it: "Segui il sistema",
  },
  "language.pending": {
    fr: "Traduction en cours — l'interface s'affiche en anglais",
    en: "Translation in progress — the interface shows English",
    de: "Übersetzung läuft — die Oberfläche zeigt Englisch",
    it: "Traduzione in corso — l'interfaccia mostra l'inglese",
  },

  // -------------------------------------------------------------- generation stack

  // ------------------------------------------------------------- recovery drawer
  "queue.recoveryTitle": {
    fr: "File d'attente récupérable",
    en: "Recoverable queue",
    de: "Wiederherstellbare Warteschlange",
    it: "Coda recuperabile",
  },
  "queue.recoverySubtitle": {
    fr: "Prompts sauvegardés suite à une annulation de file ou un arrêt imprévu de la machine.",
    en: "Prompts saved after a queue cancellation or an unexpected shutdown.",
    de: "Prompts, die nach einem Abbruch der Warteschlange oder einem unerwarteten Herunterfahren gespeichert wurden.",
    it: "Prompt salvati dopo un annullamento della coda o uno spegnimento improvviso.",
  },
  "queue.promptCount": {
    fr: "prompt",
    en: "prompt",
    de: "Prompt",
    it: "prompt",
  },
  "queue.promptsCount": {
    fr: "prompts",
    en: "prompts",
    de: "Prompts",
    it: "prompt",
  },
  "queue.restoreAll": {
    fr: "Tout réinsérer dans la file",
    en: "Re-queue all",
    de: "Alle erneut einreihen",
    it: "Reinserisci tutto",
  },
  "queue.copyAll": {
    fr: "Copier tous les prompts",
    en: "Copy all prompts",
    de: "Alle Prompts kopieren",
    it: "Copia tutti i prompt",
  },
  "queue.clearAll": {
    fr: "Tout effacer",
    en: "Clear all",
    de: "Alle löschen",
    it: "Elimina tutto",
  },
  "queue.clearAndForget": {
    fr: "Effacer et oublier",
    en: "Clear and forget",
    de: "Löschen und vergessen",
    it: "Elimina e dimentica",
  },
  "queue.restore": {
    fr: "Réinsérer",
    en: "Re-queue",
    de: "Erneut einreihen",
    it: "Reinserisci",
  },
  "queue.load": {
    fr: "Charger",
    en: "Load",
    de: "Laden",
    it: "Carica",
  },
  "queue.clearAllConfirm": {
    fr: "Effacer définitivement l'historique de récupération ?",
    en: "Permanently clear the recovery history?",
    de: "Wiederherstellungsverlauf endgültig löschen?",
    it: "Eliminare definitivamente la cronologia di ripristino?",
  },
  "queue.clearAndForgetConfirm": {
    fr: "Effacer ET oublier ?\n\nCela supprime aussi les jobs encore en attente pour qu'aucun ne soit restauré au prochain démarrage. Action définitive.",
    en: "Clear AND forget?\n\nThis also removes the jobs still waiting, so none of them can be restored at the next start. This cannot be undone.",
    de: "Löschen UND vergessen?\n\nDamit werden auch die noch wartenden Jobs entfernt, sodass beim nächsten Start keiner wiederhergestellt wird. Endgültig.",
    it: "Eliminare E dimenticare?\n\nVengono rimossi anche i lavori ancora in attesa, così nessuno verrà ripristinato al prossimo avvio. Operazione definitiva.",
  },
  "queue.deleteFailed": {
    fr: "Échec de la suppression. Vérifiez que le serveur est bien démarré, puis réessayez.",
    en: "Deletion failed. Check that the server is running, then try again.",
    de: "Löschen fehlgeschlagen. Prüfen Sie, ob der Server läuft, und versuchen Sie es erneut.",
    it: "Eliminazione non riuscita. Verifica che il server sia in esecuzione, poi riprova.",
  },
  "queue.restoreFailed": {
    fr: "Échec de la réinsertion.",
    en: "Re-queueing failed.",
    de: "Erneutes Einreihen fehlgeschlagen.",
    it: "Reinserimento non riuscito.",
  },

  // ------------------------------------------------------------------- licences
  "licences.title": {
    fr: "Licences",
    en: "Licences",
    de: "Lizenzen",
    it: "Licenze",
  },
  "licences.intro": {
    fr: "Ce que vous pouvez et ne pouvez pas faire avec DiffusionBear, et les projets auxquels il s'appuie.",
    en: "What you may and may not do with DiffusionBear, and which projects it builds on.",
    de: "Was Sie mit DiffusionBear tun dürfen und was nicht, sowie die Projekte, auf denen es aufbaut.",
    it: "Cosa puoi e cosa non puoi fare con DiffusionBear, e su quali progetti si basa.",
  },
  "licences.source": {
    fr: "Source :",
    en: "Source:",
    de: "Quelle:",
    it: "Fonte:",
  },

  // -------------------------------------------------------------------- generic

  // ------------------------------------------------------------- recovery list
  "queue.empty": {
    fr: "Aucun prompt en attente de récupération.",
    en: "No prompts waiting to be recovered.",
    de: "Keine Prompts warten auf Wiederherstellung.",
    it: "Nessun prompt in attesa di ripristino.",
  },
  "queue.emptyHint": {
    fr: "Vos générations annulées ou interrompues s'archiveront automatiquement ici.",
    en: "Your cancelled or interrupted generations will be archived here automatically.",
    de: "Ihre abgebrochenen oder unterbrochenen Generationen werden hier automatisch archiviert.",
    it: "Le tue generazioni annullate o interrotte verranno archiviate qui automaticamente.",
  },
  "queue.interruptedBadge": {
    fr: "Interrompu (Crash/Arrêt)",
    en: "Interrupted (crash/shutdown)",
    de: "Unterbrochen (Absturz/Herunterfahren)",
    it: "Interrotto (crash/spgnimento)",
  },
  "queue.cancelledBadge": {
    fr: "Annulé",
    en: "Cancelled",
    de: "Abgebrochen",
    it: "Annullato",
  },
  "queue.defaultModel": {
    fr: "modèle standard",
    en: "default model",
    de: "Standardmodell",
    it: "modello predefinito",
  },
  "queue.steps": {
    fr: "étapes",
    en: "steps",
    de: "Schritte",
    it: "passaggi",
  },
  "queue.requeueOneTitle": {
    fr: "Réinsérer ce prompt directement dans la file",
    en: "Re-queue this prompt right away",
    de: "Diesen Prompt sofort erneut einreihen",
    it: "Reinserisci subito questo prompt",
  },
  "queue.loadTitle": {
    fr: "Charger ce prompt et ses réglages dans le formulaire",
    en: "Load this prompt and its settings into the form",
    de: "Diesen Prompt mit seinen Einstellungen ins Formular laden",
    it: "Carica questo prompt e le sue impostazioni nel modulo",
  },
  "queue.copyTitle": {
    fr: "Copier le texte du prompt",
    en: "Copy the prompt text",
    de: "Prompt-Text kopieren",
    it: "Copia il testo del prompt",
  },
  "queue.deleteOneTitle": {
    fr: "Supprimer définitivement ce prompt de la liste de récupération",
    en: "Permanently delete this prompt from the recovery list",
    de: "Diesen Prompt endgültig aus der Wiederherstellungsliste löschen",
    it: "Elimina definitivamente questo prompt dalla lista di ripristino",
  },
  "queue.forgetTitle": {
    fr: "Efface aussi la file en attente : rien ne sera restauré au prochain démarrage",
    en: "Also clears the waiting queue: nothing will be restored at the next start",
    de: "Leert auch die wartende Schlange: beim nächsten Start wird nichts wiederhergestellt",
    it: "Svuota anche la coda in attesa: al prossimo avvio non verrà ripristinato nulla",
  },
  "queue.copiedAll": {
    fr: "✓ Copiés !",
    en: "✓ Copied!",
    de: "✓ Kopiert!",
    it: "✓ Copiati!",
  },
  "queue.clipboardFailed": {
    fr: "Copie impossible. Le presse-papiers est inaccessible.",
    en: "Could not copy. The clipboard is unavailable.",
    de: "Kopieren nicht möglich. Zwischenablage nicht verfügbar.",
    it: "Copia non riuscita. Appunti non disponibili.",
  },
};

import { generateStrings } from "./parts/generate.js";
import { paramsStrings } from "./parts/params.js";
import { stackStrings } from "./parts/stack.js";
import { canvasStrings } from "./parts/canvas.js";
import { galleryStrings } from "./parts/gallery.js";
import { loraStrings } from "./parts/lora.js";
import { settingsStrings } from "./parts/settings.js";
import { downloaderStrings } from "./parts/downloader.js";
import { licencesStrings } from "./parts/licences.js";
import { installerStrings } from "./parts/installer.js";
import { tokensStrings } from "./parts/tokens.js";
import { appStrings } from "./parts/app.js";
import { modelsStrings } from "./parts/models.js";
import { fillStrings } from "./parts/fill.js";

// A duplicate key means two files claim the same string, which makes the
// translation silently depend on import order. test_i18n.py fails on collisions.
for (const part of [
  generateStrings,
  paramsStrings,
  stackStrings,
  canvasStrings,
  galleryStrings,
  loraStrings,
  settingsStrings,
  downloaderStrings,
  installerStrings,
  licencesStrings,
  tokensStrings,
  appStrings,
  fillStrings,
  modelsStrings,
]) {
  for (const [key, value] of Object.entries(part)) {
    if (key in STRINGS) {
      throw new Error(`duplicate i18n key: ${key}`);
    }
    STRINGS[key] = value;
  }
}

// Full per-language overlays for languages added after the four shipped ones.
//
// They live in ./lang/<code>.js rather than ./parts/ for two reasons. parts/*.js is
// globbed as a DISJOINT catalogue of the same four languages, so a file listing all 739
// keys there would collide with every other part; and a translation is per-language
// data, not a per-component slice.
//
// Each entry is folded INTO the shipped four rather than replacing them, so translate.js
// keeps its single `entry[lang] ?? entry.en` lookup and needs no changes. An untranslated
// key is simply absent here, which is what makes it fall back to English -- so a
// half-finished translation degrades to English instead of going blank.
import { LANGUAGES as _LANGUAGES } from "./languages";
import { esStrings } from "./lang/es";
import { zhStrings } from "./lang/zh";
import { jaStrings } from "./lang/ja";
import { ptStrings } from "./lang/pt";
import { koStrings } from "./lang/ko";

const OVERLAYS = {
  es: esStrings,
  zh: zhStrings,
  ja: jaStrings,
  pt: ptStrings,
  ko: koStrings,
};

for (const [code, overlay] of Object.entries(OVERLAYS)) {
  const unknown = [];
  for (const [key, text] of Object.entries(overlay)) {
    if (!(key in STRINGS)) {
      // A stale key would otherwise sit here forever, invisible.
      unknown.push(key);
      continue;
    }
    // null or "" means "not translated". Assigning either would SHADOW the English
    // fallback, because translate.js falls back with ?? and "" is a real value.
    if (typeof text !== "string" || text.trim() === "") continue;
    STRINGS[key][code] = text;
  }
  if (unknown.length) {
    throw new Error(`i18n overlay ${code}.js has keys that no longer exist: ${unknown.join(", ")}`);
  }
}

// Coverage for the picker tooltip. This deliberately does NOT decide whether a language
// counts as shipped: `status` in languages.js is the declaration, and test_i18n.py fails
// if the declaration and the real coverage disagree. Deciding it here would let a
// half-finished translation quietly reclassify itself as done.
for (const lang of _LANGUAGES) {
  if (!OVERLAYS[lang.code]) continue;
  const done = Object.keys(OVERLAYS[lang.code]).filter((k) => OVERLAYS[lang.code][k]).length;
  lang.translationProgress = `${done}/${Object.keys(STRINGS).length}`;
}

export default STRINGS;

// LoraManagerDrawer.jsx. French is the reference language, as in strings.js.
export const loraStrings = {
  "lora.syncingInfo": {
    fr: "Calcul des empreintes et synchronisation de toutes les LoRAs locales avec Civitai…",
    en: "Hashing & syncing all local LoRAs with Civitai…",
    de: "Alle lokalen LoRAs werden gehasht und mit Civitai synchronisiert…",
    it: "Hashing e sincronizzazione di tutte le LoRA locali con Civitai…",
  },
  "lora.syncComplete": {
    fr: "Synchronisation Civitai terminée : {updated} mise à jour sur {total} LoRAs.",
    en: "Civitai sync complete: {updated} updated out of {total} LoRAs.",
    de: "Civitai-Synchronisierung abgeschlossen: {updated} von {total} LoRAs aktualisiert.",
    it: "Sincronizzazione con Civitai completata: {updated} aggiornate su {total} LoRA.",
  },
  "lora.syncFailed": {
    fr: "Échec de la synchronisation : {message}",
    en: "Sync failed: {message}",
    de: "Synchronisierung fehlgeschlagen: {message}",
    it: "Sincronizzazione non riuscita: {message}",
  },
  "lora.maxActive": {
    fr: "16 LoRAs actives au maximum par génération d'image.",
    en: "Maximum of 16 active LoRAs per image generation.",
    de: "Höchstens 16 aktive LoRAs pro Bildgenerierung.",
    it: "Massimo 16 LoRA attive per generazione di immagini.",
  },

  "lora.deleteConfirm": {
    fr: "Supprimer définitivement « {name} » du disque et du registre ?",
    en: "Permanently delete \"{name}\" from disk and registry?",
    de: "„{name}“ endgültig von der Festplatte und aus dem Register löschen?",
    it: "Eliminare definitivamente «{name}» dal disco e dal registro?",
  },
  "lora.deleteSuccess": {
    fr: "« {name} » supprimé avec succès.",
    en: "Deleted \"{name}\" successfully.",
    de: "„{name}“ erfolgreich gelöscht.",
    it: "«{name}» eliminato correttamente.",
  },
  "lora.deleteFailed": {
    fr: "Échec de la suppression : {message}",
    en: "Failed to delete: {message}",
    de: "Löschen fehlgeschlagen: {message}",
    it: "Eliminazione non riuscita: {message}",
  },

  "lora.activeLegend": {
    fr: "LoRAs actives ({count}/16 max)",
    en: "Active LoRAs ({count}/16 max)",
    de: "Aktive LoRAs ({count}/16 max.)",
    it: "LoRA attive ({count}/16 max)",
  },
  "lora.syncTitle": {
    fr: "Analyser les LoRAs locales et récupérer les identifiants Civitai officiels et les mots-clés d'activation",
    en: "Scan local LoRAs and fetch official Civitai IDs & triggers",
    de: "Lokale LoRAs scannen und offizielle Civitai-IDs sowie Trigger abrufen",
    it: "Analizza le LoRA locali e recupera ID Civitai ufficiali e trigger",
  },
  "lora.syncing": { fr: "⏳ Synchronisation…", en: "⏳ Syncing…", de: "⏳ Wird synchronisiert…", it: "⏳ Sincronizzazione…" },
  "lora.syncBtn": {
    fr: "🔄 Synchroniser avec Civitai",
    en: "🔄 Sync Civitai",
    de: "🔄 Mit Civitai synchronisieren",
    it: "🔄 Sincronizza con Civitai",
  },

  "lora.distillTitle": {
    fr: "Adaptateur de distillation pour 4 étapes ou moins",
    en: "Distillation adapter for 4 steps or fewer",
    de: "Destillations-Adapter für 4 Schritte oder weniger",
    it: "Adattatore di distillazione per 4 passaggi o meno",
  },
  "lora.distillTag": { fr: "⚡ Distillation", en: "⚡ Distill", de: "⚡ Destillation", it: "⚡ Distillazione" },
  "lora.civitaiIdTitle": {
    fr: "Identifiant Civitai : {id} ({version})",
    en: "Civitai ID: {id} ({version})",
    de: "Civitai-ID: {id} ({version})",
    it: "ID Civitai: {id} ({version})",
  },
  "lora.civitaiTagSuffix": {
    fr: "[Civitai n° {id}]",
    en: "[Civitai #{id}]",
    de: "[Civitai #{id}]",
    it: "[Civitai n. {id}]",
  },

  "lora.scaleDecreaseTitle": { fr: "-0.05", en: "-0.05", de: "−0,05", it: "−0,05" },
  "lora.scaleIncreaseTitle": { fr: "+0.05", en: "+0.05", de: "+0,05", it: "+0,05" },

  "lora.registryEmpty": {
    fr: "Aucune LoRA {format} dans le registre",
    en: "No {format} LoRAs in the registry",
    de: "Keine {format}-LoRAs im Register",
    it: "Nessuna LoRA {format} nel registro",
  },
  "lora.addOption": { fr: "+ Ajouter une LoRA…", en: "+ Add LoRA…", de: "+ LoRA hinzufügen…", it: "+ Aggiungi LoRA…" },
  "lora.installedPrefix": { fr: "Installées :", en: "Installed:", de: "Installiert:", it: "Installate:" },
  "lora.addActiveTitle": {
    fr: "Cliquez pour ajouter aux LoRAs actives",
    en: "Click to add to the active LoRAs",
    de: "Klicken, um zu den aktiven LoRAs hinzuzufügen",
    it: "Clicca per aggiungerla alle LoRA attive",
  },
  "lora.deleteChipTitle": {
    fr: "Supprimer {name} du disque et du registre",
    en: "Delete {name} from disk and registry",
    de: "{name} von der Festplatte und aus dem Register löschen",
    it: "Eliminare {name} dal disco e dal registro",
  },

  "lora.hideHub": {
    fr: "▲ Masquer le hub des LoRAs installées",
    en: "▲ Hide installed LoRAs hub",
    de: "▲ Hub der installierten LoRAs ausblenden",
    it: "▲ Nascondi l'hub delle LoRA installate",
  },
  "lora.showHub": {
    fr: "📦 Hub des LoRAs installées ({count} modèles)",
    en: "📦 Installed LoRAs hub ({count} models)",
    de: "📦 Hub der installierten LoRAs ({count} Modelle)",
    it: "📦 Hub delle LoRA installate ({count} modelli)",
  },
  "lora.hubEmpty": {
    fr: "Aucune LoRA installée pour l'instant.",
    en: "No LoRAs installed yet.",
    de: "Noch keine LoRAs installiert.",
    it: "Nessuna LoRA ancora installata.",
  },
  "lora.hubActive": { fr: "✓ Active", en: "✓ Active", de: "✓ Aktiv", it: "✓ Attiva" },
  "lora.hubAdd": { fr: "+ Ajouter", en: "+ Add", de: "+ Hinzufügen", it: "+ Aggiungi" },
  "lora.switchTitle": {
    fr: "Passer le modèle à {model}",
    en: "Switch model to {model}",
    de: "Modell auf {model} umstellen",
    it: "Passa al modello {model}",
  },
  "lora.switchBtn": {
    fr: "⚡ Passer à {model}",
    en: "⚡ Switch to {model}",
    de: "⚡ Zu {model} wechseln",
    it: "⚡ Passa a {model}",
  },
};

export default loraStrings;

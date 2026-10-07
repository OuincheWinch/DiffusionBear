// TokenManager.jsx. Written in English, so the literal is the `en` value.
//
// The two URLs in the placeholders are load-bearing -- the Civitai one carries a
// referral code -- so they are byte-identical in all four languages and only the
// prose around them moves. Reused from strings.js: app.cancel ("Cancel").
export const tokensStrings = {
  "tokens.labelCivitai": {
    fr: "Clé d'API Civitai",
    en: "Civitai API Key",
    de: "Civitai-API-Schlüssel",
    it: "Chiave API Civitai",
  },
  "tokens.labelHf": {
    fr: "Jeton Hugging Face",
    en: "Hugging Face Token",
    de: "Hugging-Face-Token",
    it: "Token Hugging Face",
  },
  "tokens.configured": {
    fr: "Configuré",
    en: "Configured",
    de: "Eingerichtet",
    it: "Configurato",
  },
  "tokens.notSet": {
    fr: "Non défini",
    en: "Not set",
    de: "Nicht gesetzt",
    it: "Non impostato",
  },
  "tokens.editBtn": {
    fr: "Modifier",
    en: "Edit",
    de: "Bearbeiten",
    it: "Modifica",
  },
  // {label} is the provider name (Civitai / Hugging Face), passed in by the
  // caller as a brand name -- it is never translated.
  "tokens.setBtn": {
    fr: "+ Définir {label}",
    en: "+ Set {label}",
    de: "+ {label} festlegen",
    it: "+ Imposta {label}",
  },
  "tokens.savingBtn": {
    fr: "Enregistrement…",
    en: "Saving…",
    de: "Wird gespeichert…",
    it: "Salvataggio…",
  },
  "tokens.saveBtn": {
    fr: "Enregistrer {label}",
    en: "Save {label}",
    de: "{label} speichern",
    it: "Salva {label}",
  },
  "tokens.clearBtn": {
    fr: "Effacer",
    en: "Clear",
    de: "Löschen",
    it: "Cancella",
  },
  "tokens.civitaiPlaceholder": {
    fr: "Clé d'API Civitai (depuis civitai.red/?ref_code=88C8VEBA)",
    en: "Civitai API Key (from civitai.red/?ref_code=88C8VEBA)",
    de: "Civitai-API-Schlüssel (von civitai.red/?ref_code=88C8VEBA)",
    it: "Chiave API Civitai (da civitai.red/?ref_code=88C8VEBA)",
  },
  "tokens.hfPlaceholder": {
    fr: "Jeton Hugging Face (depuis huggingface.co/settings/tokens)",
    en: "Hugging Face Token (from huggingface.co/settings/tokens)",
    de: "Hugging-Face-Token (von huggingface.co/settings/tokens)",
    it: "Token Hugging Face (da huggingface.co/settings/tokens)",
  },
  "tokens.savedFeedback": {
    fr: "{label} enregistré en toute sécurité (fichier local, jamais envoyé aux clients).",
    en: "{label} saved securely (local file, never sent to clients).",
    de: "{label} sicher gespeichert (lokale Datei, nie an Clients gesendet).",
    it: "{label} salvato in modo sicuro (file locale, mai inviato ai client).",
  },
  "tokens.clearedFeedback": {
    fr: "{label} effacé.",
    en: "{label} cleared.",
    de: "{label} gelöscht.",
    it: "{label} cancellato.",
  },
  // {error} is the raw exception message from the API, passed through untouched.
  "tokens.saveFailed": {
    fr: "Échec de l'enregistrement de {label} : {error}",
    en: "Failed to save {label}: {error}",
    de: "{label} konnte nicht gespeichert werden: {error}",
    it: "Salvataggio di {label} non riuscito: {error}",
  },
};

export default tokensStrings;

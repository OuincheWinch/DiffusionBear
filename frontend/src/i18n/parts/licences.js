// LicencesTab.jsx. The component was written in English, so the literal goes in
// `en` and fr/de/it are translations of it -- the opposite of strings.js, whose
// French originals came first.
//
// Deliberately NOT translated: every row of the third-party tables. The package
// names, SPDX identifiers ("Apache-2.0", "MIT-CMU", "Black Forest Labs --
// Non-Commercial") and repository URLs live in src/data/licences.js and are
// quotations of upstream terms; renaming them would misstate what the upstream
// licence actually says. Only the app's own chrome around them is here.
export const licencesStrings = {
  "licences.sectionBackend": {
    fr: "Backend — environnement MLX / mflux (Python)",
    en: "Backend — MLX / mflux runtime (Python)",
    de: "Backend — MLX-/mflux-Laufzeitumgebung (Python)",
    it: "Backend — runtime MLX / mflux (Python)",
  },
  "licences.sectionFrontend": {
    fr: "Interface web — application React / Vite",
    en: "Frontend — React / Vite SPA",
    de: "Weboberfläche — React-/Vite-SPA",
    it: "Interfaccia web — SPA React / Vite",
  },
  "licences.sectionSdxl": {
    fr: "Moteur SDXL — Juggernaut XL Lightning (Python, environnement sans torch)",
    en: "SDXL engine — Juggernaut XL Lightning (Python, torch-free runtime)",
    de: "SDXL-Engine — Juggernaut XL Lightning (Python, Laufzeitumgebung ohne torch)",
    it: "Motore SDXL — Juggernaut XL Lightning (Python, runtime senza torch)",
  },
  "licences.sectionModels": {
    fr: "Poids des modèles IA",
    en: "Models and weights",
    de: "Modelle und Gewichte",
    it: "Modelli e pesi",
  },
  "licences.noteBackend": {
    fr: "Cœur du moteur de génération pour FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo et le moteur expérimental Qwen-Image 2.1.",
    en: "Core generation stack for FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo and the Qwen-Image 2.1 experimental engine.",
    de: "Kern-Generierungsstack für FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo und die experimentelle Qwen-Image-2.1-Engine.",
    it: "Stack di generazione principale per FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo e il motore sperimentale Qwen-Image 2.1.",
  },
  "licences.noteSdxl": {
    fr: "La conversion des checkpoints utilise l'environnement principal ; le moteur SDXL MLX natif ne dépend pas de torch.",
    en: "Checkpoint conversion uses the main runtime; the native MLX SDXL engine does not require torch.",
    de: "Die Checkpoint-Konvertierung nutzt die Haupt-Laufzeitumgebung; die native MLX-SDXL-Engine benötigt kein torch.",
    it: "La conversione dei checkpoint usa il runtime principale; il motore SDXL MLX nativo non richiede torch.",
  },
  "licences.noteFrontend": {
    fr: "Interface web. Les mêmes dépendances que toute application Vite moderne.",
    en: "Browser UI. Same packages as any modern Vite app.",
    de: "Weboberfläche. Dieselben Pakete wie jede moderne Vite-App.",
    it: "Interfaccia web. Gli stessi pacchetti di qualsiasi app Vite moderna.",
  },
  "licences.noteWeights": {
    fr: "Les fichiers de poids sont téléchargés depuis Hugging Face / les fiches de modèle lors de la première utilisation — ils ne sont ni inclus ni redistribués par DiffusionBear. Chacun porte ses propres conditions.",
    en: "Weight files are downloaded on first use from Hugging Face / model cards — they are NOT bundled with, nor redistributed by, DiffusionBear. Each carries its own terms.",
    de: "Gewichtsdateien werden bei der ersten Verwendung von Hugging Face / den Modellkarten heruntergeladen — sie sind weder in DiffusionBear enthalten noch werden sie von ihm weiterverbreitet. Jede Datei hat eigene Bedingungen.",
    it: "I file di peso vengono scaricati alla prima uso da Hugging Face / dalle schede dei modelli — non sono inclusi né ridistribuiti da DiffusionBear. Ciascuno ha le proprie condizioni.",
  },

  // The <h3> and the header nav button both compose this key with "⚖" and
  // "— DiffusionBear", so the word is never spelled twice.
  "licences.introSecond": {
    fr: "Chaque paquet tiers ci-dessous conserve sa propre licence ; rien ici n'accorde ni ne révoque ces conditions.",
    en: "Every third-party package below keeps its own licence; nothing here grants or revokes those terms.",
    de: "Jedes unten aufgeführte Paket von Dritten behält seine eigene Lizenz; durch nichts hier werden diese Bedingungen gewährt oder widerrufen.",
    it: "Ogni pacchetto di terze parti elencato qui sotto mantiene la propria licenza; nulla di quanto scritto qui concede o revoca quei termini.",
  },
  "licences.thisProject": {
    fr: "Ce projet",
    en: "This project",
    de: "Dieses Projekt",
    it: "Questo progetto",
  },
  // "released under the ___" + link + ", Copyright ..." -- the article lives in
  // the connector so each language gets the one its grammar needs.
  "licences.releasedUnder": {
    fr: "est distribué sous la",
    en: "is released under the",
    de: "wird unter der",
    it: "è distribuito con la",
  },
  "licences.mitLicense": {
    fr: "licence MIT",
    en: "MIT License",
    de: "MIT-Lizenz",
    it: "licenza MIT",
  },
  // International legal notice: identical in all four languages by convention.
  "licences.copyright": {
    fr: "Copyright © {year}",
    en: "Copyright © {year}",
    de: "Copyright © {year}",
    it: "Copyright © {year}",
  },
  "licences.website": {
    fr: "Site web",
    en: "Website",
    de: "Website",
    it: "Sito web",
  },
  "licences.aiAuthored": {
    fr: "En grande partie écrit par une IA",
    en: "Heavily coded by AI",
    de: "Überwiegend von KI geschrieben",
    it: "Scritto in gran parte dall'IA",
  },
  "licences.aiCredits": {
    fr: "assisté par {credits} aux côtés de son auteur humain. Relu et mesuré à la main.",
    en: "assisted by {credits} together with its human author. Reviewed and benchmarked by hand.",
    de: "unterstützt von {credits} zusammen mit seinem menschlichen Autor. Von Hand geprüft und getestet.",
    it: "assistito da {credits} insieme al suo autore umano. Riletto e verificato a mano.",
  },
  // Split in three around the <em> run: splitting keeps the emphasis and lets each
  // language order the sentence the way it needs to (French needs "que" inside).
  "licences.mitScopeBefore": {
    fr: "La licence MIT ne couvre",
    en: "The MIT licence covers",
    de: "Die MIT-Lizenz deckt",
    it: "La licenza MIT copre",
  },
  "licences.mitScopeEm": {
    fr: "que le code source de ce projet",
    en: "this project's source code only",
    de: "nur den Quellcode dieses Projekts",
    it: "solo il codice sorgente di questo progetto",
  },
  "licences.mitScopeAfter": {
    fr: ". Les poids des modèles ne sont pas couverts par elle — voir le tableau des poids ci-dessous.",
    en: ". The model weights are not covered by it — see the weights table below.",
    de: ". Die Modellgewichte sind davon nicht abgedeckt – siehe Gewichtstabelle unten.",
    it: ". I pesi dei modelli non sono coperti — vedi la tabella dei pesi qui sotto.",
  },
  "licences.colPackage": {
    fr: "Paquet",
    en: "Package",
    de: "Paket",
    it: "Pacchetto",
  },
  "licences.colModel": {
    fr: "Modèle",
    en: "Model",
    de: "Modell",
    it: "Modello",
  },
  "licences.colLora": {
    fr: "LoRA",
    en: "LoRA",
    de: "LoRA",
    it: "LoRA",
  },
  "licences.colLicence": {
    fr: "Licence",
    en: "Licence",
    de: "Lizenz",
    it: "Licenza",
  },
  "licences.colRepository": {
    fr: "Dépôt",
    en: "Repository",
    de: "Repository",
    it: "Repository",
  },
  "licences.versions": {
    fr: "Versions",
    en: "Versions",
    de: "Versionen",
    it: "Versioni",
  },
  "licences.tabAppModels": {
    fr: "Application et modèles",
    en: "App & models",
    de: "App & Modelle",
    it: "App e modelli",
  },
  "licences.tabInstalled": {
    fr: "Installés",
    en: "Installed",
    de: "Installiert",
    it: "Installati",
  },
  "licences.installedModels": {
    fr: "Modèles installés",
    en: "Installed models",
    de: "Installierte Modelle",
    it: "Modelli installati",
  },
  "licences.installedLoras": {
    fr: "LoRA installés",
    en: "Installed LoRAs",
    de: "Installierte LoRAs",
    it: "LoRA installati",
  },
  "licences.installedEmpty": {
    fr: "Rien d'installé pour le moment.",
    en: "Nothing installed yet.",
    de: "Noch nichts installiert.",
    it: "Niente installato per ora.",
  },
  "licences.frontendVersion": {
    fr: "Interface {version}",
    en: "Frontend {version}",
    de: "Frontend {version}",
    it: "Interfaccia {version}",
  },
  "licences.backendApi": {
    fr: " · API backend {version}",
    en: " · Backend API {version}",
    de: " · Backend-API {version}",
    it: " · API backend {version}",
  },
  "licences.repoLink": {
    fr: "dépôt",
    en: "repo",
    de: "Repo",
    it: "repo",
  },
  "licences.colOutputs": {
    fr: "Images commerciales",
    en: "Commercial images",
    de: "Kommerzielle Bilder",
    it: "Immagini commerciali",
  },
  "licences.colWeights": {
    fr: "Usage commercial des poids",
    en: "Commercial weights use",
    de: "Kommerzielle Gewichte-Nutzung",
    it: "Uso commerciale dei pesi",
  },
  "licences.usageWarning": {
    fr: "Les licences des modèles couvrent les poids. Elles ne règlent pas les droits sur l'image elle-même : vérifiez le droit à l'image, les marques et le droit d'auteur avant tout usage commercial. Ceci n'est pas un avis juridique.",
    en: "Model licences cover the weights. They do not clear rights in the image itself: check likeness, trademark and copyright before commercial use. Not legal advice.",
    de: "Modelllizenzen decken die Gewichte. Sie klären keine Rechte am Bild selbst: Prüfen Sie vor kommerzieller Nutzung Persönlichkeits-, Marken- und Urheberrechte. Keine Rechtsberatung.",
    it: "Le licenze dei modelli coprono i pesi. Non chiariscono i diritti sull'immagine: verifica diritti di immagine, marchi e copyright prima dell'uso commerciale. Non è consulenza legale.",
  },
  "licences.rightsYes": {
    fr: "Oui",
    en: "Yes",
    de: "Ja",
    it: "Sì",
  },
  "licences.rightsNo": {
    fr: "Non",
    en: "No",
    de: "Nein",
    it: "No",
  },
  "licences.rightsConditional": {
    fr: "Conditionnel",
    en: "Conditional",
    de: "Bedingt",
    it: "Condizionale",
  },
  "licences.rightsUnclear": {
    fr: "Voir la source",
    en: "Check source",
    de: "Quelle prüfen",
    it: "Controlla la fonte",
  },
  "licences.conditionApache": {
    fr: "Usage commercial autorisé. Conservez la licence et les mentions si vous redistribuez le modèle. La licence n'accorde pas de droits sur le contenu de l'image.",
    en: "Commercial use allowed. Keep the licence and notices if you redistribute the model. The licence does not grant image-content rights.",
    de: "Kommerzielle Nutzung erlaubt. Lizenz und Hinweise bei Weitergabe des Modells beibehalten. Die Lizenz gewährt keine Rechte am Bildinhalt.",
    it: "Uso commerciale consentito. Conserva licenza e avvisi se ridistribuisci il modello. La licenza non concede diritti sul contenuto dell'immagine.",
  },
  "licences.conditionFluxNine": {
    fr: "Les images peuvent être vendues, mais les poids sont non commerciaux. Un travail client payant ou un hébergement commercial exige une licence BFL. Filtres et mention IA requis.",
    en: "Outputs may be sold, but the weights are non-commercial. Paid client work or commercial hosting needs a BFL licence. Add filters and AI disclosure.",
    de: "Bilder dürfen verkauft werden, die Gewichte sind aber nicht kommerziell. Bezahlte Kundenarbeit oder kommerzielles Hosting braucht eine BFL-Lizenz. Filter und KI-Kennzeichnung erforderlich.",
    it: "Le immagini possono essere vendute, ma i pesi non sono commerciali. Lavoro per clienti pagati o hosting commerciale richiede una licenza BFL. Filtri e dichiarazione IA obbligatori.",
  },
  "licences.conditionFluxKommunity": {
    fr: "Images commerciales seulement si vous et vos affiliés gagnez moins de 5 M$ US/an. Filtres et mention IA requis. L'exception couvre la génération d'images, pas la distribution commerciale des poids.",
    en: "Commercial outputs only if you and affiliates earn under US$5M/year. Filters and AI disclosure required. The exception covers output generation, not commercial weight distribution.",
    de: "Kommerzielle Bilder nur, wenn Sie und verbundene Unternehmen unter 5 Mio. US$/Jahr verdienen. Filter und KI-Kennzeichnung erforderlich. Die Ausnahme gilt für die Bilderzeugung, nicht für kommerzielle Gewichtsverteilung.",
    it: "Immagini commerciali solo se tu e le affiliate guadagnate meno di 5 milioni di USD/anno. Filtri e dichiarazione IA obbligatori. L'eccezione copre la generazione, non la distribuzione commerciale dei pesi.",
  },
  "licences.conditionKrea": {
    fr: "Images commerciales seulement si vous et vos affiliés gagnez moins de 1 M$ US/an. Filtrage du contenu et mention IA requis. Vous possédez vos images.",
    en: "Commercial outputs only if you and affiliates earn under US$1M/year. Content filtering and AI disclosure required. You own your outputs.",
    de: "Kommerzielle Bilder nur, wenn Sie und verbundene Unternehmen unter 1 Mio. US$/Jahr verdienen. Inhaltsfilter und KI-Kennzeichnung erforderlich. Die Bilder gehören Ihnen.",
    it: "Immagini commerciali solo se tu e le affiliate guadagnate meno di 1 milione di USD/anno. Filtro dei contenuti e dichiarazione IA obbligatori. Le immagini sono tue.",
  },
  "licences.conditionQwenResearch": {
    fr: "Recherche et évaluation uniquement. La génération commerciale exige une licence Qwen séparée. La conversion MLX n'accorde aucun droit commercial.",
    en: "Research and evaluation only. Commercial generation needs a separate Qwen licence. The MLX conversion grants no commercial rights.",
    de: "Nur Forschung und Evaluierung. Kommerzielle Erzeugung braucht eine separate Qwen-Lizenz. Die MLX-Konvertierung gewährt keine kommerziellen Rechte.",
    it: "Solo ricerca e valutazione. La generazione commerciale richiede una licenza Qwen separata. La conversione MLX non concede diritti commerciali.",
  },
  "licences.conditionFibo": {
    fr: "Non commercial uniquement. L'usage commercial exige un accord Bria. Attribution requise si vous partagez le modèle.",
    en: "Non-commercial only. Commercial use needs a Bria agreement. Give attribution if you share the model.",
    de: "Nur nicht kommerziell. Kommerzielle Nutzung braucht eine Bria-Vereinbarung. Namensnennung bei Weitergabe des Modells.",
    it: "Solo non commerciale. L'uso commerciale richiede un accordo Bria. Attribuzione se condividi il modello.",
  },
  "licences.conditionJuggernautXI": {
    fr: "Non commercial, aucune dérivée, aucune API payante ni service concurrent, aucune redistribution des poids. Licence commerciale RunDiffusion requise.",
    en: "Non-commercial, no derivatives, no paid API or competing service, no weight redistribution. Commercial licence from RunDiffusion required.",
    de: "Nicht kommerziell, keine Ableitungen, keine bezahlte API oder konkurrierender Dienst, keine Weitergabe der Gewichte. Kommerzielle Lizenz von RunDiffusion erforderlich.",
    it: "Non commerciale, nessuna derivata, nessuna API a pagamento o servizio concorrente, nessuna ridistribuzione dei pesi. Licenza commerciale RunDiffusion necessaria.",
  },
  "licences.conditionPony": {
    fr: "La vente des images générées est autorisée. L'hébergement d'inférence payant exige une autorisation. Les dérivés gardent la même licence et l'attribution.",
    en: "Selling generated images is allowed. Paid inference hosting needs permission. Derivatives keep the same licence and attribution.",
    de: "Verkauf erzeugter Bilder ist erlaubt. Bezahltes Inferenz-Hosting braucht Erlaubnis. Ableitungen behalten dieselbe Lizenz und Namensnennung.",
    it: "Vendere le immagini generate è consentito. L'hosting di inferenza a pagamento richiede autorizzazione. Le derivate mantengono la stessa licenza e attribuzione.",
  },
  "licences.conditionOpenRail": {
    fr: "Usage commercial autorisé sous réserve des restrictions RAIL. Vous possédez vos images, mais les restrictions s'appliquent à la génération.",
    en: "Commercial use allowed subject to RAIL use restrictions. You own your outputs, but the restrictions still apply to generation.",
    de: "Kommerzielle Nutzung unter den RAIL-Nutzungsbeschränkungen erlaubt. Die Bilder gehören Ihnen, die Beschränkungen gelten aber für die Erzeugung.",
    it: "Uso commerciale consentito con le restrizioni RAIL. Le immagini sono tue, ma le restrizioni valgono per la generazione.",
  },
  "licences.acceptTitle": {
    fr: "Licence de {model}",
    en: "Licence for {model}",
    de: "Lizenz für {model}",
    it: "Licenza per {model}",
  },
  "licences.acceptThisModel": {
    fr: "ce modèle",
    en: "this model",
    de: "dieses Modell",
    it: "questo modello",
  },
  "licences.acceptIntro": {
    fr: "Vérifiez ce que la licence autorise avant de lancer le téléchargement. Rien ne sera téléchargé sans votre accord.",
    en: "Review what the licence permits before starting the download. Nothing is downloaded without your acceptance.",
    de: "Prüfen Sie vor dem Download, was die Lizenz erlaubt. Ohne Ihre Zustimmung wird nichts heruntergeladen.",
    it: "Controlla cosa consente la licenza prima di avviare il download. Nulla viene scaricato senza la tua accettazione.",
  },
  "licences.acceptUnverified": {
    fr: "Licence non vérifiée par l'application",
    en: "Licence not verified by the app",
    de: "Lizenz von der App nicht geprüft",
    it: "Licenza non verificata dall'app",
  },
  "licences.acceptUnknown": {
    fr: "L'application n'a pas de résumé vérifié pour ce modèle. Ouvrez la source et lisez la licence avant de continuer.",
    en: "The app has no verified summary for this model. Open the source and read the licence before continuing.",
    de: "Die App hat keine geprüfte Zusammenfassung für dieses Modell. Öffnen Sie die Quelle und lesen Sie die Lizenz, bevor Sie fortfahren.",
    it: "L'app non ha un riepilogo verificato per questo modello. Apri la fonte e leggi la licenza prima di continuare.",
  },
  "licences.acceptOpenLicence": {
    fr: "Ouvrir la licence",
    en: "Open licence",
    de: "Lizenz öffnen",
    it: "Apri la licenza",
  },
  "licences.acceptCheckbox": {
    fr: "J'ai lu et j'accepte la licence de ce modèle.",
    en: "I have read and accept the licence for this model.",
    de: "Ich habe die Lizenz für dieses Modell gelesen und akzeptiere sie.",
    it: "Ho letto e accetto la licenza di questo modello.",
  },
  "licences.acceptButton": {
    fr: "Accepter et télécharger",
    en: "Accept and download",
    de: "Akzeptieren und herunterladen",
    it: "Accetta e scarica",
  },
  "licences.acceptRequired": {
    fr: "Vous devez accepter la licence avant de télécharger.",
    en: "You must accept the licence before downloading.",
    de: "Sie müssen die Lizenz vor dem Download akzeptieren.",
    it: "Devi accettare la licenza prima di scaricare.",
  },
  "licences.acceptNotLegal": {
    fr: "Ce résumé n'est pas un avis juridique. La licence source fait foi.",
    en: "This summary is not legal advice. The source licence governs.",
    de: "Diese Zusammenfassung ist keine Rechtsberatung. Maßgeblich ist die Lizenz der Quelle.",
    it: "Questo riepilogo non è consulenza legale. Fa fede la licenza della fonte.",
  },
};

export default licencesStrings;

// 한국어 translation scaffold -- GENERATED, do not hand-edit the key list.
// Regenerate: node src/i18n/scaffold/generate-scaffold.mjs
//
// Every one of the 739 interface strings, with the English source as a comment.
// Replace null with the 한국어 translation. Leave a key null if you are unsure:
// null renders as English, whereas "" renders as a blank label (see the header note in
// generate-scaffold.mjs about why the fallback uses ?? and not ||).
//
// Native terms to keep in English unless there is a settled 한국어 equivalent:
// FLUX, Krea, Z-Image, LoRA, SDXL, MLX, CoreML, Metal, mflux, safetensors, VAE, UNet,
// sampler names, and the licence identifiers in src/data/licences.js.

export const koStrings = {
  "app.generate": null, // Generate
  "app.cancel": null, // Cancel
  "app.close": null, // Close
  "app.copy": null, // Copy
  "app.copied": null, // Copied
  "app.retry": null, // Retry
  "app.dismiss": null, // Dismiss
  "language.title": null, // Interface language
  "language.auto": null, // Match system
  "language.pending": null, // Translation in progress — the interface shows English
  "queue.recoveryTitle": null, // Recoverable queue
  "queue.recoverySubtitle": null, // Prompts saved after a queue cancellation or an unexpected shutdown.
  "queue.promptCount": null, // prompt
  "queue.promptsCount": null, // prompts
  "queue.restoreAll": null, // Re-queue all
  "queue.copyAll": null, // Copy all prompts
  "queue.clearAll": null, // Clear all
  "queue.clearAndForget": null, // Clear and forget
  "queue.restore": null, // Re-queue
  "queue.load": null, // Load
  "queue.clearAllConfirm": null, // Permanently clear the recovery history?
  "queue.clearAndForgetConfirm": null, // Clear AND forget?\n\nThis also removes the jobs still waiting, so none of them can be restored at the next start. This cannot be undone.
  "queue.deleteFailed": null, // Deletion failed. Check that the server is running, then try again.
  "queue.restoreFailed": null, // Re-queueing failed.
  "licences.title": null, // Licences
  "licences.intro": null, // What you may and may not do with DiffusionBear, and which projects it builds on.
  "licences.source": null, // Source:
  "queue.empty": null, // No prompts waiting to be recovered.
  "queue.emptyHint": null, // Your cancelled or interrupted generations will be archived here automatically.
  "queue.interruptedBadge": null, // Interrupted (crash/shutdown)
  "queue.cancelledBadge": null, // Cancelled
  "queue.defaultModel": null, // default model
  "queue.steps": null, // steps
  "queue.requeueOneTitle": null, // Re-queue this prompt right away
  "queue.loadTitle": null, // Load this prompt and its settings into the form
  "queue.copyTitle": null, // Copy the prompt text
  "queue.deleteOneTitle": null, // Permanently delete this prompt from the recovery list
  "queue.forgetTitle": null, // Also clears the waiting queue: nothing will be restored at the next start
  "queue.copiedAll": null, // ✓ Copied!
  "queue.clipboardFailed": null, // Could not copy. The clipboard is unavailable.
  "generate.palette.neonPink": null, // Neon pink
  "generate.palette.cyberCyan": null, // Cyber cyan
  "generate.palette.cyberGold": null, // Cyber gold
  "generate.palette.electricViolet": null, // Electric violet
  "generate.palette.neonGreen": null, // Neon green
  "generate.palette.tangerine": null, // Tangerine
  "generate.palette.pureWhite": null, // Pure white
  "generate.palette.matteBlack": null, // Matte black
  "generate.palette.royalBlue": null, // Royal blue
  "generate.palette.lavender": null, // Lavender
  "generate.palette.crimson": null, // Crimson
  "generate.palette.emerald": null, // Emerald
  "generate.error.modelNotInRegistry": null, // The model "{model}" is not available in the current registry.
  "generate.error.enhancerUnavailable": null, // Prompt enhancement is not available for this model.
  "generate.enhanceFeedback.cancelled": null, // Prompt enhancement cancelled.
  "generate.enhanceFeedback.proseFallback": null, // Prompt enhanced with a prose fallback.
  "generate.enhanceFeedback.enhancedFor": null, // Enhanced for {engine}.
  "generate.error.enhancerNoPrompt": null, // The enhancer returned no prompt.
  "generate.error.enhancerFailed": null, // Prompt enhancer: {message}
  "generate.error.refNotSupported": null, // Reference images are not supported on {model}.
  "generate.error.maxReferencesOne": null, // This model accepts at most {max} reference image.
  "generate.error.maxReferencesMany": null, // This model accepts at most {max} reference images.
  "generate.error.refUploadFailed": null, // Reference upload failed: {message}
  "generate.error.importFailed": null, // Could not import {path}: {message}
  "generate.error.dropWebUrl": null, // Cannot import a web address ({url}). Download it first, or drag the file itself from Finder.
  "generate.error.dropNothingUsable": null, // Nothing usable was dropped.
  "generate.confirm.experimentalModel": null, // {model} is marked experimental on this machine. Launch it anyway?
  "generate.error.selectModelFirst": null, // Select an available model before generating.
  "generate.preset.defaultLabel": null, // Default
  "generate.triggers.label": null, // LoRA:
  "generate.triggers.titleRemove": null, // Click to remove from the prompt
  "generate.triggers.titleAdd": null, // Click to add to the prompt
  "generate.refTags.label": null, // Reference tags:
  "generate.refTags.titleInsert": null, // Insert "Image {n}" at the cursor
  "generate.refTags.add": null, // + Image {n}
  "generate.enhance.titleAvailable": null, // Enhance the prompt for the selected model using the local prompt engine.
  "generate.enhance.btnBusy": null, // ✨ Enhancing…
  "generate.enhance.btnIdle": null, // ✨ Enhance (experimental)
  "generate.enhance.cancelTitle": null, // Cancel prompt enhancement
  "generate.enhance.cancelBtn": null, // ✕ Cancel
  "generate.enhance.jsonTitle": null, // Enhance as a structured JSON prompt (subject / appearance / action / setting / lighting / atmosphere / composition / details / text_elements / technical / trigger word)
  "generate.enhance.jsonBtn": null, // { } JSON
  "generate.color.title": null, // Exact colour matching (#HEX)
  "generate.color.btn": null, // 🎨 Colour #{hex}
  "generate.color.popoverTitle": null, // {model} — exact colour matching (#HEX)
  "generate.color.swatchTitle": null, // {name} ({hex}) — click to insert into the prompt
  "generate.color.pickCustom": null, // Pick a custom colour
  "generate.color.insert": null, // + Insert #{hex}
  "generate.prompt.placeholder": null, // Describe the image to generate… (or drag & drop an image/LoRA here)
  "generate.hexBar.label": null, // Active #HEX colours:
  "generate.hexBar.removeTitle": null, // Remove {hex} from the prompt
  "generate.model.loading": null, // Loading models…
  "generate.civitai.title": null, // Civitai model #{modelId} (version #{versionId})
  "generate.civitai.badge": null, // Civitai #{versionId} ↗
  "generate.preset.title": null, // {width}×{height}, {steps} steps
  "generate.lora.summary": null, // ＋ Import or register a new LoRA
  "generate.lora.orRegisterLocal": null, // or register a local file
  "generate.lora.namePlaceholder": null, // name
  "generate.lora.pathPlaceholder": null, // /absolute/path/to/lora.safetensors
  "generate.lora.saveBtnBusy": null, // Saving…
  "generate.lora.saveBtnIdle": null, // Save
  "generate.lora.uploadLabel": null, // …or upload a local .safetensors file
  "generate.lora.formatHint": null, // Must be a {format}-compatible LoRA (.safetensors).
  "generate.lora.unavailableHint": null, // LoRAs are unavailable on {model}.
  "generate.btn.downloadFirstTitle": null, // Download this model first
  "generate.btn.queueWithNewParams": null, // ⚡ Queue with new parameters…
  "generate.btn.queued": null, // ⏳ Queued (waiting for the engine)…
  "generate.btn.downloadingModel": null, // 📥 Downloading model…
  "generate.btn.loadingMemory": null, // 🧠 Loading into memory…
  "generate.btn.compilingShaders": null, // ⚡ Compiling shaders…
  "generate.btn.finalizingImage": null, // 🎨 Finalizing image…
  "generate.btn.stepProgress": null, // ⚙ Step {step}/{steps}…
  "generate.btn.generating": null, // ⚙ Generating…
  "generate.btn.generate": null, // Generate
  "generate.btn.cancelTitle": null, // Cancel this generation
  "generate.btn.cancel": null, // ✕ Cancel
  "generate.progress.batchSaved": null, // Image {index}/{batch} saved ✓ ·
  "generate.progress.stepElapsed": null, // Step {step}/{steps} · elapsed
  "generate.progress.phaseDownloading": null, // Downloading model weights from the repository…
  "generate.progress.phaseLoadingMemory": null, // Loading model weights into Apple Silicon unified memory (100% local, no internet)…
  "generate.progress.phaseCompiling": null, // Compiling Metal shaders and encoding the prompt…
  "generate.progress.eta": null, //  · ETA ~{eta}
  "generate.color.hexPlaceholder": null, // #FFFFFF
  "generate.lora.uploading": null, // Uploading LoRA…
  "generate.lora.registered": null, // "{name}" registered — pick it from the dropdown.
  "generate.progress.phasePreparing": null, // Preparing generation…
  "generate.switchDialog.title": null, // Generation in progress with different settings
  "generate.switchDialog.body": null, // You modified the prompt or the parameters. Queue this new generation after the current one, or stop the current one and start now?
  "generate.switchDialog.queueBtn": null, // Queue after current
  "generate.switchDialog.switchBtn": null, // Stop current and switch
  "generate.switchDialog.cancelBtn": null, // Cancel
  "generate.enhanceWarning.titleLoading": null, // Model is loading
  "generate.enhanceWarning.titleGenerating": null, // Generation in progress
  "generate.enhanceWarning.leadLoading": null, // The engine is still loading into unified memory.
  "generate.enhanceWarning.leadGenerating": null, // An image is currently being generated.
  "generate.enhanceWarning.body": null, // The prompt enhancer runs a second local model on the same Apple Silicon GPU and unified memory, so enhancing now can slow the current job and take longer itself. You can cancel the enhancement at any time. Output is capped to the selected engine profile.
  "generate.enhanceWarning.confirmBtn": null, // Enhance anyway
  "generate.enhanceWarning.cancelBtn": null, // Cancel
  "params.size.label": null, // Size
  "params.size.baseLabel": null, // Base resolution
  "params.size.baseTitle": null, // Base {base} px
  "params.size.ratioLabel": null, // Aspect ratio
  "params.size.ratioTitle": null, // {ratio} aspect ratio
  "params.size.customShort": null, // Custom
  "params.size.customTitle": null, // Enter exact width and height
  "params.size.widthLabel": null, // Width
  "params.size.heightLabel": null, // Height
  "params.size.backToGrid": null, // Grid
  "params.size.backToGridTitle": null, // Go back to choosing by resolution and ratio
  "params.size.budgetWarning": null, // Over {max} px — generation may fail. Use a smaller size or raise the cap in Advanced Settings.
  "params.size.budgetHint": null, // Dimensions are snapped to the nearest multiple of 16, minimum 256.
  "params.size.lockRatio": null, // Keep ratio
  "params.size.lockRatioTitle": null, // Editing one side adjusts the other to preserve the proportions
  "params.size.unlockRatioTitle": null, // Both dimensions are independent
  "params.size.lockRatioHint": null, // Ratio locked. Uncheck to edit both dimensions freely.
  "params.advanced.label": null, // Advanced Settings
  "params.size.customOption": null, // ✦ {width} × {height} (preset)
  "params.size.shapePreviewTitle": null, // {width} × {height}
  "params.maxPixels.label": null, // Max pixels (hard cap)
  "params.maxPixels.oomGuardTitle": null, // Hard ceiling enforced by the backend for this model.
  "params.maxPixels.oomGuardTag": null, // ⚙ OOM guard
  "params.steps.label": null, // Steps ({steps})
  "params.steps.distillTitle": null, // 4-step distillation LoRA auto-activated in the parameters
  "params.steps.distillTag": null, // ⚡ 4-step distill
  "params.guidance.label": null, // Guidance
  "params.seed.label": null, // Seed
  "params.seed.randomTitle": null, // Randomize (empty seed)
  "params.seed.nextTitle": null, // Next seed (+1)
  "params.seed.nextBatchTitle": null, // Next batch seed (+1024)
  "params.seed.placeholder": null, // random
  "params.batch.label": null, // Batch
  "params.batch.optionOne": null, // {n} image
  "params.batch.optionMany": null, // {n} images
  "params.batch.optionSeedSuffix": null, //  (+1024 seed)
  "params.negative.label": null, // Negative prompt
  "params.negative.placeholder": null, // What to avoid: blurry, low quality…
  "params.sampler.label": null, // Sampler
  "params.deepCache.label": null, // DeepCache
  "params.deepCache.off": null, // Off (exact UNet)
  "params.deepCache.level2": null, // ⚡ DeepCache 2 (~1.6x faster)
  "params.deepCache.level3": null, // ⚡⚡ DeepCache 3 (~2x faster)
  "params.ref.legend": null, // Reference images ({count}/{max})
  "params.ref.inContextBadge": null, // FLUX.2 In-Context
  "params.ref.thumbAlt": null, // Reference {n}
  "params.ref.previewUnavailable": null, // Reference {n} preview unavailable
  "params.ref.indexBadge": null, // Image {n}
  "params.ref.removeTitle": null, // Remove reference image
  "params.ref.insertTitle": null, // Insert "Image {n}" into the prompt
  "params.ref.promptTagBtn": null, // + Prompt tag
  "params.ref.addCardTitle": null, // Add a reference image (up to 10 for FLUX.2)
  "params.ref.addTextFirst": null, // Add image
  "params.ref.addTextNext": null, // Add ({n})
  "params.ref.strengthLabel": null, // KREA image reference strength
  "params.ref.decrease01Title": null, // Decrease by 0.1
  "params.ref.decrease001Title": null, // Decrease by 0.01
  "params.ref.increase001Title": null, // Increase by 0.01
  "params.ref.increase01Title": null, // Increase by 0.1
  "params.ref.multiRefHintLead": null, // FLUX.2 In-Context conditioning:
  "params.ref.multiRefHintBody": null, // Up to 10 reference images. The transformer injects image tokens directly into cross-attention. Refer to them naturally in your prompt as Image 1, Image 2, and so on (e.g. "a portrait of the character from Image 1 in the artistic style of Image 2").
  "params.duration.minutesSeconds": null, // {m}m {s}s
  "params.duration.secondsOnly": null, // {s}s
  "stack.status.queued": null, // ⏳ Queued
  "stack.status.generating": null, // ⚙ Generating
  "stack.status.done": null, // ✓ Done
  "stack.status.error": null, // ✗ Error
  "stack.status.cancelled": null, // Cancelled
  "stack.heading": null, // Generation stack ({count})
  "stack.killCurrentBtn": null, // ✕ Kill current
  "stack.emptyQueueBtn": null, // ⌫ Empty queue
  "stack.emptyQueueTitle": null, // Remove every waiting job from the queue
  "stack.recoveryTriggerTitle": null, // Review and re-queue cancelled or interrupted prompts
  "stack.recoveryTriggerOne": null, // ↺ {count} recoverable
  "stack.recoveryTriggerMany": null, // ↺ {count} recoverable
  "stack.crashAlertOne": null, // ⚡ {count} generation interrupted during the last session
  "stack.crashAlertMany": null, // ⚡ {count} generations interrupted during the last session
  "stack.crashAlertBtn": null, // View & Restore
  "stack.phase.downloading": null, // 📥 Downloading
  "stack.phase.loadingMemory": null, // 🧠 Loading memory
  "stack.phase.compiling": null, // ⚡ Compiling
  "stack.phase.finalizing": null, // 🎨 Finalizing
  "stack.job.cancelTitle": null, // Cancel this job
  "stack.job.completedIn": null, // Completed in {time}s ({steps} steps
  "stack.job.completedBatch": null, //  · {batch} images)
  "stack.job.decoding": null, // Decoding image (VAE) · {elapsed}s elapsed
  "stack.job.stepElapsed": null, // Step {step}/{steps} · {elapsed}s elapsed
  "stack.job.eta": null, //  · ETA ~{eta}s
  "stack.job.imageProgress": null, //  · image {index}/{batch}
  "stack.job.waitingInQueue": null, // Waiting in queue
  "stack.job.waitingBatch": null, //  · batch of {batch} images
  "stack.job.phaseDownloading": null, // Downloading weights…
  "stack.job.phaseLoadingWeights": null, // Loading model weights into unified memory…
  "canvas.progress.preparing": null, // Preparing generation…
  "canvas.progress.decodingVae": null, // 🎨 Decoding VAE ({elapsed}s)…
  "canvas.progress.step": null, // Step {step}/{steps} ({elapsed}s)
  "canvas.progress.phaseDownloading": null, // 📥 Downloading model weights…
  "canvas.progress.phaseLoadingModel": null, // 🧠 Loading model into unified memory…
  "canvas.progress.phaseCompiling": null, // ⚡ Compiling Metal shaders and encoding the prompt…
  "canvas.progress.phaseSaving": null, // 🎨 Finalizing the image and saving…
  "canvas.progress.phaseStandalone": null, // Starting the engine and loading the model…
  "canvas.batchPill": null, // Image {index} / {total}
  "canvas.error.upscaleFailed": null, // Upscale failed.
  "canvas.error.downloadFailed": null, // Download failed.
  "canvas.heading": null, // Studio Canvas
  "canvas.pill.downloadingModel": null, // 📥 Downloading model
  "canvas.pill.loadingMemory": null, // 🧠 Loading into memory
  "canvas.pill.compilingShaders": null, // ⚡ Compiling shaders
  "canvas.pill.finalizingImage": null, // 🎨 Finalizing image
  "canvas.pill.step": null, // ⚙ Step {step}/{steps}
  "canvas.pill.working": null, // ⚙ Working…
  "canvas.metaPill": null, // {width}×{height} · {time}s
  "canvas.image.altFallback": null, // Generated image
  "canvas.image.dragTitle": null, // Drag for the full-resolution image
  "canvas.empty.title": null, // Your creation will appear here
  "canvas.empty.subtitle": null, // Enter a prompt and click Generate to start rendering
  "canvas.batchStrip.title": null, // Batch gallery ({count}{totalSuffix})
  "canvas.batchStrip.totalSuffix": null, // /{total}
  "canvas.batchStrip.rendering": null, // Rendering #{n}…
  "canvas.batchStrip.thumbTitle": null, // Image #{n} (seed: {seed}) — click to preview · drag for the full-resolution image
  "canvas.batchStrip.thumbAlt": null, // Batch #{n}
  "canvas.batchStrip.renderingImage": null, // Rendering image #{n}…
  "canvas.prevTag": null, // Previous result
  "canvas.prevHint": null, // Showing the last completed creation while the new image renders
  "canvas.badge.model": null, // Model:
  "canvas.badge.seed": null, // Seed:
  "canvas.badge.steps": null, // Steps:
  "canvas.badge.sampler": null, // Sampler:
  "canvas.badge.cfg": null, // CFG:
  "canvas.badge.vae": null, // VAE:
  "canvas.badge.vaeValue": null, //  SOTA ⚡
  "canvas.action.upscale2xTitle": null, // Fast resampling 2x (Lanczos + unsharp)
  "canvas.action.upscale2xBusy": null, // Fast 2x…
  "canvas.action.upscale2xIdle": null, // ⚡ Fast 2x
  "canvas.action.upscale4xTitle": null, // Fast resampling 4x (Lanczos + unsharp)
  "canvas.action.upscale4xBusy": null, // Fast 4x…
  "canvas.action.upscale4xIdle": null, // ⚡ Fast 4x
  "canvas.action.useAsRefTitle": null, // Use this image as a reference (up to {max})
  "canvas.action.useAsRef": null, // 🖼️ Use as reference
  "canvas.action.variationTitle": null, // Generate a variation with seed + 1
  "canvas.action.variation": null, // 🔄 Variation
  "canvas.action.copyPromptTitle": null, // Copy prompt
  "canvas.action.copySeedTitle": null, // Copy seed
  "canvas.action.copyImageTitle": null, // Copy the PNG pixels to the system clipboard
  "canvas.action.copyImageDone": null, // Image copied ✓
  "canvas.action.copyImageIdle": null, // 📋 Copy image
  "canvas.action.copyPromptDone": null, // Prompt copied ✓
  "canvas.action.copyPromptIdle": null, // 📋 Copy prompt
  "canvas.action.copySeedDone": null, // Seed copied ✓
  "canvas.action.copySeedIdle": null, // 📋 Copy seed
  "canvas.action.downloadTitle": null, // Download the full-resolution image
  "canvas.action.downloadBusy": null, // Downloading…
  "canvas.action.downloadIdle": null, // ⬇ Download
  "canvas.action.clipboardFailed": null, // Could not copy. The clipboard is unavailable.
  "gallery.searchPlaceholder": null, // Search prompts or seeds…
  "gallery.tagsPlaceholder": null, // tags: comma,separated
  "gallery.sortNewest": null, // Newest
  "gallery.sortOldest": null, // Oldest
  "gallery.allModels": null, // All models
  "gallery.loraFilterTitle": null, // Filter images by LoRA
  "gallery.loraOptionAll": null, // All images ({count})
  "gallery.loraOptionNone": null, // Without LoRA ({count})
  "gallery.loraOptionAny": null, // With any LoRA ({count})
  "gallery.loraOptgroup": null, // Installed & used LoRAs
  "gallery.countOne": null, // {count} image
  "gallery.countMany": null, // {count} images
  "gallery.loraPrefix": null, // LoRA:
  "gallery.loraResetTitle": null, // Reset LoRA filter
  "gallery.loraClearBtn": null, // ✕ Clear filter
  "gallery.loraTabAll": null, // All images
  "gallery.loraTabNone": null, // Without LoRA
  "gallery.loraTabAny": null, // With any LoRA
  "gallery.loraTabTitleOne": null, // Filter by {name} ({count} image)
  "gallery.loraTabTitleMany": null, // Filter by {name} ({count} images)
  "gallery.empty": null, // No images yet. Generate something!
  "gallery.cellTitle": null, // Click to view details · Drag anywhere for the full-resolution image
  "gallery.cellCopyTitle": null, // Copy the PNG image (Cmd+V on Civitai or in the chat)
  "gallery.cellRevealTitle": null, // Reveal the file in the macOS Finder
  "gallery.detailPrevTitle": null, // Previous (←)
  "gallery.detailNextTitle": null, // Next (→)
  "gallery.dragFullResTitle": null, // Drag for the full-resolution image
  "gallery.fieldModel": null, // Model
  "gallery.fieldSeed": null, // Seed
  "gallery.fieldSize": null, // Size
  "gallery.fieldSteps": null, // Steps
  "gallery.fieldGuidance": null, // Guidance
  "gallery.fieldSampler": null, // Sampler
  "gallery.fieldNegative": null, // Negative
  "gallery.fieldQuantization": null, // Quantization
  "gallery.fieldTime": null, // Time
  "gallery.fieldLoras": null, // LoRAs
  "gallery.copyImageFullTitle": null, // Copy the original PNG image to the macOS clipboard (Cmd+C / then Cmd+V on Civitai or in the chat)
  "gallery.copyImageDone": null, // Image copied ✓ (Cmd+V)
  "gallery.copyImageBtn": null, // 📋 Copy image
  "gallery.revealFullTitle": null, // Open the original image in the macOS Finder to drag and drop it to Civitai
  "gallery.revealedDone": null, // Opened in the Finder ✓
  "gallery.revealBtn": null, // 📂 Finder
  "gallery.promptCopiedDone": null, // Prompt copied ✓
  "gallery.copyPromptBtn": null, // Copy prompt
  "gallery.seedCopiedDone": null, // Copied ✓
  "gallery.copySeedBtn": null, // Copy seed
  "gallery.upscale2xTitle": null, // 2x super-resolution upscale
  "gallery.upscale4xTitle": null, // 4x super-resolution upscale
  "gallery.upscaling": null, // Upscaling…
  "gallery.upscale2xBtn": null, // ⚡ Upscale 2x
  "gallery.upscale4xBtn": null, // ⚡ Upscale 4x
  "gallery.upscaleError": null, // Upscale error: {message}
  "gallery.reuseParams": null, // Reuse params
  "gallery.download": null, // Download
  "gallery.deleteConfirm": null, // Delete this image?
  "gallery.deleteBtn": null, // Delete
  "gallery.exportBtn": null, // Export…
  "gallery.showOriginalBtn": null, // Show original
  "gallery.showOriginalTitle": null, // Open the image this one was made from
  "gallery.closeBtn": null, // Close
  "gallery.tagsLabel": null, // Tags
  "gallery.saveTags": null, // Save tags
  "lora.syncingInfo": null, // Hashing & syncing all local LoRAs with Civitai…
  "lora.syncComplete": null, // Civitai sync complete: {updated} updated out of {total} LoRAs.
  "lora.syncFailed": null, // Sync failed: {message}
  "lora.maxActive": null, // Maximum of 16 active LoRAs per image generation.
  "lora.deleteConfirm": null, // Permanently delete "{name}" from disk and registry?
  "lora.deleteSuccess": null, // Deleted "{name}" successfully.
  "lora.deleteFailed": null, // Failed to delete: {message}
  "lora.disclosureLabel": null, // LoRAs
  "lora.activeLegend": null, // Active LoRAs ({count}/16 max)
  "lora.syncTitle": null, // Scan local LoRAs and fetch official Civitai IDs & triggers
  "lora.syncing": null, // ⏳ Syncing…
  "lora.syncBtn": null, // 🔄 Sync Civitai
  "lora.distillTitle": null, // Distillation adapter for 4 steps or fewer
  "lora.distillTag": null, // ⚡ Distill
  "lora.civitaiIdTitle": null, // Civitai ID: {id} ({version})
  "lora.civitaiTagSuffix": null, // [Civitai #{id}]
  "lora.scaleDecreaseTitle": null, // -0.05
  "lora.scaleIncreaseTitle": null, // +0.05
  "lora.registryEmpty": null, // No {format} LoRAs in the registry
  "lora.addOption": null, // + Add LoRA…
  "lora.installedPrefix": null, // Installed:
  "lora.addActiveTitle": null, // Click to add to the active LoRAs
  "lora.deleteChipTitle": null, // Delete {name} from disk and registry
  "lora.hideHub": null, // ▲ Hide installed LoRAs hub
  "lora.showHub": null, // 📦 Installed LoRAs hub ({count} models)
  "lora.hubEmpty": null, // No LoRAs installed yet.
  "lora.hubActive": null, // ✓ Active
  "lora.hubAdd": null, // + Add
  "lora.switchTitle": null, // Switch model to {model}
  "lora.switchBtn": null, // ⚡ Switch to {model}
  "settings.tab.preferences": null, // Preferences
  "settings.tab.enhancer": null, // Prompt Enhancer prompts (experimental)
  "settings.tab.enhancerTitle": null, // 🧠 Prompt Enhancer — system prompts (experimental)
  "settings.tab.enhancerDesc": null, // Customize the system prompt the local LLM (Qwen2.5-0.5B-Instruct via MLX) uses when you click ✨ Enhance. One editable prompt per engine — FLUX.2 Klein, SDXL Lightning, Krea 2 Turbo and Z-Image Turbo. Saved overrides are used immediately by the enhancer. The enforced contract is prompt-only output within each engine's length cap (no preamble, no explanation), but the feature itself is experimental.
  "settings.section.defaultsTitle": null, // 🖋 Defaults & personalization
  "settings.section.defaultsDesc": null, // Default generation preferences for new images, plus the artist credit embedded in every output. Your artist name replaces the previous hard-coded credit — great for a public release.
  "settings.section.engineTitle": null, // 🖥 Engine & GPU
  "settings.section.engineDesc": null, // Live Metal usage, wired-memory budgets, resident mflux/SDXL pipelines and the idle auto-release countdown.
  "settings.section.modelsTitle": null, // 🗂 Model management
  "settings.section.modelsDesc": null, // Installed status and disk footprint of every engine. Remove weights to free space; they are re-downloaded on demand.
  "settings.section.queueTitle": null, // ⏳ Queue & pending jobs
  "settings.section.queueDesc": null, // Watch the generation queue and re-queue prompts that were interrupted or cancelled.
  "settings.section.storageTitle": null, // 📦 Storage
  "settings.section.storageDesc": null, // What is on disk and what accounts for it. Read-only: this panel never deletes anything, and never marks anything reclaimable.
  "settings.section.hfCacheTitle": null, // 💾 Hugging Face cache
  "settings.section.hfCacheDesc": null, // Local copy of every downloaded model repo. Clear entries to reclaim disk space.
  "settings.section.secretsTitle": null, // 🔐 Secret management
  "settings.section.secretsDescBefore": null, // API keys and tokens are saved by the backend into local files under
  "settings.section.secretsDescAfter": null, // for example
  "settings.section.secretsDescTail": null, // They are never sent to the browser clients, never logged, and never exposed by the API — they are used only server-side to authenticate outbound requests to Civitai / Hugging Face.
  "settings.queue.tokenFallback": null, // Note: gated Hugging Face repos also fall back to a token stored in
  "settings.defaults.artistCredit": null, // 🖋 Artist credit
  "settings.defaults.artistHint": null, // Embedded in EXIF / Civitai metadata of every generated image. Empty uses "MLX-DIFFUSION".
  "settings.defaults.saveArtist": null, // Save artist
  "settings.defaults.outputFormat": null, // Default output format
  "settings.defaults.formatPng": null, // PNG (lossless)
  "settings.defaults.formatJpeg": null, // JPEG (smaller)
  "settings.defaults.stealth": null, // Default 🥷 Stealth (no metadata)
  "settings.defaults.saveOutput": null, // Save output preferences
  "settings.defaults.globalFallbacks": null, // ◇ Global fallbacks
  "settings.defaults.globalHint": null, // Applied only when a model has no explicit per-model override. Fast-VAE (TAESD/TAEF) trades a little fidelity for much faster VAE decoding.
  "settings.defaults.defaultSampler": null, // Default sampler
  "settings.defaults.modelDefaultOption": null, // (model default)
  "settings.defaults.deepCacheInterval": null, // SDXL DeepCache interval
  "settings.defaults.fastVae": null, // ⚡ Fast VAE (all engines)
  "settings.defaults.saveGlobal": null, // Save global defaults
  "settings.defaults.perModelTitle": null, // 🎛 Per-model defaults
  "settings.defaults.perModelHint": null, // Overrides loaded automatically when you switch models in the Generate form. Leave a field blank / 0 to keep the model's built-in value.
  "settings.defaults.customBadge": null, // custom ⚙
  "settings.defaults.steps": null, // Steps
  "settings.defaults.guidance": null, // Guidance
  "settings.defaults.sampler": null, // Sampler
  "settings.defaults.defaultOption": null, // (default)
  "settings.defaults.deepCacheSdxl": null, // DeepCache (SDXL)
  "settings.defaults.fastVaeShort": null, // Fast-VAE
  "settings.defaults.width": null, // Width
  "settings.defaults.height": null, // Height
  "settings.defaults.defaultPlaceholder": null, // default
  "settings.defaults.saveForModel": null, // Save defaults for this model
  "settings.defaults.resetFeedback": null, // Model defaults reset
  "settings.defaults.resetBuiltin": null, // Reset to built-in
  "settings.defaults.savedFeedback": null, // Preferences saved
  "settings.engine.tuningNumbersError": null, // All tuning values must be numbers (0 disables).
  "settings.engine.tuningRangeError": null, // Values must be ≥ 0.
  "settings.engine.loading": null, // Loading engine status…
  "settings.engine.metalCard": null, // 🖥 Metal / GPU
  "settings.engine.gaugeGpu": null, // GPU
  "settings.engine.gaugeMemory": null, // Memory
  "settings.engine.gaugeMaxWorkingSet": null, // Max working set
  "settings.engine.gaugePeakMemory": null, // Peak memory
  "settings.engine.peakMemoryTitle": null, // Highest wired memory reached this session
  "settings.engine.gaugeCache": null, // Cache
  "settings.engine.cacheTitle": null, // mlx allocator cache
  "settings.engine.gaugeWiredLimit": null, // Wired limit
  "settings.engine.gaugeKreaWired": null, // krea2 wired
  "settings.engine.unbounded": null, // unbounded
  "settings.engine.mfluxCard": null, // ⚡ mflux pipelines
  "settings.engine.modelFallback": null, // model
  "settings.engine.residentInMemory": null, // resident in unified memory
  "settings.engine.autoReleaseIn": null, // auto-release in {time}
  "settings.engine.noPipelineResident": null, // No pipeline resident (cold reload on next generation)
  "settings.engine.gaugeIdlePolicy": null, // Idle policy
  "settings.engine.gaugePromptCache": null, // Prompt cache
  "settings.engine.gaugeWatchdog": null, // Watchdog
  "settings.engine.armed": null, // armed
  "settings.engine.cold": null, // cold
  "settings.engine.sdxlCard": null, // 🧵 SDXL daemon
  "settings.engine.engineFallback": null, // engine
  "settings.engine.alive": null, // alive
  "settings.engine.daemonNotRunning": null, // Daemon not running (spawned lazily)
  "settings.engine.hideDaemonLog": null, // Hide daemon log
  "settings.engine.showDaemonLog": null, // Daemon log tail
  "settings.engine.tuningCard": null, // ⚙️ Runtime tuning
  "settings.engine.tuningNote": null, // Saved values take effect on the next generation (or the next idle rearm) without restarting anything.
  "settings.engine.tuneWired": null, // Wired limit
  "settings.engine.tuneWiredTitle": null, // Metal allocator wired limit for FLUX.2 / SDXL (MLX_WIRED_LIMIT_GB). 0 = unbounded.
  "settings.engine.tuneKrea": null, // krea2 wired
  "settings.engine.tuneKreaTitle": null, // krea2 (13B q4) wired budget (MLX_KREA_WIRED_LIMIT_GB). 0 = unbounded.
  "settings.engine.tuneMfluxIdle": null, // mflux idle
  "settings.engine.tuneMfluxIdleTitle": null, // Idle seconds before the mflux pipeline is auto-released. 0 = keep resident.
  "settings.engine.tuneSdxlIdle": null, // sdxl idle
  "settings.engine.tuneSdxlIdleTitle": null, // Idle seconds before the SDXL daemon is killed. 0 = keep alive.
  "settings.engine.tuneQwenIdle": null, // qwen idle
  "settings.engine.tuneQwenIdleTitle": null, // Idle seconds before the Qwen daemon is killed. 0 = keep alive.
  "settings.engine.saving": null, // Saving…
  "settings.engine.save": null, // Save
  "settings.engine.saved": null, // Saved ✓
  "settings.engine.zeroHint": null, // 0 = disabled / never auto-release
  "settings.engine.storageCard": null, // 💾 Storage
  "settings.engine.gaugeGenerated": null, // Generated
  "settings.engine.imagesCount": null, // {count} images
  "settings.engine.gaugeSize": null, // Size
  "settings.engine.gaugeTaef": null, // TAEF
  "settings.engine.noneLoaded": null, // none loaded
  "settings.engine.gaugeSettingsFile": null, // Settings file
  "settings.engine.present": null, // present
  "settings.engine.absent": null, // absent
  "settings.engine.now": null, // now
  "settings.models.loading": null, // Loading model registry…
  "settings.models.colModel": null, // Model
  "settings.models.colState": null, // State
  "settings.models.colOnDisk": null, // On disk
  "settings.models.colActions": null, // Actions
  "settings.models.installed": null, // ✓ installed
  "settings.models.notInstalled": null, // not installed
  "settings.models.customDefaults": null, // custom defaults
  "settings.models.removeTitle": null, // Delete weights from disk (guard: refuses while generating/downloading)
  "settings.models.deleting": null, // Deleting…
  "settings.models.unlinkPath": null, // 🔗 Unlink path
  "settings.models.remove": null, // 🗑 Remove
  "settings.models.confirmUnlink": null, // unlink the configured local path for "{model}" (files stay on disk)?\n\nYou can re-download it later from the Generate tab.
  "settings.models.confirmRemove": null, // remove the managed weights from disk for "{model}"?\n\nYou can re-download it later from the Generate tab.
  "settings.models.feedbackUnlinked": null, // {model} local path unlinked
  "settings.models.feedbackRemoved": null, // {model} removed from disk
  "settings.models.storageHintLead": null, // Models are stored under
  "settings.models.storageHintTail": null, // and the local Hugging Face hub cache (~/.cache/huggingface). Removing a model frees disk space; it is re-downloaded lazily on first use.
  "settings.hfCache.clearConfirm": null, // Remove "{repo}" ({size}) from the Hugging Face cache?
  "settings.hfCache.freed": null, // Freed {size}
  "settings.hfCache.scanning": null, // Scanning Hugging Face cache…
  "settings.hfCache.totalBadge": null, // {size} total
  "settings.hfCache.rootMissing": null, // {path} does not exist yet
  "settings.hfCache.refresh": null, // ↺ Refresh
  "settings.hfCache.empty": null, // No cached repositories.
  "settings.hfCache.colRepository": null, // Repository
  "settings.hfCache.colSize": null, // Size
  "settings.hfCache.colDetails": null, // Details
  "settings.hfCache.colActions": null, // Actions
  "settings.hfCache.revisionOne": null, // {count} revision
  "settings.hfCache.revisionMany": null, // {count} revisions
  "settings.hfCache.files": null, //  · {count} files
  "settings.hfCache.removing": null, // Removing…
  "settings.hfCache.clear": null, // 🗑 Clear
  "settings.hfCache.footerHint": null, // Model weights live in the shared Hugging Face hub cache (~/.cache/huggingface). Clearing a repo removes snapshots + blobs; generation re-downloads it lazily. Repos currently downloading are protected.
  "settings.storage.statusRegistered": null, // in use
  "settings.storage.statusInternal": null, // loaded at runtime
  "settings.storage.statusUnrecognised": null, // unrecognised
  "settings.storage.scanning": null, // Scanning…
  "settings.storage.scanFailed": null, // Storage scan failed: {error}
  "settings.storage.onDisk": null, // on disk
  "settings.storage.storeRoot": null, // store: {path}
  "settings.storage.rootsSplit": null, // models: {models} · data: {data}
  "settings.storage.rescan": null, // Rescan
  "settings.storage.rowModels": null, // Models
  "settings.storage.directories": null, // {count} directories
  "settings.storage.rowLoras": null, // LoRAs
  "settings.storage.lorasDetailOne": null, // {files} file(s) across 1 directory
  "settings.storage.lorasDetailMany": null, // {files} file(s) across {dirs} directories
  "settings.storage.rowGallery": null, // Generated images
  "settings.storage.galleryDetail": null, // {images} image(s) + {thumbnails} thumbnail(s)
  "settings.storage.rowUploads": null, // Uploads
  "settings.storage.uploadsDetail": null, // {files} file(s)
  "settings.storage.dupeLead": null, // {size} is stored more than once.
  "settings.storage.dupeBodyOne": null, // {count} probable duplicate file exists as an identical copy in a different directory — usually a leftover from an older install alongside the current one. Nothing has been removed; this is only telling you the space is there.
  "settings.storage.dupeBodyMany": null, // {count} probable duplicate files exist as identical copies in different directories — usually leftovers from an older install alongside the current one. Nothing has been removed; this is only telling you the space is there.
  "settings.storage.missingOne": null, // {count} registered LoRA points at a file that is no longer there.
  "settings.storage.missingMany": null, // {count} registered LoRAs point at files that are no longer there.
  "settings.storage.missingTail": null, // They will fail silently if selected. Names:
  "settings.storage.unseenOne": null, // {count} model directory is not referenced by anything in the app.
  "settings.storage.unseenMany": null, // {count} model directories are not referenced by anything in the app.
  "settings.storage.unseenTail": null, // That is not a verdict — a model placed by hand, or one a different build supports, looks the same. Worth a look, not an instruction. Nothing here is ever marked reclaimable.
  "settings.storage.duplicatedFiles": null, // Duplicated files
  "settings.storage.thFile": null, // File
  "settings.storage.thEach": null, // Each
  "settings.storage.thCopies": null, // Copies
  "settings.storage.thWasted": null, // Wasted
  "settings.storage.modelDirectories": null, // Model directories
  "settings.storage.thDirectory": null, // Directory
  "settings.storage.thSize": null, // Size
  "settings.storage.thFiles": null, // Files
  "settings.storage.thStatus": null, // Status
  "settings.storage.unaccountedFor": null, // Unaccounted for: {names}
  "settings.storage.configCreds": null, // Configuration and credentials
  "settings.storage.detailSettings": null, // preferences, presets
  "settings.storage.detailLoras": null, // LoRA registry
  "settings.storage.secretPresent": null, // present on disk
  "settings.storage.secretMissing": null, // not set
  "settings.storage.credentialNote": null, // Credential files are reported by name and size only. This panel never opens them.
  "settings.queue.cancelAllConfirm": null, // Cancel the entire queue? Running generations are stopped and archived for recovery.
  "settings.queue.clearRecoveryConfirm": null, // Delete all recovery records? (interrupted prompts stay archived unless you also clear the pending queue).
  "settings.queue.counts": null, // {active} active · {recovery} recoverable
  "settings.queue.cancelAll": null, // ⌫ Cancel all
  "settings.queue.hideRecovery": null, // Hide recovery
  "settings.queue.recovery": null, // ↺ Recovery ({count})
  "settings.queue.loading": null, // Loading queue…
  "settings.queue.idle": null, // Queue idle — no generation currently running or waiting.
  "settings.queue.colPromptModel": null, // Prompt / model
  "settings.queue.colStatus": null, // Status
  "settings.queue.colQueued": null, // Queued
  "settings.queue.colActions": null, // Actions
  "settings.queue.queued": null, // queued
  "settings.queue.cancel": null, // ✕ Cancel
  "settings.queue.recoverableTitle": null, // ↺ Recoverable prompts
  "settings.queue.requeueAll": null, // ↺ Requeue all ({count})
  "settings.queue.clearHistory": null, // 🗑 Clear history
  "settings.queue.nothingRecoverable": null, // Nothing recoverable. Cancelled or interrupted generations are archived here (queue_recovery.json) and can be requeued after a restart.
  "settings.queue.interrupted": null, // ⚡ interrupted
  "settings.queue.cancelled": null, // ⌫ cancelled
  "settings.queue.requeue": null, // ↺ Requeue
  "settings.queue.loadInForm": null, // ✎ Load in form
  "settings.enhancer.savedFeedback": null, // Enhancer system prompt saved
  "settings.enhancer.loading": null, // Loading prompt-enhancer profiles…
  "settings.enhancer.noEngines": null, // No engines available.
  "settings.enhancer.customPromptTitle": null, // Custom prompt active
  "settings.enhancer.targetLength": null, // target length: {length}
  "settings.enhancer.modeText": null, // text
  "settings.enhancer.modeJson": null, // JSON
  "settings.enhancer.customText": null, // custom text
  "settings.enhancer.customJson": null, // custom JSON
  "settings.enhancer.builtinText": null, // built-in text
  "settings.enhancer.builtinJson": null, // built-in JSON
  "settings.enhancer.modeToggleAria": null, // Enhancer output mode
  "settings.enhancer.textModeBtn": null, // 📝 Text mode
  "settings.enhancer.jsonModeBtn": null, // { } JSON mode
  "settings.enhancer.jsonFieldTitle": null, // JSON system prompt — structure + fill-in guidelines
  "settings.enhancer.textFieldTitle": null, // System prompt — engine guidance (text mode)
  "settings.enhancer.jsonFieldHint": null, // This is the complete instruction sent to the local LLM in JSON mode: the engine guidance, the exact JSON structure (keys) to output, and the rules for filling each field. Edit it and Save to store a custom override, or ↺ Reset to built-in to restore the default.
  "settings.enhancer.textFieldHint": null, // This text is injected into the local LLM's system prompt when you click ✨ Enhance with an {model} model in text mode. Edit it and Save to store a custom override, or ↺ Reset to built-in to restore the default.
  "settings.enhancer.modesIndependent": null, // Text and JSON modes are stored independently.
  "settings.enhancer.saveTitleDirty": null, // Save this custom guidance
  "settings.enhancer.saveTitleClean": null, // Edit the text to enable saving
  "settings.enhancer.saving": null, // Saving…
  "settings.enhancer.saveBtn": null, // 💾 Save
  "settings.enhancer.resetTitle": null, // Restore the built-in engine guidance
  "settings.enhancer.resetBtn": null, // ↺ Reset to built-in
  "settings.enhancer.hideBuiltin": null, // Hide built-in
  "settings.enhancer.viewBuiltin": null, // View built-in
  "settings.enhancer.hidePreview": null, // Hide preview
  "settings.enhancer.previewFullPrompt": null, // Preview full prompt
  "settings.enhancer.builtinJsonTitle": null, // Built-in JSON instructions (structure + guidelines)
  "settings.enhancer.builtinTextTitle": null, // Built-in engine guidance (text mode)
  "settings.enhancer.previewTitle": null, // Full system prompt sent to the LLM ({mode} mode, no active LoRA triggers)
  "settings.enhancer.previewHintJson": null, // JSON mode appends the mandatory LoRA-trigger block (when triggers are active) after your instructions.
  "settings.enhancer.previewHintText": null, // Text mode appends the prose TASK and general prompt-engineering rules after your guidance.
  "settings.enhancer.footerHint": null, // These prompts drive the local prompt enhancer (Qwen2.5-0.5B-Instruct, MLX). Each generated model is mapped to one of these four engines: FLUX.2 Klein / SDXL / Krea 2 / Z-Image Turbo.
  "downloader.starting": null, // Starting download of "{model}" ({base})…
  "downloader.failed": null, // Download failed: {error}
  "downloader.cancelFailed": null, // Failed to cancel: {error}
  "downloader.tagDirectUrl": null, // DIRECT URL
  "downloader.tagDirect": null, // DIRECT
  "downloader.title": null, // Universal LoRA Downloader
  "downloader.detectedHf": null, // 🤗 Hugging Face detected
  "downloader.detectedCivitai": null, // ⚡ Civitai detected
  "downloader.detectedDirect": null, // 🔗 Direct Link detected
  "downloader.hint": null, // Paste a Civitai model URL or ID, a Hugging Face repo ID or URL, or any direct link to a .safetensors file.
  "downloader.urlPlaceholder": null, // Civitai URL or ID, Hugging Face repo or URL, or direct .safetensors link...
  "downloader.startingBtn": null, // Starting…
  "downloader.downloadBtn": null, // ⚡ Download & Register
  "downloader.namePlaceholder": null, // Custom name (optional)
  "downloader.triggersPlaceholder": null, // Trigger words: comma, separated (optional)
  "downloader.autoDetect": null, // Auto-Detect Architecture
  "downloader.cancelTitle": null, // Cancel download
  "downloader.cancelBtn": null, // ✕ Cancel
  "downloader.ready": null, // ✓ Ready
  "downloader.error": null, // ✖ Error
  "downloader.cancelled": null, // Cancelled
  "downloader.authRequiredHf": null, // This model requires authentication on Hugging Face.
  "downloader.authRequiredCivitai": null, // This model requires authentication on Civitai.
  "downloader.configureHfToken": null, // 🔑 Configure HF Token
  "downloader.configureApiKey": null, // 🔑 Configure API Key
  "downloader.installedFor": null, // Installed for {model}.
  "downloader.switchTo": null, // ⚡ Switch to {model}
  "installer.tagWeights": null, // weights
  "installer.statusInstalled": null, // ✓ Installed
  "installer.statusFailed": null, // ✕ Download failed
  "installer.statusCancelled": null, // Download cancelled
  "installer.statusDownloading": null, // ⬇ Downloading…
  "installer.statusNotInstalled": null, // ⬇ Not installed — download weights first
  "installer.startingBtn": null, // Starting…
  "installer.downloadBtn": null, // Download
  "installer.localBtn": null, // 📁 Local…
  "installer.localBtnTitle": null, // Install from a folder already on disk (HF cache or local)
  "installer.pathRequired": null, // Choose a cached repo or enter a folder path
  "installer.localHint": null, // Install from a copy already on your disk — nothing is downloaded or copied; the model loads directly from the folder.
  "installer.scanningCache": null, // Scanning Hugging Face cache…
  "installer.pickCachedRepo": null, // — pick a cached Hugging Face repo —
  "installer.noCachedRepo": null, // — no cached repos found —
  "installer.pathPlaceholder": null, // …or /path/to/model/folder (or an HF cache models--org--name dir)
  "installer.installingBtn": null, // Installing…
  "installer.installLocalBtn": null, // Install from local
  "licences.sectionBackend": null, // Backend — MLX / mflux runtime (Python)
  "licences.sectionFrontend": null, // Frontend — React / Vite SPA
  "licences.sectionSdxl": null, // SDXL engine — Juggernaut XL Lightning (Python, torch-free runtime)
  "licences.sectionModels": null, // Models and weights
  "licences.noteBackend": null, // Core generation stack for FLUX.2-klein 4B, Krea 2 Turbo, Z-Image Turbo and the Qwen-Image 2.1 experimental engine.
  "licences.noteSdxl": null, // Conversion tools ship in venv-sdxl only; the deployed engine runtime is torch-free.
  "licences.noteFrontend": null, // Browser UI. Same packages as any modern Vite app.
  "licences.noteWeights": null, // Weight files are downloaded on first use from Hugging Face / model cards — they are NOT bundled with, nor redistributed by, DiffusionBear. Each carries its own terms.
  "licences.introSecond": null, // Every third-party package below keeps its own licence; nothing here grants or revokes those terms.
  "licences.thisProject": null, // This project
  "licences.releasedUnder": null, // is released under the
  "licences.mitLicense": null, // MIT License
  "licences.copyright": null, // Copyright © {year}
  "licences.aiAuthored": null, // Heavily coded by AI
  "licences.aiCredits": null, // assisted by {credits} together with its human author. Reviewed and benchmarked by hand.
  "licences.mitScopeBefore": null, // The MIT licence covers
  "licences.mitScopeEm": null, // this project's source code only
  "licences.mitScopeAfter": null, // . The model weights are not covered by it — see the weights table below.
  "licences.colPackage": null, // Package
  "licences.colLicence": null, // Licence
  "licences.colRepository": null, // Repository
  "licences.versions": null, // Versions
  "licences.frontendVersion": null, // Frontend {version}
  "licences.backendApi": null, //  · Backend API {version}
  "licences.repoLink": null, // repo
  "tokens.labelCivitai": null, // Civitai API Key
  "tokens.labelHf": null, // Hugging Face Token
  "tokens.configured": null, // Configured
  "tokens.notSet": null, // Not set
  "tokens.editBtn": null, // Edit
  "tokens.setBtn": null, // + Set {label}
  "tokens.savingBtn": null, // Saving…
  "tokens.saveBtn": null, // Save {label}
  "tokens.clearBtn": null, // Clear
  "tokens.civitaiPlaceholder": null, // Civitai API Key (from civitai.red/?ref_code=88C8VEBA)
  "tokens.hfPlaceholder": null, // Hugging Face Token (from huggingface.co/settings/tokens)
  "tokens.savedFeedback": null, // {label} saved securely (local file, never sent to clients).
  "tokens.clearedFeedback": null, // {label} cleared.
  "tokens.saveFailed": null, // Failed to save {label}: {error}
  "app.logoAlt": null, // DiffusionBear
  "app.navBrowser": null, // Browser
  "app.navParams": null, // ⚙️ Parameters
  "fill.title": null, // Generative fill
  "fill.canvasLabel": null, // Region to regenerate
  "fill.prompt": null, // Prompt
  "fill.promptPlaceholder": null, // Describe what should appear in the painted area…
  "fill.tools": null, // Tools
  "fill.paint": null, // Paint
  "fill.erase": null, // Erase
  "fill.clear": null, // Clear
  "fill.brushSize": null, // Brush size
  "fill.engine": null, // Engine
  "fill.coverage": null, // {percent}% of the image selected
  "fill.hint": null, // Paint the region to regenerate, then describe what you want there.
  "fill.hintLargeRegion": null, // Large region: the result may diverge more from the rest of the image, because the engine regenerates everything and then recomposites.
  "fill.submit": null, // Fill this area
  "fill.working": null, // Generating…
  "fill.errorInvalid": null, // The request was refused. Check that the painted area matches the image.
  "fill.errorEngines": null, // Could not load the engine list. Try again.
  "fill.errorFailed": null, // The fill failed. Try again.
  "fill.cancel": null, // Cancel
  "fill.cancelled": null, // Fill cancelled.
  "fill.progressEta": null, // ~{seconds}s left
  "fill.doneBanner": null, // Fill complete — the image has been added to the gallery.
  "fill.cardButton": null, // Fill a region
};

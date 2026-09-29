import { imageUrl, fetchImageBlob, api } from "../api.js";

// In-memory cache for pre-fetched full resolution Files
// Key: image_id -> { file: File, timestamp: number }
const MAX_CACHE_SIZE = 25;
const fileCache = new Map();
const inFlightPromises = new Map();

/**
 * Pre-fetches the full-resolution image and caches it as a File object.
 * Because MLX-DIFFUSION runs locally on localhost:8001, loopback fetch takes <8ms,
 * ensuring the File is ready in memory before human drag movement reaches dragstart.
 */
export async function preloadFullImageFile(imageOrId) {
  if (!imageOrId) return null;
  const id = typeof imageOrId === "string" ? imageOrId : imageOrId.id;
  if (!id) return null;

  // Check cache
  const cached = fileCache.get(id);
  if (cached && Date.now() - cached.timestamp < 10 * 60 * 1000) {
    return cached.file;
  }

  // Deduplicate in-flight requests
  if (inFlightPromises.has(id)) {
    return inFlightPromises.get(id);
  }

  const promise = (async () => {
    try {
      const blob = await fetchImageBlob(id);
      const filename =
        typeof imageOrId === "object" && imageOrId?.file
          ? imageOrId.file
          : `${id}.${typeof imageOrId === "object" && imageOrId?.format ? imageOrId.format : "png"}`;
      const mime = blob.type || (filename.endsWith(".jpeg") || filename.endsWith(".jpg") ? "image/jpeg" : "image/png");
      const file = new File([blob], filename, { type: mime, lastModified: Date.now() });

      // Evict oldest entry if cache exceeds ceiling
      if (fileCache.size >= MAX_CACHE_SIZE) {
        const oldestKey = fileCache.keys().next().value;
        fileCache.delete(oldestKey);
      }

      fileCache.set(id, { file, timestamp: Date.now() });
      return file;
    } catch (err) {
      console.warn(`[dragDrop] Failed preloading full image ${id}:`, err);
      return null;
    } finally {
      inFlightPromises.delete(id);
    }
  })();

  inFlightPromises.set(id, promise);
  return promise;
}

/**
 * Ensures any dragged <img> element immediately swaps from its thumbnail URL
 * (?thumb=true) to the full-resolution original image URL before and during dragstart.
 *
 * In WebKit and Chromium on macOS, dragging an <img> element causes the browser's
 * native Cocoa drag controller to inspect the element's .src attribute directly.
 * By updating the DOM .src synchronously on pointerdown/dragstart, Chromium is guaranteed
 * to register the full-resolution URL on the pasteboard, so macOS Finder and dropzones
 * download and save the real {id}.png rather than the 512px thumbnail.
 */
export function ensureFullResolutionImage(e, id) {
  if (!id) return;
  const fullUrl = imageUrl(id, false);

  // 1. Direct target if it's an img
  if (e?.target && e.target.tagName === "IMG") {
    if (e.target.src !== fullUrl) {
      e.target.src = fullUrl;
    }
  }

  // 2. Current target if it's an img or contains an img
  if (e?.currentTarget) {
    if (e.currentTarget.tagName === "IMG") {
      if (e.currentTarget.src !== fullUrl) {
        e.currentTarget.src = fullUrl;
      }
    } else if (typeof e.currentTarget.querySelector === "function") {
      const img = e.currentTarget.querySelector("img");
      if (img && img.src !== fullUrl) {
        img.src = fullUrl;
      }
    }
  }
}

/**
 * Generates drag-and-drop props for any thumbnail or image element to guarantee
 * that when dragged, the FULL-RESOLUTION image lands on the drop target.
 *
 * Populates 5 channels:
 * 1. Native <img> .src -> Swapped to full resolution for macOS Finder / Desktop image drop
 * 2. dataTransfer.items.add(File) -> For web dropzones (Civitai, Discord, ChatGPT, etc.)
 * 3. DownloadURL -> For macOS Finder / Desktop / local folder drops
 * 4. text/uri-list & text/plain -> For URL-based web targets
 * 5. text/html -> For rich-text editors & Notion
 */
export function bindFullImageDrag(image, extraHandlers = {}) {
  if (!image || !image.id) return {};

  const id = image.id;
  const filename = image.file || `${id}.${image.format || "png"}`;
  const rawUrl = imageUrl(id, false); // false = full resolution, NOT thumb
  const fullUrl = typeof window !== "undefined" ? new URL(rawUrl, window.location.href).href : rawUrl;

  // Hover deliberately does NOT swap src or pre-fetch. Doing it on pointerenter
  // replaced every 265KB thumbnail in the DOM with its 689KB original (with no
  // mouseleave to restore it), so sweeping the mouse across a 24-cell gallery
  // page pulled ~16MB and forced a full-resolution PNG decode per cell. The
  // macOS drag requirement is satisfied at pointerdown, which is what the
  // browser inspects, and the loopback pre-fetch has the whole press-to-drag
  // gesture to land before dragstart.
  const handleInteraction = (e) => {
    ensureFullResolutionImage(e, id);
    preloadFullImageFile(image);
  };

  return {
    draggable: true,
    onPointerEnter: (e) => {
      extraHandlers.onPointerEnter?.(e);
    },
    onMouseEnter: (e) => {
      extraHandlers.onMouseEnter?.(e);
    },
    onPointerDown: (e) => {
      handleInteraction(e);
      extraHandlers.onPointerDown?.(e);
    },
    onMouseDown: (e) => {
      handleInteraction(e);
      extraHandlers.onMouseDown?.(e);
    },
    onDragStart: (e) => {
      ensureFullResolutionImage(e, id);
      e.dataTransfer.effectAllowed = "copyMove";

      // Channel 1: Real File object for web dropzones (Civitai, Discord, ChatGPT)
      const cached = fileCache.get(id);
      if (cached?.file) {
        try {
          e.dataTransfer.items.add(cached.file);
        } catch (err) {
          console.warn("[dragDrop] Error adding File to dataTransfer:", err);
        }
      }

      // Channel 2: DownloadURL for macOS Finder / Desktop / Local folders
      const mime = cached?.file?.type || (filename.endsWith(".jpeg") || filename.endsWith(".jpg") ? "image/jpeg" : "image/png");
      try {
        e.dataTransfer.setData("DownloadURL", `${mime}:${filename}:${fullUrl}`);
      } catch {}

      // Channel 3: Direct URL list for browser tabs & URL dropzones
      try {
        e.dataTransfer.setData("text/uri-list", fullUrl);
        e.dataTransfer.setData("text/plain", fullUrl);
      } catch {}

      // Channel 4: Rich HTML for editors
      try {
        const altText = (image.prompt || filename).replace(/"/g, "&quot;");
        e.dataTransfer.setData("text/html", `<img src="${fullUrl}" alt="${altText}" />`);
      } catch {}

      // Channel 5: App-internal image ID & metadata
      try {
        e.dataTransfer.setData("application/x-mlx-image-id", id);
        e.dataTransfer.setData("application/json", JSON.stringify(image));
      } catch {}

      extraHandlers.onDragStart?.(e);
    },
    ...extraHandlers,
  };
}

/**
 * Copies the full-resolution PNG image directly to the system clipboard
 * so the user can immediately paste (Cmd+V) it into Civitai, chat, Discord, etc.
 */
export async function copyFullImageToClipboard(imageOrId) {
  if (!imageOrId) return false;
  const id = typeof imageOrId === "string" ? imageOrId : imageOrId.id;
  if (!id) return false;

  try {
    const file = await preloadFullImageFile(imageOrId);
    if (!file) {
      const blob = await fetchImageBlob(id);
      await navigator.clipboard.write([
        new ClipboardItem({ [blob.type || "image/png"]: blob }),
      ]);
      return true;
    }
    await navigator.clipboard.write([
      new ClipboardItem({ [file.type || "image/png"]: file }),
    ]);
    return true;
  } catch (err) {
    console.error("[dragDrop] Failed to copy image to clipboard:", err);
    throw err;
  }
}

/**
 * Asks the local backend to reveal the full-resolution image in macOS Finder.
 */
export async function revealImageInFinder(imageOrId) {
  if (!imageOrId) return false;
  const id = typeof imageOrId === "string" ? imageOrId : imageOrId.id;
  if (!id) return false;

  try {
    await api(`/api/images/${id}/reveal`, { method: "POST" });
    return true;
  } catch (err) {
    console.error("[dragDrop] Failed to reveal in Finder:", err);
    return false;
  }
}


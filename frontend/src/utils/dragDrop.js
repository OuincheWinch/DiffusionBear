import { imageUrl, fetchImageBlob, api } from "../api.js";

// In-memory cache for pre-fetched full resolution Files
// Key: image_id -> { file: File, timestamp: number }
const MAX_CACHE_SIZE = 25;
const fileCache = new Map();
const inFlightPromises = new Map();

// The file:// URL the <img> src is switched to, keyed by image id.
//
// This Map is load-bearing in a way the older caches are not: see
// setDragSourceFileUrl below.
const dragSourceCache = new Map();

/**
 * Points an <img> at the file:// URL of the image on disk, and returns the element
 * afterwards so callers can restore it.
 *
 * WHY THIS EXISTS, because the obvious fix does not work:
 *
 * dataTransfer.setData("DownloadURL", ...) is the documented way to drag a file out to
 * Finder, and it is silently ignored in WKWebView. The native drag controller builds the
 * pasteboard from the <img src> it finds under the cursor, and our JS-set flavours are
 * discarded. Reported symptom: dropping a gallery card on the Desktop wrote
 * `127.0.0.1-8001:.webloc` -- a web-location stub, because macOS read an http URL and
 * did what it does with one. The DownloadURL channel was correct the entire time and
 * simply never reached the pasteboard.
 *
 * So the src itself has to become file://. Everything else in this module is unchanged:
 * the File object still serves web dropzones, and text/uri-list still carries the
 * loopback URL for browser tabs.
 */
export function setDragSourceFileUrl(element, fileUrl) {
  if (!element || !fileUrl) return null;
  const img = element.tagName === "IMG"
    ? element
    : typeof element.querySelector === "function"
      ? element.querySelector("img")
      : null;
  if (!img) return null;
  const previous = img.dataset.dragOriginalSrc || img.src;
  img.dataset.dragOriginalSrc = previous;
  img.src = fileUrl;
  dragSourceCache.set(img, previous);
  return img;
}

function restoreDragSource(element) {
  if (!element) return;
  const previous = dragSourceCache.get(element);
  if (previous) {
    element.src = previous;
    dragSourceCache.delete(element);
  }
}

/**
 * The <img> elements under a drag source, so their src can be pointed at disk.
 * Covers both the gallery cell and the detail view's single image.
 */
function findDragImages(node) {
  if (!node) return [];
  if (node.tagName === "IMG") return [node];
  if (typeof node.querySelectorAll !== "function") return [];
  return Array.from(node.querySelectorAll("img"));
}

/**
 * Pre-fetches the full-resolution image and caches it as a File object.
 * Because MLX-DIFFUSION runs locally on localhost:8001, loopback fetch takes <8ms,
 * ensuring the File is ready in memory before human drag movement reaches dragstart.
 */
// Cache for the on-disk file:// URL used by the macOS Finder drag channel.
// Key: image_id -> { fileUrl: string, filename: string, timestamp: number }
const fileUrlCache = new Map();
const MAX_FILE_URL_CACHE = 60;

/**
 * Resolves the image's real path on disk as a file:// URL.
 *
 * This is what makes dragging a card onto the Desktop or into a Finder window produce
 * the actual picture. The alternative -- putting the loopback http URL in DownloadURL --
 * does not work: macOS reads the scheme and treats http as a web link, so the drop
 * created a link stub pointing back at the backend instead of copying the image.
 *
 * Prefetched on pointerdown for the same reason the File is: the whole press-to-drag
 * gesture is the budget, and this is a loopback round-trip.
 */
export async function preloadFileUrl(imageOrId) {
  if (!imageOrId) return null;
  const id = typeof imageOrId === "string" ? imageOrId : imageOrId.id;
  if (!id) return null;

  const cached = fileUrlCache.get(id);
  if (cached) {
    cached.timestamp = Date.now();
    return cached;
  }

  try {
    const data = await api(`/api/images/${id}/file-url`);
    if (!data?.file_url) return null;
    const entry = { fileUrl: data.file_url, filename: data.filename, timestamp: Date.now() };
    if (fileUrlCache.size >= MAX_FILE_URL_CACHE) {
      fileUrlCache.delete(fileUrlCache.keys().next().value);
    }
    fileUrlCache.set(id, entry);
    return entry;
  } catch (err) {
    console.warn(`[dragDrop] Failed resolving file:// URL for ${id}:`, err);
    return null;
  }
}

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
    // Resolved here, read at dragstart. onDragStart is synchronous -- the pasteboard has
    // to be fully populated before it returns -- so anything awaited there would land too
    // late. The round-trip completes during the press, and dragstart reads the cache.
    preloadFileUrl(image);
  };

  // Native shell? Do NOT mark the element draggable.
  //
  // This, not CSS, is what stops WebKit winning. `draggable` sits on the wrapping <div>
  // (and on the detail <img>), and a CSS rule aimed at `img.gallery-thumb` never touched
  // it -- so WebKit kept building the pasteboard from the div and every drag produced an
  // http://127.0.0.1 link, which Finder saved as a .webloc. Verified by the symptom: with
  // the CSS rule in place and still getting the http URL, the suppression was simply on the
  // wrong element.
  //
  // Removing the attribute is deterministic where CSS was a guess. The AppWebView starts a
  // real NSDraggingSession from mouseDown instead, with a real file:// URL.
  const nativeDrag = Boolean(nativeBridge());

  // Suppress the element drag inline, on the images as well as the wrapper.
  //
  // Two things this replaces, both of which were assumptions:
  //   * A CSS rule scoped to body.native-shell. Whether the shell actually adds that
  //     class could not be confirmed, and if it did not, every drag silently fell back
  //     to WebKit's own and produced an http:// link again.
  //   * Setting draggable on the wrapper only. A plain <img> is draggable in WebKit on
  //     its own account; a draggable=false ancestor does not make its images
  //     undraggable.
  //
  // A plain <img> is the likely reason the http URL survived four attempts: it drags as
  // the last-loaded http resource, which is exactly what was reported.
  return {
    draggable: !nativeDrag,
    style: nativeDragSuppression(),
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
      // The load-bearing step. WKWebView's drag controller reads the <img src> to build
      // the pasteboard, so pointing it at the real file on disk is what makes Finder
      // copy the image instead of writing a .webloc for the loopback URL.
      const onDisk = fileUrlCache.get(id);
      const dragged = findDragImages(e.target);
      if (onDisk?.fileUrl) {
        dragged.forEach((img) => setDragSourceFileUrl(img, onDisk.fileUrl));
      } else {
        // Not resolved yet. The http URL is still better than nothing -- it is at least
        // a real reachable image for web targets -- and pointerdown normally wins this
        // race anyway.
        dragged.forEach((img) => ensureFullResolutionImage(img, id));
      }
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

      // Channel 2: DownloadURL.
      //
      // Kept because it is correct and it IS honoured by Safari and by any target that
      // reads the flavour directly. It is NOT the fix for WKWebView -- see
      // setDragSourceFileUrl, which is what actually reaches Finder's pasteboard. Do
      // not remove this on the assumption it is dead weight: the web targets still
      // consume it.
      const mime = cached?.file?.type || (filename.endsWith(".jpeg") || filename.endsWith(".jpg") ? "image/jpeg" : "image/png");
      const downloadTarget = onDisk?.fileUrl || fullUrl;
      const downloadName = onDisk?.filename || filename;
      try {
        e.dataTransfer.setData("DownloadURL", `${mime}:${downloadName}:${downloadTarget}`);
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
    onDragEnd: (e) => {
      // The src is left pointing at file:// after a drag, which would show the raw
      // file rather than the served image if the same node is re-rendered. Restoring
      // here also keeps the browser's own image cache coherent.
      findDragImages(e.target).forEach(restoreDragSource);
      extraHandlers.onDragEnd?.(e);
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
/**
 * Saves a full-resolution image through a native save panel.
 *
 * The reliable way out of the app, because dragging does not work for this: macOS builds
 * the drag pasteboard from the DOM element, so a gallery image can only ever be dragged
 * out as an `http://127.0.0.1:8001/...` URL, and dropping that onto the Desktop writes a
 * `.webloc` link stub -- pointing at a server that stops answering the moment the app
 * quits. `DownloadURL` is a Safari-only flavour (Chrome ignores it) and WKWebView throws
 * the JS-set variants away entirely. None of that is reachable from JavaScript, so the
 * shell owns it.
 *
 * Returns false when the bridge is absent, i.e. in a browser during development, where
 * the plain <a download> path is the fallback.
 */
export function exportImageNatively(imageOrId) {
  const id = typeof imageOrId === "string" ? imageOrId : imageOrId?.id;
  const bridge = nativeBridge();
  if (!id || !bridge) return false;
  try {
    bridge.postMessage({ action: "export", imageId: id });
    return true;
  } catch (err) {
    console.warn("[dragDrop] native export unavailable:", err);
    return false;
  }
}

function nativeBridge() {
  return typeof window !== "undefined" ? window.webkit?.messageHandlers?.native ?? null : null;
}

/**
 * Inline style that stops an <img> dragging itself.
 *
 * Applied to the images directly rather than via a stylesheet. A plain <img> is draggable
 * in WebKit on its own account -- `draggable=false` on an ancestor does not change that --
 * and an image dragging itself offers its last-loaded resource, which for this gallery is
 * an http://127.0.0.1 URL. Finder then saves that as a .webloc instead of copying the file.
 *
 * Returns undefined in a browser, where the web-layer drag should keep working.
 */
export function nativeDragSuppression() {
  return nativeBridge() ? { WebkitUserDrag: "none", WebkitUserDraggable: "no-drag" } : undefined;
}


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


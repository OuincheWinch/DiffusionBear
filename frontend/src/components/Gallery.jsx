import { useCallback, useEffect, useRef, useState } from "react";
import { api, imageUrl } from "../api";
import {
  bindFullImageDrag,
  copyFullImageToClipboard,
  revealImageInFinder,
  exportImageNatively,
  nativeDragSuppression,
} from "../utils/dragDrop";
import { copyText } from "../clipboard";
import ImageActions from "./ImageActions";
import LazyGalleryImage from "./LazyGalleryImage";
import FillBrush from "./FillBrush";
import { useI18n } from "../i18n/I18nContext";

export default function Gallery({ refreshKey, onReuse, onSetReference, maxReferenceImages = 1, activeTab = "browser", newImage = null, onImageDeleted }) {
  const { t } = useI18n();
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [query, setQuery] = useState("");
  const [tags, setTags] = useState("");
  const [sort, setSort] = useState("newest");
  const [model, setModel] = useState("");
  const [models, setModels] = useState([]);
  const [lora, setLora] = useState("");
  const [loraStats, setLoraStats] = useState({ total: 0, with_lora: 0, without_lora: 0, loras: [] });
  const [favoriteOnly, setFavoriteOnly] = useState(false);
  const [revealedBlurredIds, setRevealedBlurredIds] = useState(() => new Set());
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);
  const [galleryError, setGalleryError] = useState(null);
  const limit = 24;

  const abortRef = useRef(null);

  // Typing in the search boxes used to re-run the full gallery query per
  // keystroke, and each run copies + filters + sorts the whole index server-side
  // (~1469 entries) for a result that is thrown away a few ms later. The inputs
  // stay instant; only the request is debounced.
  const [debouncedQuery, setDebouncedQuery] = useState(query);
  const [debouncedTags, setDebouncedTags] = useState(tags);
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
      setDebouncedTags(tags);
    }, 200);
    return () => clearTimeout(timer);
  }, [query, tags]);

  useEffect(() => {
    api("/api/models").then(setModels).catch(() => {});
  }, []);

  // When a newly generated image arrives, prepend it immediately to items and reset to page 1
  useEffect(() => {
    if (!newImage || !newImage.id) return;
    setItems((prev) => {
      if (prev.some((it) => it.id === newImage.id)) return prev;
      return [newImage, ...prev];
    });
    setTotal((t) => t + 1);
    setPage(1);
  }, [newImage]);

  useEffect(() => {
    if (activeTab !== "browser") return;
    let alive = true;
    api(`/api/gallery/loras${model ? `?model=${encodeURIComponent(model)}` : ""}`)
      .then((stats) => {
        if (alive && stats) setLoraStats(stats);
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [model, refreshKey, activeTab]);

  const labelFor = (repo) =>
    models.find((m) => m.repo === repo)?.label ??
    (repo?.includes("z-image") ? "Z-Image Turbo" : repo || "FLUX.2-klein 4B");

  const lastLoadedPageRef = useRef(null);

  const fetchPage = useCallback(
    async (targetPage, signal) => {
      const params = new URLSearchParams({
        query: debouncedQuery.trim(),
        tags: debouncedTags,
        sort,
        page: targetPage,
        limit,
        model,
        lora,
        ...(favoriteOnly ? { favorite: "true" } : {}),
      });
      return api(`/api/gallery?${params}`, { signal });
    },
    [debouncedQuery, debouncedTags, sort, limit, model, lora, favoriteOnly]
  );

  const toggleFavorite = useCallback(async (imageId, newStatus) => {
    setItems((prev) =>
      prev.map((it) => (it.id === imageId ? { ...it, favorite: newStatus } : it))
    );
    setSelected((sel) => (sel?.id === imageId ? { ...sel, favorite: newStatus } : sel));
    try {
      await api(`/api/images/${imageId}/flags`, {
        method: "POST",
        body: JSON.stringify({ favorite: newStatus }),
      });
    } catch (err) {
      console.warn("[Gallery] Failed to update favorite flag:", err);
      setItems((prev) =>
        prev.map((it) => (it.id === imageId ? { ...it, favorite: !newStatus } : it))
      );
      setSelected((sel) => (sel?.id === imageId ? { ...sel, favorite: !newStatus } : sel));
    }
  }, []);

  const toggleBlur = useCallback(async (imageId, newStatus) => {
    if (newStatus) {
      setRevealedBlurredIds((prev) => {
        const next = new Set(prev);
        next.delete(imageId);
        return next;
      });
    }
    setItems((prev) =>
      prev.map((it) => (it.id === imageId ? { ...it, blurred: newStatus } : it))
    );
    setSelected((sel) => (sel?.id === imageId ? { ...sel, blurred: newStatus } : sel));
    try {
      await api(`/api/images/${imageId}/flags`, {
        method: "POST",
        body: JSON.stringify({ blurred: newStatus }),
      });
    } catch (err) {
      console.warn("[Gallery] Failed to update blur flag:", err);
      setItems((prev) =>
        prev.map((it) => (it.id === imageId ? { ...it, blurred: !newStatus } : it))
      );
      setSelected((sel) => (sel?.id === imageId ? { ...sel, blurred: !newStatus } : sel));
    }
  }, []);

  const load = useCallback(async (force = false) => {
    if (!force && lastLoadedPageRef.current === page) return;
    lastLoadedPageRef.current = page;
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    try {
      const data = await fetchPage(page, controller.signal);
      setItems(data.items || []);
      setTotal(data.total || 0);
    } catch (e) {
      if (e.name !== "AbortError") {
        /* keep previous items on transient failures */
      }
    }
  }, [fetchPage, page]);

  // Load immediately on tab switch, refreshKey or filter changes
  useEffect(() => {
    if (activeTab !== "browser") return;
    lastLoadedPageRef.current = null;
    load();
    return () => {
      abortRef.current?.abort();
    };
  }, [load, refreshKey, activeTab]);

  // Recalculate gallery immediately when an image is deleted anywhere in the app
  useEffect(() => {
    const handleImageDeleted = (e) => {
      const deletedId = e?.detail?.id;
      if (!deletedId) return;
      // deleteImage already decremented and refetched; reacting to our own
      // broadcast would decrement twice and repaint through a stale page.
      if (e.detail?.source === "gallery") return;
      setItems((prev) => prev.filter((it) => it.id !== deletedId));
      setTotal((t) => Math.max(0, t - 1));
      lastLoadedPageRef.current = null;
      if (activeTab === "browser") {
        load(true);
      }
    };
    window.addEventListener("mlx:image-deleted", handleImageDeleted);
    return () => window.removeEventListener("mlx:image-deleted", handleImageDeleted);
  }, [activeTab, load]);

  // True only inside the native shell. In a browser the bridge is absent and the
  // <a download> fallback is the correct thing to render.
  const hasNativeBridge = () =>
    typeof window !== "undefined" && Boolean(window.webkit?.messageHandlers?.native);

  // Hand the shell the on-screen rect of every image, so a drag can start natively
  // without a JavaScript round-trip at press time.
  //
  // The dependency is a SIGNATURE OF THE IDS, not items.length. That distinction was the
  // bug reported as "I grab one image and get a different one": a new generation refreshes
  // the gallery with the same page size, so the length is unchanged, the effect never
  // re-ran, and the shell kept the previous ids at the previous positions -- which is the
  // state after every generation. `selected?.id` is included because the detail view's
  // image sits above the cells and must win the hit test.
  // Report which image is under the cursor, as the cursor moves.
  //
  // This is what the shell uses to resolve a drag, in preference to the rect index it
  // used before. The index drifted from the live layout and produced the reported
  // "grab N, drop N+4" -- exactly one grid row -- and then, once the grid went to six
  // columns, cells that resolved to nothing at all. Any cache of geometry has to be
  // invalidated by every possible relayout and that list is unbounded.
  //
  // The page hit-tests instead, because WebKit exposes no way for the shell to convert a
  // mouse point into page coordinates. A pointermove always precedes the mouseDown that
  // starts a drag, so the answer is current by construction.
  //
  // Throttled with a timer rather than requestAnimationFrame: rAF stops entirely when the
  // window is occluded, and a cursor position that has silently stopped updating resolves
  // to whatever was last seen -- the very bug this removes.
  useEffect(() => {
    if (!hasNativeBridge()) return undefined;
    const bridge = window.webkit.messageHandlers.native;

    let last = null;
    let pending = null;
    let timer = null;
    const flush = () => {
      timer = null;
      const point = pending;
      pending = null;
      if (!point) return;
      const hit = document.elementFromPoint(point.x, point.y);
      const target = hit && hit.closest ? hit.closest("[data-mlx-file-url]") : null;
      const fileUrl = target?.getAttribute("data-mlx-file-url") || null;
      if (fileUrl === last) return;
      last = fileUrl;
      try {
        bridge.postMessage({
          action: "dragCandidate",
          imageId: target?.getAttribute("data-mlx-image-id") || null,
          fileUrl,
        });
      } catch (err) {
        // No native shell (the dev server in a browser), or the bridge refused the
        // message. Drag-and-drop to the Desktop is unavailable, and the user should be
        // able to find that out from the console rather than by wondering.
        console.debug("[DiffusionBear] drag bridge unavailable:", err);
      }
    };

    const onMove = (event) => {
      pending = { x: event.clientX, y: event.clientY };
      if (timer == null) timer = window.setTimeout(flush, 16);
    };
    window.addEventListener("pointermove", onMove, { passive: true });

    return () => {
      window.removeEventListener("pointermove", onMove);
      if (timer != null) window.clearTimeout(timer);
    };
  }, []);

  // The list rows are the full sidecar dicts, so opening one needs no extra
  // round-trip to /api/images/{id}.
  function openDetail(item) {
    setSelected(item);
  }

  const selectedIndex = selected
    ? items.findIndex((it) => it.id === selected.id)
    : -1;

  const pages = Math.max(1, Math.ceil(total / limit));
  const isPagingRef = useRef(false);

  const navigateDetail = useCallback(
    async (delta) => {
      if (isPagingRef.current) return;
      if (selectedIndex < 0) return;
      const targetIndex = selectedIndex + delta;

      if (targetIndex >= 0 && targetIndex < items.length) {
        setSelected(items[targetIndex]);
        return;
      }

      if (delta > 0 && page < pages) {
        isPagingRef.current = true;
        try {
          const nextPage = page + 1;
          const data = await fetchPage(nextPage);
          if (data?.items?.length) {
            lastLoadedPageRef.current = nextPage;
            setPage(nextPage);
            setItems(data.items);
            if (data.total != null) setTotal(data.total);
            setSelected(data.items[0]);
          }
        } catch (err) {
          console.warn("[Gallery] Next page navigation error:", err);
        } finally {
          isPagingRef.current = false;
        }
        return;
      }

      if (delta < 0 && page > 1) {
        isPagingRef.current = true;
        try {
          const prevPage = page - 1;
          const data = await fetchPage(prevPage);
          if (data?.items?.length) {
            lastLoadedPageRef.current = prevPage;
            setPage(prevPage);
            setItems(data.items);
            if (data.total != null) setTotal(data.total);
            setSelected(data.items[data.items.length - 1]);
          }
        } catch (err) {
          console.warn("[Gallery] Prev page navigation error:", err);
        } finally {
          isPagingRef.current = false;
        }
      }
    },
    [selectedIndex, items, page, pages, fetchPage]
  );

  useEffect(() => {
    if (!selected) return;
    function onKey(e) {
      if (["INPUT", "TEXTAREA"].includes(e.target?.tagName)) return;
      if (e.key === "ArrowLeft") navigateDetail(-1);
      else if (e.key === "ArrowRight") navigateDetail(1);
      else if (e.key === "Escape") setSelected(null);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selected, navigateDetail]);

  async function deleteImage(id) {
    try {
      await api(`/api/images/${id}`, { method: "DELETE" });
      setSelected((prev) => (prev?.id === id ? null : prev));

      // 1. Optimistic removal & count update
      setItems((prev) => prev.filter((it) => it.id !== id));
      const nextTotal = Math.max(0, total - 1);
      setTotal(nextTotal);

      // 2. Adjust page if last item on page > 1 was deleted
      const maxPages = Math.max(1, Math.ceil(nextTotal / limit));
      let targetPage = page;
      if (page > maxPages) {
        targetPage = maxPages;
        setPage(targetPage);
      }

      // 3. Clear cache ref so load() is never ignored
      lastLoadedPageRef.current = null;

      // 4. Force reload page to fill vacated slot from server
      const controller = new AbortController();
      abortRef.current?.abort();
      abortRef.current = controller;
      try {
        const data = await fetchPage(targetPage, controller.signal);
        lastLoadedPageRef.current = targetPage;
        setItems(data.items || []);
        setTotal(data.total ?? nextTotal);
      } catch (e) {
        if (e.name !== "AbortError") {
          console.warn("[Gallery] Failed to refresh page after deletion:", e);
        }
      }

      // 5. Refresh LoRA stats for accurate counts in filter dropdown
      api(`/api/gallery/loras${model ? `?model=${encodeURIComponent(model)}` : ""}`)
        .then((stats) => {
          if (stats) setLoraStats(stats);
        })
        .catch(() => {});

      // 6. Notify parent & broadcast deletion event
      onImageDeleted?.(id);
      window.dispatchEvent(new CustomEvent("mlx:image-deleted", { detail: { id, source: "gallery" } }));
    } catch (err) {
      console.error("[Gallery] Failed to delete image:", err);
      setCopyError(err?.message || t("gallery.deleteFailed", { message: err?.message || "" }) || "Delete failed");
    }
  }

  async function saveTags(id, tagString) {
    const updated = await api(`/api/images/${id}/tags`, {
      method: "POST",
      body: JSON.stringify({
        tags: tagString.split(",").map((t) => t.trim()).filter(Boolean),
      }),
    });
    // Same ordering as upscale: the list has to contain the new id before the
    // detail view points at it, or the panel renders a selection that is not on
    // screen.
    lastLoadedPageRef.current = null;
    await load(true);
    setSelected(updated);
  }

  const [copied, setCopied] = useState(false);
  const [promptCopied, setPromptCopied] = useState(false);
  const [imageCopied, setImageCopied] = useState(false);
  const [revealed, setRevealed] = useState(false);
  // The image being filled, or null. Null means the brush is closed.
  const [fillTarget, setFillTarget] = useState(null);
  const [fillResult, setFillResult] = useState(null);
  const [cardCopiedId, setCardCopiedId] = useState(null);
  // The SCALE, not a boolean: the shared action bar shows which button is busy.
  // A plain boolean makes both buttons read "Upscaling..." at once.
  const [upscaling, setUpscaling] = useState(null);

  const copyImage = useCallback(async (img = selected) => {
    if (!img?.id) return;
    try {
      await copyFullImageToClipboard(img);
      if (img.id === selected?.id) {
        setImageCopied(true);
        setTimeout(() => setImageCopied(false), 2500);
      }
      setCardCopiedId(img.id);
      setTimeout(() => setCardCopiedId(null), 2000);
    } catch {
      window.open(imageUrl(img.id), "_blank");
    }
  }, [selected]);

  const handleReveal = useCallback(async (img = selected) => {
    if (!img?.id) return;
    const ok = await revealImageInFinder(img);
    if (ok) {
      setRevealed(true);
      setTimeout(() => setRevealed(false), 2500);
    }
  }, [selected]);

  useEffect(() => {
    if (!selected) return;
    const handleKeyDown = (e) => {
      if (["INPUT", "TEXTAREA"].includes(e.target?.tagName)) return;
      if ((e.metaKey || e.ctrlKey) && e.key === "c") {
        e.preventDefault();
        copyImage(selected);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selected, copyImage]);

  // Copy failures are shown, not swallowed. A button that silently does nothing is
  // indistinguishable from a broken app.
  const [copyError, setCopyError] = useState("");
  // The prompt is clamped until asked for; a 400-word prompt should not push the
  // metadata and the buttons off the panel.
  const [promptExpanded, setPromptExpanded] = useState(false);
  useEffect(() => { setPromptExpanded(false); }, [selected?.id]);
  useEffect(() => {
    if (!copyError) return undefined;
    const timer = window.setTimeout(() => setCopyError(""), 4000);
    return () => window.clearTimeout(timer);
  }, [copyError]);

  async function copySeed() {
    // Reports failure instead of swallowing it. `navigator.clipboard` does not exist
    // outside a secure context, so from another machine on the LAN these two buttons
    // did absolutely nothing and said nothing.
    const ok = await copyText(String(selected.seed));
    if (!ok) {
      setCopyError(t("gallery.copyFailed", { what: t("gallery.seedCopiedBtn") }));
      return;
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function copyPrompt() {
    const ok = await copyText(selected.prompt || "");
    if (!ok) {
      setCopyError(t("gallery.copyFailed", { what: t("gallery.copyPromptBtn") }));
      return;
    }
    setPromptCopied(true);
    setTimeout(() => setPromptCopied(false), 1500);
  }


  async function handleUpscale(scale = 2) {
    setUpscaling(scale);
    try {
      const upscaled = await api(`/api/images/${selected.id}/upscale`, {
        method: "POST",
        body: JSON.stringify({ scale }),
      });
      // await load(), do not fire and forget. It was called without await, so the
      // detail view kept rendering the OLD id while the grid was still on the
      // previous page; the user saw nothing happen, pressed upscale again, and
      // every press made another upscale. Several near-identical upscaled images in
      // the gallery were that loop, not a backend fault.
      lastLoadedPageRef.current = null;
      await load(true);
      setSelected(upscaled);
    } catch (e) {
      setGalleryError(t("gallery.upscaleError", { message: e.message || e }));
    } finally {
      setUpscaling(null);
    }
  }

  return (
    <div className="gallery">
      {fillTarget && (
        <FillBrush
          image={fillTarget}
          imageUrl={imageUrl(fillTarget.id)}
          onClose={() => setFillTarget(null)}
          onComplete={(result) => {
            setFillTarget(null);
            setFillResult(result);
            // The new image has to reach the grid, not just the backend: a fill
            // writes a sidecar and the index, but this view renders from `items`.
            lastLoadedPageRef.current = null;
            load(true);
          }}
        />
      )}
      {fillResult && (
        <div className="fill-done-banner" role="status">
          {t("fill.doneBanner")}
          <button type="button" className="btn-mini" onClick={() => setFillResult(null)}>
            {t("app.dismiss")}
          </button>
        </div>
      )}
      {galleryError && (
        <div className="gallery-error-banner" role="alert">
          <span>{galleryError}</span>
          <button type="button" className="btn-mini" onClick={() => setGalleryError(null)}>
            {t("app.dismiss")}
          </button>
        </div>
      )}
      <div className="gallery-controls">
        <div className="gallery-search-wrap">
          <svg className="gallery-search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            className="search"
            placeholder={t("gallery.searchPlaceholder")}
            aria-label={t("gallery.searchPlaceholder")}
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
          />
        </div>
        <input
          className="gallery-tags-input"
          placeholder={t("gallery.tagsPlaceholder")}
          aria-label={t("gallery.tagsPlaceholder")}
          value={tags}
          onChange={(e) => {
            setTags(e.target.value);
            setPage(1);
          }}
        />
        <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label={t("gallery.sortNewest")}>
          <option value="newest">{t("gallery.sortNewest")}</option>
          <option value="oldest">{t("gallery.sortOldest")}</option>
        </select>
        <select
          className="model-filter"
          value={model}
          onChange={(e) => {
            setModel(e.target.value);
            setPage(1);
          }}
          aria-label={t("gallery.allModels")}
        >
          <option value="">{t("gallery.allModels")}</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <select
          className="lora-filter"
          value={lora}
          onChange={(e) => {
            setLora(e.target.value);
            setPage(1);
          }}
          title={t("gallery.loraFilterTitle")}
          aria-label={t("gallery.loraFilterTitle")}
        >
          <option value="">{t("gallery.loraOptionAll", { count: loraStats?.total ?? total })}</option>
          <option value="__none__">{t("gallery.loraOptionNone", { count: loraStats?.without_lora ?? 0 })}</option>
          <option value="__any__">{t("gallery.loraOptionAny", { count: loraStats?.with_lora ?? 0 })}</option>
          {loraStats?.loras?.length > 0 && (
            <optgroup label={t("gallery.loraOptgroup")}>
              {loraStats.loras.map((l) => (
                <option key={l.name} value={l.name}>
                  {l.name} ({l.count})
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          type="button"
          className={`gallery-fav-filter-btn ${favoriteOnly ? "active" : ""}`}
          onClick={() => {
            setFavoriteOnly((v) => !v);
            setPage(1);
          }}
          title={favoriteOnly ? t("gallery.favFilterActiveTitle") : t("gallery.favFilterTitle")}
          aria-label={t("gallery.favFilterBtn")}
          aria-pressed={favoriteOnly}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill={favoriteOnly ? "currentColor" : "none"} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
          <span>{t("gallery.favFilterBtn")}</span>
        </button>
        <span className="count">
          {total === 1 ? t("gallery.countOne", { count: total }) : t("gallery.countMany", { count: total })}
        </span>
      </div>

      <div className="gallery-lora-tabs-bar">
        <div className="gallery-lora-tabs-header">
          <span className="lora-tabs-title">{t("gallery.loraPrefix")}</span>
          {lora && (
            <button
              type="button"
              className="lora-clear-btn"
              onClick={() => {
                setLora("");
                setPage(1);
              }}
              title={t("gallery.loraResetTitle")}
            >
              {t("gallery.loraClearBtn")}
            </button>
          )}
        </div>
        <div className="gallery-lora-tabs">
          <button
            type="button"
            className={`lora-tab-btn ${lora === "" ? "active" : ""}`}
            onClick={() => {
              setLora("");
              setPage(1);
            }}
          >
            {t("gallery.loraTabAll")} <span className="lora-tab-count">{loraStats?.total ?? total}</span>
          </button>
          <button
            type="button"
            className={`lora-tab-btn ${lora === "__none__" ? "active" : ""}`}
            onClick={() => {
              setLora(lora === "__none__" ? "" : "__none__");
              setPage(1);
            }}
          >
            {t("gallery.loraTabNone")} <span className="lora-tab-count">{loraStats?.without_lora ?? 0}</span>
          </button>
          <button
            type="button"
            className={`lora-tab-btn ${lora === "__any__" ? "active" : ""}`}
            onClick={() => {
              setLora(lora === "__any__" ? "" : "__any__");
              setPage(1);
            }}
          >
            {t("gallery.loraTabAny")} <span className="lora-tab-count">{loraStats?.with_lora ?? 0}</span>
          </button>
          {loraStats?.loras?.map((l) => (
            <button
              key={l.name}
              type="button"
              className={`lora-tab-btn ${lora === l.name ? "active" : ""}`}
              onClick={() => {
                setLora(lora === l.name ? "" : l.name);
                setPage(1);
              }}
              title={l.count === 1 ? t("gallery.loraTabTitleOne", { name: l.name, count: l.count }) : t("gallery.loraTabTitleMany", { name: l.name, count: l.count })}
            >
              <span className="lora-tab-name">{l.name}</span>
              <span className="lora-tab-count">{l.count}</span>
            </button>
          ))}
        </div>
      </div>

      {items.length === 0 ? (
        <p className="hint">{favoriteOnly ? t("gallery.emptyFavorites") : t("gallery.empty")}</p>
      ) : (
        <div className="grid">
          {items.map((item) => {
            const isBlurred = Boolean(item.blurred);
            const isFav = Boolean(item.favorite);
            return (
              <div
                key={item.id}
                role="button"
                tabIndex={0}
                className={`cell ${isBlurred ? "cell-blurred" : ""}`}
                onClick={() => openDetail(item)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    openDetail(item);
                  }
                }}
                title={t("gallery.cellTitle")}
                {...bindFullImageDrag(item)}
              >
                <LazyGalleryImage item={item} isBlurred={isBlurred} />
                {isFav && (
                  <div className="cell-fav-badge" title={t("gallery.favFilterBtn")}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="#eab308" stroke="#ca8a04" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                    </svg>
                  </div>
                )}
                {isBlurred && (
                  <div className="cell-blurred-badge" title={t("gallery.blurredBadge")}>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                    <span>{t("gallery.blurredBadge")}</span>
                  </div>
                )}
                <div className="cell-quick-actions" onClick={(e) => e.stopPropagation()}>
                  <button
                    type="button"
                    className={`cell-quick-btn cell-fav-btn ${isFav ? "active" : ""}`}
                    title={isFav ? t("gallery.favoriteRemove") : t("gallery.favoriteAdd")}
                    aria-label={isFav ? t("gallery.favoriteRemove") : t("gallery.favoriteAdd")}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleFavorite(item.id, !isFav);
                    }}
                  >
                    <svg width="13" height="13" viewBox="0 0 24 24" fill={isFav ? "#eab308" : "none"} stroke={isFav ? "#ca8a04" : "currentColor"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    className={`cell-quick-btn cell-blur-btn ${isBlurred ? "active" : ""}`}
                    title={isBlurred ? t("gallery.blurToggleOff") : t("gallery.blurToggleOn")}
                    aria-label={isBlurred ? t("gallery.blurToggleOff") : t("gallery.blurToggleOn")}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleBlur(item.id, !isBlurred);
                    }}
                  >
                    {isBlurred ? (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    ) : (
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                        <line x1="1" y1="1" x2="23" y2="23" />
                      </svg>
                    )}
                  </button>
                  <button
                    type="button"
                    className="cell-quick-btn"
                    title={t("gallery.cellCopyTitle")}
                    aria-label={t("gallery.cellCopyTitle")}
                    onClick={(e) => {
                      e.stopPropagation();
                      copyImage(item);
                    }}
                  >
                  {cardCopiedId === item.id ? (
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  ) : (
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                    </svg>
                  )}
                </button>
                <button
                  type="button"
                  className="cell-quick-btn"
                  title={t("gallery.cellRevealTitle")}
                  aria-label={t("gallery.cellRevealTitle")}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleReveal(item);
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                </button>
                <button
                  type="button"
                  className="cell-quick-btn cell-delete-btn"
                  title={t("gallery.deleteBtn")}
                  aria-label={`${t("gallery.deleteBtn")} — ${item.prompt || item.id}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (window.confirm(t("gallery.deleteConfirm"))) {
                      deleteImage(item.id);
                    }
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
                {/* Fill is deliberately NOT inside .cell-quick-actions. Those are
                    opacity:0 until hover, which suits copy and reveal but not an
                    action you have to see in order to start. A fill also needs the
                    full-resolution image under the brush -- a thumbnail would make
                    the user judge the result at thumbnail resolution. */}
                <button
                  type="button"
                  className="cell-fill-btn"
                  title={t("fill.cardButton")}
                  aria-label={`${t("fill.cardButton")} — ${labelFor(item.model)}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    setFillTarget(item);
                  }}
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="m18 15-6-6" />
                    <path d="m14 2 8 8" />
                    <path d="m11 5 8 8" />
                    <path d="m9 13-5 5v3h3l5-5" />
                  </svg>
                </button>
              </div>
              <span className="cell-model-badge">{labelFor(item.model)}</span>
              <div className="cell-caption">{item.prompt}</div>
            </div>
          );
        })}
        </div>
      )}

      {pages > 1 && (
        <div className="pagination">
          <button disabled={page <= 1} onClick={() => setPage(page - 1)}>
            ←
          </button>
          <span>
            {page} / {pages}
          </span>
          <button disabled={page >= pages} onClick={() => setPage(page + 1)}>
            →
          </button>
        </div>
      )}

      {selected && (
        <div className="modal" onClick={() => setSelected(null)}>
          {(selectedIndex > 0 || page > 1) && (
            <button
              className="nav-arrow nav-prev"
              onClick={(e) => {
                e.stopPropagation();
                navigateDetail(-1);
              }}
              title={t("gallery.detailPrevTitle")}
            >
              ←
            </button>
          )}
          {selectedIndex >= 0 && (selectedIndex < items.length - 1 || page < pages) && (
            <button
              className="nav-arrow nav-next"
              onClick={(e) => {
                e.stopPropagation();
                navigateDetail(1);
              }}
              title={t("gallery.detailNextTitle")}
            >
              →
            </button>
          )}
<div className="modal-body" onClick={(e) => e.stopPropagation()}>
              {(() => {
                const isModalBlurred = Boolean(selected.blurred && !revealedBlurredIds.has(selected.id));
                return (
                  <div
                    className="gallery-detail-frame"
                    data-is-blurred={isModalBlurred ? "true" : undefined}
                    onClick={
                      isModalBlurred
                        ? () => setRevealedBlurredIds((prev) => new Set(prev).add(selected.id))
                        : undefined
                    }
                  >
                    <img
                      className={`gallery-detail-img${isModalBlurred ? " detail-img-blurred" : ""}`}
                      style={nativeDragSuppression()}
                      data-mlx-image-id={selected.id}
                      data-mlx-file-url={selected.file_url || ""}
                      src={imageUrl(selected.id)}
                      alt={selected.prompt}
                      title={isModalBlurred ? t("gallery.clickToUnblur") : t("gallery.dragFullResTitle")}
                      {...(!isModalBlurred ? bindFullImageDrag(selected) : {})}
                    />
                    {isModalBlurred && (
                      <div
                        className="detail-blur-overlay"
                        role="button"
                        tabIndex={0}
                        aria-label={t("gallery.clickToUnblur")}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            setRevealedBlurredIds((prev) => new Set(prev).add(selected.id));
                          }
                        }}
                      >
                        <span className="blur-overlay-pill">
                          <svg
                            width="14"
                            height="14"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                            <circle cx="12" cy="12" r="3" />
                          </svg>
                          {t("gallery.clickToUnblur")}
                        </span>
                      </div>
                    )}
                  </div>
                );
              })()}
            <div className="detail">
              <p className={`detail-prompt${promptExpanded ? " expanded" : ""}`}>
                {selected.prompt}
              </p>
              {(selected.prompt || "").length > 280 && (
                <button
                  type="button"
                  className="btn-mini"
                  onClick={() => setPromptExpanded((v) => !v)}
                >
                  {promptExpanded ? t("gallery.promptLess") : t("gallery.promptMore")}
                </button>
              )}
              <dl>
                <dt>{t("gallery.fieldModel")}</dt>
                <dd>{labelFor(selected.model)}</dd>
                <dt>{t("gallery.fieldSeed")}</dt>
                <dd>{selected.seed}</dd>
                <dt>{t("gallery.fieldSize")}</dt>
                <dd>
                  {selected.width} × {selected.height}
                </dd>
                <dt>{t("gallery.fieldSteps")}</dt>
                <dd>{selected.steps}</dd>
                <dt>{t("gallery.fieldGuidance")}</dt>
                <dd>{selected.guidance}</dd>
                {selected.sampler && (
                  <>
                    <dt>{t("gallery.fieldSampler")}</dt>
                    <dd>{selected.sampler}</dd>
                  </>
                )}
                {selected.negative_prompt && (
                  <>
                    <dt>{t("gallery.fieldNegative")}</dt>
                    <dd>{selected.negative_prompt}</dd>
                  </>
                )}
                {selected.quantization != null && (
                  <>
                    <dt>{t("gallery.fieldQuantization")}</dt>
                    <dd>{selected.quantization}-bit</dd>
                  </>
                )}
                <dt>{t("gallery.fieldTime")}</dt>
                <dd>{selected.generation_time}s</dd>
                {selected.loras?.length > 0 && (
                  <>
                    <dt>{t("gallery.fieldLoras")}</dt>
                    <dd>
                      {selected.loras
                        .map((l) => {
                          const name = l.name || (l.path ? l.path.split("/").pop() : String(l));
                          return l.scale !== undefined ? `${name} (${l.scale})` : name;
                        })
                        .join(", ")}
                    </dd>
                  </>
                )}
              </dl>
              <TagEditor key={selected.id} item={selected} onSave={(t) => saveTags(selected.id, t)} />
              {copyError && <p className="error" role="alert">{copyError}</p>}
              <ImageActions
                onCopyImage={() => copyImage(selected)}
                onCopyPrompt={copyPrompt}
                onCopySeed={copySeed}
                onReveal={() => handleReveal(selected)}
                onUpscale2={() => handleUpscale(2)}
                onUpscale4={() => handleUpscale(4)}
                onReuse={() => { onReuse(selected); setSelected(null); }}
                onFill={() => setFillTarget(selected)}
                onSetReference={onSetReference ? () => {
                  const filename = selected.file || `${selected.id}.${selected.format || "png"}`;
                  onSetReference({ path: filename, preview: imageUrl(selected.id) });
                  setSelected(null);
                } : undefined}
                referenceTitle={t("canvas.action.useAsRefTitle", { max: maxReferenceImages })}
                onDownload={() => {
                  /* "Export..." opens a real save panel and writes the file at full
                     resolution, which is the one dependable way out of the app:
                     dragging a card to the Desktop writes a .webloc link stub instead
                     of the image. The <a download> stays as the fallback for the dev
                     server in a browser, where there is no native bridge. */
                  if (hasNativeBridge()) exportImageNatively(selected);
                  else {
                    const a = document.createElement("a");
                    a.href = imageUrl(selected.id);
                    a.download = selected.file || `${selected.id}.${selected.format || "png"}`;
                    a.click();
                  }
                }}
                onDelete={() => {
                  if (window.confirm(t("gallery.deleteConfirm"))) {
                    deleteImage(selected.id);
                  }
                }}
                isFavorite={Boolean(selected.favorite)}
                onToggleFavorite={() => toggleFavorite(selected.id, !selected.favorite)}
                isBlurred={Boolean(selected.blurred)}
                onToggleBlur={() => toggleBlur(selected.id, !selected.blurred)}
                upscaling={upscaling}
                copied={imageCopied ? "image" : promptCopied ? "prompt" : copied ? "seed" : null}
                revealed={revealed}
              />
              <div className="detail-actions">
                <button type="button" className="action-btn" onClick={() => setSelected(null)}>
                  {t("gallery.closeBtn")}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TagEditor({ item, onSave }) {
  const { t } = useI18n();
  const [value, setValue] = useState((item.tags || []).join(", "));
  return (
    <div className="tag-editor">
      <label>
        {t("gallery.tagsLabel")}
        <input value={value} onChange={(e) => setValue(e.target.value)} />
      </label>
      <button onClick={() => onSave(value)}>{t("gallery.saveTags")}</button>
    </div>
  );
}

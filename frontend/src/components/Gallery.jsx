import { useCallback, useEffect, useRef, useState } from "react";
import { api, imageUrl } from "../api";
import { bindFullImageDrag, copyFullImageToClipboard, revealImageInFinder } from "../utils/dragDrop";
import LazyGalleryImage from "./LazyGalleryImage";
import FillBrush from "./FillBrush";
import { useI18n } from "../i18n/I18nContext";

export default function Gallery({ refreshKey, onReuse, activeTab = "browser", newImage = null }) {
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
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);
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

  const load = useCallback(async () => {
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    try {
      const params = new URLSearchParams({
        query: debouncedQuery.trim(),
        tags: debouncedTags,
        sort,
        page,
        limit,
        model,
        lora,
      });
      const data = await api(`/api/gallery?${params}`, {
        signal: controller.signal,
      });
      setItems(data.items);
      setTotal(data.total);
    } catch (e) {
      if (e.name !== "AbortError") {
        /* keep previous items on transient failures */
      }
    }
  }, [debouncedQuery, debouncedTags, sort, page, model, lora]);

  // Load immediately on tab switch, refreshKey or filter changes
  useEffect(() => {
    if (activeTab !== "browser") return;
    load();
    return () => {
      abortRef.current?.abort();
    };
  }, [load, refreshKey, activeTab]);

  // The list rows are the full sidecar dicts, so opening one needs no extra
  // round-trip to /api/images/{id}.
  function openDetail(item) {
    setSelected(item);
  }

  const selectedIndex = selected
    ? items.findIndex((it) => it.id === selected.id)
    : -1;

  const navigateDetail = useCallback(
    (delta) => {
      if (selectedIndex < 0) return;
      const next = items[selectedIndex + delta];
      if (!next) return;
      setSelected(next);
    },
    [selectedIndex, items]
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
    await api(`/api/images/${id}`, { method: "DELETE" });
    setSelected(null);
    load();
  }

  async function saveTags(id, tagString) {
    const updated = await api(`/api/images/${id}/tags`, {
      method: "POST",
      body: JSON.stringify({
        tags: tagString.split(",").map((t) => t.trim()).filter(Boolean),
      }),
    });
    setSelected(updated);
    load();
  }

  const pages = Math.max(1, Math.ceil(total / limit));

  const [copied, setCopied] = useState(false);
  const [promptCopied, setPromptCopied] = useState(false);
  const [imageCopied, setImageCopied] = useState(false);
  const [revealed, setRevealed] = useState(false);
  // The image being filled, or null. Null means the brush is closed.
  const [fillTarget, setFillTarget] = useState(null);
  const [fillResult, setFillResult] = useState(null);
  const [cardCopiedId, setCardCopiedId] = useState(null);
  const [upscaling, setUpscaling] = useState(false);

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

  async function copySeed() {
    try {
      await navigator.clipboard.writeText(String(selected.seed));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(selected.prompt || "");
      setPromptCopied(true);
      setTimeout(() => setPromptCopied(false), 1500);
    } catch {}
  }


  async function handleUpscale(scale = 2) {
    setUpscaling(true);
    try {
      const upscaled = await api(`/api/images/${selected.id}/upscale`, {
        method: "POST",
        body: JSON.stringify({ scale }),
      });
      setSelected(upscaled);
      load();
    } catch (e) {
      alert(t("gallery.upscaleError", { message: e.message || e }));
    } finally {
      setUpscaling(false);
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
            load();
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
      <div className="gallery-controls">
        <input
          className="search"
          placeholder={t("gallery.searchPlaceholder")}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setPage(1);
          }}
        />
        <input
          placeholder={t("gallery.tagsPlaceholder")}
          value={tags}
          onChange={(e) => {
            setTags(e.target.value);
            setPage(1);
          }}
        />
        <select value={sort} onChange={(e) => setSort(e.target.value)}>
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
        <p className="hint">{t("gallery.empty")}</p>
      ) : (
        <div className="grid">
          {items.map((item) => (
            <div
              key={item.id}
              role="button"
              tabIndex={0}
              className="cell"
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
              <LazyGalleryImage item={item} />
              <div className="cell-quick-actions" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="cell-quick-btn"
                  title={t("gallery.cellCopyTitle")}
                  onClick={(e) => {
                    e.stopPropagation();
                    copyImage(item);
                  }}
                >
                  {cardCopiedId === item.id ? "✓" : "📋"}
                </button>
                <button
                  type="button"
                  className="cell-quick-btn"
                  title={t("gallery.cellRevealTitle")}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleReveal(item);
                  }}
                >
                  📂
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
                  🖌
                </button>
              </div>
              <span className="cell-model-badge">{labelFor(item.model)}</span>
              <div className="cell-caption">{item.prompt}</div>
            </div>
          ))}
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
          {selectedIndex > 0 && (
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
          {selectedIndex >= 0 && selectedIndex < items.length - 1 && (
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
            <img
              src={imageUrl(selected.id)}
              alt={selected.prompt}
              title={t("gallery.dragFullResTitle")}
              {...bindFullImageDrag(selected)}
            />
            <div className="detail">
              <p className="detail-prompt">{selected.prompt}</p>
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
              <div className="detail-actions">
                <button
                  className="btn-accent"
                  onClick={() => copyImage(selected)}
                  title={t("gallery.copyImageFullTitle")}
                >
                  {imageCopied ? t("gallery.copyImageDone") : t("gallery.copyImageBtn")}
                </button>
                <button
                  onClick={() => handleReveal(selected)}
                  title={t("gallery.revealFullTitle")}
                >
                  {revealed ? t("gallery.revealedDone") : t("gallery.revealBtn")}
                </button>
                <button onClick={copyPrompt}>
                  {promptCopied ? t("gallery.promptCopiedDone") : t("gallery.copyPromptBtn")}
                </button>
                <button onClick={copySeed}>
                  {copied ? t("gallery.seedCopiedDone") : t("gallery.copySeedBtn")}
                </button>
                <button
                  onClick={() => handleUpscale(2)}
                  disabled={upscaling}
                  title={t("gallery.upscale2xTitle")}
                >
                  {upscaling ? t("gallery.upscaling") : t("gallery.upscale2xBtn")}
                </button>
                <button
                  onClick={() => handleUpscale(4)}
                  disabled={upscaling}
                  title={t("gallery.upscale4xTitle")}
                >
                  {upscaling ? t("gallery.upscaling") : t("gallery.upscale4xBtn")}
                </button>
                <button
                  onClick={() => {
                    onReuse(selected);
                    setSelected(null);
                  }}
                >
                  {t("gallery.reuseParams")}
                </button>
                <a className="btn" href={imageUrl(selected.id)} download={selected.file || `${selected.id}.${selected.format || 'png'}`}>
                  {t("gallery.download")}
                </a>
                <button
                  className="danger"
                  onClick={() =>
                    confirm(t("gallery.deleteConfirm")) && deleteImage(selected.id)
                  }
                >
                  {t("gallery.deleteBtn")}
                </button>
                <button onClick={() => setSelected(null)}>{t("gallery.closeBtn")}</button>
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

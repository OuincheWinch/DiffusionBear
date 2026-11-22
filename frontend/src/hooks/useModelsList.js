import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

/**
 * The installed-model list, shared by every consumer.
 *
 * This existed as a copy-pasted fetch in ParametersTab, and moving the Models section out
 * to its own tab would have produced a second identical copy in ModelsTab -- two sources of
 * truth for the same inventory, which is exactly the kind of drift that shows up later as
 * "the Models tab says a model is missing but Generate can use it".
 *
 * Returns the raw `/api/models` array rather than the normalised shape from
 * `useModelConfig`, because DefaultsSection indexes it directly and changing that contract
 * is a separate piece of work.
 */
export function useModelsList() {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const list = await api("/api/models");
      setModels(Array.isArray(list) ? list : []);
    } catch {
      // Leave the previous list in place: a failed poll should not blank a populated table.
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { models, loading, refresh };
}
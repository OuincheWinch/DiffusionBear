import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

export function useSettings() {
  const [settings, setSettings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api("/api/settings");
      setSettings(data);
      setError(null);
      return data;
    } catch (e) {
      setError(e.message || String(e));
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const update = useCallback(async (updates) => {
    const data = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify(updates),
    });
    setSettings(data);
    return data;
  }, []);

  const artistName = (settings?.artist_name || "").trim() || "DiffusionBear";

  return { settings, loading, error, refresh, update, artistName };
}

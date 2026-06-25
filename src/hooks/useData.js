// src/hooks/useData.js
// Purpose: Shared hook for loading pre-exported JSON data files from /public/dashboard-data/.
// Returns { data, loading, error } — components render graceful empty states on failure.

import { useState, useEffect } from "react";

// Resolve the correct base path whether running locally or on GitHub Pages.
// Vite injects import.meta.env.BASE_URL at build time from vite.config.js.
const BASE = import.meta.env.BASE_URL;

export function useData(filename) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    if (!filename) return;
    const url = `${BASE}dashboard-data/${filename}`;
    let cancelled = false;

    async function fetchData() {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status} — ${url}`);
        const json = await res.json();
        if (!cancelled) {
          setData(json);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, [filename]);

  return { data, loading, error };
}
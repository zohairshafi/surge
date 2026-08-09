#!/usr/bin/env python3
"""
Shared gene-name resolution for LOC-prefixed and other non-standard symbols.

Resolution order (cascading):
1. **TSV** — pre-computed ``gene_name_to_locid.tsv`` (42 044 entries, instant)
2. **NCBI JSON cache** — previously resolved genes, persisted to disk
3. **mygene.info** — batch REST API (stickleback taxid 69293), resolves
   many LOC genes in a single POST by stripping the LOC prefix and
   querying by numeric entrez ID
4. **NCBI Entrez API** — one-at-a-time fallback for genes mygene.info
   doesn't know about, rate-limited at 0.35 s/call

Results from steps 3-4 are persisted to the NCBI JSON cache, so each
gene is looked up externally at most once.

Usage::

    from scripts.loc_resolver import GeneNameResolver

    resolver = GeneNameResolver(
        tsv_path='output/gene_name_to_locid.tsv',
        ncbi_cache_path='output/ncbi_loc_cache.json',
    )
    proper = resolver.resolve('LOC100174865')   # → 'tas1r3'
    proper = resolver.resolve('si:ch1073-...')  # → None  (filtered)
    proper = resolver.resolve('spi1b.H')        # → 'spi1b'

    # Batch mode (uses mygene.info for all unresolved LOC genes in one call):
    names = resolver.resolve_many(['LOC100174865', 'si:foo', 'gata3.H'])
"""

import json
import os
import time
import requests


class GeneNameResolver:
    """Resolve matrix gene names to proper symbols.

    Parameters
    ----------
    tsv_path : str or None
        Path to ``gene_name_to_locid.tsv``.  If None, TSV lookup is
        skipped (all LOC genes fall through to NCBI or are dropped).
    ncbi_cache_path : str or None
        Path to ``ncbi_loc_cache.json`` for persisting NCBI API results.
    ncbi_delay : float
        Seconds to sleep between NCBI API calls (default 0.35).
    """

    def __init__(self, tsv_path=None, ncbi_cache_path=None,
                 ncbi_delay=0.35, use_mygene=True):
        self._tsv = None
        if tsv_path is not None and os.path.exists(tsv_path):
            self._tsv = self._load_tsv(tsv_path)

        self._ncbi_cache = {}
        self._ncbi_cache_path = ncbi_cache_path
        if ncbi_cache_path is not None and os.path.exists(ncbi_cache_path):
            with open(ncbi_cache_path) as f:
                self._ncbi_cache = json.load(f)

        self._ncbi_delay = ncbi_delay
        self._use_mygene = use_mygene

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, gene_name):
        """Resolve a single gene name.  Returns the proper symbol or None.

        None means the gene should be dropped (si./trna prefix, or
        unresolvable LOC).
        """
        # Strip .H suffix (haplotype annotation in stickleback)
        name = str(gene_name).replace('.H', '')

        # Filter non-informative prefixes
        lo = name.lower()
        if lo.startswith('si.') or lo.startswith('si:') or lo.startswith('trna'):
            return None

        # Non-LOC genes pass through as-is
        if not name.upper().startswith('LOC'):
            return name

        # --- LOC resolution ---
        # 1. TSV lookup
        if self._tsv is not None:
            tsv_symbol = self._tsv.get(name)
            if tsv_symbol is not None:
                if not tsv_symbol.upper().startswith('LOC'):
                    return tsv_symbol
                # TSV says it's still LOC → genuinely unresolved
                return None

        # 2. NCBI JSON cache.  Use `in` (not `.get(...) is not None`): a cached
        # None means "previously queried and confirmed unresolved" and must be
        # treated as a HIT — otherwise unresolvable LOC genes were re-queried
        # on every call.
        if name in self._ncbi_cache:
            return self._ncbi_cache[name]  # may be None (confirmed unresolved)

        # 3. mygene.info batch-friendly lookup (single-gene fallback)
        if self._use_mygene:
            mygene_results = self._query_mygene_batch([name])
            if mygene_results:
                return mygene_results.get(name)

        # 4. NCBI API (with persistence)
        return self._query_ncbi(name)

    def resolve_many(self, gene_names, verbose=True):
        """Batch-resolve a list of gene names.

        Uses TSV → NCBI JSON cache → mygene.info batch → NCBI Entrez.
        Unresolved LOC genes are collected and sent to mygene.info in a
        single HTTP request before falling back to one-at-a-time NCBI.

        Returns a list of resolved names (may contain None entries for
        filtered / unresolvable genes).
        """
        # --- Pass 1: resolve via TSV + cache (instant) ---
        resolved = []
        unresolved_loc = []  # (index, loc_symbol)
        for i, name in enumerate(gene_names):
            # Strip .H suffix
            clean = str(name).replace('.H', '')
            lo = clean.lower()
            if lo.startswith('si.') or lo.startswith('si:') or lo.startswith('trna'):
                resolved.append(None)
                continue
            if not clean.upper().startswith('LOC'):
                resolved.append(clean)
                continue

            # --- LOC resolution ---
            # 1. TSV
            tsv_symbol = self._tsv.get(clean) if self._tsv is not None else None
            if tsv_symbol is not None:
                if not tsv_symbol.upper().startswith('LOC'):
                    resolved.append(tsv_symbol)
                else:
                    resolved.append(None)
                continue

            # 2. NCBI JSON cache
            cached = self._ncbi_cache.get(clean)
            if cached is not None:
                resolved.append(cached)  # may be None
                continue

            # Needs further resolution
            resolved.append(None)  # placeholder
            unresolved_loc.append((i, clean))

        if not unresolved_loc:
            return resolved

        # --- Pass 2: mygene.info batch ---
        if self._use_mygene:
            loc_symbols = [s for _, s in unresolved_loc]
            if verbose and loc_symbols:
                print(f'  Querying mygene.info for {len(loc_symbols)} '
                      f'LOC genes...', end=' ', flush=True)
            mygene_results = self._query_mygene_batch(loc_symbols)
            if verbose and loc_symbols:
                n_resolved = sum(1 for v in mygene_results.values() if v)
                print(f'{n_resolved} resolved')

            still_unresolved = []
            for idx, loc_sym in unresolved_loc:
                result = mygene_results.get(loc_sym)
                if result is not None:
                    resolved[idx] = result
                else:
                    # mygene didn't resolve — check if cache was populated
                    # with None (meaning mygene confirmed it's unresolvable)
                    if loc_sym in mygene_results:
                        resolved[idx] = None  # confirmed unresolvable
                    else:
                        still_unresolved.append((idx, loc_sym))
            unresolved_loc = still_unresolved

        # --- Pass 3: NCBI Entrez (one-at-a-time, rate-limited) ---
        if unresolved_loc:
            if verbose:
                print(f'  Falling back to NCBI Entrez for '
                      f'{len(unresolved_loc)} genes...')
            for idx, loc_sym in unresolved_loc:
                result = self._query_ncbi(loc_sym)
                resolved[idx] = result

        return resolved

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load_tsv(tsv_path):
        """Load gene_name_to_locid.tsv → {matrix_name: ncbi_symbol}."""
        lookup = {}
        with open(tsv_path) as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 4:
                    lookup[parts[0]] = parts[3]  # matrix_name → ncbi_symbol
        return lookup

    # ------------------------------------------------------------------
    # mygene.info batch lookup
    # ------------------------------------------------------------------

    MYGENE_URL = 'https://mygene.info/v3/gene/'

    def _query_mygene_batch(self, loc_symbols):
        """Batch-query mygene.info for LOC gene symbols.

        Strips the LOC prefix to get numeric entrez IDs, queries
        mygene.info in a single POST, and returns a dict mapping
        ``{loc_symbol: proper_name_or_None}``.  Results are also
        persisted to the NCBI JSON cache so subsequent lookups hit
        the cache.

        Parameters
        ----------
        loc_symbols : list of str
            LOC-prefixed gene symbols, e.g. ``['LOC100174880']``.

        Returns
        -------
        dict
            Mapping from original LOC symbol to resolved name (or None).
        """
        if not loc_symbols:
            return {}

        # Build mapping: numeric_id → loc_symbol
        id_to_sym = {}
        for s in loc_symbols:
            gene_id = s.upper().replace('LOC', '', 1)
            if gene_id.isdigit():
                id_to_sym[gene_id] = s

        if not id_to_sym:
            return {}

        numeric_ids = list(id_to_sym.keys())
        try:
            resp = requests.post(
                self.MYGENE_URL,
                json={
                    'ids': numeric_ids,
                    'species': '69293',        # Gasterosteus aculeatus
                    'fields': 'symbol',
                },
                timeout=60,
            )
            if resp.status_code != 200:
                return {}

            results = {}
            for entry in resp.json():
                gene_id = entry.get('_id', '')
                if isinstance(gene_id, int):
                    gene_id = str(gene_id)
                loc_sym = id_to_sym.get(gene_id)
                if loc_sym is None:
                    continue

                symbol = entry.get('symbol', '')
                if symbol and not symbol.upper().startswith('LOC'):
                    results[loc_sym] = symbol
                    self._ncbi_cache[loc_sym] = symbol
                else:
                    results[loc_sym] = None
                    self._ncbi_cache[loc_sym] = None

            self._save_cache()
            return results

        except Exception:
            return {}

    def _query_ncbi(self, loc_symbol):
        """Query NCBI Entrez for a single LOC gene.  Updates the cache."""
        gene_id = loc_symbol.upper().replace('LOC', '')
        url = (
            'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
            f'esummary.fcgi?db=gene&id={gene_id}&retmode=json'
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                self._ncbi_cache[loc_symbol] = None
                self._save_cache()
                return None
            data = resp.json()
            result = data.get('result', {})
            uids = result.get('uids', [])
            if not uids:
                self._ncbi_cache[loc_symbol] = None
                self._save_cache()
                return None
            gene_info = result.get(uids[0], {})
            name = gene_info.get('name', '')
            if name and not name.upper().startswith('LOC'):
                self._ncbi_cache[loc_symbol] = name
                self._save_cache()
                return name
            # Named but still LOC → unresolved
            self._ncbi_cache[loc_symbol] = None
            self._save_cache()
            return None
        except Exception:
            self._ncbi_cache[loc_symbol] = None
            self._save_cache()
            return None
        finally:
            time.sleep(self._ncbi_delay)

    def _save_cache(self):
        if self._ncbi_cache_path is not None:
            os.makedirs(os.path.dirname(self._ncbi_cache_path) or '.',
                        exist_ok=True)
            with open(self._ncbi_cache_path, 'w') as f:
                json.dump(self._ncbi_cache, f, indent=2)


# ------------------------------------------------------------------
# Convenience: module-level singleton for scripts that just need a
# quick resolver without managing paths.
# ------------------------------------------------------------------

_default_resolver = None


def get_resolver(tsv_path='output/gene_name_to_locid.tsv',
                 ncbi_cache_path='output/ncbi_loc_cache.json'):
    """Return a cached GeneNameResolver singleton."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = GeneNameResolver(
            tsv_path=tsv_path,
            ncbi_cache_path=ncbi_cache_path,
        )
    return _default_resolver

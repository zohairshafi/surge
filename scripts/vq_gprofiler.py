#!/usr/bin/env python3
"""
Run g:Profiler enrichment on each VQ code's gene set individually.

For each significant VQ code in the input CSV, extracts the named genes
(using gene_mappings.pkl), converts them to Ensembl IDs via g:Convert,
and runs g:Profiler GO/KEGG/WP/REAC enrichment. Results are saved as CSV.

LOC genes are resolved to proper gene names via NCBI Entrez before
enrichment, recovering ~10% of otherwise-discarded genes.

Usage:
  python scripts/vq_gprofiler.py \
      --codes rol/output/figures/infection_genes_for_go.csv \
      --gene-mappings rol/output/gene_mappings.pkl \
      --output rol/output/infection_vq_gprofiler.csv \
      --p-threshold 0.001
"""

import argparse, csv, json, os, pickle, sys, time
import requests


# ---------------------------------------------------------------------------
# Gene extraction
# ---------------------------------------------------------------------------

def load_gene_mappings(path):
    """Return (gene_names, vq_to_gene) from a gene_mappings.pkl file."""
    with open(path, "rb") as f:
        gm = pickle.load(f)
    gene_names = gm["gene_names"]
    vq_to_gene = gm["vq_to_gene"]
    return gene_names, vq_to_gene


def get_all_loc_genes(gene_names):
    """Return the set of unique LOC gene symbols (without .H suffix)."""
    loc_genes = set()
    for name in gene_names:
        clean = name.replace(".H", "")
        if clean.startswith("LOC"):
            loc_genes.add(clean)
    return loc_genes


def build_loc_name_cache(loc_genes, cache_path, delay=0.35):
    """
    Query NCBI Entrez for each LOC gene and cache the result.

    For each LOC gene, queries NCBI esummary. If NCBI returns a proper
    (non-LOC) gene name, stores it. Otherwise stores null.

    Returns dict: {LOC_SYMBOL: proper_name_or_None}
    """
    cache = {}
    loc_list = sorted(loc_genes)
    n_total = len(loc_list)
    n_named = 0
    n_described = 0

    print(f"Building NCBI LOC cache for {n_total} genes...")
    print(f"  Estimated time: ~{n_total * delay / 60:.0f} min "
          f"(rate-limited to {1/delay:.0f} req/s)")

    for i, gene in enumerate(loc_list):
        gene_id = gene.replace("LOC", "")
        url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
            f"esummary.fcgi?db=gene&id={gene_id}&retmode=json"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                result = data.get("result", {})
                uids = result.get("uids", [])
                if uids:
                    info = result.get(uids[0], {})
                    ncbi_name = info.get("name", "")
                    ncbi_desc = info.get("description", "")
                    if ncbi_name and not ncbi_name.startswith("LOC"):
                        cache[gene] = ncbi_name
                        n_named += 1
                    else:
                        cache[gene] = None
                        if ncbi_desc and "uncharacterized" not in ncbi_desc.lower():
                            n_described += 1
                else:
                    cache[gene] = None
            else:
                cache[gene] = None
        except Exception:
            cache[gene] = None

        # Progress
        if (i + 1) % 200 == 0:
            pct = (i + 1) / n_total * 100
            print(f"  {i+1}/{n_total} ({pct:.0f}%): "
                  f"{n_named} named, {n_described} described, "
                  f"{(i+1)-n_named-n_described} uncharacterized")

        time.sleep(delay)

    # Save cache
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"  Done: {n_named} proper names, {n_described} described, "
          f"{n_total - n_named - n_described} uncharacterized")
    print(f"  Cache saved to {cache_path}")
    return cache


def get_named_genes_for_code(vq_to_gene, gene_names, code,
                              resolver=None):
    """Return sorted list of named genes assigned to a VQ code.

    A gene is "named" if its symbol does not start with LOC, si., or trna.
    LOC genes are included if they resolve to a proper name via loc_cache.
    The .H suffix is stripped.
    """
    all_genes = set()
    for strat_key, code_to_genes in vq_to_gene.items():
        if code in code_to_genes:
            for g in code_to_genes[code]:
                all_genes.add(int(g))

    # Collect gene names and resolve in batch
    idx_to_name = {}
    for idx in sorted(all_genes):
        idx_to_name[idx] = gene_names[idx]

    if resolver is not None:
        # Batch-resolve all genes at once (uses mygene.info batch)
        raw_names = [gene_names[i] for i in sorted(all_genes)]
        resolved_names = resolver.resolve_many(raw_names, verbose=False)
    else:
        # Fallback: basic filtering without resolver
        resolved_names = []
        for i in sorted(all_genes):
            name = gene_names[i]
            clean = name.replace('.H', '')
            if clean.lower().startswith('si.') or clean.lower().startswith('trna'):
                resolved_names.append(None)
            elif clean.upper().startswith('LOC'):
                resolved_names.append(None)
            else:
                resolved_names.append(clean)

    named = []
    loc_recovered = 0
    for idx, resolved in zip(sorted(all_genes), resolved_names):
        if resolved is not None:
            named.append(resolved)
            clean = gene_names[idx].replace('.H', '')
            if clean.upper().startswith('LOC'):
                loc_recovered += 1

    return named, loc_recovered


# ---------------------------------------------------------------------------
# g:Profiler API
# ---------------------------------------------------------------------------

GPROFILER_CONVERT_URL = "https://biit.cs.ut.ee/gprofiler/api/convert/convert/"
GPROFILER_GOST_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def convert_to_ensembl(genes, organism="gaculeatus", delay=0.0):
    """Convert gene symbols to Ensembl IDs via g:Convert.

    Returns list of Ensembl IDs (strings).
    """
    payload = {
        "organism": organism,
        "query": genes,
        "target": "ENSG",
    }
    resp = requests.post(GPROFILER_CONVERT_URL, json=payload, timeout=60)
    if resp.status_code != 200:
        return []
    ensembl_ids = []
    for r in resp.json()["result"]:
        c = r.get("converted", "")
        if c and c != "None":
            ensembl_ids.append(c)
    if delay:
        time.sleep(delay)
    return ensembl_ids


def run_enrichment(ensembl_ids, organism="gaculeatus",
                   sources=None, user_threshold=0.05, background=None):
    """Run g:Profiler enrichment and return significant terms.

    background : list[str] or None
        Ensembl IDs of the eligible gene universe (the pre-filtered set the
        query was drawn from). When provided, domain_scope='custom' restricts
        the statistical background to this universe instead of the whole
        genome, giving correct p-values for queries from a filtered subset.

    Returns list of dicts with keys: source, term_id, term_name, p_value,
    intersection_size, term_size, precision, recall.
    """
    if sources is None:
        sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "WP", "REAC"]

    payload = {
        "organism": organism,
        "query": ensembl_ids,
        "sources": sources,
        "user_threshold": user_threshold,
        "no_evidences": True,
    }
    if background is not None:
        payload["domain_scope"] = "custom"
        payload["background"] = background
    resp = requests.post(GPROFILER_GOST_URL, json=payload, timeout=60)
    if resp.status_code != 200:
        return []

    results = []
    for r in resp.json().get("result", []):
        results.append({
            "source": r["source"],
            "term_id": r["native"],
            "term_name": r["name"],
            "p_value": r["p_value"],
            "intersection_size": r["intersection_size"],
            "term_size": r["term_size"],
            "precision": r.get("precision", 0),
            "recall": r.get("recall", 0),
        })
    results.sort(key=lambda x: x["p_value"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run g:Profiler enrichment per VQ code"
    )
    parser.add_argument("--codes", required=True,
                        help="CSV with VQ codes (must have 'vq_code' and 'p_value' columns)")
    parser.add_argument("--gene-mappings", required=True,
                        help="Path to gene_mappings.pkl")
    parser.add_argument("--output", required=True,
                        help="Output CSV path")
    parser.add_argument("--p-threshold", type=float, default=0.05,
                        help="Only process codes with p_value < this threshold")
    parser.add_argument("--organism", default="gaculeatus",
                        help="g:Profiler organism name")
    parser.add_argument("--sources", default="GO:BP,GO:MF,GO:CC,KEGG,WP,REAC",
                        help="Comma-separated g:Profiler sources")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay in seconds between g:Profiler calls")
    parser.add_argument("--loc-tsv", default="rol/output/gene_name_to_locid.tsv",
                        help="Pre-computed gene name → LOC ID TSV")
    parser.add_argument("--loc-cache", default="rol/output/ncbi_loc_cache.json",
                        help="JSON cache file for NCBI LOC→name resolutions "
                             "(fallback, built automatically if TSV misses genes)")
    parser.add_argument("--no-custom-background", action="store_true",
                        help="Disable custom background (use g:Profiler "
                             "genome-wide default instead of the filtered "
                             "model gene set)")
    args = parser.parse_args()

    # Load gene names only (gene lists come from the CSV, not vq_to_gene)
    gene_names, _ = load_gene_mappings(args.gene_mappings)

    # Set up gene name resolver (TSV first, NCBI JSON cache fallback)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from scripts.loc_resolver import GeneNameResolver
    resolver = GeneNameResolver(
        tsv_path=args.loc_tsv if os.path.exists(args.loc_tsv) else None,
        ncbi_cache_path=args.loc_cache,
    )
    if resolver._tsv is not None:
        print(f"Loaded TSV with {len(resolver._tsv)} gene mappings")
    if resolver._ncbi_cache:
        n_named = sum(1 for v in resolver._ncbi_cache.values() if v)
        print(f"Loaded NCBI cache: {len(resolver._ncbi_cache)} genes, "
              f"{n_named} with proper names")

    # Custom background universe = all genes that entered the model (the
    # filtered 10k/25k set), converted to Ensembl. Restricting the g:Profiler
    # statistical domain to this universe (domain_scope='custom') instead of
    # the whole genome gives correct enrichment p-values for queries drawn
    # from a pre-filtered subset.
    background_ensembl = None
    if not args.no_custom_background:
        bg_named = (resolver.resolve_many(gene_names, verbose=False)
                    if resolver is not None else
                    [g for g in gene_names
                     if not g.upper().startswith('LOC')
                     and not g.lower().startswith(('si.', 'si:', 'trna'))])
        bg_named = [g for g in bg_named if g is not None]
        background_ensembl = convert_to_ensembl(bg_named, args.organism)
        if background_ensembl:
            print(f"Custom background: {len(background_ensembl)} Ensembl IDs "
                  f"(from {len(gene_names)} model genes)")
        else:
            print("  WARNING: background conversion failed — falling back to "
                  "g:Profiler genome-wide default")

    # The gene_indices / gene_names columns can be very large (all genes
    # assigned to a VQ code, semicolon-separated), exceeding Python's
    # default csv.field_size_limit of 131 KB.
    csv.field_size_limit(sys.maxsize)

    with open(args.codes) as f:
        all_rows = list(csv.DictReader(f))

    rows = [r for r in all_rows if float(r["p_value"]) < args.p_threshold]
    print(f"Processing {len(rows)} codes (p < {args.p_threshold}) "
          f"from {args.codes}")

    sources = [s.strip() for s in args.sources.split(",")]

    # Helper: resolve gene indices from a CSV row to named genes.
    # Uses the gene_indices column (semicolon-separated) so the gene list
    # matches exactly what step 9 exported for this comparison — no
    # cross-strata leakage from vq_to_gene.
    def _resolve_gene_indices(row, gene_names, resolver):
        gene_idx_str = row.get('gene_indices', '')
        if not gene_idx_str:
            return [], 0
        gene_indices = [int(g) for g in gene_idx_str.split(';') if g.strip()]
        raw_names = [gene_names[gi] for gi in gene_indices
                     if gi < len(gene_names)]
        if resolver is not None:
            resolved = resolver.resolve_many(raw_names, verbose=False)
        else:
            resolved = [g for g in raw_names
                        if not g.upper().startswith('LOC')
                        and not g.lower().startswith(('si.', 'si:', 'trna'))]
        named = [r for r in resolved if r is not None]
        loc_recovered = sum(1 for raw, r in zip(raw_names, resolved)
                            if r is not None
                            and raw.replace('.H', '').upper().startswith('LOC'))
        return named, loc_recovered

    # Process each code
    output_rows = []
    total_loc_recovered = 0
    for i, row in enumerate(rows):
        code = int(row["vq_code"])
        p_code = float(row["p_value"])

        # Use gene_indices from the CSV (matching step 9's export for this
        # comparison) instead of aggregating across all strata in vq_to_gene.
        named, loc_recovered = _resolve_gene_indices(row, gene_names, resolver)
        total_loc_recovered += loc_recovered

        if not named:
            print(f"  [{i+1}/{len(rows)}] Code {code}: 0 named genes, skipping")
            continue

        # Step 3: convert to Ensembl
        ensembl_ids = convert_to_ensembl(named, args.organism)
        if not ensembl_ids:
            print(f"  [{i+1}/{len(rows)}] Code {code}: {len(named)} named, "
                  f"0 Ensembl, skipping")
            continue

        # Step 4: enrich
        terms = run_enrichment(ensembl_ids, args.organism, sources,
                               background=background_ensembl)
        time.sleep(args.delay)

        n_terms = len(terms)
        loc_str = f" ({loc_recovered} from LOC)" if loc_recovered else ""
        print(f"  [{i+1}/{len(rows)}] Code {code}: p={p_code:.2e}, "
              f"{len(named)} named{loc_str}, {len(ensembl_ids)} Ensembl, "
              f"{n_terms} terms")

        for t in terms[:5]:
            print(f"         [{t['source']}] {t['term_name']}: "
                  f"p={t['p_value']:.2e}, "
                  f"{t['intersection_size']}/{t['term_size']}")

        # Step 5: collect output
        for t in terms:
            output_rows.append({
                "vq_code": code,
                "code_p_value": p_code,
                "n_named_genes": len(named),
                "n_loc_recovered": loc_recovered,
                "n_ensembl_ids": len(ensembl_ids),
                "source": t["source"],
                "term_id": t["term_id"],
                "term_name": t["term_name"],
                "p_value": t["p_value"],
                "intersection_size": t["intersection_size"],
                "term_size": t["term_size"],
            })

    print(f"\nTotal LOC genes recovered across all codes: {total_loc_recovered}")

    # Write CSV
    if output_rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "vq_code", "code_p_value", "n_named_genes", "n_loc_recovered",
                "n_ensembl_ids",
                "source", "term_id", "term_name", "p_value",
                "intersection_size", "term_size",
            ])
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"Saved {len(output_rows)} enrichment rows to {args.output}")
    else:
        print("\nNo enriched terms found.")


if __name__ == "__main__":
    main()

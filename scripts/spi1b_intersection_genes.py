#!/usr/bin/env python3
"""
Re-run g:Profiler enrichment for spi1b Code 32 genes with intersection gene lists.

For each enriched term, extracts the specific gene names from the intersections
field (mapping indices back to query gene order), then picks representative genes
for manuscript table inclusion.

Usage:
  python scripts/spi1b_intersection_genes.py \
      --gene-mappings output/gene_mappings.pkl \
      --output output/spi1b_gprofiler_with_genes.csv \
      --top-n 10
"""

import argparse, csv, json, os, pickle, sys, time
import requests

# ---------------------------------------------------------------------------
# g:Profiler API
# ---------------------------------------------------------------------------

GPROFILER_CONVERT_URL = "https://biit.cs.ut.ee/gprofiler/api/convert/convert/"
GPROFILER_GOST_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"


def convert_to_ensembl(genes, organism="gaculeatus"):
    """Convert gene symbols to Ensembl IDs. Returns (ensembl_ids, ensembl_to_name)."""
    payload = {"organism": organism, "query": genes, "target": "ENSG"}
    resp = requests.post(GPROFILER_CONVERT_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        return [], {}
    ensembl_ids = []
    ensembl_to_name = {}
    for r in resp.json()["result"]:
        name = r["incoming"]
        c = r.get("converted", "")
        if c and c != "None":
            ensembl_ids.append(c)
            ensembl_to_name[c] = name
    return ensembl_ids, ensembl_to_name


def run_enrichment_with_intersections(ensembl_ids, organism="gaculeatus",
                                       sources=None, user_threshold=0.05):
    """Run g:Profiler with no_evidences=False to get intersection gene lists.

    Returns list of dicts. Each dict has an extra 'intersection_genes' key:
    list of gene names (from the original query) in the intersection.
    """
    if sources is None:
        sources = ["GO:BP", "GO:MF", "GO:CC", "KEGG", "WP", "REAC"]

    payload = {
        "organism": organism,
        "query": ensembl_ids,
        "sources": sources,
        "user_threshold": user_threshold,
        "no_evidences": False,  # required to get intersections
    }
    resp = requests.post(GPROFILER_GOST_URL, json=payload, timeout=120)
    if resp.status_code != 200:
        print(f"  g:Profiler error: {resp.status_code}")
        return []

    results = []
    for term in resp.json().get("result", []):
        intersections = term.get("intersections", [])
        # Map indices with non-empty evidence lists back to gene names
        # intersections[i] corresponds to ensembl_ids[i] (same order)
        # Non-empty inner list = gene i is in the intersection
        term_genes = []
        for i, evidence_list in enumerate(intersections):
            if evidence_list and len(evidence_list) > 0:
                if i < len(ensembl_ids):
                    term_genes.append(ensembl_ids[i])

        results.append({
            "source": term["source"],
            "term_id": term["native"],
            "term_name": term["name"],
            "p_value": term["p_value"],
            "intersection_size": term["intersection_size"],
            "term_size": term["term_size"],
            "intersection_ensembl": term_genes,
        })

    results.sort(key=lambda x: x["p_value"])
    return results


# ---------------------------------------------------------------------------
# Gene loading (mirrors vq_gprofiler.py)
# ---------------------------------------------------------------------------

def load_gene_mappings(path):
    with open(path, "rb") as f:
        gm = pickle.load(f)
    return gm["gene_names"], gm["vq_to_gene"]


def get_genes_always_in_code(vq_to_gene, gene_names, code, resolver=None,
                             exclude_gene=None):
    """Return sorted list of named genes that are in *every* stratum for `code`.

    This computes the intersection across all strata that contain the code,
    yielding genes that always co-occur with the focal gene (if one is specified).
    """
    # Collect the set of gene indices in `code` for each stratum
    gene_sets = []
    for strat_key, code_to_genes in vq_to_gene.items():
        if code in code_to_genes:
            gene_sets.append(set(int(g) for g in code_to_genes[code]))

    if not gene_sets:
        return [], {}

    # Intersection across all strata
    always = gene_sets[0]
    for s in gene_sets[1:]:
        always = always & s

    # Exclude the focal gene if requested
    if exclude_gene is not None:
        always.discard(exclude_gene)

    # Collect gene names and resolve in batch
    if resolver is not None:
        raw_names = [gene_names[i] for i in sorted(always)]
        resolved_names = resolver.resolve_many(raw_names, verbose=False)
    else:
        # Fallback: basic filtering without resolver
        resolved_names = []
        for idx in sorted(always):
            name = gene_names[idx]
            clean = name.replace('.H', '')
            if clean.lower().startswith('si.') or clean.lower().startswith('trna'):
                resolved_names.append(None)
            elif clean.upper().startswith('LOC'):
                resolved_names.append(None)
            else:
                resolved_names.append(clean)

    named = []
    for resolved in resolved_names:
        if resolved is not None:
            named.append(resolved)

    return named


def find_gene_index(gene_names, symbol):
    """Find the index of a gene symbol in the gene_names array."""
    sym_lower = symbol.lower().replace('.h', '')
    for i, name in enumerate(gene_names):
        if name.replace('.H', '').lower() == sym_lower:
            return i
    return None


# Gene family prefixes that are well-known and informative in manuscripts
KNOWN_FAMILIES = [
    # Ribosomal
    "rpl", "rps", "mrpl", "mrps",
    # Splicing / snRNP
    "snrp", "prpf", "sf3a", "sf3b", "srsf", "lsm",
    # RNA binding / processing
    "rbm", "hnrnp", "cstf", "cpsf", "pcf",
    # Translation
    "eif",
    # Chromatin / structure
    "smc", "hsp", "hmg",
    # Transcription factors (common families)
    "tbx", "fox", "sox", "gata", "nfkb", "stat",
    # Cohesin / condensin
    "smc",
    # Chaperone / folding
    "hsp", "dnaj", "calr", "canx",
    # DNA replication / repair
    "mcm", "orc", "rpa", "rad",
    # Proteasome / ubiquitin
    "psm", "usp",
    # Transport
    "cop", "clt", "ap",
    # Kinases (common)
    "camk", "cdk", "mapk",
]


def _gene_score(g):
    """Score a gene name for representativeness. Higher = better pick.

    Rewards: membership in a known gene family, moderate name length (4-7 chars),
    alphanumeric readability.
    Penalises: LOC/si/trna prefixes, very short names (likely cryptic),
    very long names.
    """
    s = 0.0
    lo = g.lower()

    # Reject non-informative prefixes
    if lo.startswith("loc"):
        return -100.0
    if lo.startswith("si.") or lo.startswith("trna"):
        return -50.0

    # Bonus for known gene families
    for fam in KNOWN_FAMILIES:
        if lo.startswith(fam):
            s += 5.0
            break  # one bonus per gene

    # Prefer informative length: 4-7 characters is typical for well-known genes
    n = len(g)
    if 4 <= n <= 7:
        s += 3.0
    elif n <= 3:
        s += 1.0  # could be fine (e.g. "rel") but often cryptic
    elif n <= 10:
        s += 1.5
    else:
        s -= 0.5

    # Penalise numeric-heavy names (less human-readable)
    digits = sum(1 for c in g if c.isdigit())
    if digits == 0:
        s += 1.0
    elif digits <= 2:
        s += 0.0
    else:
        s -= 2.0

    return s


def _gene_family(g):
    """Return the family prefix for a gene, or the gene itself if no family."""
    lo = g.lower()
    for fam in KNOWN_FAMILIES:
        if lo.startswith(fam):
            return fam
    return g  # no family — gene itself serves as its own family


def pick_representative_genes(intersection_genes, n=4):
    """Pick up to n representative genes from the intersection list.

    Scores genes by recognisability, preferring known gene families.
    Caps each family at 1 gene to ensure diversity across families.
    Returns a comma-separated string.
    """
    if not intersection_genes:
        return ""

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for g in intersection_genes:
        if g.lower() not in seen:
            seen.add(g.lower())
            unique.append(g)

    scored = sorted(unique, key=_gene_score, reverse=True)

    # Greedy pick: take top-scoring gene, then next from a different family
    picked = []
    used_families = set()
    for g in scored:
        fam = _gene_family(g)
        if fam not in used_families:
            picked.append(g)
            used_families.add(fam)
        if len(picked) >= n:
            break

    return ", ".join(picked)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-run g:Profiler for spi1b Code 32 and extract intersection genes"
    )
    parser.add_argument("--gene-mappings", required=True,
                        help="Path to gene_mappings.pkl")
    parser.add_argument("--output", required=True,
                        help="Output CSV with intersection gene lists")
    parser.add_argument("--loc-tsv", default="output/gene_name_to_locid.tsv",
                        help="Pre-computed gene name → LOC ID TSV")
    parser.add_argument("--loc-cache", default="output/ncbi_loc_cache.json",
                        help="NCBI LOC cache (fallback)")
    parser.add_argument("--code", type=int, default=32,
                        help="VQ code to analyse")
    parser.add_argument("--top-n", type=int, default=4,
                        help="Number of representative genes per term")
    parser.add_argument("--sources", default="GO:BP,GO:MF,GO:CC,KEGG,WP,REAC",
                        help="g:Profiler sources")
    args = parser.parse_args()

    # Load
    gene_names, vq_to_gene = load_gene_mappings(args.gene_mappings)

    # Set up gene name resolver
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from scripts.loc_resolver import GeneNameResolver
    resolver = GeneNameResolver(
        tsv_path=args.loc_tsv,
        ncbi_cache_path=args.loc_cache,
    )

    # Find spi1b index and exclude it
    spi1b_idx = find_gene_index(gene_names, "spi1b")
    print(f"spi1b gene index: {spi1b_idx}")

    # Get genes that are in Code 32 in EVERY stratum (intersection)
    named = get_genes_always_in_code(
        vq_to_gene, gene_names, args.code, resolver,
        exclude_gene=spi1b_idx
    )

    print(f"Genes always in Code {args.code} (named, excl spi1b): {len(named)}")
    n_loc = sum(1 for g in named if g.startswith("LOC"))
    print(f"  LOC genes recovered: {n_loc}")

    # Convert to Ensembl
    print("Converting to Ensembl...")
    ensembl_ids, ensembl_to_name = convert_to_ensembl(named)
    print(f"  Ensembl IDs: {len(ensembl_ids)}")

    # We need name lookup from Ensembl ID back to name
    # Build reverse map from the conversion
    # But we also need the original name for genes that didn't convert
    # Use the ensembl_to_name from conversion; supplement with our list
    # Build a map: Ensembl ID -> original gene name
    # (ensembl_to_name already has this from the conversion response)
    # For any Ensembl IDs not in the map, fall back to the ID itself
    for eid in ensembl_ids:
        if eid not in ensembl_to_name:
            ensembl_to_name[eid] = eid

    # Run enrichment
    sources = [s.strip() for s in args.sources.split(",")]
    print(f"Running g:Profiler with {len(ensembl_ids)} query genes...")
    terms = run_enrichment_with_intersections(ensembl_ids, sources=sources)

    print(f"\nSignificant terms: {len(terms)}")

    # Map Ensembl IDs in intersections back to gene names
    output_rows = []
    for t in terms:
        intersection_names = []
        for eid in t["intersection_ensembl"]:
            name = ensembl_to_name.get(eid, eid)
            intersection_names.append(name)

        representatives = pick_representative_genes(intersection_names, args.top_n)

        output_rows.append({
            "source": t["source"],
            "term_id": t["term_id"],
            "term_name": t["term_name"],
            "p_value": t["p_value"],
            "intersection_size": t["intersection_size"],
            "term_size": t["term_size"],
            "intersection_genes": ", ".join(intersection_names),
            "representative_genes": representatives,
        })

        print(f"  [{t['source']}] {t['term_name']}: "
              f"p={t['p_value']:.2e}, "
              f"{t['intersection_size']}/{t['term_size']}, "
              f"repr: {representatives}")

    # Write output
    if output_rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source", "term_id", "term_name", "p_value",
                "intersection_size", "term_size",
                "intersection_genes", "representative_genes",
            ])
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"\nSaved {len(output_rows)} terms to {args.output}")


if __name__ == "__main__":
    main()

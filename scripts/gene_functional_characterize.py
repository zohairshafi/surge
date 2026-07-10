#!/usr/bin/env python3
"""
Functional characterization of stickleback gene lists via Ensembl REST API.

For each gene, looks up the Ensembl description (if available) and classifies
it using curated immune-related and other-interesting keyword patterns.

Outputs a CSV with columns:
  gene, description, immune_category, other_category

Usage:
  python scripts/gene_functional_characterize.py \
      --input infection_genes_p001.txt \
      --gene-names output/gene_mappings.pkl \
      --output infection_characterized.csv
"""

import argparse, csv, os, pickle, re, sys, time
import requests

# ---------------------------------------------------------------------------
# Pattern libraries (match gene symbols, species-agnostic)
# ---------------------------------------------------------------------------

IMMUNE_PATTERNS = {
    "TLR signaling": [
        r'^tlr\d+$', r'^tlr\d+[a-z]?$',
        r'^myd88$', r'^tirap$', r'^tram1?$', r'^trif$', r'^sarm1$',
        r'^irak[1-4]$', r'^traf[2-6]$',
    ],
    "NLR / inflammasome": [
        r'^nod[12]$', r'^nlrp\d+$', r'^nlrc\d+$', r'^naip',
        r'^aim2$', r'^casp[1-9]$', r'^casp[1-9]\d?$',
    ],
    "RIG-I / MDA5 / cGAS-STING": [
        r'^rigi$', r'^ddx58$', r'^ifih1$', r'^mda5$', r'^dhx58$',
        r'^mavs$', r'^sting1?$', r'^cgas$', r'^mb21d1$', r'^tmem173$',
    ],
    "Interleukins & receptors": [
        r'^il\d+[a-z]?$', r'^il\d+r[abc]?$', r'^il\d+ra[bc]?$',
    ],
    "Chemokines & receptors": [
        r'^ccl\d+[a-z]?$', r'^cxcl\d+[a-z]?$', r'^xcl\d?$',
        r'^ccr\d+$', r'^cxcr\d+$', r'^xcr\d+$', r'^ackr\d+$',
    ],
    "Interferons": [
        r'^ifn[abgklwz]', r'^ifnar\d?$', r'^ifngr\d?$', r'^ifnlr\d?$',
        r'^irf[1-9]$', r'^irf[1-9][ab]?$',
    ],
    "TNF family": [
        r'^tnf$', r'^tnfa$', r'^tnfb$', r'^lt[ab]$',
        r'^tnfrsf\d+[a-z]?$', r'^tnfsf\d+[a-z]?$',
        r'^traf[1-6]$', r'^tradd$', r'^fadd$',
    ],
    "Complement": [
        r'^c[1-9][ab]?$', r'^c[1-9]$',
        r'^cf[bdhimp]$', r'^cfb$', r'^cfd$', r'^cfh$', r'^cfi$', r'^cfp$',
        r'^c1q[abc]?$', r'^c1qtnf\d?$',
    ],
    "Antimicrobial peptides": [
        r'^defb\d+[a-z]?$', r'^defa\d+[a-z]?$', r'^def$',
        r'^camp$', r'^cathelicidin$', r'^lyz\d?$', r'^lzp',
        r'^bpi$', r'^bpi[lL]', r'^lactoferrin$', r'^ltf$',
    ],
    "JAK / STAT": [
        r'^jak[123]$', r'^tyk2$',
        r'^stat[1-6]$', r'^stat[1-6][ab]$',
    ],
    "NF-κB pathway": [
        r'^nfkb[12]?$', r'^nfkb[ia]$', r'^rel[ab]?$',
        r'^ikb[abgkez]', r'^ikbkb$', r'^ikbkg$', r'^ch(u)?k$',
        r'^tab[123]$', r'^tak1$', r'^map3k7$',
    ],
    "Antigen processing / MHC": [
        r'^mhc[12]', r'^hla-[a-g]', r'^b2m$',
        r'^tap[12]$', r'^tapbp$', r'^erap[12]$',
        r'^psmb[89]$', r'^psmb[89][ab]?$', r'^psme[12]$',
    ],
    "CD markers": [
        r'^cd[1-9]\d*[a-z]?$', r'^cd[1-9]$',
    ],
    "T / B cell signaling": [
        r'^cd3[deglz]$', r'^cd247$',
        r'^zap70$', r'^syk$', r'^lck$', r'^fyn$', r'^lyn$', r'^blk$',
        r'^btk$', r'^itk$', r'^txk$', r'^tec$',
        r'^plcg[12]$', r'^plc[gz]',
        r'^lat$', r'^slp76$', r'^lcp2$', r'^blnk$', r'^slp65$',
        r'^vav[123]$', r'^rac[12]$', r'^cdc42$',
        r'^card11$', r'^bcl10$', r'^malt1$',
        r'^nfatc[1-4]$', r'^nfat5$',
        r'^pi3k', r'^pik3c[adg]', r'^pik3r[1-6]',
    ],
    "Immune transcription factors": [
        r'^foxp3$', r'^foxn1$', r'^gata3$', r'^tbx21$', r'^tbet$',
        r'^rorc$', r'^bcl6$', r'^prdm1$', r'^blimp1?$',
        r'^batf$', r'^batf3$', r'^xbp1$', r'^irf[48]$',
    ],
    "Phagocytosis / respiratory burst": [
        r'^nox[1-4]$', r'^cybb$', r'^cyba$',
        r'^ncf[1-4]$', r'^ncf[1-4][a-z]?$',
        r'^mp[og]$', r'^mpo$', r'^elane$', r'^prtn3$', r'^ctsg$',
        r'^marco$', r'^sr-[ai]', r'^scara[3-5]', r'^cd36$',
        r'^fcgr[123]', r'^fcgr[123][ab]?$', r'^fcer1g$', r'^fcer1a$',
    ],
    "Cytokine / growth factor receptors": [
        r'^csf[123]r$', r'^csf\d+r[abc]?$',
        r'^il[1-9]\d*r[abc]?$',
        r'^epor$', r'^tpor$', r'^gcsfr$', r'^gmcsfr$',
    ],
    "Ubiquitin / ISGylation": [
        r'^usp\d+$', r'^usp\d+[a-z]?$',
        r'^isg15$', r'^herc[56]$', r'^ube2l6$', r'^hectd\d?$',
    ],
    "MAP kinase cascades": [
        r'^map3k\d+[a-z]?$', r'^map2k\d+[a-z]?$', r'^mapk\d+[a-z]?$',
        r'^mapk[1-4]\d$',
    ],
}

OTHER_PATTERNS = {
    "Collagen / ECM": [
        r'^col\d+[ab]?\d?$', r'^col\d+a\d+$',
        r'^lamb\d', r'^lamc\d', r'^lama\d',
    ],
    "MMP / TIMP": [
        r'^mmp\d+[ab]?$', r'^mmp\d+[a-z]$',
        r'^timp[1-4]$',
    ],
    "Fox transcription factors": [r'^fox[ao-z]\d?', r'^fox[a-z]\d+[a-z]?$'],
    "Sox / Hox / Pax / T-box": [
        r'^sox\d+[a-z]?$', r'^hox[a-z]\d+[a-z]?$',
        r'^pax\d+[a-z]?$', r'^tbx\d+[a-z]?$',
    ],
    "Wnt / FGF / Hedgehog / Notch": [
        r'^wnt\d+[a-z]?$', r'^fgf\d+[a-z]?$', r'^fgfr\d+[a-z]?$',
        r'^shh$', r'^ihh$', r'^dhh$', r'^ptch\d?$', r'^smo$', r'^gli[123]$',
        r'^notch\d?$', r'^jag\d?$', r'^dll[1-4]$', r'^hes\d?$',
    ],
    "Solute carriers (SLC)": [r'^slc\d+[a-z]?\d*[a-z]?$', r'^slco\d+[a-z]?$'],
    "ABC transporters": [r'^abc[abcde]\d+[a-z]?$'],
    "Cytochrome P450": [r'^cyp\d+[a-z]\d+[a-z]?$', r'^cyp\d+[a-z]?$'],
    "GST / antioxidant": [
        r'^gst[amptkz]\d?$', r'^gsta\d?$',
        r'^sod[123]$', r'^cat$', r'^prdx\d?$', r'^txn$', r'^txnrd\d?$',
        r'^gpx\d?$', r'^gcl[cm]$', r'^gsr$',
    ],
    "Heat shock proteins": [
        r'^hsp[abdegh]\d+[a-z]?$', r'^hsp\d+[a-z]?$',
        r'^hsp[abdegh]$', r'^dnaj', r'^dnaja', r'^dnajb', r'^dnajc',
        r'^hsf[1-4]$',
    ],
    "Nuclear receptors": [
        r'^nr[1-5][a-z]\d$', r'^ppar[ag]$', r'^r[ax]r[abg]$',
        r'^esr[12]$', r'^ar$', r'^pr$', r'^gr$', r'^nr3c1$',
    ],
    "G-protein coupled receptors": [r'^gpr\d+[a-z]?$', r'^gprc\d+[a-z]?$'],
    "Ion channels": [
        r'^kcn[abghjkmqstv]\d+[a-z]?$', r'^scn\d+[a-z]?$',
    ],
    "Cadherins / Protocadherins": [r'^cdh\d+[a-z]?$', r'^pcdh\d+[a-z]?$'],
    "Zinc finger TFs": [r'^znf\d+[a-z]?$', r'^zbtb\d+[a-z]?$', r'^zfp\d+[a-z]?$'],
    "Kinases (other)": [
        r'^camk\d+[a-z]?$', r'^cdk\d+[a-z]?$', r'^aurka?[a-z]?$',
        r'^plk\d+$', r'^nek\d+$', r'^dyrk\d+[a-z]?$', r'^gsk3[ab]?$',
    ],
    "RAS / RHO / small GTPases": [
        r'^kras$', r'^hras$', r'^nras$', r'^rras\d?$',
        r'^rho[abcu]$', r'^rac[123]$', r'^cdc42$',
        r'^rab\d+[a-z]?$', r'^arf\d+[a-z]?$', r'^ran$',
    ],
    "Apoptosis / BCL2 family": [
        r'^bcl2$', r'^bcl2l\d?$', r'^bcl2l\d+$',
        r'^bax$', r'^bak1$', r'^bad$', r'^bid$', r'^bim$', r'^bik$',
        r'^mcl1$', r'^bcl-xl$', r'^bclxl$',
    ],
}


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def lookup_ensembl_descriptions(genes, cache_path=None, delay=0.1):
    """
    Look up Ensembl descriptions for a list of gene symbols.

    Parameters
    ----------
    genes : list[str]
        Gene symbols (without .H suffix).
    cache_path : str or None
        Path to a JSON cache file (read/write) so we don't re-query every run.
    delay : float
        Seconds between API calls.

    Returns
    -------
    dict {gene_symbol: description_string}
    """
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            import json
            cache = json.load(f)

    descriptions = {}
    to_query = [g for g in genes if g not in cache]

    if to_query:
        print(f"  Querying Ensembl for {len(to_query)} genes...")
        for i, gene in enumerate(to_query):
            # Single call: lookup/symbol gives description + Ensembl ID
            url = (f"https://rest.ensembl.org/lookup/symbol/"
                   f"gasterosteus_aculeatus/{gene}?"
                   f"content-type=application/json")
            desc = ''
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    desc = data.get('description', '') or ''
                cache[gene] = desc
            except Exception:
                cache[gene] = ''

            if (i + 1) % 50 == 0:
                found = sum(1 for v in cache.values() if v)
                print(f"    {i+1}/{len(to_query)}: {found} with descriptions so far...")
            time.sleep(delay)

    # Apply cache
    for gene in genes:
        descriptions[gene] = cache.get(gene, '')

    # Save updated cache
    if cache_path:
        with open(cache_path, 'w') as f:
            import json
            json.dump(cache, f, indent=2)

    return descriptions


def classify_gene(gene_symbol, patterns):
    """Return list of category names matching *gene_symbol*."""
    hits = []
    g = gene_symbol.lower()
    for category, regexes in patterns.items():
        for pat in regexes:
            if re.match(pat, g):
                hits.append(category)
                break
    return hits


def characterize_genes(genes, descriptions=None, cache_path=None):
    """
    Characterize a list of gene symbols.

    Parameters
    ----------
    genes : list[str]
        Gene symbols (without .H suffix).
    descriptions : dict or None
        Pre-computed gene→description mapping. If None, queries Ensembl.
    cache_path : str or None
        Path for Ensembl description cache.

    Returns
    -------
    list[dict] with keys: gene, description, immune_categories, other_categories
    """
    if descriptions is None:
        descriptions = lookup_ensembl_descriptions(genes, cache_path=cache_path)

    results = []
    for gene in genes:
        desc = descriptions.get(gene, '')
        immune = classify_gene(gene, IMMUNE_PATTERNS)
        other = classify_gene(gene, OTHER_PATTERNS)
        results.append({
            'gene': gene,
            'description': desc,
            'immune_categories': '; '.join(immune) if immune else '',
            'other_categories': '; '.join(other) if other else '',
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Functional characterization of stickleback gene lists"
    )
    parser.add_argument('--input', required=True,
                        help='File with one gene symbol per line (no .H suffix)')
    parser.add_argument('--output', required=True,
                        help='Output CSV path')
    parser.add_argument('--cache', default='output/ensembl_description_cache.json',
                        help='JSON cache for Ensembl descriptions')
    parser.add_argument('--no-ensembl', action='store_true',
                        help='Skip Ensembl lookup (pattern-match only)')
    args = parser.parse_args()

    # Load genes
    with open(args.input) as f:
        genes = sorted(set(
            line.strip() for line in f if line.strip() and not line.startswith('#')
        ))

    print(f"Loaded {len(genes)} genes from {args.input}")

    # Characterize
    if args.no_ensembl:
        results = characterize_genes(genes, descriptions={})
    else:
        results = characterize_genes(genes, cache_path=args.cache)

    # Summary
    immune_count = sum(1 for r in results if r['immune_categories'])
    other_count = sum(1 for r in results if r['other_categories'])
    with_desc = sum(1 for r in results if r['description'])
    print(f"  Immune-related: {immune_count}")
    print(f"  Other interesting: {other_count}")
    print(f"  With Ensembl description: {with_desc}")

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'gene', 'description', 'immune_categories', 'other_categories'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} rows to {args.output}")

    # Print immune genes
    immune_results = [r for r in results if r['immune_categories']]
    if immune_results:
        print(f"\nImmune-related genes ({len(immune_results)}):")
        for r in immune_results:
            print(f"  {r['gene']:30s} [{r['immune_categories']}]")


if __name__ == '__main__':
    main()

"""
Analysis tools for lake embeddings and gene co-occurrence networks.

LakeAnalyzer
------------
- PCA projection of VQ code histogram embeddings
- Wasserstein distance for temporal drift (year-over-year change)
- Silhouette scores for evaluating group separability
- Temporal slope computation with Source vs Recipient statistical tests
- Hierarchical clustering (scipy linkage)
- Permutation tests (label shuffling) for Source/Recipient separability

GeneNetworkAnalyzer
-------------------
- Gene co-occurrence in VQ code assignments across stratifications
- Ego network construction for a gene of interest
- Second-order gene-gene network expansion
- Permutation-based null models (sample shuffling) for gene co-occurrence
- Empirical p-value computation for co-occurrence significance
"""

import numpy as np
from scipy.stats import wasserstein_distance, ttest_ind, mannwhitneyu, fisher_exact
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
from tqdm import tqdm
import networkx as nx
import re


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR q-values for a sequence of p-values.

    Returns an array the same length/order as the input, where each entry is
    the BH-adjusted q-value (the minimum false discovery rate at which that
    test can be called significant). Controls FDR at level alpha when tests
    with q <= alpha are rejected.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    # Enforce monotonicity (step-up): q_(i) = min_{j>=i} n*p_(j)/j
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def _attach_qvalues(results, pkey='p_value'):
    """Attach BH-FDR 'q_value' to each entry of {key: {pkey: float}}.

    Treats all entries as one test family and mutates each entry in place by
    adding 'q_value'. Returns ``results`` unchanged. Empty dicts pass through.
    """
    if not results:
        return results
    keys = list(results.keys())
    pvals = [results[k].get(pkey, 1.0) for k in keys]
    qvals = benjamini_hochberg(pvals)
    for k, q in zip(keys, qvals):
        results[k]['q_value'] = float(q)
    return results


def _hypergeom_enrichment(in_cluster, cluster_size, bg_size, bg_hit):
    """Upper-tail hypergeometric p-value for trait over-representation.

    P(X >= in_cluster) with X ~ Hypergeom(N=bg_size, K=bg_hit, n=cluster_size)
    — the exact ``phyper(..., lower.tail=FALSE)`` enrichment test.

    Why NOT Fisher's exact test (the previous implementation): Fisher
    conditions on BOTH margins, so when a cluster is exactly the full set of a
    trait's lakes (in_other == 0, e.g. a cluster containing all 4
    Source-Benthic lakes), the margins force the single table
    [[n, 0], [0, bg-n]] and Fisher returns p = 1.0 — the STRONGEST possible
    enrichment reported as non-significant. The hypergeometric conditions only
    on the cluster size and the background trait count, so full containment
    yields the tiny probability 1 / C(bg, n). This is the standard enrichment
    test used by GO/DAVID-style tools.
    """
    from scipy.stats import hypergeom
    if in_cluster <= 0 or bg_hit <= 0 or cluster_size <= 0:
        return 1.0
    # P(X >= in_cluster) == survival at (in_cluster - 1).
    return float(hypergeom.sf(in_cluster - 1, bg_size, bg_hit, cluster_size))


# ---------------------------------------------------------------------------
# PERMANOVA helpers: multivariable marginal (Type-III) model, block-restricted
# permutations, and betadisper. Pure numpy so the analysis needs no extra
# stats dependency.
# ---------------------------------------------------------------------------

def _permute_within_blocks(labels, blocks, rng):
    """Permute `labels` independently within each block value (restricted
    permutation respecting repeated-measures nesting, e.g. years/sex/
    infection within lake)."""
    labels = np.asarray(labels)
    blocks = np.asarray(blocks)
    out = labels.copy()
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        if len(idx) > 1:
            out[idx] = labels[rng.permutation(idx)]
    return out


def _factor_design_column(labels):
    """Full (no-intercept) one-hot design matrix for a categorical factor.

    PERMANOVA omits the intercept: the Gower matrix is already centered, so
    the grand-mean direction lies in its null space. Returns (X, n_levels).
    """
    labels = np.asarray(labels)
    uniq = sorted(set(labels.tolist()))
    idx = {lab: i for i, lab in enumerate(uniq)}
    X = np.zeros((len(labels), len(uniq)), dtype=np.float64)
    for i, lab in enumerate(labels):
        X[i, idx[lab]] = 1.0
    return X, len(uniq)


def _explained_ss(X, G):
    """PERMANOVA explained sum-of-squares for design X on Gower matrix G:
    trace(H G) = trace((X'X)^{-1} X' G X), H = X(X'X)^{-1} X'. Clamped to >= 0.
    """
    if X.shape[1] == 0:
        return 0.0
    # No try/except → 0.0: if the SVD in pinv genuinely fails (e.g. a
    # non-finite design), that is a real error and must propagate loudly —
    # silently reporting SS=0 would poison every factor's F and p.
    XtX_inv = np.linalg.pinv(X.T @ X)
    return max(float(np.trace(XtX_inv @ (X.T @ G @ X))), 0.0)


def _oneway_F(z, groups):
    """One-way ANOVA F-statistic of values `z` grouped by `groups`, or None
    if undefined (<2 groups, non-positive within-SS, or no residual df)."""
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    if len(uniq) < 2:
        return None
    grand = z.mean()
    ss_b = ss_w = 0.0
    for g in uniq:
        zi = z[groups == g]
        ss_b += len(zi) * (zi.mean() - grand) ** 2
        ss_w += float(((zi - zi.mean()) ** 2).sum())
    df_b = len(uniq) - 1
    df_w = len(z) - len(uniq)
    if df_w <= 0 or ss_w <= 0:
        return None
    return float((ss_b / df_b) / (ss_w / df_w))


def _betadisper(D, groups, n_perm, rng):
    """Anderson PERMDISP2: distance-to-centroid in PCoA space, one-way ANOVA
    F + permutation p. Lets centroid (location) effects be distinguished from
    dispersion effects. Returns (F, p), or (None, None) if undefined."""
    groups = np.asarray(groups)
    uniq = sorted(set(groups.tolist()))
    if len(uniq) < 2:
        return None, None
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(G)
    pos = w > 0
    coords = (V[:, pos] * np.sqrt(w[pos])[None, :]
              if pos.any() else np.zeros((n, 1)))
    z = np.zeros(n, dtype=np.float64)
    for g in uniq:
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        z[idx] = np.linalg.norm(coords[idx] - coords[idx].mean(axis=0), axis=1)
    F_obs = _oneway_F(z, groups)
    if F_obs is None:
        return None, None
    null = []
    for _ in range(n_perm):
        Fp = _oneway_F(z, rng.permutation(groups))
        if Fp is not None:
            null.append(Fp)
    p = (float((np.sum(np.array(null) >= F_obs) + 1) / (len(null) + 1))
         if null else 1.0)
    return float(F_obs), p


_EMD_A_EQ_CACHE = {}


def _emd_constraint_matrix(n):
    """Sparse equality-constraint matrix for the n×n transport LP (row sums = a,
    col sums = b). Depends only on n, so cache it."""
    if n in _EMD_A_EQ_CACHE:
        return _EMD_A_EQ_CACHE[n]
    from scipy.sparse import vstack, csr_matrix
    # Row constraints: for each i, sum_j P[i,j] = a[i].  P is flattened row-major.
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(n):
            rows.append(i); cols.append(i * n + j); vals.append(1.0)
    row_block = csr_matrix((vals, (rows, cols)), shape=(n, n * n))
    # Column constraints: for each j, sum_i P[i,j] = b[j].
    rows, cols, vals = [], [], []
    for j in range(n):
        for i in range(n):
            rows.append(j); cols.append(i * n + j); vals.append(1.0)
    col_block = csr_matrix((vals, (rows, cols)), shape=(n, n * n))
    A_eq = vstack([row_block, col_block]).tocsr()
    _EMD_A_EQ_CACHE[n] = A_eq
    return A_eq


def _emd_exact(a, b, C):
    """Exact earth-mover's distance between distributions a, b over cost matrix
    C, via the transportation LP (scipy HiGHS). No epsilon / no numerical
    underflow — preferred over Sinkhorn for codebook costs in [0, 2]."""
    from scipy.optimize import linprog
    n = len(a)
    A_eq = _emd_constraint_matrix(n)
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([a, b]),
                  bounds=[(0, None)] * (n * n), method='highs')
    if not res.success:
        # No silent NaN: a NaN distance would propagate through every slope,
        # mean and p-value downstream (compute_temporal_slopes / ttest / MWU)
        # with no indication. A HiGHS failure on a valid LP is a real error.
        raise RuntimeError(
            f"[_emd_exact] transport LP failed to converge: "
            f"{res.message.strip()} (n={n}). Refusing to return a NaN "
            f"distance.")
    return float(res.fun)


class LakeAnalyzer:
    """
    Downstream analysis of lake VQ-code histogram embeddings.

    Parameters
    ----------
    embeddings : dict {str: np.ndarray}
        Maps lake keys to VQ code histograms (each shape: (codebook_size,)).
    data : SticklebackData, optional
        Reference to the data loader for lake classification lookups.
    """

    def __init__(self, embeddings, data=None, codebook=None):
        self.embeddings = embeddings
        self.keys = sorted(embeddings.keys())
        self.data = data
        # Codebook vectors [K, dim] for the Wasserstein ground metric (cosine
        # distance between codes). Optional; when absent, wasserstein_temporal
        # falls back to a 1-D Wasserstein on histogram values (legacy).
        self.codebook = None
        self._cost_matrix = None
        if codebook is not None:
            self.codebook = np.asarray(codebook, dtype=np.float32)
            self._cost_matrix = self._cosine_cost_matrix(self.codebook)

    @staticmethod
    def _cosine_cost_matrix(codebook):
        """K×K cosine-distance matrix between codebook vectors, used as the
        optimal-transport ground cost for comparing VQ-code distributions."""
        cb = np.asarray(codebook, dtype=np.float64)
        norms = np.linalg.norm(cb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = cb / norms
        sim = unit @ unit.T  # cosine similarity
        return 1.0 - sim      # cosine distance ∈ [0, 2]

    def _wasserstein_code_distance(self, a, b):
        """Distance between two VQ-code histograms.

        With a codebook: the exact earth-mover's (optimal-transport) distance
        over the cosine ground cost between codebook vectors. Code indices are
        categorical, so the transport cost must come from codebook geometry,
        not index order. Without a codebook: falls back to the legacy 1-D
        Wasserstein on histogram values (informative only)."""
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if self._cost_matrix is not None:
            sa, sb = a.sum(), b.sum()
            if sa <= 0 or sb <= 0:
                # A zero-mass histogram is NOT "identical to everything": OT
                # between a zero-mass and a positive-mass distribution is
                # undefined, and reporting distance 0.0 fabricates "no drift".
                raise ValueError(
                    f"[_wasserstein_code_distance] zero-mass histogram "
                    f"(sum_a={sa}, sum_b={sb}). Embeddings must sum to a "
                    f"positive codebook weight.")
            wa, wb = a / sa, b / sb  # normalized distributions
            return _emd_exact(wa, wb, self._cost_matrix)
        return float(wasserstein_distance(a, b))

    # ------------------------------------------------------------------
    # Stratification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def stratification_from_key(key):
        """
        Infer stratification type from canonical key formatting.

        Returns one of: 'sex_year_lake', 'infection_year_lake',
        'year_lake', or 'lake'.
        """
        key_str = str(key)
        if re.search(r'\)-[fFmM]$', key_str):
            return 'sex_year_lake'
        if re.search(r'\)-[01]$', key_str):
            return 'infection_year_lake'
        if re.search(r'\(\d{4}(?:\.\d+)?\)$', key_str):
            return 'year_lake'
        return 'lake'

    def keys_by_stratification(self):
        """Return {stratification_type: [keys]} for the current embeddings."""
        grouped = defaultdict(list)
        for k in self.keys:
            grouped[self.stratification_from_key(k)].append(k)
        return dict(grouped)

    def for_stratification(self, strat_type):
        """
        Return a new LakeAnalyzer containing only embeddings for the
        given stratification type.
        """
        filtered = {k: self.embeddings[k] for k in self.keys
                    if self.stratification_from_key(k) == strat_type}
        return LakeAnalyzer(filtered, data=self.data, codebook=self.codebook)

    def for_subset(self, keys):
        """Return a new LakeAnalyzer containing only the given keys."""
        filtered = {k: self.embeddings[k] for k in keys if k in self.embeddings}
        return LakeAnalyzer(filtered, data=self.data, codebook=self.codebook)

    def for_role(self, role):
        """Return a new LakeAnalyzer with only Source or Recipient lake keys.

        Parameters
        ----------
        role : str
            ``'Source'`` or ``'Recipient'``.
        """
        if self.data is None:
            raise ValueError("LakeAnalyzer.for_role() requires data=SticklebackData")
        subset = []
        for k in self.keys:
            lake = str(k).split(' (')[0] if ' (' in str(k) else str(k)
            if self.data.get_lake_role(lake) == role:
                subset.append(k)
        return self.for_subset(subset)

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    def pca_project(self, n_components=2):
        """
        Project embeddings into PCA space.

        Returns
        -------
        projected : np.ndarray (n_lakes × n_components)
        pca : sklearn PCA object (fitted)
        """
        X = np.vstack([self.embeddings[k] for k in self.keys])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(X)
        # Min-max normalize each component to [0, 1]
        for i in range(projected.shape[1]):
            pmin, pmax = projected[:, i].min(), projected[:, i].max()
            if pmax > pmin:
                projected[:, i] = (projected[:, i] - pmin) / (pmax - pmin)
        return projected, pca

    # ------------------------------------------------------------------
    # Wasserstein temporal drift
    # ------------------------------------------------------------------

    def wasserstein_temporal(self, base_year=2019, suffix=None):
        """
        For each lake, compute Wasserstein distance between the base year's
        embedding and each subsequent year's embedding.

        This measures how much a lake's gene co-expression network diverges
        over time from its initial state — the key experimental signal:
        recipient lakes should drift more than source lakes.

        Parameters
        ----------
        base_year : int
            Reference year for comparison (default 2019, the transplant year).
        suffix : str or None
            When given (e.g. '-f', '-m', '-0', '-1'), build the per-lake
            temporal series using ONLY keys carrying this suffix — i.e. within
            a single sex or infection status.  This is how per-status grids
            ('do female networks drift differently from male networks?') are
            computed.  When None (default), only unsuffixed year_lake keys are
            used, and suffixed keys are skipped loudly (they would otherwise
            collide at (lake, year)).

        Returns
        -------
        distances : dict {lake: [(year, distance), ...]}
            Sorted by year.
        """
        lake_years = defaultdict(dict)
        n_skipped_suffixed = 0
        n_mismatched = 0
        for key, hist in self.embeddings.items():
            if ' (' not in key:
                continue
            lake, rest = key.split(' (', 1)
            if ')' not in rest:
                continue
            year_part, _, key_suffix = rest.partition(')')
            key_suffix = key_suffix.strip()
            if suffix is not None:
                # Per-status series: keep only keys with THIS suffix; keys of
                # another status (or unsuffixed) are not part of the series.
                if key_suffix != suffix:
                    n_mismatched += 1
                    continue
            elif key_suffix:
                # Lake-level series: only year_lake stratifications (e.g.
                # 'Lake (2021)') belong.  sex/infection-suffixed keys
                # ('Lake (2021)-f', 'Lake (2021)-0') share the same
                # (lake, year) and would silently OVERWRITE each other in the
                # dict, so they are skipped loudly instead.
                n_skipped_suffixed += 1
                continue
            try:
                year = int(float(year_part))
            except ValueError:
                continue
            lake_years[lake][year] = hist

        if not lake_years and self.embeddings:
            if suffix is not None:
                raise ValueError(
                    f"[wasserstein_temporal] No embeddings matched "
                    f"suffix='{suffix}' in this analyzer — cannot build a "
                    f"per-status temporal series.")
            raise ValueError(
                "[wasserstein_temporal] No year_lake embeddings found — the "
                "analyzer holds only suffixed (sex/infection) stratifications. "
                "Call for_stratification('year_lake') or build embeddings with "
                "by='year_lake'.")
        if n_skipped_suffixed:
            print(f"[wasserstein_temporal] Skipped {n_skipped_suffixed} "
                  f"non-year_lake keys (sex/infection-suffixed); temporal "
                  f"series uses only year_lake embeddings.")
        if suffix is not None and n_mismatched:
            print(f"[wasserstein_temporal] Ignored {n_mismatched} keys that "
                  f"do not match suffix='{suffix}'; temporal series uses "
                  f"'{suffix}' embeddings only.")

        result = {}
        n_fallback_base = 0
        for lake, year_hists in lake_years.items():
            if base_year in year_hists:
                base_yr = base_year
            else:
                base_yr = min(year_hists)
                n_fallback_base += 1
            base_hist = year_hists[base_yr]
            dists = [(yr, self._wasserstein_code_distance(base_hist, hist))
                     for yr, hist in sorted(year_hists.items())
                     if yr != base_yr]
            if dists:
                result[lake] = dists

        if n_fallback_base:
            # Lakes first sampled after base_year use their earliest sample as
            # the reference, so their distances are NOT on the same scale as
            # lakes with a true base_year baseline.  Loudly flag this — the
            # endpoint Source/Recipient comparison would otherwise compare
            # lakes at unequal baselines (and unequal horizons).
            print(f"[wasserstein_temporal] {n_fallback_base} lake(s) had no "
                  f"base_year={base_year} sample and used their earliest year "
                  f"as baseline instead: "
                  f"{', '.join(sorted(l for l in lake_years if base_year not in lake_years[l]))}. "
                  f"Endpoint drift comparisons across lakes are on differing "
                  f"baselines.")

        return result

    def wasserstein_temporal_or_skip(self, base_year=2019, suffix=None):
        """Like ``wasserstein_temporal`` but returns ``{}`` with a loud note
        instead of raising when this analyzer's keys don't define an
        unambiguous per-lake temporal series (e.g. a sex/infection-suffixed
        sub-analyzer with no ``suffix``, where 'Lake (2021)-f' and
        'Lake (2021)-m' collide per lake-year).  ``suffix`` is forwarded to
        ``wasserstein_temporal`` so per-status series ('-f', '-m', '-0', '-1')
        can be requested.  Per-stratification figure loops use this so a
        legitimately-undefined stratification is skipped loudly rather than
        crashing the pipeline."""
        try:
            return self.wasserstein_temporal(base_year=base_year, suffix=suffix)
        except ValueError as exc:
            print(f"  [wasserstein_temporal] Skipped: {exc}")
            return {}

    # ------------------------------------------------------------------
    # Silhouette scores
    # ------------------------------------------------------------------

    def silhouette(self, labels_dict):
        """
        Compute silhouette score for a given labeling of the embeddings.

        Parameters
        ----------
        labels_dict : dict {key: int_label}
            Maps embedding keys to integer cluster labels.

        Returns
        -------
        score : float
            Silhouette score in [-1, 1]. Higher = better separated clusters.
        """
        X = np.vstack([self.embeddings[k] for k in self.keys
                       if k in labels_dict])
        labels = np.array([labels_dict[k] for k in self.keys
                          if k in labels_dict])
        if len(set(labels)) < 2:
            return float('nan')
        return silhouette_score(X, labels)

    # ------------------------------------------------------------------
    # Label builders
    # ------------------------------------------------------------------

    def build_labels(self, label_type):
        """
        Build integer labels for common biological groupings.

        Parameters
        ----------
        label_type : str
            One of:
            - 'year': label by year
            - 'source_recipient': 0 = Source, 1 = Recipient
            - 'genotype_benthic_limnetic': 0 = LimneticPool, 1 = BenthicPool
            - 'sex': 0 = female, 1 = male
            - 'infection': 0 = not infected, 1 = infected

        Returns
        -------
        labels : dict {key: int}
        """
        labels = {}

        for key in self.keys:
            if label_type == 'year':
                # Extract year from key like "Crystal (2021)"
                if ' (' in key:
                    year_str = key.split('(')[1].split(')')[0].split('-')[0]
                    try:
                        labels[key] = int(float(year_str))
                    except ValueError:
                        pass

            elif label_type == 'source_recipient' and self.data:
                lake = key.split(' (')[0] if ' (' in key else key
                role = self.data.get_lake_role(lake)
                if role == 'Source':
                    labels[key] = 0
                elif role == 'Recipient':
                    labels[key] = 1

            elif label_type == 'genotype_benthic_limnetic' and self.data:
                lake = key.split(' (')[0] if ' (' in key else key
                gen = self.data.get_genotype(lake)
                if gen == 'LimneticPool':
                    labels[key] = 0
                elif gen == 'BenthicPool':
                    labels[key] = 1

            elif label_type == 'ecotype' and self.data:
                lake = key.split(' (')[0] if ' (' in key else key
                eco = self.data.get_lake_ecotype(lake)
                if eco == 'Benthic':
                    labels[key] = 0
                elif eco == 'Limnetic':
                    labels[key] = 1

            elif label_type == 'sex':
                if key.endswith('-f'):
                    labels[key] = 0
                elif key.endswith('-m'):
                    labels[key] = 1

            elif label_type == 'infection':
                if key.endswith('-0'):
                    labels[key] = 0
                elif key.endswith('-1'):
                    labels[key] = 1

        return labels

    # ------------------------------------------------------------------
    # Wasserstein scaling
    # ------------------------------------------------------------------

    @staticmethod
    def scale_wasserstein_p95(wasserstein_distances):
        """
        Scale all Wasserstein distances by the 95th percentile across all
        lakes and years, mapping them to a [0, ~1] range where 0 = identical
        distributions and 1.0 ≈ top-5% drift magnitude.

        This makes slopes comparable across analyses: a slope of 0.05 / year
        means the lake drifts 5% of the P95 distance per year.

        Parameters
        ----------
        wasserstein_distances : dict {lake: [(year, distance), ...]}

        Returns
        -------
        scaled : dict {lake: [(year, scaled_distance), ...]}
        p95 : float
            The 95th-percentile value used as the denominator.
        """
        all_dists = [d for lake_dists in wasserstein_distances.values()
                     for _, d in lake_dists]
        if not all_dists:
            return wasserstein_distances, 1.0
        p95 = float(np.percentile(all_dists, 95))
        if p95 == 0:
            return wasserstein_distances, 1.0
        scaled = {
            lake: [(yr, d / p95) for yr, d in dists]
            for lake, dists in wasserstein_distances.items()
        }
        return scaled, p95

    # ------------------------------------------------------------------
    # Temporal slopes
    # ------------------------------------------------------------------

    @staticmethod
    def compute_temporal_slopes(wasserstein_distances):
        """
        Compute mean temporal slope for each lake's Wasserstein drift
        via least-squares linear regression.

        Parameters
        ----------
        wasserstein_distances : dict {lake: [(year, distance), ...]}

        Returns
        -------
        slopes : dict {lake: float}
            Mean change in Wasserstein distance per year.
        """
        slopes = {}
        skipped = []
        for lake, year_dists in wasserstein_distances.items():
            if len(year_dists) < 2:
                # A <2-point lake has no estimable slope.  Fabricating 0.0
                # polluted the Source/Recipient drift comparison with fake
                # zeros — exclude it loudly instead.
                skipped.append(lake)
                continue
            years = np.array([y for y, _ in year_dists])
            dists = np.array([d for _, d in year_dists])
            A = np.vstack([years, np.ones_like(years)]).T
            slope, _ = np.linalg.lstsq(A, dists, rcond=None)[0]
            slopes[lake] = float(slope)
        if skipped:
            print(f"[compute_temporal_slopes] Excluded {len(skipped)} lake(s) "
                  f"with <2 timepoints (no estimable slope): "
                  f"{', '.join(sorted(skipped))}")
        return slopes

    def source_vs_recipient_slope_test(self, wasserstein_distances):
        """
        Compute temporal slopes and test whether Source and Recipient
        lakes differ in their rate of network divergence.

        Runs both a t-test (parametric) and Mann-Whitney U (non-parametric).
        Requires at least 2 lakes per group for the statistics to run.

        Parameters
        ----------
        wasserstein_distances : dict {lake: [(year, distance), ...]}

        Returns
        -------
        result : dict
            'slopes': {lake: float}
            'source_slopes': list[float]
            'recipient_slopes': list[float]
            'ttest_stat': float or None
            'ttest_pvalue': float or None
            'mannwhitney_stat': float or None
            'mannwhitney_pvalue': float or None
        """
        slopes = self.compute_temporal_slopes(wasserstein_distances)
        result = {'slopes': slopes}

        if self.data is None:
            return result

        source_slopes = []
        recipient_slopes = []
        for lake, slope in slopes.items():
            role = self.data.get_lake_role(lake)
            if role == 'Source':
                source_slopes.append(slope)
            elif role == 'Recipient':
                recipient_slopes.append(slope)

        result['source_slopes'] = source_slopes
        result['recipient_slopes'] = recipient_slopes

        if len(source_slopes) >= 2 and len(recipient_slopes) >= 2:
            t_stat, t_pval = ttest_ind(source_slopes, recipient_slopes,
                                       alternative='two-sided')
            mw_stat, mw_pval = mannwhitneyu(source_slopes, recipient_slopes,
                                            alternative='two-sided')
            result['ttest_stat'] = float(t_stat)
            result['ttest_pvalue'] = float(t_pval)
            result['mannwhitney_stat'] = float(mw_stat)
            result['mannwhitney_pvalue'] = float(mw_pval)
        else:
            result['ttest_stat'] = None
            result['ttest_pvalue'] = None
            result['mannwhitney_stat'] = None
            result['mannwhitney_pvalue'] = None

        return result

    # ------------------------------------------------------------------
    # Hierarchical clustering
    # ------------------------------------------------------------------

    def hierarchical_clustering(self, method='ward', n_clusters=None):
        """
        Hierarchical clustering of lake embeddings using scipy linkage.

        Parameters
        ----------
        method : str
            Linkage method ('ward', 'average', 'complete', 'single').
        n_clusters : int or None
            If provided, cut the tree to return flat cluster labels.

        Returns
        -------
        result : dict
            'linkage': np.ndarray (linkage matrix)
            'labels': dict {key: int} (cluster labels, if n_clusters given)
            'keys': list[str] (embedding keys in linkage order)
        """
        X = np.vstack([self.embeddings[k] for k in self.keys])
        Z = linkage(X, method=method)
        result = {'linkage': Z, 'keys': self.keys}

        if n_clusters is not None:
            cluster_ids = fcluster(Z, n_clusters, criterion='maxclust')
            result['labels'] = dict(zip(self.keys, cluster_ids))

        return result

    # ------------------------------------------------------------------
    # Cluster composition analysis
    # ------------------------------------------------------------------

    def analyze_clusters(self, n_clusters=4):
        """
        Cut the dendrogram into *n_clusters* and test whether each cluster
        is enriched for a biological label (Source/Recipient, Benthic/Limnetic,
        sex, infection year).

        Returns a dict with per-cluster composition and significant enrichments.
        """
        clust = self.hierarchical_clustering(n_clusters=n_clusters)
        labels = clust['labels']  # {key: cluster_id}

        # Build per-cluster lists
        clusters = defaultdict(list)
        for key, cid in labels.items():
            clusters[cid].append(key)

        result = {'n_clusters': n_clusters, 'clusters': {}}

        for cid in sorted(clusters.keys()):
            keys = clusters[cid]
            n = len(keys)
            info = {'size': n, 'all_keys': keys, 'keys': keys[:5]}

            # --- Lake-level enrichment (Role × Ecotype, Role, Ecotype).
            # The background must be UNIQUE biological lakes: self.keys holds
            # every stratification of the same lake (year_lake + sex_year_lake
            # + infection_year_lake), so counting keys double-counted each
            # lake up to 3× and inflated the Fisher tables.
            cluster_lakes = sorted(
                {k.split(' (')[0] if ' (' in k else k for k in keys})
            all_lakes = sorted(
                {k.split(' (')[0] if ' (' in k else k for k in self.keys})
            n_lakes = len(cluster_lakes)
            n_bg = len(all_lakes)

            # --- Role × Ecotype (Source Benthic, Source Limnetic,
            #     Recipient Benthic, Recipient Limnetic) ---
            if self.data:
                combo_counts = Counter(
                    f'{self.data.get_lake_role(l)} '
                    f'{self.data.get_lake_ecotype(l)}' for l in cluster_lakes)
                info['role_ecotype_counts'] = dict(combo_counts)
                bg_combos = Counter(
                    f'{self.data.get_lake_role(l)} '
                    f'{self.data.get_lake_ecotype(l)}' for l in all_lakes)
                info['role_ecotype_enrichment'] = {}
                for combo in ['Source Benthic', 'Source Limnetic',
                              'Recipient Benthic', 'Recipient Limnetic']:
                    in_cluster = combo_counts.get(combo, 0)
                    if in_cluster == 0:
                        continue
                    in_other = bg_combos.get(combo, 0) - in_cluster
                    # Hypergeometric upper-tail enrichment (NOT Fisher's exact):
                    # the fully-contained case (in_other == 0) is the STRONGEST
                    # enrichment and must get the tiny p it deserves — Fisher
                    # conditions on both margins and reports p = 1.0 there.
                    p = _hypergeom_enrichment(
                        in_cluster, n_lakes, n_bg, bg_combos.get(combo, 0))
                    info['role_ecotype_enrichment'][combo] = {
                        'count': in_cluster, 'pct': in_cluster / n_lakes,
                        'fisher_p': float(p),
                    }

                # --- Marginal role (Source / Recipient) ---
                role_counts = Counter(
                    self.data.get_lake_role(l) for l in cluster_lakes)
                info['role_counts'] = dict(role_counts)
                bg_roles = Counter(
                    self.data.get_lake_role(l) for l in all_lakes)
                info['role_enrichment'] = {}
                for role in ['Source', 'Recipient']:
                    in_cluster = role_counts.get(role, 0)
                    if in_cluster == 0:
                        continue
                    in_other = bg_roles.get(role, 0) - in_cluster
                    p = _hypergeom_enrichment(
                        in_cluster, n_lakes, n_bg, bg_roles.get(role, 0))
                    info['role_enrichment'][role] = {
                        'count': in_cluster, 'pct': in_cluster / n_lakes,
                        'fisher_p': float(p),
                    }

                # --- Marginal ecotype (Benthic / Limnetic) ---
                eco_counts = Counter(
                    self.data.get_lake_ecotype(l) for l in cluster_lakes)
                info['ecotype_counts'] = dict(eco_counts)
                bg_eco = Counter(
                    self.data.get_lake_ecotype(l) for l in all_lakes)
                info['ecotype_enrichment'] = {}
                for eco in ['Benthic', 'Limnetic']:
                    in_cluster = eco_counts.get(eco, 0)
                    if in_cluster == 0:
                        continue
                    in_other = bg_eco.get(eco, 0) - in_cluster
                    p = _hypergeom_enrichment(
                        in_cluster, n_lakes, n_bg, bg_eco.get(eco, 0))
                    info['ecotype_enrichment'][eco] = {
                        'count': in_cluster, 'pct': in_cluster / n_lakes,
                        'fisher_p': float(p),
                    }

                # --- Sex (Male / Female) — only when keys carry sex suffixes ---
                # The enrichment universe is the SEX-SUFFIXED keys only: a
                # year_lake or infection-suffixed key cannot be "not male",
                # so counting it in the background biased the 2×2 toward
                # whatever strata dominate self.keys.
                sexes = []
                for k in keys:
                    if str(k).endswith('-f'):
                        sexes.append('Female')
                    elif str(k).endswith('-m'):
                        sexes.append('Male')
                if sexes:
                    info['sex_counts'] = dict(Counter(sexes))
                    sex_keys_all = [k for k in self.keys
                                    if str(k).endswith('-f')
                                    or str(k).endswith('-m')]
                    bg_sex = Counter(
                        'Female' if str(k).endswith('-f') else 'Male'
                        for k in sex_keys_all)
                    n_sex = len(sex_keys_all)
                    n_sex_cluster = len(sexes)
                    info['sex_enrichment'] = {}
                    for s in ['Male', 'Female']:
                        in_cluster = sexes.count(s)
                        in_other = bg_sex.get(s, 0) - in_cluster
                        not_in_cluster = n_sex_cluster - in_cluster
                        not_in_other = n_sex - n_sex_cluster - in_other
                        if in_cluster > 0 and in_other > 0:
                            _, p = fisher_exact([[in_cluster, not_in_cluster],
                                                 [in_other, not_in_other]])
                            info['sex_enrichment'][s] = {
                                'count': in_cluster,
                                'pct': in_cluster / n_sex_cluster,
                                'fisher_p': float(p),
                            }

                # --- Infection (0=non-infected / 1=infected) — only when
                #     keys carry infection suffixes ---
                # Same suffixed-keys-only universe as sex (see above).
                infs = []
                for k in keys:
                    m = re.search(r'\)-([01])$', str(k))
                    if m:
                        infs.append('Infected' if int(m.group(1)) == 1
                                    else 'Non-infected')
                if infs:
                    info['infection_counts'] = dict(Counter(infs))
                    inf_keys_all = [k for k in self.keys
                                    if re.search(r'\)-([01])$', str(k))]
                    bg_inf = Counter(
                        'Infected' if re.search(r'\)-([01])$', str(k))
                        .group(1) == '1' else 'Non-infected'
                        for k in inf_keys_all)
                    n_inf = len(inf_keys_all)
                    n_inf_cluster = len(infs)
                    info['infection_enrichment'] = {}
                    for lab in ['Infected', 'Non-infected']:
                        in_cluster = infs.count(lab)
                        in_other = bg_inf.get(lab, 0) - in_cluster
                        not_in_cluster = n_inf_cluster - in_cluster
                        not_in_other = n_inf - n_inf_cluster - in_other
                        if in_cluster > 0 and in_other > 0:
                            _, p = fisher_exact([[in_cluster, not_in_cluster],
                                                 [in_other, not_in_other]])
                            info['infection_enrichment'][lab] = {
                                'count': in_cluster,
                                'pct': in_cluster / n_inf_cluster,
                                'fisher_p': float(p),
                            }

                # --- BH-FDR across every Fisher test in this cluster.  Dozens
                #     of tests with no correction would produce spurious
                #     'significant' enrichments. ---
                fisher_families = ('role_ecotype_enrichment', 'role_enrichment',
                                   'ecotype_enrichment', 'sex_enrichment',
                                   'infection_enrichment')
                all_fisher = []
                for enr_key in fisher_families:
                    for label, d in info.get(enr_key, {}).items():
                        all_fisher.append((enr_key, label, d['fisher_p']))
                if all_fisher:
                    qvals = benjamini_hochberg([x[2] for x in all_fisher])
                    for (enr_key, label, _), q in zip(all_fisher, qvals):
                        info[enr_key][label]['fisher_q'] = float(q)

            # --- Year distribution ---
            years = []
            for k in keys:
                if ' (' in k:
                    yr_str = k.split('(')[1].split(')')[0].split('-')[0]
                    try:
                        years.append(int(float(yr_str)))
                    except ValueError:
                        pass
            if years:
                info['year_range'] = f'{min(years)}–{max(years)}'
                info['year_mean'] = float(np.mean(years))

            result['clusters'][cid] = info

        return result

    # ------------------------------------------------------------------
    # Silhouette analysis for optimal k
    # ------------------------------------------------------------------

    def silhouette_scan(self, max_k=10):
        """
        Compute mean silhouette score for k = 2..max_k.

        Returns
        -------
        dict : {k: silhouette_score}
        """
        from sklearn.metrics import silhouette_score
        X = np.vstack([self.embeddings[k] for k in self.keys])
        scores = {}
        for k in range(2, min(max_k + 1, len(self.keys))):
            Z = linkage(X, method='ward')
            labels = fcluster(Z, k, criterion='maxclust')
            if len(set(labels)) < 2:
                scores[k] = float('nan')
            else:
                scores[k] = float(silhouette_score(X, labels))
        return scores

    # ------------------------------------------------------------------
    # Per-code infection enrichment
    # ------------------------------------------------------------------

    def infection_code_enrichment(self, codebook_size=100):
        """
        For each VQ code, test whether it is differentially used in
        infected vs non-infected strata within infection_year_lake keys.

        Returns
        -------
        dict : {code: {'infected_mean': float, 'noninfected_mean': float,
                        'fold_change': float, 'fisher_p': float}}
            Keyed by code (in code order); p-values are attached as
            'p_value'/'q_value' on each entry.
        """
        # Filter to infection_year_lake keys only
        inf_keys = [k for k in self.keys
                    if re.search(r'\)-([01])$', str(k))]
        if not inf_keys:
            return {}

        infected = []
        noninfected = []
        for k in inf_keys:
            m = re.search(r'\)-([01])$', str(k))
            is_inf = int(m.group(1)) == 1
            hist = self.embeddings[k]
            if is_inf:
                infected.append(hist)
            else:
                noninfected.append(hist)

        if not infected or not noninfected:
            return {}

        import numpy as np
        inf_arr = np.array(infected)    # (n_inf, K)
        ninf_arr = np.array(noninfected)  # (n_ninf, K)

        results = {}
        for code in range(codebook_size):
            inf_usage = inf_arr[:, code]
            ninf_usage = ninf_arr[:, code]
            inf_mean = float(np.mean(inf_usage))
            ninf_mean = float(np.mean(ninf_usage))

            # Fold change (add small epsilon to avoid div by zero)
            fc = (inf_mean + 1e-8) / (ninf_mean + 1e-8)

            # Mann-Whitney U test (non-parametric, handles zero-inflated data).
            # No blanket try/except → p=1.0: a genuine failure (shape mismatch,
            # etc.) must propagate loudly.  The only legitimate skip is when
            # ALL values are identical across both groups, which makes
            # Mann-Whitney undefined (scipy raises) and carries no signal.
            pooled = np.concatenate([inf_usage, ninf_usage])
            if np.all(pooled == pooled[0]):
                print(f"  [infection_code_enrichment] code {code}: identical "
                      f"usage across all strata — skipping.")
                continue
            from scipy.stats import mannwhitneyu
            u_stat, p = mannwhitneyu(inf_usage, ninf_usage,
                                     alternative='two-sided')

            results[code] = {
                'infected_mean': inf_mean,
                'noninfected_mean': ninf_mean,
                'fold_change': fc,
                'p_value': float(p),
            }

        # Multiple-testing correction: one family per codebook across all codes.
        return _attach_qvalues(results)

    def sex_code_enrichment(self, codebook_size=100):
        """Test each VQ code for differential usage in Male vs Female.

        Uses sex_year_lake keys only.  Returns dict keyed by code (in code
        order) with 'p_value'/'q_value' attached per entry.
        """
        sex_keys = [k for k in self.keys
                    if re.search(r'\)-[fFmM]$', str(k))]
        if not sex_keys:
            return {}

        male_hist = []
        female_hist = []
        for k in sex_keys:
            m = re.search(r'\)-([fFmM])$', str(k))
            is_male = m.group(1).lower() == 'm'
            hist = self.embeddings[k]
            if is_male:
                male_hist.append(hist)
            else:
                female_hist.append(hist)

        if not male_hist or not female_hist:
            return {}

        import numpy as np
        male_arr = np.array(male_hist)
        female_arr = np.array(female_hist)

        results = {}
        for code in range(codebook_size):
            m_usage = male_arr[:, code]
            f_usage = female_arr[:, code]
            m_mean = float(np.mean(m_usage))
            f_mean = float(np.mean(f_usage))
            fc = (m_mean + 1e-8) / (f_mean + 1e-8)
            # No blanket try/except → p=1.0 (see infection_code_enrichment).
            pooled = np.concatenate([m_usage, f_usage])
            if np.all(pooled == pooled[0]):
                print(f"  [sex_code_enrichment] code {code}: identical usage "
                      f"across all strata — skipping.")
                continue
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(m_usage, f_usage, alternative='two-sided')
            results[code] = {
                'male_mean': m_mean, 'female_mean': f_mean,
                'fold_change': fc, 'p_value': float(p),
            }
        # Multiple-testing correction: one family per codebook across all codes.
        return _attach_qvalues(results)

    # Hardcoded fallback when sd/data is unavailable
    _LAKE_ROLES = {
        'Finger Lake': 'Source', 'Long Lake': 'Source',
        'Spirit Lake': 'Source', 'South Rolly Lake': 'Source',
        'Tern Lake': 'Source', 'Walby Lake': 'Source',
        'Wik Lake': 'Source', 'Watson Lake': 'Source',
        'CC Lake': 'Recipient', 'Crystal Lake': 'Recipient',
        'Fred Lake': 'Recipient', 'Hope Lake': 'Recipient',
        'Leisure Lake': 'Recipient', 'Leisure Pond': 'Recipient',
        'Loon Lake': 'Recipient', 'Ranchero Lake': 'Recipient',
    }

    def _get_lake_role(self, key):
        """Resolve lake role from data object or hardcoded fallback."""
        lake = str(key).split(' (')[0] if ' (' in str(key) else str(key)
        # No try/except here: a bug in data.get_lake_role must propagate, not
        # be silently swallowed into the hardcoded fallback.
        if self.data is not None:
            return self.data.get_lake_role(lake)
        return self._LAKE_ROLES.get(lake, 'Other')

    def role_code_enrichment(self, codebook_size=100):
        """Test each VQ code for differential usage in Source vs Recipient.

        Uses year_lake keys only (no sex/infection split).
        """
        yl_keys = [k for k in self.keys
                   if re.search(r'\(\d{4}(?:\.\d+)?\)$', str(k))]
        if not yl_keys:
            return {}

        src_hist = []
        rec_hist = []
        for k in yl_keys:
            role = self._get_lake_role(k)
            hist = self.embeddings[k]
            if role == 'Source':
                src_hist.append(hist)
            elif role == 'Recipient':
                rec_hist.append(hist)

        if not src_hist or not rec_hist:
            return {}

        import numpy as np
        src_arr = np.array(src_hist)
        rec_arr = np.array(rec_hist)

        results = {}
        for code in range(codebook_size):
            s_usage = src_arr[:, code]
            r_usage = rec_arr[:, code]
            s_mean = float(np.mean(s_usage))
            r_mean = float(np.mean(r_usage))
            fc = (s_mean + 1e-8) / (r_mean + 1e-8)
            # No blanket try/except → p=1.0 (see infection_code_enrichment).
            pooled = np.concatenate([s_usage, r_usage])
            if np.all(pooled == pooled[0]):
                print(f"  [role_code_enrichment] code {code}: identical usage "
                      f"across all strata — skipping.")
                continue
            from scipy.stats import mannwhitneyu
            _, p = mannwhitneyu(s_usage, r_usage, alternative='two-sided')
            results[code] = {
                'source_mean': s_mean, 'recipient_mean': r_mean,
                'fold_change': fc, 'p_value': float(p),
            }
        # Multiple-testing correction: one family per codebook across all codes.
        return _attach_qvalues(results)

    # ------------------------------------------------------------------
    # Permutation tests
    # ------------------------------------------------------------------

    def _permute_labels_lake_level(self, true_labels, rng, n_permutations):
        """Permutation null of the silhouette score, permuting labels at the
        LAKE level (block permutation).

        ``self.keys`` holds multiple stratifications of each lake (year_lake +
        sex_year_lake + infection_year_lake), so permuting keys directly would
        treat every lake-year as an independent sample — pseudo-replication
        that inflates the effective sample size.  Instead each lake is assigned
        a label once per permutation and that label propagates to all of the
        lake's keys, preserving within-lake correlation under the null.

        Returns
        -------
        (null_scores : list[float], n_lakes : int)
        """
        labeled_keys = [k for k in self.keys if k in true_labels]
        key_to_lake = {k: (k.split(' (')[0] if ' (' in k else k)
                       for k in labeled_keys}
        lakes = sorted({l for l in key_to_lake.values()})

        # Lake-level block permutation is ONLY valid when the label is
        # constant within a lake (source_recipient / ecotype / genotype).
        # For labels that vary per key of the same lake (year, sex,
        # infection), collapsing each lake to its first key's label would
        # destroy the very signal being tested, and the null would measure
        # lake-block separation instead of the label of interest.  Detect
        # that and fall back to plain key-level permutation (there is no
        # pseudo-replication to guard against when labels vary per key).
        label_varies_within_lake = False
        for l in lakes:
            lake_keys = [k for k in labeled_keys if key_to_lake[k] == l]
            if len({true_labels[k] for k in lake_keys}) > 1:
                label_varies_within_lake = True
                break

        if label_varies_within_lake:
            key_values = [true_labels[k] for k in labeled_keys]
            null_scores = []
            for _ in range(n_permutations):
                perm_labels = dict(zip(labeled_keys,
                                       rng.permutation(key_values)))
                score = self.silhouette(perm_labels)
                if not np.isnan(score):
                    null_scores.append(score)
            return null_scores, len(lakes)

        lake_label = {}
        for l in lakes:
            first_key = next(k for k in labeled_keys if key_to_lake[k] == l)
            lake_label[l] = true_labels[first_key]
        lake_values = [lake_label[l] for l in lakes]

        null_scores = []
        for _ in range(n_permutations):
            perm_vals = rng.permutation(lake_values)
            perm_lake = dict(zip(lakes, perm_vals))
            perm_labels = {k: perm_lake[key_to_lake[k]] for k in labeled_keys}
            score = self.silhouette(perm_labels)
            if not np.isnan(score):
                null_scores.append(score)
        return null_scores, len(lakes)

    def source_recipient_permutation_test(self, n_permutations=1000,
                                          random_seed=42):
        """
        Permutation test: is the silhouette score between Source and
        Recipient lakes larger than expected by random label assignment?

        Shuffles Source/Recipient labels to build a null distribution.
        Does NOT require rebuilding graphs — only shuffles labels.
        Labels are permuted at the LAKE level (block permutation) so multiple
        stratifications of the same lake do not inflate the effective n.

        Parameters
        ----------
        n_permutations : int
            Number of label permutations.
        random_seed : int
            Seed for reproducibility.

        Returns
        -------
        result : dict
            'observed': float (silhouette with true labels)
            'null': list[float] (silhouette with permuted labels)
            'p_value': float (fraction of null >= observed)
        """
        rng = np.random.RandomState(random_seed)
        true_labels = self.build_labels('source_recipient')
        if not true_labels:
            raise ValueError(
                "[source_recipient_permutation_test] No Source/Recipient "
                "labels could be built (requires data=SticklebackData with "
                "the lake classification map).")
        observed = self.silhouette(true_labels)

        null_scores, n_lakes = self._permute_labels_lake_level(
            true_labels, rng, n_permutations)

        p_value = (np.sum(np.array(null_scores) >= observed) + 1) / (len(null_scores) + 1)

        return {
            'observed': observed,
            'null': null_scores,
            'p_value': p_value,
            'label_type': 'source_recipient',
            'n_lakes': n_lakes,
        }

    def label_permutation_test(self, label_type, n_permutations=1000,
                                random_seed=42):
        """Permutation test for an arbitrary label type.

        Builds labels via ``build_labels(label_type)``, computes the
        observed silhouette, then shuffles labels to build a null
        distribution.  Returns the same dict shape as
        ``source_recipient_permutation_test``.

        Parameters
        ----------
        label_type : str
            Passed to ``build_labels()`` (e.g. 'ecotype',
            'genotype_benthic_limnetic').
        n_permutations : int
        random_seed : int

        Returns
        -------
        dict with keys: observed, null, p_value, label_type, n_labeled
        """
        rng = np.random.RandomState(random_seed)
        true_labels = self.build_labels(label_type)
        if not true_labels:
            raise ValueError(
                f"[label_permutation_test] No keys were labeled for "
                f"label_type='{label_type}'. Check the labels are present "
                f"in the embedding keys.")
        observed = self.silhouette(true_labels)

        # Lake-level block permutation (see _permute_labels_lake_level): do not
        # treat every stratification of the same lake as an independent sample.
        null_scores, n_lakes = self._permute_labels_lake_level(
            true_labels, rng, n_permutations)

        if len(null_scores) > 0:
            p_value = ((np.sum(np.array(null_scores) >= observed) + 1)
                        / (len(null_scores) + 1))
        else:
            p_value = float('nan')

        return {
            'observed': observed,
            'null': null_scores,
            'p_value': p_value,
            'label_type': label_type,
            'n_lakes': n_lakes,
        }


    def permanova_decomposition(self, metadata, n_permutations=1000,
                                 random_seed=42, strata=None):
        """
        Multivariable PERMANOVA with marginal (Type-III) R².

        All factors are fit JOINTLY in one design; each factor's R² is its
        PARTIAL variance — adjusted for all the other factors (the adonis2
        ``by='margin'`` analog) — so confounded factors (Year/Lake/Ecotype/
        Ancestry) share variance instead of each claiming it in full
        (McArdle & Anderson 2001).

        Permutations are restricted within ``strata`` blocks when provided
        (e.g. lake, respecting the years/sex/infection-within-lake nesting);
        unrestricted otherwise.

        A betadisper (PERMDISP2; Anderson 2006) check is reported per factor
        (``dispersion_F``, ``dispersion_p``) so centroid (location) effects
        can be distinguished from dispersion effects.

        Factors with any missing labels are dropped from that call
        (complete-case analysis) and logged — e.g. Ancestry (recipient-only)
        is assessed in the recipient-only call.

        Parameters
        ----------
        metadata : list[dict]
            One dict per embedding (in self.keys order), with keys like
            'Lake', 'Year', 'Ecotype', 'Lake Category', 'Ancestry', 'Sex'.
        n_permutations : int
        random_seed : int
        strata : dict {key: block_label} or None
            Block labels for restricted permutations (default: unrestricted).

        Returns
        -------
        dict {factor: {'r2','p_value','df','n_levels','q_value','f_stat',
                       'dispersion_F','dispersion_p'}}, sorted by R² desc.
        """
        rng = np.random.default_rng(random_seed)
        X = np.vstack([self.embeddings[k] for k in self.keys])
        n = X.shape[0]
        if n < 3:
            return {}
        if len(metadata) != n:
            # Positional pairing of metadata rows with self.keys is the whole
            # contract here; a length mismatch would silently mislabel every
            # embedding row and produce wrong PERMANOVA results. Fail loudly.
            raise ValueError(
                f"[permanova] len(metadata)={len(metadata)} does not match "
                f"n={n} embedding keys. Metadata must be built in the exact "
                f"order of self.keys (one dict per key).")

        # Euclidean distance matrix + Gower centering
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        J = np.eye(n) - np.ones((n, n)) / n
        G = -0.5 * J @ (D ** 2) @ J
        ss_total = float(np.trace(G))
        if ss_total <= 0:
            return {}

        # Blocking labels for restricted permutations (None → unrestricted)
        blocks = None
        if strata is not None:
            blocks = np.array([str(strata.get(k, k)) for k in self.keys])

        # ---- Factor labels; drop factors with any missing values ----
        factor_names = list(metadata[0].keys())
        for m in metadata[1:n]:
            for k in m.keys():
                if k not in factor_names:
                    factor_names.append(k)
        factor_labels = {}
        dropped = []
        for factor in factor_names:
            raw = [metadata[i].get(factor) for i in range(n)]
            if any(v is None or (isinstance(v, float) and np.isnan(v))
                   for v in raw):
                dropped.append(factor)
                continue
            factor_labels[factor] = np.array([str(v) for v in raw])
        modeled = list(factor_labels.keys())
        if not modeled:
            return {}
        if dropped:
            print(f"  [permanova] complete-case: dropped factors with missing "
                  f"labels: {dropped}")

        # ---- Full joint design + residual SS ----
        X_full = np.hstack([_factor_design_column(factor_labels[f])[0]
                            for f in modeled])
        # Residual df must use the RANK of the design, not the nominal column
        # count — nested/aliased factors (e.g. Ecotype within Lake) inflate the
        # column count and would understate df_resid, biasing every F.
        rank_full = np.linalg.matrix_rank(X_full)
        df_resid = n - rank_full
        if df_resid <= 0:
            raise ValueError(
                f"[permanova] Residual df = {df_resid} <= 0 (n={n} samples, "
                f"design rank={rank_full}). The design is over-determined, so "
                f"every factor would silently report F=0 / p=1. Reduce the "
                f"factor set or use more samples.")
        ss_full = _explained_ss(X_full, G)
        ss_resid = ss_total - ss_full
        if ss_resid <= 1e-12:
            print(f"  [permanova] WARNING: residual SS ≈ 0 — the joint design "
                  f"explains all variation; F statistics are degenerate.")

        results = {}
        for factor in modeled:
            others = [f for f in modeled if f != factor]
            X_red = (np.hstack([_factor_design_column(factor_labels[f])[0]
                                for f in others]) if others
                     else np.zeros((n, 0)))
            ss_red = _explained_ss(X_red, G)
            ss_factor = max(ss_full - ss_red, 0.0)   # marginal (partial) SS
            n_levels = _factor_design_column(factor_labels[factor])[1]
            df_factor = n_levels - 1
            r2 = ss_factor / ss_total if ss_total > 0 else 0.0
            if df_resid > 0 and ss_resid > 0 and df_factor > 0:
                f_obs = (ss_factor / df_factor) / (ss_resid / df_resid)
            else:
                f_obs = 0.0

            # Permutation null for THIS factor's marginal effect: permute its
            # labels (within strata blocks if given; else unrestricted), hold
            # all other factors at their observed values, recompute marginal F.
            null_f = []
            fcol = factor_labels[factor]
            # If this factor does NOT vary within any block (e.g. Lake is the
            # blocking variable), block-restricted permutation is degenerate
            # (all permuted labels equal the observed ones → null ≡ observed
            # → p = 1). Fall back to unrestricted — the factor IS the block,
            # and its test compares against a global reallocation across blocks,
            # matching adonis2 handling of the strata variable itself.
            varies_within = False
            if blocks is not None:
                for b in np.unique(blocks):
                    idx = np.where(blocks == b)[0]
                    if len(np.unique(fcol[idx])) > 1:
                        varies_within = True
                        break
            use_blocked = blocks is not None and varies_within
            for _ in range(n_permutations):
                perm = (_permute_within_blocks(fcol, blocks, rng)
                        if use_blocked else rng.permutation(fcol))
                perm_col = _factor_design_column(perm)[0]
                X_perm = (np.hstack([perm_col, X_red]) if X_red.shape[1]
                          else perm_col)
                ss_perm = _explained_ss(X_perm, G)
                ss_f_perm = max(ss_perm - ss_red, 0.0)
                if df_resid > 0 and ss_resid > 0 and df_factor > 0:
                    null_f.append((ss_f_perm / df_factor)
                                  / (ss_resid / df_resid))
            p_value = ((np.sum(np.array(null_f) >= f_obs) + 1)
                       / (len(null_f) + 1)) if null_f else 1.0

            disp_F, disp_p = _betadisper(D, factor_labels[factor],
                                         n_permutations, rng)

            results[factor] = {
                'r2': r2,
                'p_value': float(p_value),
                'df': df_factor,
                'n_levels': n_levels,
                'f_stat': float(f_obs),
                'dispersion_F': (float(disp_F) if disp_F is not None else None),
                'dispersion_p': (float(disp_p) if disp_p is not None else None),
            }

        # Multiple-testing correction: all factors in this decomposition form
        # one family (BH-FDR q_value per factor).
        _attach_qvalues(results)
        return dict(sorted(results.items(), key=lambda x: -x[1]['r2']))


# DEAD CODE (commented out): GeneNetworkAnalyzer class (never instantiated anywhere)
#class GeneNetworkAnalyzer:
#    """
#    Analyzes gene co-occurrence patterns in VQ code assignments.

#    When VQGNN assigns genes to discrete codes, genes that consistently
#    share the same VQ code across different lake/year/sex/infection
#    stratifications are likely co-regulated or functionally related.

#    The core analysis traces a gene of interest (e.g., spi1b.H, a
#    hematopoietic transcription factor) through the VQ codebook to find
#    its network neighbors.

#    Parameters
#    ----------
#    gene_to_vq : dict {stratification_key: {gene_idx: [vq_codes]}}
#        Maps each stratification to gene→VQ-code assignments.
#    vq_to_gene : dict {stratification_key: {vq_code: [gene_indices]}}
#        Maps each stratification to VQ-code→gene assignments.
#    gene_names : list[str], optional
#        Gene names indexed by gene_idx. If provided, enables gene-name lookups.
#    """

#    def __init__(self, gene_to_vq, vq_to_gene, gene_names=None):
#        self.gene_to_vq = gene_to_vq
#        self.vq_to_gene = vq_to_gene
#        self.gene_names = gene_names
#        self.stratifications = sorted(gene_to_vq.keys())

#    def _resolve_gene(self, gene):
#        """Resolve a gene identifier (name or index) to an index."""
#        if isinstance(gene, str):
#            if self.gene_names:
#                try:
#                    return self.gene_names.index(gene)
#                except ValueError:
#                    raise ValueError(f"Gene '{gene}' not found in gene_names.")
            # gene_names=None: _gene_label produced the stringified index,
            # so map it back to an int (a non-numeric string is an error).
#            if gene.isdigit():
#                return int(gene)
#            raise ValueError(
#                f"Gene '{gene}' cannot be resolved without gene_names.")
#        return gene

#    def _gene_label(self, gene_idx):
#        """Return gene name if available, else index string."""
#        if self.gene_names and gene_idx < len(self.gene_names):
#            return self.gene_names[gene_idx]
#        return str(gene_idx)

    # ------------------------------------------------------------------
    # Co-occurrence counting
    # ------------------------------------------------------------------

#    def co_occurrence_counts(self, gene, stratification_key=None):
#        """
#        Count how many times each other gene shares a VQ code with the
#        target gene, across the specified stratification(s).

#        Parameters
#        ----------
#        gene : int or str
#            Target gene index or name.
#        stratification_key : str or None
#            Specific stratification to query (e.g., 'Crystal (2021)-1').
#            If None, aggregates across all stratifications.

#        Returns
#        -------
#        counts : dict {gene_idx: int}
#            Co-occurrence counts for each gene that appears with the target.
#        """
#        gene_idx = self._resolve_gene(gene)
#        keys = [stratification_key] if stratification_key else self.stratifications
#        counts = Counter()

#        for key in keys:
#            if key not in self.gene_to_vq:
#                continue
#            if gene_idx not in self.gene_to_vq[key]:
#                continue
            # VQ codes the target gene belongs to in this stratification
#            target_codes = self.gene_to_vq[key][gene_idx]
#            for code in target_codes:
#                if code in self.vq_to_gene.get(key, {}):
#                    for neighbor in self.vq_to_gene[key][code]:
#                        if neighbor != gene_idx:
#                            counts[neighbor] += 1

#        return dict(counts)

    # ------------------------------------------------------------------
    # Ego network
    # ------------------------------------------------------------------

#    def build_ego_network(self, gene, stratification_key=None,
#                          percentile=99):
#        """
#        Build a NetworkX ego network centered on a gene.

#        Only genes with co-occurrence count above the percentile threshold
#        are included as neighbors.

#        Parameters
#        ----------
#        gene : int or str
#            Target gene.
#        stratification_key : str or None
#            Stratification to query.
#        percentile : int
#            Percentile threshold for co-occurrence count (0-100).

#        Returns
#        -------
#        G : networkx.Graph
#            Ego network (star centered on target gene).
#        """
#        gene_idx = self._resolve_gene(gene)
#        counts = self.co_occurrence_counts(gene_idx, stratification_key)

#        if not counts:
#            return nx.Graph()

        # Threshold at percentile
#        count_values = np.array(list(counts.values()))
#        threshold = np.percentile(count_values, percentile)

#        G = nx.Graph()
#        center_label = self._gene_label(gene_idx)
#        G.add_node(center_label)

#        for neighbor_idx, count in counts.items():
#            if count >= threshold:
#                neighbor_label = self._gene_label(neighbor_idx)
#                G.add_node(neighbor_label)
#                G.add_edge(center_label, neighbor_label, weight=count)

#        return G

    # ------------------------------------------------------------------
    # Expanded gene graph (second-order connections)
    # ------------------------------------------------------------------

#    def build_gene_graph(self, gene, stratification_key=None,
#                         percentile=99, min_degree=0):
#        """
#        Build an expanded gene network: start with the ego network, then
#        add edges between all ego-network neighbors that co-occur together
#        in VQ codes (second-order connections).

#        This reveals the full co-regulation module around the target gene,
#        not just direct connections.

#        Parameters
#        ----------
#        gene : int or str
#            Target gene.
#        stratification_key : str or None
#            Stratification to query.
#        percentile : int
#            Percentile threshold for co-occurrence.
#        min_degree : int
#            Remove nodes with degree less than this.

#        Returns
#        -------
#        G : networkx.Graph
#            Expanded gene co-occurrence network.
#        """
#        gene_idx = self._resolve_gene(gene)

        # Step 1: Build ego network
#        G = self.build_ego_network(gene_idx, stratification_key, percentile)
#        if G.number_of_nodes() < 2:
#            return G

#        center_label = self._gene_label(gene_idx)
#        neighbor_labels = [n for n in G.nodes() if n != center_label]
#        neighbor_indices = [
#            self._resolve_gene(n) for n in neighbor_labels
#        ]

        # Step 2: Add edges between neighbors using pre-computed
        # co-occurrence counts.  co_occurrence_counts already aggregates
        # across all stratification keys, so a simple dict lookup gives
        # the pairwise weight — no need for an inner loop over keys.
#        for i, ni in enumerate(neighbor_indices):
#            ni_label = neighbor_labels[i]
#            ni_counts = self.co_occurrence_counts(ni, stratification_key)
#            for j, nj in enumerate(neighbor_indices):
#                if j <= i:
#                    continue
#                nj_label = neighbor_labels[j]
#                weight = ni_counts.get(nj, 0)
#                if weight > 0:
#                    G.add_edge(ni_label, nj_label, weight=weight)

        # Step 3: Filter by min degree
#        if min_degree > 0:
#            low_degree = [n for n, d in G.degree() if d < min_degree]
#            G.remove_nodes_from(low_degree)

#        return G

    # ------------------------------------------------------------------
    # VQ code statistics
    # ------------------------------------------------------------------

#    def code_sizes(self, stratification_key=None):
#        """
#        Get the number of genes assigned to each VQ code.

#        Returns
#        -------
#        sizes : dict {vq_code: int}
#        """
#        keys = [stratification_key] if stratification_key else self.stratifications
#        sizes = Counter()
#        for key in keys:
#            if key in self.vq_to_gene:
#                for code, genes in self.vq_to_gene[key].items():
#                    sizes[code] += len(genes)
#        return dict(sizes)

#    def gene_code_diversity(self, stratification_key=None):
#        """
#        Get the number of distinct VQ codes each gene appears in.
#        Genes appearing in many codes span multiple network contexts.

#        Returns
#        -------
#        diversity : dict {gene_idx: int}
#        """
#        keys = [stratification_key] if stratification_key else self.stratifications
#        diversity = Counter()
#        for key in keys:
#            if key in self.gene_to_vq:
#                for gene_idx, codes in self.gene_to_vq[key].items():
#                    diversity[gene_idx] += len(set(codes))
#        return dict(diversity)

    # ------------------------------------------------------------------
    # Permutation null models (sample shuffling)
    # ------------------------------------------------------------------

#    def co_occurrence_null(self, gene, permuted_gene_to_vq_list,
#                           stratification_key=None):
#        """
#        Compute null distribution of co-occurrence counts using permuted
#        VQ assignments (generated by shuffling fish labels before graph
#        construction).

#        Parameters
#        ----------
#        gene : int or str
#            Target gene.
#        permuted_gene_to_vq_list : list[dict]
#            List of gene_to_vq dicts from each permutation run.
#        stratification_key : str or None
#            Specific stratification to query. If None, aggregates across
#            all stratifications.

#        Returns
#        -------
#        null_dist : dict {gene_idx: list[int]}
#            Null co-occurrence counts for each gene across permutations.
#        """
#        gene_idx = self._resolve_gene(gene)
#        null_dist = defaultdict(list)

#        for perm_gene_to_vq in permuted_gene_to_vq_list:
#            keys = [stratification_key] if stratification_key else sorted(perm_gene_to_vq.keys())

            # Build per-key vq_to_gene for this permutation, so the neighbour
            # lookup below is restricted to the SAME stratification (graph) as
            # the target — matching co_occurrence_counts exactly.  NOT pooled
            # across keys: pooling would let a gene co-occur with the target in
            # graphs it was never present in (merely sharing a code somewhere),
            # inflating the null and invalidating the empirical p-values.
#            perm_vq_to_gene = {}
#            for key in keys:
#                per_key = defaultdict(list)
#                for g_idx, codes in perm_gene_to_vq.get(key, {}).items():
#                    for c in codes:
#                        per_key[c].append(g_idx)
#                perm_vq_to_gene[key] = dict(per_key)

            # Compute co-occurrence with target, per key
#            counts = Counter()
#            for key in keys:
#                target_codes = perm_gene_to_vq.get(key, {}).get(gene_idx, [])
#                for code in target_codes:
#                    for neighbor in perm_vq_to_gene.get(key, {}).get(code, []):
#                        if neighbor != gene_idx:
#                            counts[neighbor] += 1

#            for neighbor, count in counts.items():
#                null_dist[neighbor].append(count)

        # Pad permutations in which a neighbor was ABSENT (co-occurrence 0)
        # so empirical_pvalues' denominator is the TRUE number of
        # permutations, not just the count of permutations in which the
        # neighbor happened to appear.  Without this, genes that co-occur
        # only occasionally got artificially small denominators.
#        n_permutations = len(permuted_gene_to_vq_list)
#        for neighbor in list(null_dist.keys()):
#            missing = n_permutations - len(null_dist[neighbor])
#            if missing > 0:
#                null_dist[neighbor].extend([0] * missing)

#        return dict(null_dist)

#    def empirical_pvalues(self, gene, observed_counts, null_dist):
#        """
#        Compute empirical p-values by comparing observed co-occurrence
#        counts against a null distribution from permutations.

#        p = (n_null >= observed + 1) / (n_permutations + 1)

#        Parameters
#        ----------
#        gene : int or str
#            Target gene.
#        observed_counts : dict {gene_idx: int}
#            Observed co-occurrence counts from real data.
#        null_dist : dict {gene_idx: list[int]}
#            Null co-occurrence counts from permutations.

#        Returns
#        -------
#        pvalues : dict {gene_idx: float}
#            Empirical p-value for each gene that appears in observed_counts.
#        """
#        gene_idx = self._resolve_gene(gene)
#        pvalues = {}

#        for neighbor, obs_count in observed_counts.items():
#            null_counts = null_dist.get(neighbor, [])
#            n_perm = len(null_counts)
#            if n_perm == 0:
#                pvalues[neighbor] = 1.0
#                continue
#            p = (np.sum(np.array(null_counts) >= obs_count) + 1) / (n_perm + 1)
#            pvalues[neighbor] = float(p)

#        return pvalues

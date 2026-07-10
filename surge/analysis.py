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

    def __init__(self, embeddings, data=None):
        self.embeddings = embeddings
        self.keys = sorted(embeddings.keys())
        self.data = data

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
        return LakeAnalyzer(filtered, data=self.data)

    def for_subset(self, keys):
        """Return a new LakeAnalyzer containing only the given keys."""
        filtered = {k: self.embeddings[k] for k in keys if k in self.embeddings}
        return LakeAnalyzer(filtered, data=self.data)

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

    def wasserstein_temporal(self, base_year=2019):
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

        Returns
        -------
        distances : dict {lake: [(year, distance), ...]}
            Sorted by year.
        """
        lake_years = defaultdict(dict)
        for key, hist in self.embeddings.items():
            if ' (' not in key:
                continue
            lake, rest = key.split(' (', 1)
            year_str = rest.split(')')[0].split('-')[0]
            try:
                year = int(float(year_str))
            except ValueError:
                continue
            lake_years[lake][year] = hist

        result = {}
        for lake, year_hists in lake_years.items():
            if base_year in year_hists:
                base_yr = base_year
            else:
                base_yr = min(year_hists)
            base_hist = year_hists[base_yr]
            dists = [(yr, wasserstein_distance(base_hist, hist))
                     for yr, hist in sorted(year_hists.items())
                     if yr != base_yr]
            if dists:
                result[lake] = dists

        return result

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
        for lake, year_dists in wasserstein_distances.items():
            if len(year_dists) < 2:
                slopes[lake] = 0.0
                continue
            years = np.array([y for y, _ in year_dists])
            dists = np.array([d for _, d in year_dists])
            A = np.vstack([years, np.ones_like(years)]).T
            slope, _ = np.linalg.lstsq(A, dists, rcond=None)[0]
            slopes[lake] = float(slope)
        return slopes

    def source_vs_recipient_slope_test(self, wasserstein_distances):
        """
        Compute temporal slopes and test whether Source and Recipient
        lakes differ in their rate of network divergence.

        Runs both a t-test (parametric) and Mann-Whitney U (non-parametric).
        Requires at least 3 lakes per group for meaningful statistics.

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

            # --- Role × Ecotype (Source Benthic, Source Limnetic,
            #     Recipient Benthic, Recipient Limnetic) ---
            if self.data:
                combos = []
                for k in keys:
                    lake = k.split(' (')[0] if ' (' in k else k
                    role = self.data.get_lake_role(lake)
                    eco = self.data.get_lake_ecotype(lake)
                    combos.append(f'{role} {eco}')
                combo_counts = Counter(combos)
                info['role_ecotype_counts'] = dict(combo_counts)
                # Background
                all_combos = []
                for k in self.keys:
                    lake = k.split(' (')[0] if ' (' in k else k
                    role = self.data.get_lake_role(lake)
                    eco = self.data.get_lake_ecotype(lake)
                    all_combos.append(f'{role} {eco}')
                bg_combos = Counter(all_combos)
                info['role_ecotype_enrichment'] = {}
                for combo in ['Source Benthic', 'Source Limnetic',
                              'Recipient Benthic', 'Recipient Limnetic']:
                    in_cluster = combo_counts.get(combo, 0)
                    in_other = bg_combos.get(combo, 0) - in_cluster
                    not_in_cluster = n - in_cluster
                    not_in_other = len(self.keys) - n - in_other
                    if in_cluster > 0 and in_other > 0:
                        _, p = fisher_exact([[in_cluster, not_in_cluster],
                                             [in_other, not_in_other]])
                        info['role_ecotype_enrichment'][combo] = {
                            'count': in_cluster, 'pct': in_cluster / n,
                            'fisher_p': float(p),
                        }

                # --- Marginal role (Source / Recipient) ---
                roles = []
                for k in keys:
                    lake = k.split(' (')[0] if ' (' in k else k
                    roles.append(self.data.get_lake_role(lake))
                role_counts = Counter(roles)
                info['role_counts'] = dict(role_counts)
                all_roles_bg = []
                for k in self.keys:
                    lake = k.split(' (')[0] if ' (' in k else k
                    all_roles_bg.append(self.data.get_lake_role(lake))
                bg_roles = Counter(all_roles_bg)
                info['role_enrichment'] = {}
                for role in ['Source', 'Recipient']:
                    in_cluster = role_counts.get(role, 0)
                    in_other = bg_roles.get(role, 0) - in_cluster
                    not_in_cluster = n - in_cluster
                    not_in_other = len(self.keys) - n - in_other
                    if in_cluster > 0 and in_other > 0:
                        _, p = fisher_exact([[in_cluster, not_in_cluster],
                                             [in_other, not_in_other]])
                        info['role_enrichment'][role] = {
                            'count': in_cluster, 'pct': in_cluster / n,
                            'fisher_p': float(p),
                        }

                # --- Marginal ecotype (Benthic / Limnetic) ---
                ecotypes = []
                for k in keys:
                    lake = k.split(' (')[0] if ' (' in k else k
                    ecotypes.append(self.data.get_lake_ecotype(lake))
                eco_counts = Counter(ecotypes)
                info['ecotype_counts'] = dict(eco_counts)
                all_eco_bg = []
                for k in self.keys:
                    lake = k.split(' (')[0] if ' (' in k else k
                    all_eco_bg.append(self.data.get_lake_ecotype(lake))
                bg_eco = Counter(all_eco_bg)
                info['ecotype_enrichment'] = {}
                for eco in ['Benthic', 'Limnetic']:
                    in_cluster = eco_counts.get(eco, 0)
                    in_other = bg_eco.get(eco, 0) - in_cluster
                    not_in_cluster = n - in_cluster
                    not_in_other = len(self.keys) - n - in_other
                    if in_cluster > 0 and in_other > 0:
                        _, p = fisher_exact([[in_cluster, not_in_cluster],
                                             [in_other, not_in_other]])
                        info['ecotype_enrichment'][eco] = {
                            'count': in_cluster, 'pct': in_cluster / n,
                            'fisher_p': float(p),
                        }

                # --- Sex (Male / Female) — only when keys carry sex suffixes ---
                sexes = []
                for k in keys:
                    if str(k).endswith('-f'):
                        sexes.append('Female')
                    elif str(k).endswith('-m'):
                        sexes.append('Male')
                if sexes:
                    info['sex_counts'] = dict(Counter(sexes))
                    all_sex = []
                    for k in self.keys:
                        if str(k).endswith('-f'):
                            all_sex.append('Female')
                        elif str(k).endswith('-m'):
                            all_sex.append('Male')
                    bg_sex = Counter(all_sex)
                    info['sex_enrichment'] = {}
                    for s in ['Male', 'Female']:
                        in_cluster = sexes.count(s)
                        in_other = bg_sex.get(s, 0) - in_cluster
                        not_in_cluster = n - in_cluster
                        not_in_other = len(self.keys) - n - in_other
                        if in_cluster > 0 and in_other > 0:
                            _, p = fisher_exact([[in_cluster, not_in_cluster],
                                                 [in_other, not_in_other]])
                            info['sex_enrichment'][s] = {
                                'count': in_cluster, 'pct': in_cluster / n,
                                'fisher_p': float(p),
                            }

                # --- Infection (0=non-infected / 1=infected) — only when
                #     keys carry infection suffixes ---
                infs = []
                for k in keys:
                    m = re.search(r'\)-([01])$', str(k))
                    if m:
                        infs.append('Infected' if int(m.group(1)) == 1
                                    else 'Non-infected')
                if infs:
                    info['infection_counts'] = dict(Counter(infs))
                    all_inf = []
                    for k in self.keys:
                        m = re.search(r'\)-([01])$', str(k))
                        if m:
                            all_inf.append('Infected' if int(m.group(1)) == 1
                                           else 'Non-infected')
                    bg_inf = Counter(all_inf)
                    info['infection_enrichment'] = {}
                    for lab in ['Infected', 'Non-infected']:
                        in_cluster = infs.count(lab)
                        in_other = bg_inf.get(lab, 0) - in_cluster
                        not_in_cluster = n - in_cluster
                        not_in_other = len(self.keys) - n - in_other
                        if in_cluster > 0 and in_other > 0:
                            _, p = fisher_exact([[in_cluster, not_in_cluster],
                                                 [in_other, not_in_other]])
                            info['infection_enrichment'][lab] = {
                                'count': in_cluster, 'pct': in_cluster / n,
                                'fisher_p': float(p),
                            }

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
            Sorted by significance (smallest p first).
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

            # Mann-Whitney U test (non-parametric, handles zero-inflated data)
            try:
                from scipy.stats import mannwhitneyu
                u_stat, p = mannwhitneyu(inf_usage, ninf_usage,
                                         alternative='two-sided')
            except Exception:
                p = 1.0

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

        Uses sex_year_lake keys only.  Returns dict sorted by p-value.
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
            try:
                from scipy.stats import mannwhitneyu
                _, p = mannwhitneyu(m_usage, f_usage, alternative='two-sided')
            except Exception:
                p = 1.0
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
        if self.data is not None:
            try:
                return self.data.get_lake_role(lake)
            except Exception:
                pass
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
            try:
                from scipy.stats import mannwhitneyu
                _, p = mannwhitneyu(s_usage, r_usage, alternative='two-sided')
            except Exception:
                p = 1.0
            results[code] = {
                'source_mean': s_mean, 'recipient_mean': r_mean,
                'fold_change': fc, 'p_value': float(p),
            }
        # Multiple-testing correction: one family per codebook across all codes.
        return _attach_qvalues(results)

    # ------------------------------------------------------------------
    # Permutation tests
    # ------------------------------------------------------------------

    def source_recipient_permutation_test(self, n_permutations=1000,
                                          random_seed=42):
        """
        Permutation test: is the silhouette score between Source and
        Recipient lakes larger than expected by random label assignment?

        Shuffles Source/Recipient labels to build a null distribution.
        Does NOT require rebuilding graphs — only shuffles labels.

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
        observed = self.silhouette(true_labels)

        # Collect keys that have Source/Recipient labels
        labeled_keys = [k for k in self.keys if k in true_labels]
        label_values = [true_labels[k] for k in labeled_keys]

        null_scores = []
        for _ in tqdm(range(n_permutations)):
            permuted_vals = rng.permutation(label_values)
            perm_labels = dict(zip(labeled_keys, permuted_vals))
            score = self.silhouette(perm_labels)
            if not np.isnan(score):
                null_scores.append(score)

        p_value = (np.sum(np.array(null_scores) >= observed) + 1) / (len(null_scores) + 1)

        return {
            'observed': observed,
            'null': null_scores,
            'p_value': p_value,
            'label_type': 'source_recipient',
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
        observed = self.silhouette(true_labels)

        labeled_keys = [k for k in self.keys if k in true_labels]
        label_values = [true_labels[k] for k in labeled_keys]

        null_scores = []
        for _ in tqdm(range(n_permutations),
                      desc=f"  Permuting {label_type}"):
            permuted_vals = rng.permutation(label_values)
            perm_labels = dict(zip(labeled_keys, permuted_vals))
            score = self.silhouette(perm_labels)
            if not np.isnan(score):
                null_scores.append(score)

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
            'n_labeled': len(labeled_keys),
        }


    def permanova_decomposition(self, metadata, n_permutations=1000,
                                 random_seed=42):
        """
        Marginal PERMANOVA: for each categorical factor, compute the
        fraction of variance in the VQ histogram distance matrix that
        the factor alone explains (marginal R²), with a permutation-based
        p-value.

        Parameters
        ----------
        metadata : list[dict]
            One dict per embedding, with keys like 'role', 'ecotype',
            'year', 'sex', 'infection', 'lake'.
        n_permutations : int
        random_seed : int

        Returns
        -------
        dict : {factor_name: {'r2': float, 'p_value': float, 'df': int}}
            Sorted by R² descending.
        """
        rng = np.random.default_rng(random_seed)
        X = np.vstack([self.embeddings[k] for k in self.keys])
        n = X.shape[0]
        if n < 3:
            return {}

        # Euclidean distance matrix + Gower centering
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        J = np.eye(n) - np.ones((n, n)) / n
        G = -0.5 * J @ (D ** 2) @ J
        ss_total = float(np.trace(G))
        if ss_total <= 0:
            return {}

        # Include a factor if it appears in ANY sample. The per-factor
        # valid_idx filter below drops rows that are missing/NaN for that
        # factor. (Previously a factor missing from even one row — e.g.
        #  Ancestry absent for source lakes — was dropped from the entire
        #  decomposition, even though it was present for most rows.)
        factor_names = list(metadata[0].keys())
        for m in metadata[1:]:
            for k in m.keys():
                if k not in factor_names:
                    factor_names.append(k)
        results = {}
        for factor in factor_names:
            raw_labels = [m.get(factor) for m in metadata]
            # Drop NaN / None labels (can arise from missing metadata)
            valid_idx = [i for i, v in enumerate(raw_labels)
                         if v is not None
                         and (not isinstance(v, float) or not np.isnan(v))]
            if len(valid_idx) < 3:
                continue
            labels = np.array([raw_labels[i] for i in valid_idx])
            # Subset distance submatrix to valid rows
            G_sub = G[np.ix_(valid_idx, valid_idx)]
            ss_total_sub = float(np.trace(G_sub))
            if ss_total_sub <= 0:
                continue
            unique = sorted(set(labels))
            n_levels = len(unique)
            if n_levels < 2:
                continue
            n_sub = len(valid_idx)
            df_factor = n_levels - 1

            # One-hot design matrix (valid rows only)
            X_design = np.zeros((n_sub, n_levels))
            for i, lab in enumerate(labels):
                X_design[i, unique.index(lab)] = 1.0

            # Hat matrix H = X (X'X)^-1 X'
            XtX = X_design.T @ X_design
            try:
                XtX_inv = np.linalg.pinv(XtX)
            except np.linalg.LinAlgError:
                continue
            H = X_design @ XtX_inv @ X_design.T
            ss_factor = float(np.trace(H @ G_sub @ H))
            r2 = ss_factor / ss_total_sub if ss_total_sub > 0 else 0.0

            # Pseudo-F
            ss_resid = ss_total_sub - ss_factor
            df_resid = n_sub - n_levels
            if df_resid > 0 and ss_resid > 0 and df_factor > 0:
                f_obs = (ss_factor / df_factor) / (ss_resid / df_resid)
            else:
                f_obs = 0.0

            # Permutation null — restricted to the SAME valid subset as the
            # observed statistic, so observed and permuted F are exchangeable.
            # (Previously the null used the full n-row G + ss_total and wrote
            #  the n_sub permuted labels into rows 0..n_sub-1 of an n-row
            #  design, landing them in the wrong rows whenever any row was
            #  dropped — making the permutation test invalid.)
            null_f = []
            for _ in range(n_permutations):
                perm_labels = rng.permutation(labels)
                X_perm = np.zeros((n_sub, n_levels))
                for i, lab in enumerate(perm_labels):
                    X_perm[i, unique.index(lab)] = 1.0
                XtX_p = X_perm.T @ X_perm
                try:
                    XtX_p_inv = np.linalg.pinv(XtX_p)
                except np.linalg.LinAlgError:
                    continue
                H_p = X_perm @ XtX_p_inv @ X_perm.T
                ss_p = float(np.trace(H_p @ G_sub @ H_p))
                ss_r = ss_total_sub - ss_p
                if df_resid > 0 and ss_r > 0:
                    null_f.append((ss_p / df_factor) / (ss_r / df_resid))

            if null_f:
                p_value = (np.sum(np.array(null_f) >= f_obs) + 1) / (len(null_f) + 1)
            else:
                p_value = 1.0

            results[factor] = {
                'r2': r2,
                'p_value': float(p_value),
                'df': df_factor,
                'n_levels': n_levels,
            }

        # Multiple-testing correction: all factors tested in this
        # decomposition form one family (BH-FDR q_value per factor).
        _attach_qvalues(results)

        # Sort by R² descending
        return dict(sorted(results.items(), key=lambda x: -x[1]['r2']))


class GeneNetworkAnalyzer:
    """
    Analyzes gene co-occurrence patterns in VQ code assignments.

    When VQGNN assigns genes to discrete codes, genes that consistently
    share the same VQ code across different lake/year/sex/infection
    stratifications are likely co-regulated or functionally related.

    The core analysis traces a gene of interest (e.g., spi1b.H, a
    hematopoietic transcription factor) through the VQ codebook to find
    its network neighbors.

    Parameters
    ----------
    gene_to_vq : dict {stratification_key: {gene_idx: [vq_codes]}}
        Maps each stratification to gene→VQ-code assignments.
    vq_to_gene : dict {stratification_key: {vq_code: [gene_indices]}}
        Maps each stratification to VQ-code→gene assignments.
    gene_names : list[str], optional
        Gene names indexed by gene_idx. If provided, enables gene-name lookups.
    """

    def __init__(self, gene_to_vq, vq_to_gene, gene_names=None):
        self.gene_to_vq = gene_to_vq
        self.vq_to_gene = vq_to_gene
        self.gene_names = gene_names
        self.stratifications = sorted(gene_to_vq.keys())

    def _resolve_gene(self, gene):
        """Resolve a gene identifier (name or index) to an index."""
        if isinstance(gene, str) and self.gene_names:
            try:
                return self.gene_names.index(gene)
            except ValueError:
                raise ValueError(f"Gene '{gene}' not found in gene_names.")
        return gene

    def _gene_label(self, gene_idx):
        """Return gene name if available, else index string."""
        if self.gene_names and gene_idx < len(self.gene_names):
            return self.gene_names[gene_idx]
        return str(gene_idx)

    # ------------------------------------------------------------------
    # Co-occurrence counting
    # ------------------------------------------------------------------

    def co_occurrence_counts(self, gene, stratification_key=None):
        """
        Count how many times each other gene shares a VQ code with the
        target gene, across the specified stratification(s).

        Parameters
        ----------
        gene : int or str
            Target gene index or name.
        stratification_key : str or None
            Specific stratification to query (e.g., 'Crystal (2021)-1').
            If None, aggregates across all stratifications.

        Returns
        -------
        counts : dict {gene_idx: int}
            Co-occurrence counts for each gene that appears with the target.
        """
        gene_idx = self._resolve_gene(gene)
        keys = [stratification_key] if stratification_key else self.stratifications
        counts = Counter()

        for key in keys:
            if key not in self.gene_to_vq:
                continue
            if gene_idx not in self.gene_to_vq[key]:
                continue
            # VQ codes the target gene belongs to in this stratification
            target_codes = self.gene_to_vq[key][gene_idx]
            for code in target_codes:
                if code in self.vq_to_gene.get(key, {}):
                    for neighbor in self.vq_to_gene[key][code]:
                        if neighbor != gene_idx:
                            counts[neighbor] += 1

        return dict(counts)

    # ------------------------------------------------------------------
    # Ego network
    # ------------------------------------------------------------------

    def build_ego_network(self, gene, stratification_key=None,
                          percentile=99):
        """
        Build a NetworkX ego network centered on a gene.

        Only genes with co-occurrence count above the percentile threshold
        are included as neighbors.

        Parameters
        ----------
        gene : int or str
            Target gene.
        stratification_key : str or None
            Stratification to query.
        percentile : int
            Percentile threshold for co-occurrence count (0-100).

        Returns
        -------
        G : networkx.Graph
            Ego network (star centered on target gene).
        """
        gene_idx = self._resolve_gene(gene)
        counts = self.co_occurrence_counts(gene_idx, stratification_key)

        if not counts:
            return nx.Graph()

        # Threshold at percentile
        count_values = np.array(list(counts.values()))
        threshold = np.percentile(count_values, percentile)

        G = nx.Graph()
        center_label = self._gene_label(gene_idx)
        G.add_node(center_label)

        for neighbor_idx, count in counts.items():
            if count >= threshold:
                neighbor_label = self._gene_label(neighbor_idx)
                G.add_node(neighbor_label)
                G.add_edge(center_label, neighbor_label, weight=count)

        return G

    # ------------------------------------------------------------------
    # Expanded gene graph (second-order connections)
    # ------------------------------------------------------------------

    def build_gene_graph(self, gene, stratification_key=None,
                         percentile=99, min_degree=0):
        """
        Build an expanded gene network: start with the ego network, then
        add edges between all ego-network neighbors that co-occur together
        in VQ codes (second-order connections).

        This reveals the full co-regulation module around the target gene,
        not just direct connections.

        Parameters
        ----------
        gene : int or str
            Target gene.
        stratification_key : str or None
            Stratification to query.
        percentile : int
            Percentile threshold for co-occurrence.
        min_degree : int
            Remove nodes with degree less than this.

        Returns
        -------
        G : networkx.Graph
            Expanded gene co-occurrence network.
        """
        gene_idx = self._resolve_gene(gene)

        # Step 1: Build ego network
        G = self.build_ego_network(gene_idx, stratification_key, percentile)
        if G.number_of_nodes() < 2:
            return G

        center_label = self._gene_label(gene_idx)
        neighbor_labels = [n for n in G.nodes() if n != center_label]
        neighbor_indices = [
            self._resolve_gene(n) for n in neighbor_labels
        ]

        # Step 2: Add edges between neighbors using pre-computed
        # co-occurrence counts.  co_occurrence_counts already aggregates
        # across all stratification keys, so a simple dict lookup gives
        # the pairwise weight — no need for an inner loop over keys.
        for i, ni in enumerate(neighbor_indices):
            ni_label = neighbor_labels[i]
            ni_counts = self.co_occurrence_counts(ni, stratification_key)
            for j, nj in enumerate(neighbor_indices):
                if j <= i:
                    continue
                nj_label = neighbor_labels[j]
                weight = ni_counts.get(nj, 0)
                if weight > 0:
                    G.add_edge(ni_label, nj_label, weight=weight)

        # Step 3: Filter by min degree
        if min_degree > 0:
            low_degree = [n for n, d in G.degree() if d < min_degree]
            G.remove_nodes_from(low_degree)

        return G

    # ------------------------------------------------------------------
    # VQ code statistics
    # ------------------------------------------------------------------

    def code_sizes(self, stratification_key=None):
        """
        Get the number of genes assigned to each VQ code.

        Returns
        -------
        sizes : dict {vq_code: int}
        """
        keys = [stratification_key] if stratification_key else self.stratifications
        sizes = Counter()
        for key in keys:
            if key in self.vq_to_gene:
                for code, genes in self.vq_to_gene[key].items():
                    sizes[code] += len(genes)
        return dict(sizes)

    def gene_code_diversity(self, stratification_key=None):
        """
        Get the number of distinct VQ codes each gene appears in.
        Genes appearing in many codes span multiple network contexts.

        Returns
        -------
        diversity : dict {gene_idx: int}
        """
        keys = [stratification_key] if stratification_key else self.stratifications
        diversity = Counter()
        for key in keys:
            if key in self.gene_to_vq:
                for gene_idx, codes in self.gene_to_vq[key].items():
                    diversity[gene_idx] += len(set(codes))
        return dict(diversity)

    # ------------------------------------------------------------------
    # Permutation null models (sample shuffling)
    # ------------------------------------------------------------------

    def co_occurrence_null(self, gene, permuted_gene_to_vq_list,
                           stratification_key=None):
        """
        Compute null distribution of co-occurrence counts using permuted
        VQ assignments (generated by shuffling fish labels before graph
        construction).

        Parameters
        ----------
        gene : int or str
            Target gene.
        permuted_gene_to_vq_list : list[dict]
            List of gene_to_vq dicts from each permutation run.
        stratification_key : str or None
            Specific stratification to query. If None, aggregates across
            all stratifications.

        Returns
        -------
        null_dist : dict {gene_idx: list[int]}
            Null co-occurrence counts for each gene across permutations.
        """
        gene_idx = self._resolve_gene(gene)
        null_dist = defaultdict(list)

        for perm_gene_to_vq in permuted_gene_to_vq_list:
            keys = [stratification_key] if stratification_key else sorted(perm_gene_to_vq.keys())

            # Build vq_to_gene for this permutation across all relevant keys
            perm_vq_to_gene = defaultdict(list)
            for key in keys:
                for g_idx, codes in perm_gene_to_vq.get(key, {}).items():
                    for c in codes:
                        perm_vq_to_gene[c].append(g_idx)

            # Compute co-occurrence with target
            counts = Counter()
            for key in keys:
                target_codes = perm_gene_to_vq.get(key, {}).get(gene_idx, [])
                for code in target_codes:
                    for neighbor in perm_vq_to_gene.get(code, []):
                        if neighbor != gene_idx:
                            counts[neighbor] += 1

            for neighbor, count in counts.items():
                null_dist[neighbor].append(count)

        return dict(null_dist)

    def empirical_pvalues(self, gene, observed_counts, null_dist):
        """
        Compute empirical p-values by comparing observed co-occurrence
        counts against a null distribution from permutations.

        p = (n_null >= observed + 1) / (n_permutations + 1)

        Parameters
        ----------
        gene : int or str
            Target gene.
        observed_counts : dict {gene_idx: int}
            Observed co-occurrence counts from real data.
        null_dist : dict {gene_idx: list[int]}
            Null co-occurrence counts from permutations.

        Returns
        -------
        pvalues : dict {gene_idx: float}
            Empirical p-value for each gene that appears in observed_counts.
        """
        gene_idx = self._resolve_gene(gene)
        pvalues = {}

        for neighbor, obs_count in observed_counts.items():
            null_counts = null_dist.get(neighbor, [])
            n_perm = len(null_counts)
            if n_perm == 0:
                pvalues[neighbor] = 1.0
                continue
            p = (np.sum(np.array(null_counts) >= obs_count) + 1) / (n_perm + 1)
            pvalues[neighbor] = float(p)

        return pvalues

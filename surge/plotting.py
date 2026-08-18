"""
Visualization utilities for lake embeddings and gene networks.

All plotting functions operate on already-computed data — they don't
do analysis, just rendering.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def _clean_year(s):
    """Strip '.0' from 4-digit years in labels.

    'Finger Lake (2019.0)' -> 'Finger Lake (2019)'
    '2023.0' -> '2023'
    """
    return re.sub(r'(\d{4})\.0', r'\1', str(s))


class LakePlotter:
    """
    Static methods for plotting lake analysis results.
    """

    # ------------------------------------------------------------------
    # PCA scatter
    # ------------------------------------------------------------------

    @staticmethod
    def pca_scatter(projected, keys, data, title="PCA of Lake Embeddings",
                    color_by='genotype', ax=None, label_points=True):
        """
        2D scatter plot of PCA-projected embeddings.

        Parameters
        ----------
        projected : np.ndarray (n_points, 2)
            PCA coordinates (already min-max scaled).
        keys : list[str]
            Labels for each point.
        data : SticklebackData
            For color/lake lookups.
        color_by : str
            'genotype' or 'lake' — how to color points.
        ax : matplotlib Axes, optional
        label_points : bool
            Whether to annotate points with lake names.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 10))

        for i, key in enumerate(keys):
            lake = key.split(' (')[0] if ' (' in key else key

            if color_by == 'genotype':
                color = data.get_genotype_color(lake)
            else:
                color = data.get_lake_color(lake)

            ax.scatter(projected[i, 0], projected[i, 1],
                       c=color, s=60, edgecolors='black', linewidth=0.5,
                       zorder=5)

            if label_points:
                label = _clean_year(key)
                ax.annotate(label, (projected[i, 0], projected[i, 1]),
                           fontsize=7, ha='center', va='bottom',
                           xytext=(0, 5), textcoords='offset points')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title)

        # Legend for genotype colors
        if color_by == 'genotype':
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='BenthicPool',
                       markerfacecolor=data.GENOTYPE_COLORS['BenthicPool'],
                       markersize=10),
                Line2D([0], [0], marker='o', color='w', label='LimneticPool',
                       markerfacecolor=data.GENOTYPE_COLORS['LimneticPool'],
                       markersize=10),
                Line2D([0], [0], marker='o', color='w', label='MixedPool',
                       markerfacecolor=data.GENOTYPE_COLORS['MixedPool'],
                       markersize=10),
            ]
            ax.legend(handles=legend_elements, loc='best')

        return ax

    # ------------------------------------------------------------------
    # Wasserstein temporal grid
    # ------------------------------------------------------------------

    # Colour map for lake role × LAKE HABITAT (the physical environment the
    # fish live in — what shapes the co-expression network).  Habitat is
    # distinct from ecotype (ancestry): a recipient lake can be Limnetic-
    # ancestry but Benthic-habitat (e.g. Fred/Ranchero), which the old
    # (role, ecotype) colouring could not express.  Source lakes fall back to
    # habitat == ecotype.
    _ROLE_HABITAT_COLORS = {
        ('Source',    'Benthic'):            '#e41a1c',  # red
        ('Source',    'Limnetic'):           '#377eb8',  # blue
        ('Recipient', 'Benthic'):            '#ff7f00',  # orange
        ('Recipient', 'Limnetic'):           '#984ea3',  # purple
        ('Source',    'Unknown'):            '#2ca02c',
        ('Recipient', 'Unknown'):            '#2ca02c',
    }
    _ROLE_HABITAT_COLORS_DEFAULT = '#999999'   # grey for Other/Unknown

    @staticmethod
    def wasserstein_grid(wasserstein_distances, ncols=4, figsize=(16, 12),
                         ymax=None, data=None, ylabel='Wasserstein Distance',
                         suptitle=None):
        """
        Grid of line charts showing Wasserstein distance over time for
        each lake.

        Parameters
        ----------
        wasserstein_distances : dict {lake: [(year, distance), ...]}
        ncols : int
            Number of columns in the grid.
        figsize : tuple
        ymax : float or None
            Uniform y-axis maximum for comparison across lakes.
            If None, uses the 95th percentile of all distances.
        data : SticklebackData or None
            If provided, subplots are coloured by lake role × ecotype.
        ylabel : str
            Y-axis label.
        suptitle : str or None
            Optional figure-level title (e.g. for p-values).
        """
        lakes = sorted(wasserstein_distances.keys())
        n_lakes = len(lakes)
        if n_lakes == 0:
            # The loop below never binds `i`, and `plt.subplots(0, ncols)` is
            # degenerate — return an empty figure loudly instead of NameError.
            print("  [wasserstein_grid] no lakes to plot — returning empty "
                  "figure.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No temporal data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.axis('off')
            return fig
        nrows = int(np.ceil(n_lakes / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])

        # Auto-compute ymax from data if not provided
        if ymax is None:
            all_dists = [d for lake in lakes
                         for _, d in wasserstein_distances[lake]]
            if all_dists:
                ymax = float(np.max(all_dists)) * 1.1

        # Track which (role, lake habitat) combos actually appear.  Habitat is
        # the physical environment (== ecotype for source lakes; distinct for
        # some recipients).
        seen_combos = set()

        for i, lake in enumerate(lakes):
            ax = axes[i]

            # Determine colour from lake role × lake habitat
            color = LakePlotter._ROLE_HABITAT_COLORS_DEFAULT
            label_suffix = ''
            if data is not None:
                role = data.get_lake_role(lake)
                habitat = data.get_lake_habitat(lake)
                seen_combos.add((role, habitat))
                color = LakePlotter._ROLE_HABITAT_COLORS.get(
                    (role, habitat), LakePlotter._ROLE_HABITAT_COLORS_DEFAULT)
                label_suffix = f' [{role} {habitat}]'

            years = [int(y) for y, _ in wasserstein_distances[lake]]
            dists = [d for _, d in wasserstein_distances[lake]]

            ax.plot(years, dists, 'o-', color=color, linewidth=2,
                   markersize=6)
            ax.set_title(_clean_year(lake) + label_suffix, fontsize=9, color=color)
            ax.set_xlabel('Year')
            ax.set_xticks(years)
            ax.set_xticklabels([str(y) for y in years])
            ax.set_ylabel(ylabel)
            if ymax is not None:
                ax.set_ylim(0, ymax)
            ax.grid(True, alpha=0.3)

            # Subtle background tint
            ax.set_facecolor(color + '15')  # 15 = ~8% opacity hex alpha

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Legend — only show combos that actually appear in the data
        if data is not None:
            legend_elements = []
            for (role, hab), col in LakePlotter._ROLE_HABITAT_COLORS.items():
                if (role, hab) in seen_combos:
                    legend_elements.append(
                        Line2D([0], [0], color=col, linewidth=2, marker='o',
                               label=f'{role} {hab}')
                    )
            fig.legend(handles=legend_elements, loc='lower center',
                       fontsize=11, ncol=5)

        if suptitle:
            fig.suptitle(suptitle, fontsize=11, y=0.98)

        # Reserve room: bottom 8% for legend, top 6% for suptitle
        top = 0.94 if suptitle else 1.0
        bottom = 0.08 if data is not None else 0.0
        plt.tight_layout(rect=[0, bottom, 1, top])
        return fig

    # ------------------------------------------------------------------
    # Gene network
    # ------------------------------------------------------------------

    @staticmethod
# DEAD CODE (commented out): LakePlotter.gene_network (never called)
#    def gene_network(G, central_gene, ax=None, figsize=(10, 10),
#                     max_edges=500, max_nodes=None, label_offset=0.12,
#                     font_size=6):
#        """
#        Draw a gene co-occurrence network with circular layout.

#        The central gene is highlighted in red; all other genes are blue.
#        Node sizes are proportional to degree.  Labels are placed radially
#        outside nodes and rotated to follow the circle.

#        All edges incident to *central_gene* are always kept; *max_edges*
#        only limits edges between other nodes.  Isolated nodes (degree 0)
#        are dropped before rendering.

#        Parameters
#        ----------
#        G : networkx.Graph
#            Gene co-occurrence network.
#        central_gene : str
#            Name of the central gene to highlight.
#        ax : matplotlib Axes, optional
#        figsize : tuple
#        max_edges : int or None
#            Maximum edges to draw.  Central-gene edges are always kept;
#            only edges between non-central nodes count toward this limit.
#            Set to None to draw all edges.
#        max_nodes : int or None
#            Maximum number of nodes to show.  Keeps the central gene plus
#            the top *max_nodes - 1* nodes by degree.  Set to None to show
#            all nodes.
#        label_offset : float
#            Radial distance beyond the layout circle for labels.
#        font_size : int
#            Font size for gene labels.
#        """
#        if ax is None:
#            _, ax = plt.subplots(figsize=figsize)

        # ---- edge filtering: preserve central-gene edges ----
#        if max_edges is not None and G.number_of_edges() > max_edges:
#            central_edges = []
#            other_edges = []
#            for u, v, d in G.edges(data=True):
#                w = d.get('weight', 1)
#                if u == central_gene or v == central_gene:
#                    central_edges.append((u, v, w))
#                else:
#                    other_edges.append((u, v, w))
            # Keep all central-gene edges; limit only the other edges
#            other_edges.sort(key=lambda x: -x[2])
#            kept_other = other_edges[:max_edges]
#            G_sub = nx.Graph()
#            G_sub.add_nodes_from(G.nodes(data=True))
#            G_sub.add_weighted_edges_from(central_edges + kept_other)
#            G = G_sub

        # ---- drop nodes orphaned by edge filtering ----
#        isolates = [n for n, d in G.degree() if d == 0 and n != central_gene]
#        if isolates:
#            G = G.copy()
#            G.remove_nodes_from(isolates)

        # ---- limit node count to max_nodes (by degree) ----
#        if max_nodes is not None and G.number_of_nodes() > max_nodes:
#            degrees = dict(G.degree())
            # Sort non-central nodes by degree descending, keep top max_nodes-1
#            other_nodes = sorted(
#                [n for n in G.nodes() if n != central_gene],
#                key=lambda n: -degrees.get(n, 0),
#            )
#            keep = set(other_nodes[:max_nodes - 1]) | {central_gene}
#            G = G.subgraph(keep).copy()

#        pos = nx.circular_layout(G)
#        degrees = dict(G.degree())

        # Node colors: central gene in red, others in cerulean
#        node_colors = [
#            '#e41a1c' if n == central_gene else '#2b7bba'
#            for n in G.nodes()
#        ]
        # Node sizes proportional to degree
#        max_deg = max(degrees.values()) if degrees else 1
#        node_sizes = [300 + (degrees[n] / max_deg) * 1000 for n in G.nodes()]

#        nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
#        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
#                               node_size=node_sizes, alpha=0.9, ax=ax)

        # ---- radial labels (outside nodes, rotated to follow circle) ----
#        xs = [p[0] for p in pos.values()]
#        ys = [p[1] for p in pos.values()]
#        cx, cy = np.mean(xs), np.mean(ys)
#        max_r = max(np.hypot(x - cx, y - cy) for x, y in pos.values())

#        for node, (x, y) in pos.items():
#            dx, dy = x - cx, y - cy
#            angle = np.arctan2(dy, dx)
#            deg = np.degrees(angle)

#            label_r = max_r + label_offset
#            lx = cx + label_r * np.cos(angle)
#            ly = cy + label_r * np.sin(angle)

#            if -90 < deg < 90:
#                rotation = deg
#                ha = 'left'
#            else:
#                rotation = deg + 180
#                ha = 'right'

#            ax.text(lx, ly, str(node),
#                    fontsize=font_size, ha=ha, va='center',
#                    rotation=rotation, rotation_mode='anchor')

#        ax.axis('off')
#        ax.set_xlim(cx - label_r - 0.15, cx + label_r + 0.15)
#        ax.set_ylim(cy - label_r - 0.15, cy + label_r + 0.15)
#        return ax

    # ------------------------------------------------------------------
    # VQ code distribution
    # ------------------------------------------------------------------

#    @staticmethod
    def code_distribution(vq_to_gene, gene_to_vq, ax=None):
        """
        Two-panel figure:
        (A) Bar chart: how many genes per VQ code.
        (B) Histogram: how many VQ codes per gene.

        Parameters
        ----------
        vq_to_gene : dict {code: [gene_indices]}
        gene_to_vq : dict {gene_idx: [vq_codes]}
        """
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: genes per VQ code
        code_sizes = sorted([len(genes) for genes in vq_to_gene.values()],
                            reverse=True)
        ax[0].bar(range(len(code_sizes)), code_sizes, color='steelblue',
                  edgecolor='none')
        ax[0].set_xlabel('VQ Code (sorted by size)')
        ax[0].set_ylabel('Number of Genes')
        ax[0].set_title('A: Genes per VQ Code')

        # Panel B: VQ codes per gene
        gene_diversity = sorted([len(codes) for codes in gene_to_vq.values()],
                                reverse=True)
        ax[1].hist(gene_diversity, bins=100, color='darkorange',
                   edgecolor='black', linewidth=0.3)
        ax[1].set_xlabel('Number of Distinct VQ Codes')
        ax[1].set_ylabel('Number of Genes')
        ax[1].set_title('B: VQ Codes per Gene')

        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Silhouette vs k
    # ------------------------------------------------------------------

    @staticmethod
    def silhouette_scan(scores_dicts, ax=None):
        """
        Line plot of silhouette score vs number of clusters k.

        Parameters
        ----------
        scores_dicts : dict {strat_name: {k: score}}
            Silhouette scores per stratification.
        ax : matplotlib Axes, optional
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        colors = {'year_lake': '#1f77b4', 'sex_year_lake': '#ff7f0e',
                  'infection_year_lake': '#2ca02c'}
        for label, scores in scores_dicts.items():
            ks = sorted(scores.keys())
            vals = [scores[k] for k in ks]
            ax.plot(ks, vals, 'o-', label=label,
                    color=colors.get(label, None), linewidth=2, markersize=6)

        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Analysis — Optimal k')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # Infection code enrichment bar chart
    # ------------------------------------------------------------------

    @staticmethod
    def infection_codes(code_results, codebook_size=100, top_n=20, ax=None):
        """
        Bar chart of top infection-associated VQ codes.

        Parameters
        ----------
        code_results : dict {code: {infected_mean, noninfected_mean, fold_change, p_value}}
        ax : matplotlib Axes, optional
        top_n : int
            Number of top codes to show (by significance).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))

        import numpy as np
        # Sort by p-value
        sorted_codes = sorted(code_results.items(),
                              key=lambda x: x[1]['p_value'])[:top_n]

        codes = [c for c, _ in sorted_codes]
        inf_means = [r['infected_mean'] for _, r in sorted_codes]
        ninf_means = [r['noninfected_mean'] for _, r in sorted_codes]
        pvals = [r['p_value'] for _, r in sorted_codes]

        x = np.arange(len(codes))
        width = 0.35

        bars1 = ax.bar(x - width/2, inf_means, width, label='Infected',
                       color='#e41a1c', alpha=0.8)
        bars2 = ax.bar(x + width/2, ninf_means, width, label='Non-infected',
                       color='#377eb8', alpha=0.8)

        ax.set_xlabel('VQ Code')
        ax.set_ylabel('Mean Code Usage')
        ax.set_title(f'Top {top_n} Infection-Associated VQ Codes')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in codes])
        ax.legend()

        # Add significance stars above bars
        for i, p in enumerate(pvals):
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                continue
            y_max = max(inf_means[i], ninf_means[i])
            ax.text(i, y_max + 0.0002, stars, ha='center', fontsize=7)

        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        return ax

    @staticmethod
    def sex_codes(code_results, top_n=20, ax=None):
        """Bar chart of top sex-associated VQ codes (Male vs Female)."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        import numpy as np
        sorted_codes = sorted(code_results.items(),
                              key=lambda x: x[1]['p_value'])[:top_n]
        codes = [c for c, _ in sorted_codes]
        m_means = [r['male_mean'] for _, r in sorted_codes]
        f_means = [r['female_mean'] for _, r in sorted_codes]
        pvals = [r['p_value'] for _, r in sorted_codes]

        x = np.arange(len(codes))
        width = 0.35
        ax.bar(x - width/2, m_means, width, label='Male',
               color='#377eb8', alpha=0.8)
        ax.bar(x + width/2, f_means, width, label='Female',
               color='#e41a1c', alpha=0.8)
        ax.set_xlabel('VQ Code')
        ax.set_ylabel('Mean Code Usage')
        ax.set_title(f'Top {top_n} Sex-Associated VQ Codes')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in codes])
        ax.legend()
        for i, p in enumerate(pvals):
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                continue
            y_max = max(m_means[i], f_means[i])
            ax.text(i, y_max + 0.0002, stars, ha='center', fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        return ax

    @staticmethod
    def role_codes(code_results, top_n=20, ax=None):
        """Bar chart of top Source-vs-Recipient VQ codes."""
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 5))
        import numpy as np
        sorted_codes = sorted(code_results.items(),
                              key=lambda x: x[1]['p_value'])[:top_n]
        codes = [c for c, _ in sorted_codes]
        s_means = [r['source_mean'] for _, r in sorted_codes]
        r_means = [r['recipient_mean'] for _, r in sorted_codes]
        pvals = [r['p_value'] for _, r in sorted_codes]

        x = np.arange(len(codes))
        width = 0.35
        ax.bar(x - width/2, s_means, width, label='Source',
               color='#2196F3', alpha=0.8)
        ax.bar(x + width/2, r_means, width, label='Recipient',
               color='#F44336', alpha=0.8)
        ax.set_xlabel('VQ Code')
        ax.set_ylabel('Mean Code Usage')
        ax.set_title(f'Top {top_n} Lake Category VQ Codes')
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in codes])
        ax.legend()
        for i, p in enumerate(pvals):
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            else:
                continue
            y_max = max(s_means[i], r_means[i])
            ax.text(i, y_max + 0.0002, stars, ha='center', fontsize=7)
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        return ax

    # ------------------------------------------------------------------
    # Lake embedding PCA
    # ------------------------------------------------------------------

    @staticmethod
    def lake_embedding_pca(lake_names, lake_embeddings, data=None, ax=None):
        """
        PCA scatter plot of learned lake embeddings from joint training.

        Visual encoding:
          - Marker shape encodes lake role (● Source, ▲ Recipient)
          - Colour encodes lake habitat (green = Benthic, blue = Limnetic),
            which for recipient lakes can differ from ecotype/ancestry.

        Parameters
        ----------
        lake_names : list[str]
            Lake names in order matching embeddings.
        lake_embeddings : np.ndarray (n_lakes, embed_dim)
            Learned lake embedding vectors.
        data : SticklebackData, optional
            For role / ecotype lookups.
        ax : matplotlib Axes, optional
        """
        import numpy as np
        from sklearn.decomposition import PCA

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 10))

        # PCA to 2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(lake_embeddings)

        # Min-max normalise
        for i in range(2):
            pmin, pmax = projected[:, i].min(), projected[:, i].max()
            if pmax > pmin:
                projected[:, i] = (projected[:, i] - pmin) / (pmax - pmin)

        # --- Visual encoding ---
        # Marker shape  → lake role  (● Source, ▲ Recipient)
        # Marker colour → lake HABITAT (physical environment; for recipients
        # it can differ from the ecotype/ancestry — see data.get_lake_habitat)
        role_marker  = {'Source': 'o', 'Recipient': '^', 'Other': 's'}
        hab_color    = {'Benthic': '#2ca02c',    # green
                        'Limnetic': '#1f77b4',   # blue
                        'Unknown': '#999999'}
        default_marker = 's'
        default_color  = '#999999'

        # Track which roles/habitats appear (for legend)
        plotted_roles = set()
        plotted_habs  = set()

        for i, name in enumerate(lake_names):
            role = data.get_lake_role(name) if data else 'Unknown'
            hab  = data.get_lake_habitat(name) if data else 'Unknown'

            marker = role_marker.get(role, default_marker)
            color  = hab_color.get(hab, default_color)

            ax.scatter(projected[i, 0], projected[i, 1],
                       c=color, marker=marker, s=150,
                       edgecolors='black', linewidth=0.8,
                       zorder=5)
            ax.annotate(_clean_year(name),
                        (projected[i, 0], projected[i, 1]),
                        fontsize=8, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points',
                        color=color)

            plotted_roles.add((role, marker))
            plotted_habs.add((hab, color))

        # --- Legend ---
        legend_handles = []
        from matplotlib.lines import Line2D
        # Role (marker) legend
        for role, marker in sorted(plotted_roles):
            legend_handles.append(
                Line2D([0], [0], marker=marker, color='black',
                       markersize=8, linestyle='None',
                       label=f'{role}'))
        # Habitat (colour) legend
        for hab, color in sorted(plotted_habs):
            legend_handles.append(
                Line2D([0], [0], marker='o', color=color,
                       markersize=8, linestyle='None',
                       label=f'{hab}'))
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA of Learned Lake Embeddings (Joint Training)')
        ax.legend(handles=legend_handles, fontsize=9, loc='best',
                  title='Lake Category (shape)  |  Lake Habitat (colour)')

        ax.grid(True, alpha=0.2)
        return ax

    # ------------------------------------------------------------------
    # PERMANOVA variance decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def permanova_bar(permanova_results, figsize=(16, 6)):
        """
        3-panel horizontal bar chart of marginal R² from PERMANOVA.

        Each panel shows one subset (all lakes, source only, recipient only)
        with bars for each factor, grouped by stratification.

        Parameters
        ----------
        permanova_results : dict
            {key: {factor: {'r2': float, 'p_value': float, ...}}}
            Keys containing '_source' or '_recipient' are routed to the
            corresponding panel; all others go to the "all" panel.
        """
        # Partition results into three subsets
        panels = {'All lakes': {}, 'Source only': {}, 'Recipient only': {}}
        for key, factors in permanova_results.items():
            if ' (Source only)' in key:
                panels['Source only'][
                    key.replace(' (Source only)', '')] = factors
            elif ' (Recipient only)' in key:
                panels['Recipient only'][
                    key.replace(' (Recipient only)', '')] = factors
            else:
                panels['All lakes'][key] = factors

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=False)
        strat_colors = {'year_lake': '#1f77b4',
                        'sex_year_lake': '#ff7f0e',
                        'infection_year_lake': '#2ca02c'}

        for ax, (title, strat_dict) in zip(axes, panels.items()):
            if not strat_dict:
                ax.set_title(f'{title}\n(no data)', fontsize=11)
                continue

            # Collect all factors across strats in this panel
            all_factors = sorted(set().union(*[d.keys() for d in
                                                strat_dict.values()]))
            n_factors = len(all_factors)
            n_strats = len(strat_dict)
            if n_strats == 0 or n_factors == 0:
                ax.set_title(f'{title}\n(no data)', fontsize=11)
                continue

            bar_height = 0.8 / n_strats
            y_positions = np.arange(n_factors)

            for i, (strat_name, factors) in enumerate(strat_dict.items()):
                r2_vals = [factors.get(f, {}).get('r2', 0)
                           for f in all_factors]
                p_vals = [factors.get(f, {}).get('p_value', 1.0)
                          for f in all_factors]
                offset = (i - (n_strats - 1) / 2) * bar_height
                ax.barh(y_positions + offset, r2_vals, bar_height,
                        label=strat_name,
                        color=strat_colors.get(strat_name, '#999999'),
                        alpha=0.85)

                # Significance stars
                for j, (r2, p) in enumerate(zip(r2_vals, p_vals)):
                    if p < 0.001:
                        star = '***'
                    elif p < 0.01:
                        star = '**'
                    elif p < 0.05:
                        star = '*'
                    else:
                        star = ''
                    if star:
                        ax.text(r2 + 0.003,
                                y_positions[j] + offset,
                                star, va='center', fontsize=7)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(all_factors)
            ax.set_xlabel('Marginal R²')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='x')
            ax.set_xlim(left=0)
            if n_strats > 1:
                ax.legend(loc='lower right', fontsize=7)

        fig.suptitle('PERMANOVA — Variance Explained by Each Factor',
                     fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout()
        return fig

"""
SticklebackData: Load and organize all stickleback transplantation experiment data.

Data Sources
------------
- Heart & Kidney transcriptome (gene expression for 28,135 genes per fish)
- Metadata (Fish_ID, Lake, GenotypePool, Year)
- Morphology (sex)
- Infection status (fibrosis score 0-4)

The core abstraction: for any stratification (lake, year, sex, infection),
return a normalized expression matrix M (fish × genes) where each row sums to 1.
"""

import numpy as np
import pandas as pd
from itertools import product


class SticklebackData:
    """
    Loads and manages stickleback transplant experiment data.

    Parameters
    ----------
    transcriptome_path : str
        Path to heart & kidney transcriptome CSV. Rows = fish, cols = genes.
    metadata_path : str
        Path to metadata CSV. Must have columns: Lake, GenotypePool, Year, Fish_ID.
    morphology_path : str, optional
        Path to morphology CSV. Uses column 'Sex_f_m_NA'.
    infection_path : str, optional
        Path to infection CSV. Uses column 'Fibrosis_score_0_1_2_3_4'.

    Key Maps (built on init)
    ------------------------
    fish_to_lake : dict Fish_ID -> str (lake name)
    fish_to_sex : dict Fish_ID -> str ('m' or 'f')
    fish_to_infection : dict Fish_ID -> int (fibrosis score 0-4)
    lake_to_genotype : dict str -> str (BenthicPool / LimneticPool / MixedPool)
    lake_classification : dict str -> dict with 'role' (Source/Recipient/Other)
                           and 'ecotype' (Benthic/Limnetic) and, for recipient
                           lakes where it can differ from ecotype, 'lake habitat'
                           (Benthic/Limnetic).  Source lakes have no 'lake
                           habitat' key — habitat == ecotype there.
    """

    # Colour palette for the physical lake habitat.  Distinct from ecotype
    # (ancestry): a recipient lake's habitat can differ from the ancestry of
    # its transplanted fish (e.g. Fred/Ranchero: Limnetic ancestry, Benthic
    # habitat).  Source lakes fall back to their ecotype as habitat.
    HABITAT_COLORS = {
        'Benthic':  '#8dd3c7',   # teal
        'Limnetic': '#80b1d3',   # steel blue
        'Unknown':  '#b3b3b3',
    }

    # Manual classification of lakes into experimental roles.
    # Source = long-established populations (monitoring controls, should be stable).
    # Recipient = previously fishless lakes that received transplants in 2019.
    LAKE_CLASSIFICATION = {
        # Source lakes
        'Finger':     {'role': 'Source',    'ecotype': 'Benthic'},
        'Long':       {'role': 'Source',    'ecotype': 'Limnetic'},
        'Spirit':     {'role': 'Source',    'ecotype': 'Limnetic'},
        'South Rolly':{'role': 'Source',    'ecotype': 'Limnetic'},
        'Tern':       {'role': 'Source',    'ecotype': 'Benthic'},
        'Walby':      {'role': 'Source',    'ecotype': 'Benthic'},
        'Wik':        {'role': 'Source',    'ecotype': 'Limnetic'},
        'Watson':     {'role': 'Source',    'ecotype': 'Benthic'},
        # Recipient lakes
        'CC Lake':    {'role': 'Recipient', 'ecotype': 'Benthic', 'lake habitat': 'Benthic'},
        'Crystal':    {'role': 'Recipient', 'ecotype': 'Limnetic', 'lake habitat': 'Limnetic'},
        'Fred':       {'role': 'Recipient', 'ecotype': 'Limnetic', 'lake habitat': 'Benthic'},
        'Hope':       {'role': 'Recipient', 'ecotype': 'Limnetic', 'lake habitat': 'Limnetic'},
        'Leisure':    {'role': 'Recipient', 'ecotype': 'Benthic', 'lake habitat': 'Limnetic'},
        'Loon':       {'role': 'Recipient', 'ecotype': 'Limnetic+Benthic', 'lake habitat': 'Benthic'},
        'Leisure Pond':{'role': 'Recipient', 'ecotype': 'Benthic', 'lake habitat': 'Benthic'},
        'Ranchero':   {'role': 'Recipient', 'ecotype': 'Limnetic', 'lake habitat': 'Benthic'},
        # Other
        'G Lake':     {'role': 'Other',     'ecotype': 'Limnetic+Benthic'},
        'Jean Lake':  {'role': 'Other',     'ecotype': 'Unknown'},
    }

    GENOTYPE_COLORS = {
        'BenthicPool':  '#e41a1c',
        'LimneticPool': '#377eb8',
        'MixedPool':    '#4daf4a',
        'nan':          '#000000',
    }

    LAKE_COLORS = {
        'CC Lake': '#1f77b4', 'Crystal': '#ff7f0e', 'Finger': '#2ca02c',
        'Fred': '#d62728', 'G Lake': '#9467bd', 'Hope': '#8c564b',
        'Jean Lake': '#e377c2', 'Leisure': '#7f7f7f', 'Leisure Pond': '#bcbd22',
        'Long': '#17becf', 'Loon': '#aec7e8', 'Ranchero': '#ffbb78',
        'South Rolly': '#98df8a', 'Spirit': '#ff9896', 'Tern': '#c5b0d5',
        'Walby': '#c49c94', 'Watson': '#f7b6d2', 'Wik': '#dbdb8d',
    }

    def __init__(self, transcriptome_path, metadata_path,
                 morphology_path=None, infection_path=None,
                 input_scale='linear'):
        """input_scale : {'linear', 'log2cpm'}
            Domain of the transcriptome values.  'linear' = raw/CPM counts
            (row-sum normalization yields relative abundance directly).
            'log2cpm' = log2(CPM+1) values (e.g. ComBat-corrected CSV from
            step_batch_correct); these are exponentiated back to linear CPM
            (2**x - 1, clipped at 0) BEFORE row-sum normalization, so the
            resulting matrices are batch-corrected relative abundance —
            consistent with the linear path.  Dividing log-scale values by
            their row sum directly is NOT relative abundance and was a bug.
        """
        self.input_scale = input_scale
        # -- Load transcriptome -------------------------------------------------
        self.hk_data = pd.read_csv(transcriptome_path)
        # Drop the first column if it is an unnamed index (always present in
        # the source CSV).  Must happen BEFORE dropna so the NaN check only
        # considers Fish_ID + gene columns.
        first_col = self.hk_data.columns[0]
        if first_col != 'Fish_ID':
            self.hk_data = self.hk_data.drop(columns=[first_col])
        # A fish with ANY missing gene value would otherwise produce an
        # all-NaN row after row-sum normalization (NaN propagates silently
        # through the sum).  Drop those fish loudly rather than let the NaN
        # poison the matrices.  Rows where every gene is NaN are also dropped.
        gene_cols = list(self.hk_data.columns[1:])
        n_before = len(self.hk_data)
        self.hk_data = self.hk_data.dropna(subset=gene_cols, how='any') \
                                   .reset_index(drop=True)
        n_dropped = n_before - len(self.hk_data)
        if n_dropped > 0:
            print(f"[data] Dropped {n_dropped} fish with ≥1 missing gene value "
                  f"(would otherwise produce all-NaN normalized rows).")

        # Duplicate Fish_IDs would silently inflate every matrix (pandas .loc
        # returns ALL matching rows).  This is a data-integrity failure — fail
        # loudly instead of silently double-counting fish.
        dupes = self.hk_data['Fish_ID'].duplicated(keep=False)
        dup_ids = sorted(self.hk_data.loc[dupes, 'Fish_ID'].astype(str).unique())
        if dup_ids:
            raise ValueError(
                f"[data] Duplicate Fish_ID rows in transcriptome: {dup_ids}. "
                f"Refusing to proceed — duplicates would silently inflate "
                f"expression matrices.")

        self.gene_names = list(self.hk_data.columns[1:])
        self.n_genes = len(self.gene_names)
        self.fish_ids = list(self.hk_data['Fish_ID'])

        # -- Load metadata ------------------------------------------------------
        self.metadata = pd.read_csv(metadata_path)

        # ------- Build fish_to_lake map --------
        self.fish_to_lake = dict(zip(self.metadata['Fish_ID'],
                                     self.metadata['Lake']))

        # ------- Build lake_to_genotype map -------
        # A lake can genuinely contain fish from >1 genotype pool (the
        # transplantation design).  Collapsing via drop_duplicates would
        # silently pick one pool.  Instead: report mixed-genotype lakes loudly
        # and label them 'MixedPool' (which has its own color), keeping the
        # modal pool only as a fallback.
        lake_geno = self.metadata[['Lake', 'GenotypePool']].dropna()
        geno_by_lake = {lake: list(grp['GenotypePool'].unique())
                        for lake, grp in lake_geno.groupby('Lake')}
        self.lakes_with_mixed_genotype = {}
        self.lake_to_genotype = {}
        for lake, pools in geno_by_lake.items():
            if len(pools) > 1:
                self.lakes_with_mixed_genotype[str(lake)] = [str(p) for p in pools]
                self.lake_to_genotype[lake] = 'MixedPool'
            else:
                self.lake_to_genotype[lake] = pools[0]
        if self.lakes_with_mixed_genotype:
            print(f"[data] Lakes with multiple genotype pools → labeled 'MixedPool': "
                  f"{self.lakes_with_mixed_genotype}")

        # ------- Build year -> [fish_ids] map -------
        self.year_to_fish = {}
        for yr, grp in self.metadata.groupby('Year'):
            self.year_to_fish[yr] = list(grp['Fish_ID'])

        # Sort by string representation to avoid type-comparison issues if
        # metadata contains mixed year types.
        self.years = sorted(self.year_to_fish.keys(), key=lambda v: str(v))

        # -- Load morphology (sex) ----------------------------------------------
        self.fish_to_sex = {}
        self.morphology = None
        if morphology_path:
            self.morphology = pd.read_csv(morphology_path)
            self.fish_to_sex = {f: str(s) for f, s in
                                zip(self.morphology['Fish_ID'],
                                    self.morphology['Sex_f_m_NA'])}

        # -- Load infection (fibrosis) ------------------------------------------
        # Raw fibrosis scores are 0-4.  The analysis only stratifies into
        # non-infected (0) vs infected (>=1); keeping the raw score would
        # silently drop fish with scores 2-4 from BOTH strata.  Binarize at
        # load: score >= 1 == infected.  Assumption stated explicitly: any
        # nonzero fibrosis score counts as infected.
        self.fish_to_infection = {}
        self.infection = None
        if infection_path:
            self.infection = pd.read_csv(infection_path)
            raw = dict(zip(self.infection['Fish_ID'],
                           self.infection['Fibrosis_score_0_1_2_3_4']))
            n_inf = 0
            n_noninf = 0
            n_unscored = 0
            for f, score in raw.items():
                # Fish without a recorded fibrosis score (NaN) carry no
                # infection measurement — classifying them as either infected
                # or non-infected would fabricate a label.  They stay in
                # lake/year/sex strata but are excluded from BOTH infection
                # strata (fish_to_infection[f] = None matches neither 0 nor 1
                # in _fish_subset).
                if pd.isna(score) or (
                        isinstance(score, str) and not score.strip()):
                    self.fish_to_infection[f] = None
                    n_unscored += 1
                    continue
                s = int(score)
                if s < 0:
                    raise ValueError(f"[data] Negative fibrosis score {s} for "
                                     f"fish {f}")
                self.fish_to_infection[f] = 1 if s >= 1 else 0
                if s >= 1:
                    n_inf += 1
                else:
                    n_noninf += 1
            print(f"[data] Infection binarized: {n_inf} infected "
                  f"(fibrosis >= 1), {n_noninf} non-infected "
                  f"(fibrosis == 0), {n_unscored} unscored "
                  f"(excluded from infection strata).")

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_lake(lake, lookup_dict):
        """Look up a lake name, trying exact match, with/without ' Lake' suffix."""
        if lake in lookup_dict:
            return lookup_dict[lake]
        if lake.endswith(' Lake'):
            stripped = lake[:-len(' Lake')]
            if stripped in lookup_dict:
                return lookup_dict[stripped]
        else:
            with_lake = lake + ' Lake'
            if with_lake in lookup_dict:
                return lookup_dict[with_lake]
        return None

    def get_lake_role(self, lake):
        """Return 'Source', 'Recipient', or 'Other' for a lake."""
        entry = self._normalize_lake(lake, self.LAKE_CLASSIFICATION)
        return entry.get('role', 'Unknown') if entry else 'Unknown'

    def get_lake_ecotype(self, lake):
        """Return 'Benthic' or 'Limnetic' for a lake (the ancestry/genotype
        of the lake's fish)."""
        entry = self._normalize_lake(lake, self.LAKE_CLASSIFICATION)
        return entry.get('ecotype', 'Unknown') if entry else 'Unknown'

    def get_lake_habitat(self, lake):
        """Return the physical lake habitat for a lake: 'Benthic' or 'Limnetic'.

        Recipient lakes may carry an explicit 'lake habitat' that differs from
        the ecotype (ancestry) of their transplanted fish (e.g. Fred/Ranchero
        are Limnetic-ancestry but Benthic-habitat).  Source lakes have no
        'lake habitat' key because habitat == ecotype there, so this falls
        back to ecotype.  Returns 'Unknown' for unclassified lakes.
        """
        entry = self._normalize_lake(lake, self.LAKE_CLASSIFICATION)
        if entry:
            habitat = entry.get('lake habitat')
            if habitat:
                return habitat
            return entry.get('ecotype', 'Unknown')
        return 'Unknown'

    def get_genotype(self, lake):
        """Return genotype pool string (BenthicPool / LimneticPool / MixedPool)."""
        entry = self._normalize_lake(lake, self.lake_to_genotype)
        return entry if entry else 'nan'

# DEAD CODE (commented out): SticklebackData.get_transplant_match (never called)
#    def get_transplant_match(self, lake):
#        """Classify a lake by transplant match status.

#        Returns
#        -------
#        str
#            ``'source'`` — natural population, no transplant.
#            ``'matched'`` — recipient lake where fish ancestry matches habitat.
#            ``'mismatched'`` — recipient lake where ancestry ≠ habitat.
#            ``'mixed'`` — MixedPool ancestry or mixed-ecotype habitat.
#            ``'other'`` — neither Source nor Recipient (e.g. Jean Lake).
#        """
#        role = self.get_lake_role(lake)
#        if role == 'Source':
#            return 'source'

#        genotype = self.get_genotype(lake)
        # Compare ancestry against the lake's PHYSICAL HABITAT, not its
        # ecotype: the ecotype field is itself ancestry, so comparing
        # genotype→ecotype measured ancestry-vs-ancestry and could never
        # classify a lake as 'mismatched' (e.g. Fred/Ranchero are Limnetic-
        # ancestry but Benthic-habitat).
#        hab = self.get_lake_habitat(lake)

        # Non-Source lakes without genotype data → 'other'
#        if genotype == 'nan' or hab == 'Unknown':
#            return 'other'

        # Mixed ancestry or mixed habitat → 'mixed'
#        is_mixed_hab = ('Benthic' in hab and 'Limnetic' in hab)
#        if genotype == 'MixedPool' or is_mixed_hab:
#            return 'mixed'

        # Map genotype pool to expected ecotype for matched/mismatched
#        genotype_eco = {
#            'BenthicPool': 'Benthic',
#            'LimneticPool': 'Limnetic',
#        }.get(genotype)

#        if genotype_eco is None:
#            return 'other'

#        return 'matched' if genotype_eco == hab else 'mismatched'

    def get_genotype_color(self, lake):
        """Return color for the lake's genotype pool."""
        return self.GENOTYPE_COLORS.get(str(self.get_genotype(lake)), '#000000')

    def get_lake_color(self, lake):
        """Return assigned color for a given lake.

        Uses the same name normalization as the other getters so a lake
        referenced as 'Crystal Lake' (with suffix) resolves to the color
        keyed under 'Crystal' (and vice versa) instead of silently falling
        back to black.
        """
        color = self._normalize_lake(lake, self.LAKE_COLORS)
        return color if color else '#000000'

    # ------------------------------------------------------------------
    # Expression matrix extraction
    # ------------------------------------------------------------------

    def _fish_subset(self, lake=None, year=None, sex=None, infection=None):
        """
        Return list of fish IDs matching ALL specified filters.
        Each filter is optional — if None, that dimension is not constrained.
        """
        ids = set(self.fish_ids)

        if lake is not None:
            ids &= {f for f in ids if self.fish_to_lake.get(f) == lake}
        if year is not None:
            ids &= set(self.year_to_fish.get(year, []))
        if sex is not None:
            ids &= {f for f in ids if self.fish_to_sex.get(f) == sex}
        if infection is not None:
            ids &= {f for f in ids
                    if self.fish_to_infection.get(f) == infection}

        return sorted(ids, key=lambda v: str(v))

    def get_expression_matrix(self, lake=None, year=None, sex=None,
                              infection=None, min_fish=7):
        """
        Return a row-normalized expression matrix M (n_fish × n_genes)
        for the fish matching the given filters.

        Each row is divided by its sum so rows represent relative abundance
        of gene expression across genes.  All-zero rows (failed samples)
        are dropped loudly, and any residual NaN after normalization raises
        instead of propagating silently.

        Returns None if fewer than `min_fish` fish match the criteria.
        """
        fish = self._fish_subset(lake, year, sex, infection)
        if len(fish) < min_fish:
            return None

        rows = self.hk_data.set_index('Fish_ID').loc[fish]
        M = rows.values.astype(np.float32)
        # If the transcriptome is in log2(CPM+1) space (e.g. the ComBat-
        # corrected CSV), invert to linear CPM before row-normalizing so the
        # result is genuine relative abundance. Dividing log-scale values by
        # their row sum directly is not relative abundance.
        if getattr(self, 'input_scale', 'linear') == 'log2cpm':
            M = np.power(2.0, M) - 1.0
            np.clip(M, 0.0, None, out=M)
        # Row-wise normalization to relative abundance
        row_sums = M.sum(axis=1, keepdims=True)
        # All-zero rows are failed samples (no expression at all).  The old
        # guard row_sums[row_sums == 0] = 1.0 silently turned them into
        # uniform rows — fabricating data.  Drop them loudly instead.
        bad_rows = (row_sums[:, 0] == 0) | (~np.isfinite(row_sums[:, 0]))
        if bad_rows.any():
            n_bad = int(bad_rows.sum())
            dropped_ids = [str(f) for f, b in zip(fish, bad_rows) if b]
            print(f"[data] Dropping {n_bad} fish with zero/non-finite total "
                  f"expression: {dropped_ids[:5]}{' ...' if n_bad > 5 else ''}")
            fish = [f for f, b in zip(fish, bad_rows) if not b]
            M = M[~bad_rows]
            row_sums = row_sums[~bad_rows]
            # Re-check min_fish AFTER dropping failed samples.
            if len(fish) < min_fish:
                return None
        M = M / row_sums
        if not np.isfinite(M).all():
            raise ValueError(
                f"[data] Non-finite values in normalized matrix after "
                f"filtering lake={lake}, year={year}, sex={sex}, "
                f"infection={infection}. Refusing to return a poisoned matrix.")
        return M

    # ------------------------------------------------------------------
    # Bulk stratification
    # ------------------------------------------------------------------

    def _unique_lakes(self):
        """Return deterministic non-null lake labels from metadata."""
        lakes = {
            lake for lake in self.fish_to_lake.values()
            if pd.notna(lake) and str(lake).strip() != ''
        }
        # Sort by string representation to handle mixed metadata dtypes safely.
        return sorted(lakes, key=lambda v: str(v))

    def stratify(self, by='lake', year=None, sex=None, infection=None):
        """
        Generate all expression matrices at a given stratification level.

        Parameters
        ----------
        by : str, one of {'lake', 'year_lake', 'sex_year_lake', 'infection_year_lake'}
            The stratification level.
        year, sex, infection : optional filters applied in addition to `by`.

        Yields
        ------
        (key, matrix) tuples where `key` is a descriptive string.
        """
        lakes = self._unique_lakes()

        # NOTE (correctness fix): the optional year/sex/infection filters must
        # apply in ADDITION to `by`, regardless of which dimension `by` varies
        # over.  Previously each branch dropped the filters for the dimensions
        # it did not iterate, so stratify(by='lake', infection=1) silently
        # included non-infected fish.  Every branch now passes all filters.
        if by == 'lake':
            for lake in lakes:
                M = self.get_expression_matrix(lake=lake, year=year,
                                               sex=sex, infection=infection)
                if M is not None:
                    yield lake, M

        elif by == 'year_lake':
            for yr, lake in product(self.years, lakes):
                M = self.get_expression_matrix(lake=lake, year=yr,
                                               sex=sex, infection=infection)
                if M is not None:
                    yield f'{lake} ({yr})', M

        elif by == 'sex_year_lake':
            for sex_val, yr, lake in product(
                ['f', 'm'], self.years, lakes
            ):
                M = self.get_expression_matrix(lake=lake, year=yr,
                                               sex=sex_val,
                                               infection=infection)
                if M is not None:
                    yield f'{lake} ({yr})-{sex_val}', M

        elif by == 'infection_year_lake':
            for inf, yr, lake in product(
                [0, 1], self.years, lakes
            ):
                M = self.get_expression_matrix(lake=lake, year=yr, sex=sex,
                                               infection=inf)
                if M is not None:
                    yield f'{lake} ({yr})-{inf}', M
        else:
            raise ValueError(f"Unknown stratification: {by}")

    def build_all_matrices(self, by='lake'):
        """Return a dict {key: matrix} for a given stratification level."""
        return dict(self.stratify(by=by))

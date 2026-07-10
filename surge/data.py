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
                           and 'ecotype' (Benthic/Limnetic)
    """

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
        'CC Lake':    {'role': 'Recipient', 'ecotype': 'Benthic'},
        'Crystal':    {'role': 'Recipient', 'ecotype': 'Limnetic'},
        'Fred':       {'role': 'Recipient', 'ecotype': 'Limnetic'},
        'Hope':       {'role': 'Recipient', 'ecotype': 'Limnetic'},
        'Leisure':    {'role': 'Recipient', 'ecotype': 'Limnetic'},
        'Loon':       {'role': 'Recipient', 'ecotype': 'Limnetic+Benthic'},
        'Leisure Pond':{'role': 'Recipient', 'ecotype': 'Benthic'},
        'Ranchero':   {'role': 'Recipient', 'ecotype': 'Limnetic'},
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
        # Remove rows where every gene is NaN (original code checks gene
        # columns only, not the Fish_ID column).
        gene_cols = list(self.hk_data.columns[1:])
        self.hk_data = self.hk_data.dropna(how='all', subset=gene_cols).reset_index(drop=True)

        self.gene_names = list(self.hk_data.columns[1:])
        self.n_genes = len(self.gene_names)
        self.fish_ids = list(self.hk_data['Fish_ID'])

        # -- Load metadata ------------------------------------------------------
        self.metadata = pd.read_csv(metadata_path)

        # ------- Build fish_to_lake map --------
        self.fish_to_lake = dict(zip(self.metadata['Fish_ID'],
                                     self.metadata['Lake']))

        # ------- Build lake_to_genotype map -------
        lake_genotype = self.metadata[['Lake', 'GenotypePool']].drop_duplicates()
        self.lake_to_genotype = dict(zip(lake_genotype['Lake'],
                                         lake_genotype['GenotypePool']))

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
        self.fish_to_infection = {}
        self.infection = None
        if infection_path:
            self.infection = pd.read_csv(infection_path)
            self.fish_to_infection = dict(
                zip(self.infection['Fish_ID'],
                    self.infection['Fibrosis_score_0_1_2_3_4'])
            )

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
        """Return 'Benthic' or 'Limnetic' for a lake."""
        entry = self._normalize_lake(lake, self.LAKE_CLASSIFICATION)
        return entry.get('ecotype', 'Unknown') if entry else 'Unknown'

    def get_genotype(self, lake):
        """Return genotype pool string (BenthicPool / LimneticPool / MixedPool)."""
        entry = self._normalize_lake(lake, self.lake_to_genotype)
        return entry if entry else 'nan'

    def get_transplant_match(self, lake):
        """Classify a lake by transplant match status.

        Returns
        -------
        str
            ``'source'`` — natural population, no transplant.
            ``'matched'`` — recipient lake where fish ancestry matches habitat.
            ``'mismatched'`` — recipient lake where ancestry ≠ habitat.
            ``'mixed'`` — MixedPool ancestry or mixed-ecotype habitat.
            ``'other'`` — neither Source nor Recipient (e.g. Jean Lake).
        """
        role = self.get_lake_role(lake)
        if role == 'Source':
            return 'source'

        genotype = self.get_genotype(lake)
        eco = self.get_lake_ecotype(lake)

        # Non-Source lakes without genotype data → 'other'
        if genotype == 'nan' or eco == 'Unknown':
            return 'other'

        # Mixed ancestry or mixed-ecotype habitat → 'mixed'
        is_mixed_eco = ('Benthic' in eco and 'Limnetic' in eco)
        if genotype == 'MixedPool' or is_mixed_eco:
            return 'mixed'

        # Map genotype pool to expected ecotype for matched/mismatched
        genotype_eco = {
            'BenthicPool': 'Benthic',
            'LimneticPool': 'Limnetic',
        }.get(genotype)

        if genotype_eco is None:
            return 'other'

        return 'matched' if genotype_eco == eco else 'mismatched'

    def get_genotype_color(self, lake):
        """Return color for the lake's genotype pool."""
        return self.GENOTYPE_COLORS.get(str(self.get_genotype(lake)), '#000000')

    def get_lake_color(self, lake):
        """Return assigned color for a given lake."""
        return self.LAKE_COLORS.get(lake, '#000000')

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

        return sorted(ids)

    def get_expression_matrix(self, lake=None, year=None, sex=None,
                              infection=None, min_fish=1):
        """
        Return a row-normalized expression matrix M (n_fish × n_genes)
        for the fish matching the given filters.

        Each row is divided by its sum so rows represent relative abundance
        of gene expression across genes.

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
        row_sums[row_sums == 0] = 1.0
        M = M / row_sums
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

        if by == 'lake':
            for lake in lakes:
                M = self.get_expression_matrix(lake=lake)
                if M is not None:
                    yield lake, M

        elif by == 'year_lake':
            for yr, lake in product(self.years, lakes):
                M = self.get_expression_matrix(lake=lake, year=yr)
                if M is not None:
                    yield f'{lake} ({yr})', M

        elif by == 'sex_year_lake':
            for sex_val, yr, lake in product(
                ['f', 'm'], self.years, lakes
            ):
                M = self.get_expression_matrix(lake=lake, year=yr, sex=sex_val)
                if M is not None:
                    yield f'{lake} ({yr})-{sex_val}', M

        elif by == 'infection_year_lake':
            for inf, yr, lake in product(
                [0, 1], self.years, lakes
            ):
                M = self.get_expression_matrix(lake=lake, year=yr, infection=inf)
                if M is not None:
                    yield f'{lake} ({yr})-{inf}', M
        else:
            raise ValueError(f"Unknown stratification: {by}")

    def build_all_matrices(self, by='lake'):
        """Return a dict {key: matrix} for a given stratification level."""
        return dict(self.stratify(by=by))

import itertools
from tqdm import tqdm
from joblib import Parallel, parallel_config, delayed
import sys
import signal
from contextlib import contextmanager

import numpy as np
from scipy.spatial.distance import pdist, cdist

from pymatgen.core.structure import Structure, Lattice
from pymatgen.core.composition import Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
from pyxtal.util import symmetrize
import smact
from smact.screening import pauling_test
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from aliases import *
from constants import *
from utils.io_utils import get_project_dir
import global_vars

PROJECT_DIR = get_project_dir()
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')


class TimeoutError(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Cancel alarm no matter what


def get_matches(
    structure: pmg_structure, alternatives: List[pmg_structure], matcher: StructureMatcher
) -> tuple[list[int], list[float]]:
    """For measuring uniqueness"""
    matches, rms_dists = [], []
    for structure_idx, alt in enumerate(alternatives):
        rms_dist = matcher.get_rms_dist(structure, alt)
        if rms_dist is not None:
            rms_dist, *_ = rms_dist
            rms_dists.append(rms_dist)
            matches.append(structure_idx)
    return matches, rms_dists


def get_unique_structures(
    structures: List[pmg_structure], matcher: StructureMatcher
) -> Tuple[List[pmg_structure], List[int]]:
    matches_and_rms_dists: List[Tuple[List[int], List[float]]] = joblib_map(
        lambda structure: get_matches(structure, structures, matcher),
        structures,
        n_jobs=-4,  # uses (n_cpus + 1 + n_jobs) cpus
        inner_max_num_threads=1,
        total=len(structures),
        desc="Getting unique structures."
    )

    unique_structures = []
    unique_structure_idxs = []
    for matches, rms_dists in matches_and_rms_dists:
        matches = sorted(matches)
        if len(matches) == 0:
            # This conditional should not execute because a structure should
            # match itself
            breakpoint()
            match = matches
        else:
            # only keep first structure from group of duplicates
            match = matches[0]

        if match not in unique_structure_idxs:
            unique_structure_idxs.append(match)
            unique_structures.append(structures[match])
    return unique_structures, unique_structure_idxs


def get_chemsys(structure: Structure) -> str:
    return str(sorted(set(list(e.name for e in structure.composition.elements))))


def get_novel_structures(
    structures: List[Structure],
    reference_structures: List[Structure],
    matcher: StructureMatcher
) -> Tuple[List[Structure], List[int]]:
    # composition spaces present in crystals must match to compare structures
    generated_compositions: ndarray = np.array(list(map(get_chemsys, structures)))
    reference_compositions: ndarray = np.array(list(map(get_chemsys, reference_structures)))
    (
        intersection,
        idx_of_gen_xtals_in_intersection,
        idx_of_ref_xtals_in_intersection,
    ) = np.intersect1d(
        generated_compositions, reference_compositions, return_indices=True
    )
    gen_to_compare = np.isin(generated_compositions, intersection)
    ref_to_compare = np.isin(reference_compositions, intersection)
    reference_structures = [  # reduce reference structures to the intersection
        s for compare, s in zip(ref_to_compare, reference_structures) if compare
    ]

    gen_xtals_to_compare: List[Structure] = []
    novel_structures: List[Structure] = []
    novel_structure_idxs: List[int] = []
    for i, (compare, s) in enumerate(zip(gen_to_compare, structures)):
        if compare:
            gen_xtals_to_compare.append(s)
        else:
            novel_structures.append(s)
            novel_structure_idxs.append(i)

    # Compare structures with the same composition
    matches_and_rms_dists: List[Tuple[List[int], List[float]]] = joblib_map(
        lambda structure: get_matches(structure, reference_structures, matcher),
        gen_xtals_to_compare,
        n_jobs=-4,  # uses (n_cpus + 1 + n_jobs) cpus
        inner_max_num_threads=1,
        total=len(gen_xtals_to_compare),
        desc="Getting novel structures."
    )  # length: len(gen_xtals_to_compare)

    for gen_xtal_idx, s, (matches, rms_dists) in zip(
        idx_of_gen_xtals_in_intersection,
        gen_xtals_to_compare,
        matches_and_rms_dists,
    ):
        if len(matches) == 0:
            novel_structures.append(s)
            novel_structure_idxs.append(gen_xtal_idx)
    return novel_structures, novel_structure_idxs


def load_dataset_pymatgen_structures(dataset_name: str, split: str = "train") -> List[pmg_structure]:
    assert split in ["train", "test"]
    import csv
    import pickle
    from utils.io_utils import DATA_DIRECTORY, remove_suffix, extract

    print(f"Loading {split} crystals.")
    filepath = Path(DATA_DIRECTORY / (dataset_name + f"_{split}_pmg_structures.pkl"))
    filename: str = filepath.as_posix()
    if filepath.exists():
        with open(filename, "rb") as f:
            structures = pickle.load(f)
    else:
        if dataset_name == "mp20":
            archive: Path = Path(DATA_DIRECTORY / "mp_20.tar.bz")
        elif dataset_name == "mpts52":
            archive: Path = Path(DATA_DIRECTORY / "mpts_52.tar.bz")
        else:
            raise AttributeError

        extracted_data: Path = DATA_DIRECTORY / remove_suffix(archive)
        extract(archive)
        csvs: List[Path] = extracted_data.glob("**/*csv*")

        split_csv_location = None
        for csv_location in csvs:
            if f"{split}.csv" in csv_location.as_posix():
                split_csv_location = csv_location
                break

        handle = open(split_csv_location, newline="")
        entries: List[dict] = list(csv.DictReader(handle))
        handle.close()

        structures = []
        for entry in tqdm(entries):
            with HiddenPrints():
            # suppress annoying print statements from pymatgen
                structures.append(Structure.from_str(entry["cif"], fmt="cif"))

        with open(filename, "wb") as f:
            pickle.dump(structures, f)

    print(f"Loaded {split} crystals.")

    return structures


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    # If the number of possible combinations is big, it'd take too much time to
    # run the smact checker. In this case, we assume that at least one of the
    # combinations is valid (i.e., don't filter it out now, but later with DFT).
    n_charge_state_combinations = np.prod([len(ls) for ls in ox_combos])
    if n_charge_state_combinations > 1e6:
        return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(crystal: pmg_structure, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none


class Crystal(object):

    def __init__(
        self,
        crys_array_dict: dict = None,
        convert_to_primitive: bool = False,
        atom_types_are_logits: bool = False,
        structure: pmg_structure = None,
    ):
        self.frac_coords: ndarray = crys_array_dict.get('frac_coords', None)
        self.atom_types: ndarray = crys_array_dict.get('atom_types', None)
        # atomic numbers (1-indexed)
        if (
            self.atom_types is not None and
            (atom_types_are_logits or len(self.atom_types.shape) > 1)
        ):  # diffcsp does this
            self.atom_types: ndarray = np.argmax(self.atom_types, axis=-1) + 1
        self.lengths: ndarray = crys_array_dict.get('lengths', None)
        self.angles: ndarray = crys_array_dict.get('angles', None)
        self.lattice_matrix: ndarray = crys_array_dict.get('lattice_matrix', None)
        self.wyckoff_dims: ndarray = None  #crys_array_dict.get('wyckoff_dims', None)
        self.space_group_number: int = None  #crys_array_dict.get('space_group_number', None)
        self.dict = crys_array_dict
        assert self.lattice_matrix is not None or (
            self.lengths is not None and self.angles is not None
        ) or structure is not None

        # Attributes to be filled by self.get_structure()
        if structure is None:
            self.structure: pmg_structure = None
        else:
            self.frac_coords = structure.frac_coords
            self.atom_types = np.array([site.specie.Z for site in structure.sites])
            self.lengths = np.array(list(structure.lattice.lengths))
            self.angles = np.array(list(structure.lattice.angles))

        self.get_structure(convert_to_primitive)  # Fill None attributes
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()
        self.get_symmetry_info()

    def get_structure(self, convert_to_primitive: bool = False):
        self.constructed = None
        if (1 > self.atom_types).any() or (self.atom_types > 98).any():
            self.constructed = False
            self.invalid_reason = f"{self.atom_types=} are not with range"
        if len(self.atom_types) == 0:
            self.constructed = False
            self.invalid_reason = 'empty'

        if self.lengths is not None:
            if min(self.lengths.tolist()) < 0:
                self.constructed = False
                self.invalid_reason = 'non_positive_lattice'
            elif np.isnan(self.lengths).any() or np.isnan(self.angles).any() or np.isnan(self.frac_coords).any():
                self.constructed = False
                self.invalid_reason = 'nan_value'
            elif np.isinf(self.lengths).any() or np.isinf(self.angles).any() or np.isinf(self.frac_coords).any():
                self.constructed = False
                self.invalid_reason = 'inf_value'
            elif (self.lengths > 1000).any():
                self.constructed = False
                self.invalid_reason = 'bad_value'
            elif (self.angles >= 180).any() or (self.angles <= 0).any():
                self.constructed = False
                self.invalid_reason = 'bad_value'
            elif (self.frac_coords > 1).any() or (self.frac_coords < 0).any():
                self.constructed = False
                self.invalid_reason = 'bad_value'
            elif self.lengths.min() < 1:
                self.constructed = False
                self.invalid_reason = 'bad_value'

        if self.constructed is None:
            try:
                if self.lattice_matrix is None:
                    self.structure = Structure(
                        lattice=Lattice.from_parameters(
                            *(self.lengths.tolist() + self.angles.tolist())),
                        species=self.atom_types,
                        coords=self.frac_coords,
                        coords_are_cartesian=False
                    )
                else:
                    self.structure = Structure(
                        lattice=Lattice(self.lattice_matrix),
                        species=self.atom_types,
                        coords=self.frac_coords,
                        coords_are_cartesian=False
                    )
                self.constructed = True
                if self.structure.volume < 0.1:
                    self.constructed = False
                    self.invalid_reason = 'unrealistically_small_lattice'
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'

            if convert_to_primitive:
                self.structure = self.structure.get_reduced_structure()  # niggli
                self.atom_types = np.array([e.Z for e in self.structure.species])
                self.frac_coords = self.structure.frac_coords

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        if len(elem_counter) == 0:
            self.elems = ()
            self.comps = ()
            return
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        if self.constructed:
            if len(self.elems) == 0:
                self.comp_valid = False
            else:
                self.comp_valid = smact_validity(self.elems, self.comps)
            self.struct_valid = structure_validity(self.structure)
        else:
            self.comp_valid = False
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        if self.constructed and self.valid:
            elem_counter = Counter(self.atom_types)
            comp = Composition(elem_counter)
            self.comp_fp = CompFP.featurize(comp)
            try:
                if len(self.structure) > 100:
                    raise Exception
                site_fps = [CrystalNNFP.featurize(
                    self.structure, i) for i in range(len(self.structure))]
                self.struct_fp = np.array(site_fps).mean(axis=0)
            except Exception:
                # counts crystal as invalid if fingerprint cannot be constructed.
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
                return
        else:
            self.comp_fp = None
            self.struct_fp = None

    def get_symmetry_info(self):
        # Get Wyckoff dimensionality of each crystallographic orbit
        if self.constructed and self.struct_valid:
            try:
                symmetrized_structure: pmg_structure = symmetrize(
                    self.structure, tol=0.1, a_tol=5.0, style="pyxtal"
                )[0]
                (
                    conventional_symmetrized_structure,
                    self.space_group_number
                ) = pyxtal.util.get_symmetrized_pmg(
                    symmetrized_structure, tol=0.1, a_tol=5.0, style="pyxtal"
                )
                wyckoff_letters: List[str] = [
                    wyckoff_symbol[-1] for wyckoff_symbol in conventional_symmetrized_structure.wyckoff_symbols
                ]
            except:
                self.space_group_number = 1
                wyckoff_letters = ['a' for _ in range(len(self.structure))]
        else:
            # Dummy info
            self.space_group_number = 1
            wyckoff_letters = ['a' for _ in range(self.frac_coords.shape[0])]

        self.wyckoff_dims = np.array([
            int(
                global_vars.asu_wyckoff_dict[str(self.space_group_number)][wyckoff_letter]["dim"]
            ) for wyckoff_letter in wyckoff_letters
        ])

        # For counting "templates", defined by SymmCD as a tuple:
        # (space group, Wyckoff letter multiset)
        self.immutable_template = (
            self.space_group_number, frozenset(Counter(wyckoff_letters).items())
        )


def get_fingerprint_pairwise_dist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()


def get_fingerprint_central_moment_discrepancy(
    p_fingerprints: Union[Tensor, ndarray],
    q_fingerprints: Union[Tensor, ndarray],
    n_moments: int = 50,
) -> float:
    """
    See papers on central moment discrepancy:
    https://arxiv.org/pdf/1702.08811.pdf
    https://github.com/wzell/mann/blob/7864c3b6fb511e5e1278770fdf94b58b1835a063/artificial_example.py#L80

    Args:
        p_fingerprints: shape (n_samples, n_features)
        q_fingerprints: shape (n_samples, n_features)
        n_moments: Number of distribution central moments to compare between
            p_fingerprints and q_fingerprints.

    Returns:
        cmd
    """
    x1 = p_fingerprints
    x2 = q_fingerprints

    def l2diff(x1, x2):
        """
        standard euclidean norm
        """
        if torch.is_tensor(x1):
            return ((x1-x2)**2).sum().sqrt()
        else:
            return np.sqrt(np.sum((x1-x2)**2))

    def moment_diff(sx1, sx2, k):
        if torch.is_tensor(sx1):
            ss1 = (sx1**k).mean(dim=0)
            ss2 = (sx2**k).mean(dim=0)
        else:
            ss1 = np.mean(sx1**k, axis=0)
            ss2 = np.mean(sx2**k, axis=0)
        return l2diff(ss1, ss2)

    if torch.is_tensor(x1):
        x1_mean = x1.mean(dim=0)
        x2_mean = x2.mean(dim=0)
    else:
        x1_mean = np.mean(x1, axis=0)
        x2_mean = np.mean(x2, axis=0)

    sx1 = x1 - x1_mean
    sx2 = x2 - x2_mean
    dm = l2diff(x1_mean, x2_mean)
    scms = dm
    for i in range(n_moments-1):
        # moment diff of centralized samples
        scms += moment_diff(sx1, sx2, i+2)
    return float(scms)


def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys if c.struct_fp is not None]
    comp_fps = [c.comp_fp for c in crys if c.comp_fp is not None]
    gt_struc_fps = [c.struct_fp for c in gt_crys if c.struct_fp is not None]
    gt_comp_fps = [c.comp_fp for c in gt_crys if c.comp_fp is not None]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fingerprints(struc_fps, comp_fps)

    comp_fps = CompositionScaler.transform(comp_fps)
    gt_comp_fps = CompositionScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict


def filter_fingerprints(struc_fps, comp_fps):
    """Remove fingerprints that are None"""
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps


# https://gist.github.com/tsvikas/5f859a484e53d4ef93400751d0a116de
class ParallelTqdm(Parallel):
    """joblib.Parallel, but with a tqdm progressbar

    Additional parameters:
    ----------------------
    total_tasks: int, default: None
        the number of expected jobs. Used in the tqdm progressbar.
        If None, try to infer from the length of the called iterator, and
        fallback to use the number of remaining items as soon as we finish
        dispatching.
        Note: use a list instead of an iterator if you want the total_tasks
        to be inferred from its length.

    desc: str, default: None
        the description used in the tqdm progressbar.

    disable_progressbar: bool, default: False
        If True, a tqdm progressbar is not used.

    show_joblib_header: bool, default: False
        If True, show joblib header before the progressbar.

    Removed parameters:
    -------------------
    verbose: will be ignored


    Usage:
    ------
    >>> from joblib import delayed
    >>> from time import sleep
    >>> ParallelTqdm(n_jobs=-1)([delayed(sleep)(.1) for _ in range(10)])
    80%|████████  | 8/10 [00:02<00:00,  3.12tasks/s]

    """

    def __init__(
        self,
        *,
        total_tasks: Union[int, None] = None,
        desc: Union[str, None] = None,
        disable_progressbar: bool = False,
        show_joblib_header: bool = False,
        **kwargs,
    ):
        if "verbose" in kwargs:
            raise ValueError(
                "verbose is not supported. "
                "Use show_progressbar and show_joblib_header instead."
            )
        super().__init__(verbose=(1 if show_joblib_header else 0), **kwargs)
        self.total_tasks = total_tasks
        self.desc = desc
        self.disable_progressbar = disable_progressbar
        self.progress_bar: Union[tqdm, None] = None

    def __call__(self, iterable):
        try:
            if self.total_tasks is None:
                # try to infer total_tasks from the length of the called iterator
                try:
                    self.total_tasks = len(iterable)
                except (TypeError, AttributeError):
                    pass
            # call parent function
            return super().__call__(iterable)
        finally:
            # close tqdm progress bar
            if self.progress_bar is not None:
                self.progress_bar.close()

    __call__.__doc__ = Parallel.__call__.__doc__

    def dispatch_one_batch(self, iterator):
        # start progress_bar, if not started yet.
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                desc=self.desc,
                total=self.total_tasks,
                disable=self.disable_progressbar,
                unit="tasks",
            )
        # call parent function
        return super().dispatch_one_batch(iterator)

    dispatch_one_batch.__doc__ = Parallel.dispatch_one_batch.__doc__

    def print_progress(self):
        """Display the process of the parallel execution using tqdm"""
        # if we finish dispatching, find total_tasks from the number of remaining items
        if self.total_tasks is None and self._original_iterator is None:
            self.total_tasks = self.n_dispatched_tasks
            self.progress_bar.total = self.total_tasks
            self.progress_bar.refresh()
        # update progressbar
        self.progress_bar.update(self.n_completed_tasks - self.progress_bar.n)


def joblib_map(
    func: callable,
    iterable: Iterable,
    n_jobs: int = 1,
    inner_max_num_threads: Union[int, None] = None,
    desc: Union[str, None] = None,
    total: Union[int, None] = None,
    backend: Literal["sequential", "loky", "threading", "multiprocessing"] = "loky",
) -> list:
    """
    n_jobs:
        The maximum number of concurrently running jobs, such as the number
        of Python worker processes when ``backend="loky"`` or the size of
        the thread-pool when ``backend="threading"``.
        This argument is converted to an integer, rounded below for float.
        If -1 is given, `joblib` tries to use all CPUs. The number of CPUs
        ``n_cpus`` is obtained with :func:`~cpu_count`.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. For instance,
        using ``n_jobs=-2`` will result in all CPUs but one being used.
        This argument can also go above ``n_cpus``, which will cause
        oversubscription. In some cases, slight oversubscription can be
        beneficial, e.g., for tasks with large I/O operations.
        If 1 is given, no parallel computing code is used at all, and the
        behavior amounts to a simple python `for` loop. This mode is not
        compatible with ``timeout``.
        None is a marker for 'unset' that will be interpreted as n_jobs=1
        unless the call is performed under a :func:`~parallel_config`
        context manager that sets another value for ``n_jobs``.
        If n_jobs = 0 then a ValueError is raised.
    inner_max_num_threads:
        If not None, overwrites the limit set on the number of threads
        usable in some third-party library threadpools like OpenBLAS,
        MKL or OpenMP. This is only used with the ``loky`` backend.
    """
    if backend != "loky" and inner_max_num_threads is not None:
        print(f"{backend=} does not support {inner_max_num_threads=}, setting to None.")
        inner_max_num_threads = None

    if backend != "sequential":
        with parallel_config(
            backend=backend, inner_max_num_threads=inner_max_num_threads
        ):
            if backend == "loky":
                results = ParallelTqdm(n_jobs=n_jobs, total_tasks=total, desc=desc)(
                    delayed(func)(i) for i in iterable
                )
            else:
                results = Parallel(n_jobs=n_jobs)(delayed(func)(i) for i in iterable)
    else:
        results = [func(i) for i in tqdm(iterable, desc=desc, total=total)]
    return results


CompScalerMeans = [
    21.194441759304013,
    58.20212663122281,
    37.0076848719188,
    36.52738520455582,
    13.350626389725019,
    29.468922184630255,
    28.71735137747704,
    78.8868535524408,
    50.16950217496375,
    59.56764743604155,
    19.020429484306277,
    61.335572740454325,
    47.14515893344343,
    141.75135923307818,
    94.60620029962553,
    85.95794070476977,
    34.07300576173523,
    68.06189371516912,
    637.9862061297893,
    1817.2394155466848,
    1179.2532094169414,
    1127.2743149568837,
    431.51034284549826,
    909.1060025135899,
    3.7744320927984534,
    13.673707104881585,
    9.899275012083132,
    9.620186927095652,
    3.8426065581251856,
    9.96950217496375,
    3.305461575640406,
    5.483035282745288,
    2.1775737071048815,
    4.215114560306594,
    0.8206087101824266,
    3.732092798453359,
    109.16732721121315,
    179.5570323827936,
    70.38970517158047,
    136.0978305229613,
    27.027545809538527,
    119.16713388110198,
    1.2721433060967857,
    2.4614001837260617,
    1.1892568776289631,
    1.9844483610247092,
    0.4691462290494881,
    2.100143582306204,
    1.4829869502174964,
    1.9899951667472209,
    0.5070082165297245,
    1.7956250375970633,
    0.2056251946617602,
    1.745867568873852,
    0.05650072498791687,
    2.3618656355727405,
    2.3053649105848235,
    1.2829636137262992,
    0.9995555685850794,
    1.5150314161430642,
    0.7731271145480909,
    7.4648139197680035,
    6.691686805219913,
    4.010677272036105,
    2.612307566507693,
    3.303528274528758,
    0.2739487675205413,
    5.889753504108265,
    5.615804736587724,
    2.3244356612494683,
    2.1426251769710905,
    1.4464475592073465,
    4.739246012566457,
    14.578395360077332,
    9.839149347510874,
    9.413701584608935,
    3.537059747455868,
    8.550410826486225,
    0.008119864668922184,
    0.43286611889801835,
    0.4247462542290962,
    0.16687837041055423,
    0.17139889490813626,
    0.10898985016916385,
    0.06283228612856452,
    2.6573707104881583,
    2.594538424359594,
    1.219602938224228,
    1.0596390454742999,
    1.1120831319478008,
    0.14842919284678588,
    3.8473658772353794,
    3.6989366843885936,
    1.4541605082183982,
    1.3862277372859781,
    0.8018849685838569,
    0.03542774287095215,
    2.4474625422909617,
    2.4120347994200095,
    0.7745217539010397,
    0.9145812330586208,
    0.3198646689221846,
    1.552730787820203,
    6.910681488641856,
    5.357950700821653,
    3.615163570754227,
    1.9072256165179793,
    2.6702271628806185,
    14.608536589568727,
    34.83222477045747,
    20.223688180890715,
    22.47901710732293,
    7.17674504190757,
    18.641837024143584,
    0.009066988883518605,
    0.9185191396809959,
    0.9094521507974755,
    0.4368550481994018,
    0.38905942883427047,
    0.48375558240695804,
    0.0012985909686158003,
    0.21708593995837092,
    0.21578734898975546,
    0.08167977375391729,
    0.08155386250705281,
    0.06036340747305611,
    116.32010633156113,
    217.5905751570807,
    101.27046882551957,
    162.87154200548844,
    41.920624308665566,
    136.4664572257129]
CompScalerStds = [
    16.35781741152948,
    20.189540126474725,
    20.516298414514758,
    16.816765336550194,
    7.966591328222124,
    22.270791076753067,
    21.802116630115243,
    12.804546460581966,
    24.756629388687983,
    13.930306216047477,
    10.214535652334533,
    27.801612936980938,
    39.74031558353379,
    54.269739685575814,
    53.70466607591569,
    42.852342044453444,
    20.78341194242935,
    56.28783510219931,
    563.8004405882157,
    732.0722574247563,
    736.2122907972664,
    606.351603075103,
    272.62646060896407,
    810.6156779688841,
    3.0362262146833428,
    3.2075174256751606,
    4.0633818989245665,
    2.9738244769894764,
    1.7805586029644034,
    5.643243225066782,
    1.1994336274579853,
    0.8939013979423364,
    1.2297581799896975,
    1.0066021334519983,
    0.49129747526397105,
    1.4159553146070951,
    31.754756468836774,
    28.054241463256226,
    38.16336054795611,
    25.83485338379922,
    15.388376641904662,
    39.67137484594156,
    0.31988340032011076,
    0.6833658037760536,
    0.7464197945553585,
    0.4881349085029781,
    0.3176591553643101,
    0.8601748146737138,
    0.5864801661863596,
    0.10048913710210677,
    0.5836289120986499,
    0.2811748167435902,
    0.2468696279341553,
    0.5007375747433073,
    0.37237566669029587,
    1.7235989187720187,
    1.7058836077743305,
    1.1558859351244697,
    0.7677842566598179,
    1.9203550253462733,
    2.1289400248865182,
    3.5326064169848332,
    3.708508303762512,
    2.8709941136664567,
    1.6110681295257014,
    4.310192504023775,
    1.6644182118209292,
    6.228287671164213,
    6.1200848808512305,
    3.1986202996110302,
    2.4492978142248867,
    4.030497343977163,
    3.662028270049814,
    6.8192125550358345,
    6.614243783887738,
    4.334987449618594,
    2.568319610320196,
    5.9494890200106925,
    0.08974370432893491,
    0.4954725441517777,
    0.494304434278516,
    0.2309340434963803,
    0.2072873961103969,
    0.31162647950590266,
    0.39805702757060923,
    1.8111691089355726,
    1.7973395144505941,
    0.9486995373104102,
    0.7538753151875139,
    1.5233177017753785,
    0.7952606701778913,
    3.711190225170556,
    3.638721437232604,
    1.7171165424006831,
    1.4307904413917036,
    2.1047820817622904,
    0.49193748323158065,
    4.064840532426175,
    4.035286619587313,
    1.4858577214526643,
    1.5799117659864677,
    1.6130080156145745,
    1.555249156140194,
    4.776932951077492,
    4.569790780459629,
    2.224617778217326,
    1.7217507416156546,
    2.5969733650703763,
    7.215001918238936,
    19.252513469778584,
    18.775394044177858,
    9.447222764774764,
    6.7467931836261235,
    11.106825644766616,
    0.27206794253092115,
    1.6449321034573106,
    1.6236282792648686,
    0.8506917026741503,
    0.7020945355184042,
    1.2281895279350408,
    0.04134438177238229,
    0.5508855867341717,
    0.5486095551438679,
    0.24239297524046477,
    0.2127779137935831,
    0.3036750942874694,
    80.06063945615361,
    21.345794811194104,
    80.16475677581042,
    52.58533928558554,
    35.40836791039412,
    85.980205895116
]
CompositionScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.
)

from typing import Any, Iterable, List, Tuple, Set, Dict, Union, Mapping, Optional
from collections import defaultdict
from hashlib import blake2b
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger
from tqdm.auto import tqdm

RDLogger.DisableLog("rdApp.*")


class NoReactionError(Exception):
    """Raised when the encoder attempts to encode a non-reaction SMILES.

    Attributes:
        message: a message containing the non-reaction SMILES
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DrfpEncoder:
    """A class for encoding SMILES as drfp fingerprints."""

    @staticmethod
    def shingling_from_mol(
        in_mol: Mol, radius: int = 3, rings: bool = True, min_radius: int = 0
    ) -> List[str]:
        """Creates a molecular shingling from a RDKit molecule (rdkit.Chem.rdchem.Mol).

        Arguments:
            in_mol: A RDKit molecule instance
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            rings: Whether or not to include rings in the shingling
            min_radius: The minimum radius that is used to extract n-grams

        Returns:
            The molecular shingling.
        """

        shingling = []

        if rings:
            for ring in AllChem.GetSymmSSSR(in_mol):
                bonds = set()
                ring = list(ring)
                for i in ring:
                    for j in ring:
                        if i != j:
                            bond = in_mol.GetBondBetweenAtoms(i, j)
                            if bond is not None:
                                bonds.add(bond.GetIdx())
                shingling.append(
                    AllChem.MolToSmiles(
                        AllChem.PathToSubmol(in_mol, list(bonds)),
                        canonical=True,
                        allHsExplicit=True,
                    ).encode("utf-8")
                )

        if min_radius == 0:
            for i, atom in enumerate(in_mol.GetAtoms()):
                shingling.append(atom.GetSmarts().encode("utf-8"))

        for index, _ in enumerate(in_mol.GetAtoms()):
            for i in range(1, radius + 1):
                p = AllChem.FindAtomEnvironmentOfRadiusN(in_mol, i, index)
                amap = {}
                submol = AllChem.PathToSubmol(in_mol, p, atomMap=amap)

                if index not in amap:
                    continue

                smiles = AllChem.MolToSmiles(
                    submol,
                    rootedAtAtom=amap[index],
                    canonical=True,
                    allHsExplicit=True,
                )

                if smiles != "":
                    shingling.append(smiles.encode("utf-8"))

        # Set ensures that the same shingle is not hashed multiple times
        # (which would not change the hash, since there would be no new minima)
        return list(set(shingling))

    @staticmethod
    def internal_encode(
        in_smiles: str,
        radius: int = 3,
        min_radius: int = 0,
        rings: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates an drfp array from a reaction SMILES string.

        Arguments:
            in_smiles: A valid reaction SMILES string
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            min_radius: The minimum radius that is used to extract n-grams
            rings: Whether or not to include rings in the shingling

        Returns:
            A tuple with two arrays, the first containing the drfp hash values, the second the substructure SMILES
        """

        sides = in_smiles.split(">")
        if len(sides) < 3:
            raise NoReactionError(
                f"The following is not a valid reaction SMILES: '{in_smiles}'"
            )

        if len(sides[1]) > 0:
            sides[0] += "." + sides[1]

        left = sides[0].split(".")
        right = sides[2].split(".")

        left_shingles = set()
        right_shingles = set()

        for l in left:
            mol = AllChem.MolFromSmiles(l)

            if not mol:
                continue

            sh = DrfpEncoder.shingling_from_mol(
                mol,
                radius=radius,
                rings=rings,
                min_radius=min_radius,
            )

            for s in sh:
                right_shingles.add(s)

        for r in right:
            mol = AllChem.MolFromSmiles(r)

            if not mol:
                continue

            sh = DrfpEncoder.shingling_from_mol(
                mol,
                radius=radius,
                rings=rings,
                min_radius=min_radius,
            )

            for s in sh:
                left_shingles.add(s)

        s = right_shingles.symmetric_difference(left_shingles)

        if len(s) == 0:
            s = left_shingles

        return DrfpEncoder.hash(list(s)), list(s)

    @staticmethod
    def hash(shingling: List[str]) -> np.ndarray:
        """Directly hash all the SMILES in a shingling to a 32-bit integerself.

        Arguments:
            shingling: A list of n-grams

        Returns:
            A list of hashed n-grams
        """

        hash_values = []

        for t in shingling:
            hash_values.append(int(blake2b(t, digest_size=4).hexdigest(), 16))

        return np.array(hash_values, dtype=np.int32)

    @staticmethod
    def fold(
        hash_values: np.ndarray, length: int = 2048
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Folds the hash values to a binary vector of a given length.

        Arguments:
            hash_value: An array containing the hash values
            length: The length of the folded fingerprint

        Returns:
            A tuple containing the folded fingerprint and the indices of the on bits
        """

        folded = np.zeros(length, dtype=np.uint8)
        on_bits = hash_values % length
        folded[on_bits] = 1

        return folded, on_bits

    @staticmethod
    def encode(
        X: Union[Iterable, str],
        n_folded_length: int = 2048,
        min_radius: int = 0,
        radius: int = 3,
        rings: bool = True,
        mapping: bool = False,
        use_tqdm: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], Dict[int, Set[str]]]]:
        """Encodes a list of reaction SMILES using the drfp fingerprint.

        Args:
            X: An iterable (e.g. List) of reaction SMILES or a single reaction SMILES to be encoded
            n_folded_length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            radius: The maximum radius of a substructure
            rings: Whether to include full rings as substructures
            mapping: Return a feature to substructure mapping in addition to the fingerprints
            use_tqdm: Should a progress bar be displayed while encoding the given reactions?
            tqdm_kwargs: If using :mod:`tqdm`, override default kwargs.

        Returns:
            A list of drfp fingerprints or, if mapping is enabled, a tuple containing a list of drfp fingerprints and a mapping dict.
        """
        if isinstance(X, str):
            X = [X]

        _tqdm_kwargs = dict(
            unit_scale=True, 
            unit="reaction", 
            desc="encoding reactions", 
            disable=not use_tqdm,
        )
        if tqdm_kwargs:
            _tqdm_kwargs.update(tqdm_kwargs)

        result = []
        result_map = defaultdict(set)
        for x in tqdm(X, **_tqdm_kwargs):
            hashed_diff, smiles_diff = DrfpEncoder.internal_encode(
                x, min_radius=min_radius, radius=radius, rings=rings
            )

            difference_folded, on_bits = DrfpEncoder.fold(
                hashed_diff,
                length=n_folded_length,
            )

            if mapping:
                for unfolded_index, folded_index in enumerate(on_bits):
                    result_map[folded_index].add(
                        smiles_diff[unfolded_index].decode("utf-8")
                    )

            result.append(difference_folded)

        if mapping:
            return result, result_map
        else:
            return result

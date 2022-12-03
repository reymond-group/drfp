import pytest
import numpy as np
from drfp import DrfpEncoder
from rdkit.Chem.AllChem import MolFromSmiles
from rdkit.Chem.rdchem import Mol

__author__ = "Daniel Probst"
__license__ = "MIT"


@pytest.fixture
def rxn() -> str:
    return "C=Cc1ccccn1.Cc1ccccn1.O.Oc1ccc(O)cc1.[Na]>>c1ccc(CCCc2ccccn2)nc1"


@pytest.fixture
def rxn_with_reagent() -> str:
    return "C=Cc1ccccn1.Cc1ccccn1.O.Oc1ccc(O)cc1>[Na]>c1ccc(CCCc2ccccn2)nc1"


@pytest.fixture
def mol() -> Mol:
    return MolFromSmiles("C=Cc1ccccn1")


def test_shingling_from_mol(mol):
    shingling = DrfpEncoder.shingling_from_mol(mol)
    # The n-grams in the shingling are unordered
    sorted(shingling) == [
        b"C",
        b"[CH2]=[CH2]",
        b"[CH2]=[CH][cH3]",
        b"[CH2]=[CH][c]([cH2])[nH]",
        b"[CH](=[CH2])[cH3]",
        b"[CH](=[CH2])[c]([cH2])[nH]",
        b"[CH](=[CH2])[c]([cH][cH2])[n][cH2]",
        b"[cH]([cH2])[cH2]",
        b"[cH]([cH2])[nH]",
        b"[cH]([cH][cH2])[cH][cH2]",
        b"[cH]([cH][cH2])[cH][nH]",
        b"[cH]([cH][cH2])[c]([CH3])[nH]",
        b"[cH]([cH][cH2])[n][cH2]",
        b"[cH]1[cH][cH][cH][c]([CH3])[n]1",
        b"[cH]1[cH][cH][cH][n][cH]1",
        b"[cH]1[cH][cH][cH][n][c]1[CH]=[CH2]",
        b"[cH]1[cH][cH][n][cH][cH]1",
        b"[cH]1[cH][cH][n][c]([CH3])[cH]1",
        b"[c]([CH]=[CH2])([cH][cH2])[n][cH2]",
        b"[c]([cH2])([CH3])[nH]",
        b"[c]1([CH]=[CH2])[cH][cH][cH][cH][n]1",
        b"[n]([cH2])[cH2]",
        b"[n]([cH][cH2])[c]([cH2])[CH3]",
        b"[n]1[cH][cH][cH][cH][c]1[CH]=[CH2]",
        b"c",
        b"n",
    ]


def test_internal_encode(rxn, rxn_with_reagent):
    fp, ngrams = DrfpEncoder.internal_encode(rxn)
    fp_2, ngrams_2 = DrfpEncoder.internal_encode(rxn_with_reagent)

    assert list(np.sort(fp)) == list(np.sort(fp_2))
    assert list(np.sort(fp)) == [
        -1876452362,
        -1809832425,
        -1391908306,
        -1283681952,
        -1111805688,
        -1060185789,
        -1015671755,
        -1010266043,
        -1005875332,
        -938670376,
        -872170529,
        -822153746,
        -804611022,
        -594678541,
        -519888274,
        -449525048,
        -371351030,
        -300125176,
        -136930381,
        -44607364,
        -4328230,
        -430849,
        151290598,
        252134860,
        467079089,
        602550852,
        660419725,
        706432844,
        820058193,
        870463523,
        1129042833,
        1222858596,
        1351818365,
        1410321859,
        1415739784,
        1490864970,
        1503035893,
        1619269640,
        1862109967,
    ]

    assert sorted(ngrams) == [
        b"O",
        b"[CH2]([CH2][CH2][cH3])[c]([cH][cH2])[n][cH2]",
        b"[CH2]([CH2][CH3])[c]([cH2])[nH]",
        b"[CH2]([CH2][cH3])[CH2][cH3]",
        b"[CH2]([CH2][c]([cH2])[nH])[CH2][c]([cH2])[nH]",
        b"[CH2]([CH3])[CH3]",
        b"[CH2]([CH3])[cH3]",
        b"[CH2]([cH3])[CH3]",
        b"[CH2]=[CH2]",
        b"[CH2]=[CH][cH3]",
        b"[CH2]=[CH][c]([cH2])[nH]",
        b"[CH3][cH3]",
        b"[CH3][c]([cH2])[nH]",
        b"[CH3][c]([cH][cH2])[n][cH2]",
        b"[CH](=[CH2])[cH3]",
        b"[CH](=[CH2])[c]([cH2])[nH]",
        b"[CH](=[CH2])[c]([cH][cH2])[n][cH2]",
        b"[Na]",
        b"[OH][cH3]",
        b"[OH][c]([cH2])[cH2]",
        b"[OH][c]([cH][cH2])[cH][cH2]",
        b"[cH]([cH][cH2])[c]([cH2])[OH]",
        b"[cH]1[cH][cH][cH][cH][cH]1",
        b"[cH]1[cH][cH][cH][n][c]1[CH2][CH3]",
        b"[cH]1[cH][cH][cH][n][c]1[CH3]",
        b"[cH]1[cH][cH][cH][n][c]1[CH]=[CH2]",
        b"[cH]1[cH][c]([OH])[cH][cH][c]1[OH]",
        b"[c]([CH3])([cH][cH2])[n][cH2]",
        b"[c]([CH]=[CH2])([cH][cH2])[n][cH2]",
        b"[c]([OH])([cH][cH2])[cH][cH2]",
        b"[c]([cH2])([cH2])[OH]",
        b"[c]([cH][cH2])([CH2][CH3])[n][cH2]",
        b"[c]1([CH2][CH2][CH3])[cH][cH][cH][cH][n]1",
        b"[c]1([CH3])[cH][cH][cH][cH][n]1",
        b"[c]1([CH]=[CH2])[cH][cH][cH][cH][n]1",
        b"[c]1([OH])[cH][cH][cH][cH][cH]1",
        b"[n]1[cH][cH][cH][cH][c]1[CH2][CH3]",
        b"[n]1[cH][cH][cH][cH][c]1[CH3]",
        b"[n]1[cH][cH][cH][cH][c]1[CH]=[CH2]",
    ]

    assert sorted(ngrams) == sorted(ngrams_2)


def test_hash():
    assert list(DrfpEncoder.hash([b"hello", b"world"])) == [-362346761, -2134290508]


def test_fold():
    fp, on_bits = DrfpEncoder.fold(np.array([6, 22, 42, 69]), 8)
    assert list(fp) == [
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
    ]
    assert list(on_bits) == [6, 6, 2, 5]


def test_encode(rxn):
    fp = DrfpEncoder.encode(rxn)
    assert len(fp) == 1
    assert len(fp[0]) == 2048

    fp = DrfpEncoder.encode([rxn, rxn])
    assert len(fp) == 2


def test_encode_mapping(rxn):
    fp, mapping = DrfpEncoder.encode(rxn, mapping=True)
    assert len(fp) == 1
    assert len(fp[0]) == 2048
    assert len(mapping) == 39

    fp, mapping = DrfpEncoder.encode([rxn, rxn], mapping=True)
    assert len(fp) == 2
    assert len(mapping) == 39


def test_encode_atom_index_mapping(rxn):
    fp, _, aidx_mapping = DrfpEncoder.encode(rxn, atom_index_mapping=True)
    assert len(fp) == 1
    assert len(fp[0]) == 2048
    assert len(aidx_mapping) == 1
    assert len(aidx_mapping[0]) == 2
    assert len(aidx_mapping[0]["reactants"]) == 5

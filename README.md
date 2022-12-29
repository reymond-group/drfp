![test workflow](https://github.com/reymond-group/drfp/actions/workflows/tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5268144.svg)](https://doi.org/10.5281/zenodo.5268144)


# DRFP


An NLP-inspired chemical reaction fingerprint based on basic set arithmetic.

Read the associated [open access article](https://pubs.rsc.org/en/content/articlehtml/2022/dd/d1dd00006c)


## Description

Predicting the nature and outcome of reactions using computational methods is an important tool to accelerate chemical research. The recent application of deep learning-based learned fingerprints to reaction classification and reaction yield prediction has shown an impressive increase in performance compared to previous methods such as DFT- and structure-based fingerprints. However, learned fingerprints require large training data sets, are inherently biased, and are based on complex deep learning architectures. Here we present the differential reaction fingerprint *DRFP*. The *DRFP* algorithm takes a reaction SMILES as an input and creates a binary fingerprint based on the symmetric difference of two sets containing the circular molecular n-grams generated from the molecules listed left and right from the reaction arrow, respectively, without the need for distinguishing between reactants and reagents. We show that *DRFP* outperforms DFT-based fingerprints in reaction yield prediction and other structure-based fingerprints in reaction classification, and reaching the performance of state-of-the-art learned fingerprints in both tasks while being data-independent.

## Getting Started
The best way to start exploring DRFP is on binder.
A notebook that gets you started on creating and using DRFP:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/reymond-group/drfp/HEAD?filepath=notebooks%2F01_fingerprinting.ipynb)

A notbook that explains how you can use SHAP to analyse and interpret your machine learning models when using DRFP:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/reymond-group/drfp/HEAD?filepath=notebooks%2F02_model_explainability.ipynb)

## Installation and Usage
*DRFP* can be installed from pypi using `pip install drfp`. However, it depends on [RDKit](https://www.rdkit.org/) which is best [installed using conda](https://www.rdkit.org/docs/Install.html).

Once DRFP is installed, there are two ways you can use it. You can use the cli app `drfp` or the library provided by the package.

### CLI
```bash
drfp my_rxn_smiles.txt my_rxn_fps.pkl -d 512
```

This will create a pickle dump containing an numpy ndarray containing DRFP fingerprints with a dimensionality of 512. To also export the mapping, use the flag `--mapping`. This will create the additional file `my_rxn_fps.map.pkl`. You can call `drfp --help` to show all available flags and options.

### Library
Following is a basic exmple of how to use DRFP in a Python script.
```python
from drfp import DrfpEncoder

rxn_smiles = [
    "CO.O[C@@H]1CCNC1.[C-]#[N+]CC(=O)OC>>[C-]#[N+]CC(=O)N1CC[C@@H](O)C1",
    "CCOC(=O)C(CC)c1cccnc1.Cl.O>>CCC(C(=O)O)c1cccnc1",
]

fps = DrfpEncoder.encode(rxn_smiles)
```

The variable `fps` now points to a list containing the fingerprints for the two reaction SMILES as numpy arrays.


## Documentation

The library contains the class `DrfpEncoder` with one public method `encode`.

| `DrfpEncoder.encode()` | Description | Type | Default |
|-|-|-|-|
| `X` | An iterable (e.g. a list) of reaction SMILES or a single reaction SMILES to be encoded | `Iterable` or `str` |  |
| `n_folded_length` | The folded length of the fingerprint (the parameter for the modulo hashing) | `int` | `2048` |
| `min_radius` | The minimum radius of a substructure (0 includes single atoms) | `int` | `0` |
| `radius` | The maximum radius of a substructure | `int` | `3` |
| `rings` | Whether to include full rings as substructures | `bool` | `True` |
| `mapping` |  Return a feature to substructure mapping in addition to the fingerprints. If true, the return signature of this method is `Tuple[List[np.ndarray], Dict[int, Set[str]]]` | `bool` | `False` |
| `atom_index_mapping` | Return the atom indices of mapped substructures for each reaction | `bool` | `False` |
| `root_central_atom` | Whether to root the central atom of substructures when generating SMILES | `bool` | `True` |
| `include_hydrogens` | Whether to explicitly include hydrogens in the molecular graph | `bool` | `False` |
| `show_progress_bar` | Whether to show a progress bar when encoding reactions | `bool` | `False` |

# Reproduce
Want to reproduce the results in our paper? You can find all the data in the `data` folder and encoding and training scripts in the `scripts` folder.

# Cite Us
```
@article{probst2022reaction,
  title={Reaction Classification and Yield Prediction using the Differential Reaction Fingerprint DRFP},
  author={Probst, Daniel and Schwaller, Philippe and Reymond, Jean-Louis},
  journal={Digital Discovery},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```

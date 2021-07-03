====
drfp
====


An NLP-inspired chemical reaction fingerprint based on basic set arithmetic.


Description
===========

Predicting the nature and outcome of reactions using computational methods is an important tool to accelerate chemical research. The recent application of deep learning-based learned fingerprints to reaction classification and reaction yield prediction has shown an impressive increase in performance compared to previous methods such as DFT- and structure-based fingerprints. However, learned fingerprints require large training data sets, are inherently biased, and are based on complex deep learning architectures. Here we present the differential reaction fingerprint \textit{DRFP}. The \textit{DRFP} algorithm takes a reaction SMILES as an input and creates a binary fingerprint based on the symmetric difference of two sets containing the circular molecular n-grams generated from the molecules listed left and right from the reaction arrow, respectively, without the need for distinguishing between reactants and reagents. We show that \textit{DRFP} outperforms DFT-based fingerprints in reaction yield prediction and other structure-based fingerprints in reaction classification, and reaching the performance of state-of-the-art learned fingerprints in both tasks while being data-independent.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.

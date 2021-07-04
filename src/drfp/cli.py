import pickle
from typing import TextIO
import click
from drfp import DrfpEncoder


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("wb+"))
@click.option(
    "--n_folded_length",
    "-d",
    default=2048,
    help="The lenght / dimensionality of the fingerprint. Good values are between 128 and 2048.",
)
@click.option(
    "--min_radius",
    "-m",
    default=0,
    help="The minimum radius used to extract circular substructures from molecules. 0 includes single atoms",
)
@click.option(
    "--radius",
    "-r",
    default=3,
    help="The radius, or maximum radius used to extract circular substructures form molecules.",
)
@click.option(
    "--rings/--no-rings",
    default=True,
    help="Whether or not to extract whole rings as substructures.",
)
@click.option(
    "--mapping/--no-mapping",
    default=False,
    help="Whether or not to also export a mapping to help interpret the fingerprint.",
)
def main(
    input_file: TextIO,
    output_file: TextIO,
    n_folded_length: int,
    min_radius: int,
    radius: int,
    rings: True,
    mapping: False,
):
    """Creates fingerprints from a file containing one reaction SMILES per line.

    INPUT_FILE is the file containing one reaction SMILES per line.

    OUTPUT_FILE will be a pickle file containing the corresponding list of fingerprints. If mapping is chosen, an addition file with the suffix .map will be created.
    """
    smiles = []
    for line in input_file:
        smiles.append(line.strip())

    fps = None
    fragment_map = None

    if mapping:
        fps, fragment_map = DrfpEncoder.encode(
            smiles, n_folded_length, min_radius, radius, rings, mapping
        )
    else:
        fps = DrfpEncoder.encode(
            smiles, n_folded_length, min_radius, radius, rings, mapping
        )

    pickle.dump(fps, output_file)

    if mapping:
        filename_parts = output_file.name.split(".")
        filename_parts.insert(len(filename_parts) - 1, "map")
        with open(".".join(filename_parts), "wb+") as f:
            pickle.dump(fragment_map, f)


if __name__ == "__main__":
    main()

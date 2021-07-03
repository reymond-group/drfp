import pickle
from typing import TextIO
import click
from drfp import DrfpEncoder


@click.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("wb+"))
@click.option("--n_folded_length", "-d", default=2048)
@click.option("--min_radius", "-m", default=0)
@click.option("--radius", "-r", default=3)
@click.option("--rings/--no-rings", default=True)
@click.option("--mapping/--no-mapping", default=False)
def main(
    input_file: TextIO,
    output_file: TextIO,
    n_folded_length: int,
    min_radius: int,
    radius: int,
    rings: True,
    mapping: False,
):
    smiles = []
    for line in input_file:
        smiles.append(line.strip())

    fps, fragment_map = DrfpEncoder.encode_list(
        smiles, n_folded_length, min_radius, radius, rings, mapping
    )

    pickle.dump(fps, output_file)

    if mapping:
        with open(f"{output_file.name}.map", "wb+") as f:
            pickle.dump(fragment_map, f)


if __name__ == "__main__":
    main()

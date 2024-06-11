import numpy as np

from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import (
    MolFromXYZFile,
    MolToXYZFile,
    MolFromSmiles,
    MolToPDBFile,
)
from rdkit.Chem import AddHs

import MDAnalysis as mda
import MDAnalysis.transformations as transformations

import ase.io

import os, sys


def angle_between(vectorA: np.ndarray, vectorB: np.ndarray) -> float:
    """Returns the angle in degrees between two vectors vectorA and vectorB"""
    cosAngle = np.dot(vectorA, vectorB.T)[0][0] / (
        np.linalg.norm(vectorA) * np.linalg.norm(vectorB)
    )
    angleRad = np.arccos(np.clip(cosAngle, -1, 1))
    angleDeg = angleRad * 180 / np.pi
    return angleDeg


def latticeVectorsToParameters(vectors: list[list[float]]) -> list[float]:
    """Convert the three lattice vectors in cartesian coordinates to 6 lattice parameters (a, b, c, alpha, beta, gamma)"""
    xVec, yVec, zVec = np.split(vectors, 3, axis=0)
    a, b, c = np.linalg.norm(xVec), np.linalg.norm(yVec), np.linalg.norm(zVec)
    alpha = angle_between(yVec, zVec)
    beta = angle_between(xVec, zVec)
    gamma = angle_between(xVec, yVec)
    return [a, b, c, alpha, beta, gamma]


# THE BELOW FUNCTION WAS ADAPTED FROM https://gist.github.com/richardjgowers/b16b871259451e85af0bd2907d30de91
def make_universe(pbdFile: str, nDesiredAtoms: int):
    """Create a universe with the pdb file for connectivity, and a certain amount of desired atoms."""
    u = mda.Universe(pbdFile)
    nAtomsInOneMolecule = len(u.atoms)
    factor = nDesiredAtoms // nAtomsInOneMolecule
    unis = []
    for i in range(factor):
        u = mda.Universe(pbdFile)
        unis.append(u.atoms)
    new_u = mda.Merge(*unis)
    return new_u


def unwrapMolecule(
    input: str,
    output: str,
    smiles: str,
    latticeVectors: list[list[float]] | None = None,
) -> None:
    """Unwrap molecules within a unit cell such that individual molecules are not wrapped into the cell.

    Inputs:
        input: filepath to the file (readable by ASE) that is to be unwrapped
        output: filepath to the unwrapped XYZ file
        smiles: SMILES identifier of the molecule
        latticeVectors: list of three-dimensional lattice vectors
    """
    input_ase = ase.io.read(input)
    try:
        latticeParameters = ase.geometry.cell_to_cellpar(input_ase.get_cell())
    except:
        latticeParameters = latticeVectorsToParameters(latticeVectors)

    # Create a temporary pdb file from the smiles code (with explicit hydrogens)
    tmp_pdb = "tmp.pdb"
    mol = MolFromSmiles(smiles)
    mol = AddHs(mol)
    pdb = MolToPDBFile(mol, tmp_pdb)

    # Read coordinates in ase instance
    positions = input_ase.get_positions()
    nDesiredAtoms = len(positions)
    # mol_xyz = MolFromXYZFile(input)
    # nDesiredAtoms = mol_xyz.GetNumAtoms()

    # Set up new universe
    newU = make_universe(tmp_pdb, nDesiredAtoms)
    newU.atoms.positions = positions
    newU.dimensions = latticeParameters

    # Unwrap coordinates
    atoms = newU.atoms
    unwrap = transformations.unwrap(atoms)
    newU.trajectory.add_transformations(unwrap)

    # Write atoms to xyz file
    XYZwriter = mda.coordinates.XYZ.XYZWriter(output)
    XYZwriter.write(newU)
    XYZwriter.close()


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if os.path.isfile(sys.argv[3]):
        with open(sys.argv[3], "r") as file:
            smiles_code = file.read()
            smiles_code.strip()
    else:
        smiles_code = sys.argv[3]

    unwrapMolecule(input_file, output_file, smiles_code)

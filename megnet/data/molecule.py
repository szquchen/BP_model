"""
Tools for creating graph inputs from molecule data
"""

import os
import sys
import itertools
from typing import List
from functools import partial
from collections import deque
from multiprocessing import Pool

import numpy as np
from pymatgen.core.structure import Molecule
from pymatgen.core.periodic_table import Element
from pymatgen.io.babel import BabelMolAdaptor

from megnet.utils.general import fast_label_binarize
from megnet.data.graph import (StructureGraph, GaussianDistance,
                               BaseGraphBatchGenerator, GraphBatchGenerator)

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolTransforms import GetBondLength

'''
#original
try:
    import pybel
except ImportError:
    pybel = None

try:
    from rdkit import Chem
except ImportError:
    Chem = None
'''

__date__ = '12/01/2018'

# List of features to use by default for each atom
_ATOM_FEATURES = ['element', 'formal_charge', 'ring_sizes',
                  'hybridization', 'aromatic']

# List of features to use by default for each bond
_BOND_FEATURES = ['bond_type', 'same_ring', 'spatial_distance', 'graph_distance']

# List of elements in library to use by default
_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']


class SimpleMolGraph(StructureGraph):
    """
    Default using all atom pairs as bonds. The distance between atoms are used
    as bond features. By default the distance is expanded using a Gaussian
    expansion with centers at np.linspace(0, 4, 20) and width of 0.5
    """
    def __init__(self,
                 nn_strategy='AllAtomPairs',
                 atom_converter=None,
                 bond_converter=None
                 ):
        if bond_converter is None:
            bond_converter = GaussianDistance(np.linspace(0, 4, 20), 0.5)
        super().__init__(nn_strategy=nn_strategy, atom_converter=atom_converter,
                         bond_converter=bond_converter)


class MolecularGraph(StructureGraph):
    """Class for generating the graph inputs from a molecule

    Computes many different features for the atoms and bonds in a molecule, and prepares them
    in a form compatible with MEGNet models. The :meth:`convert` method takes a OpenBabel molecule
    and, besides computing features, also encodes them in a form compatible with machine learning.
    Namely, the `convert` method one-hot encodes categorical variables and concatenates
    the atomic features

    ## Atomic Features

    This class can compute the following features for each atom

    - `atomic_num`: The atomic number
    - `element`: (categorical) Element identity. (Unlike `atomic_num`, element is one-hot-encoded)
    - `formal_charge`: Formal charge of the atom
    - `ring_sizes`: For rings with 9 or fewer atoms, how many unique rings
    of each size include this atom
    - `hybridization`: (categorical) Hybridization of atom: sp, sp2, sp3, sq.
    planer, trig, octahedral, or hydrogen
    - `aromatic`: (boolean) Whether the atom is part of an aromatic system

    ## Atom Pair Features

    The class also computes features for each pair of atoms

    - `bond_type`: (categorical) Whether the pair are unbonded, or in a single, double, triple, or aromatic bond
    - `same_ring`: (boolean) Whether the atoms are in the same aromatic ring
    - `graph_distance`: Distance of shortest path between atoms on the bonding graph
    - `spatial_distance`: Euclidean distance between the atoms. By default, this distance is expanded into
        a vector of 20 different values computed using the `GaussianDistance` converter

    """
    def __init__(self, atom_features=None, bond_features=None, distance_converter=None,
                 known_elements=None, cutoff=5):
        """
        Args:
            atom_features ([str]): List of atom features to compute
            bond_features ([str]): List of bond features to compute
            distance_converter (DistanceCovertor): Tool used to expand distances
                from a single scalar vector to an array of values
            known_elements ([str]): List of elements expected to be in dataset. Used only if the
                feature `element` is used to describe each atom
        """

        super().__init__()
        if bond_features is None:
            bond_features = _BOND_FEATURES
        if atom_features is None:
            atom_features = _ATOM_FEATURES
        if distance_converter is None:
            distance_converter = GaussianDistance(np.linspace(0, 4, 20), 0.5)
        if known_elements is None:
            known_elements = _ELEMENTS

        # Check if all feature names are valid
        if any(i not in _ATOM_FEATURES for i in atom_features):
            bad_features = set(atom_features).difference(_ATOM_FEATURES)
            raise ValueError('Unrecognized atom features: {}'.format(', '.join(bad_features)))
        self.atom_features = atom_features
        if any(i not in _BOND_FEATURES for i in bond_features):
            bad_features = set(bond_features).difference(_BOND_FEATURES)
            raise ValueError('Unrecognized bond features: {}'.format(', '.join(bad_features)))
        self.bond_features = bond_features
        self.known_elements = known_elements
        self.distance_converter = distance_converter
        self.cutoff = cutoff

    def convert(self, mol, state_attributes=None):
        """
        Compute the representation for a molecule

        Args:
            mol (rdkit.chem.rdchem.Mol): Molecule to generate features for
            state_attributes (list): State attributes. Uses average mass and number of bonds per atom as default
        Returns:
            (dict): Dictionary of features
        """

        # Get the features for all atoms and bonds
        atom_features = self.get_atom_feature(mol)

        atom_pairs = []
        num_atoms = mol.GetNumAtoms()  # Chen Qu
        for i, j in itertools.combinations(range(0, num_atoms), 2):
            bond_feature = self.get_pair_feature(mol, i, j)
            if bond_feature:
                atom_pairs.append(bond_feature)
            else:
                continue

        # Compute the graph distance, if desired
        if 'graph_distance' in self.bond_features:
            graph_dist = self._dijkstra_distance(atom_pairs)
            for i in atom_pairs:
                i.update({'graph_distance': graph_dist[i['a_idx'], i['b_idx']]})

        new_atom_pairs = []
        for pair in atom_pairs:
            if pair["graph_distance"] <= self.cutoff:
                new_atom_pairs.append(pair)
             
        # Generate the state attributes (that describe the whole network)
        state_attributes = state_attributes or [
            [num_atoms / 20.0, MolWt(mol) / num_atoms / 12.0,
             len([i for i in atom_pairs if i['bond_type'] > 0]) / num_atoms]
        ]

        # Get the atom features in the order they are requested by the user as a 2D array
        atoms = []
        for atom in atom_features:
            atoms.append(self._create_atom_feature_vector(atom))

        # Get the bond features in the order request by the user
        bonds = []
        index1_temp = []
        index2_temp = []
        for bond in new_atom_pairs:
            # Store the index of each bond
            index1_temp.append(bond.pop('a_idx'))
            index2_temp.append(bond.pop('b_idx'))

            # Get the desired bond features
            bonds.append(self._create_pair_feature_vector(bond))

        # Given the bonds (i,j), make it so (i,j) == (j, i)
        index1 = index1_temp + index2_temp
        index2 = index2_temp + index1_temp
        bonds = bonds + bonds

        # Sort the arrays by the beginning index
        sorted_arg = np.argsort(index1)
        index1 = np.array(index1)[sorted_arg].tolist()
        index2 = np.array(index2)[sorted_arg].tolist()
        bonds = np.array(bonds)[sorted_arg].tolist()

        return {'atom': atoms,
                'bond': bonds,
                'state': state_attributes,
                'index1': index1,
                'index2': index2}

    def _create_pair_feature_vector(self, bond: dict) -> List[float]:
        """Generate the feature vector from the bond feature dictionary

        Handles the binarization of categorical variables, and performing the distance conversion

        Args:
            bond (dict): Features for a certain pair of atoms
        Returns:
            ([float]) Values converted to a vector
            """
        bond_temp = []
        for i in self.bond_features:
            # Some features require conversion (e.g., binarization)
            if i in bond:
                if i == "bond_type":
                    bond_temp.extend(fast_label_binarize(bond[i], [0, 1, 2, 3, 4]))
                elif i == "same_ring":
                    bond_temp.append(int(bond[i]))
                elif i == "spatial_distance":
                    expanded = self.distance_converter.convert([bond[i]])[0]
                    if isinstance(expanded, np.ndarray):
                        # If we use a distance expansion
                        bond_temp.extend(expanded.tolist())
                    else:
                        # If not
                        bond_temp.append(expanded)
                else:
                    bond_temp.append(bond[i])
        return bond_temp

    def _create_atom_feature_vector(self, atom: dict) -> List[int]:
        """Generate the feature vector from the atomic feature dictionary

        Handles the binarization of categorical variables, and transforming the ring_sizes to a list

        Args:
            atom (dict): Dictionary of atomic features
        Returns:
            ([int]): Atomic feature vector
        """
        atom_temp = []
        for i in self.atom_features:
            if i == 'element':
                atom_temp.extend(fast_label_binarize(atom[i], self.known_elements))
            elif i == 'aromatic':
                atom_temp.append(int(atom[i]))
            elif i == 'hybridization':
                atom_temp.extend(fast_label_binarize(atom[i], [1, 2, 3, 4, 5, 6]))
            elif i == 'ring_sizes':
                atom_temp.extend(ring_to_vector(atom[i]))
            else:  # It is a scalar
                atom_temp.append(atom[i])
        return atom_temp

    def _dijkstra_distance(self, pairs):
        """
        Compute the graph distance between each pair of atoms,
        using the network defined by the bonded atoms.

        Args:
            pairs ([dict]): List of bond information
        Returns:
            ([int]) Distance for each pair of bonds
        """
        bonds = []
        for p in pairs:
            if p['bond_type'] > 0:
                bonds.append([p['a_idx'], p['b_idx']])
        return dijkstra_distance(bonds)

    def get_atom_feature(self, mol):
        """
        Generate all features of a particular atom

        Args:
            mol (pybel.Molecule): Molecule being evaluated
            atom (pybel.Atom): Specific atom being evaluated
        Return:
            (dict): All features for that atom
        """

        feature = []

        # Find the rings, if desired
        if 'ring_sizes' in self.atom_features:
            rinfo = mol.GetRingInfo()  # Chen Qu
            rings = rinfo.AtomRings()  # Chen Qu

        for idx, atom in enumerate(mol.GetAtoms()):  # Chen Qu
            # Get the element
            element = Element.from_Z(atom.GetAtomicNum()).symbol  # Chen Qu

            # Get the fast-to-compute properties
            output = {"element": element,
                      "atomic_num": atom.GetAtomicNum(),
                      "formal_charge": atom.GetFormalCharge(),
                      "hybridization": atom.GetHybridization(),
                      "aromatic": atom.GetIsAromatic()}

            if 'ring_sizes' in self.atom_features:
                output['ring_sizes'] = [len(r) for r in rings if (idx in r)]

            feature.append(output)

        return feature

    def create_bond_feature(self, mol, bid, eid):
        """
        Create information for a bond for a pair of atoms that are not actually bonded

        Args:
            mol (pybel.Molecule): Molecule being featurized
            bid (int): Index of atom beginning of the bond
            eid (int): Index of atom at the end of the bond

        """

        same_ring = False
        ri = mol.GetRingInfo().AtomRings()
        for ring in ri:
            same_ring = set([bid, eid]).issubset(set(ring))
            if same_ring:
                break

        return {"a_idx": bid,
                "b_idx": eid,
                "bond_type": 0,
                "same_ring": same_ring}

    def get_pair_feature(self, mol, bid, eid):
        """
        Get the features for a certain bond

        Args:
            mol (pybel.Molecule): Molecule being featurized
            bid (int): Index of atom beginning of the bond
            eid (int): Index of atom at the end of the bond
        """

        # Find the bonded pair of atoms
        bond = mol.GetBondBetweenAtoms(bid, eid)

        # If the atoms are not bonded
        if not bond:
            return self.create_bond_feature(mol, bid, eid)

        bond_type = bond.GetBondTypeAsDouble()
        if bond_type == 1.5:
             bt = 4
        else:
             bt = int(bond_type)

        # Compute bond features
        same_ring = False
        ri = mol.GetRingInfo().AtomRings()
        for ring in ri:
            same_ring = set([bid, eid]).issubset(set(ring))
            if same_ring:
                break

        return {"a_idx": bid,
                "b_idx": eid,
                "bond_type": bt,
                "same_ring": same_ring}

def dijkstra_distance(bonds):
    """
    Compute the graph distance based on the dijkstra algorithm

    Args:
        bonds: (list of list), for example [[0, 1], [1, 2]] means two bonds formed by atom 0, 1 and atom 1, 2

    Returns:
        full graph distance matrix
    """
    nb_atom = max(itertools.chain(*bonds)) + 1
    graph_dist = np.ones((nb_atom, nb_atom), dtype=np.int32) * np.infty
    for bond in bonds:
        graph_dist[bond[0], bond[1]] = 1
        graph_dist[bond[1], bond[0]] = 1

    queue = deque()  # Queue used in all loops
    visited = set()  # Used in all loops
    for i in range(nb_atom):
        graph_dist[i, i] = 0
        visited.clear()
        queue.append(i)
        while queue:
            s = queue.pop()
            visited.add(s)

            for k in np.where(graph_dist[s, :] == 1)[0]:
                if k not in visited:
                    queue.append(k)
                    graph_dist[i, k] = min(graph_dist[i, k],
                                           graph_dist[i, s] + 1)
                    graph_dist[k, i] = graph_dist[i, k]
    return graph_dist

def ring_to_vector(l):
    """
    Convert the ring sizes vector to a fixed length vector
    For example, l can be [3, 5, 5], meaning that the atom is involved
    in 1 3-sized ring and 2 5-sized ring. This function will convert it into
    [ 0, 0, 1, 0, 2, 0, 0, 0, 0, 0].

    Args:
        l: (list of integer) ring_sizes attributes

    Returns:
        (list of integer) fixed size list with the i-1 th element indicates number of
            i-sized ring this atom is involved in.
    """
    return_l = [0] * 9
    if l:
        for i in l:
            if (i < 9):
                return_l[i - 1] += 1
    return return_l

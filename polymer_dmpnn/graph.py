"""Target-blind periodic-polymer graph construction and a compact D-MPNN.

This file defines graph construction and model operations only. It has no data
loader for kappa labels, training loop, optimizer, checkpoint selector, or test
reader. The trimer-plus-periodic-edge representation follows the published
polymer-graph construction; the directed bond-message equations follow the
Chemprop/D-MPNN family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from rdkit import Chem, rdBase


# The model was trained with RDKit 2026.03.3. Runtime compatibility is checked by the predictor.


@dataclass(frozen=True)
class PolymerGraph:
    atom_features: np.ndarray      # [N, atom_dim], float32
    edge_index: np.ndarray         # [2, E], directed source and destination
    edge_features: np.ndarray      # [E, bond_dim], float32
    reverse_edge: np.ndarray      # [E], reverse directed-edge index
    repeat_unit_atoms: int
    periodic_closure: bool
    reference_kekulize_fallback: bool


@dataclass(frozen=True)
class GraphBatch:
    atom_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    reverse_edge: torch.Tensor
    graph_index: torch.Tensor
    graph_count: int


_ATOM_NUMBERS = tuple(range(100))       # Chemprop convention: atomic number minus one
_DEGREES = (0, 1, 2, 3, 4, 5)
_FORMAL_CHARGES = (-1, -2, 1, 2, 0)
_CHIRAL_TAGS = (0, 1, 2, 3)
_TOTAL_HS = (0, 1, 2, 3, 4)
_HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
_BOND_TYPES = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
_STEREO = tuple(range(6))


def _one_hot_unknown(value: object, choices: Sequence[object]) -> list[float]:
    out = [0.0] * (len(choices) + 1)
    try:
        out[choices.index(value)] = 1.0
    except ValueError:
        out[-1] = 1.0
    return out


def _atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    """Chemprop-style atom features computed while the endpoint wildcard exists."""
    return (
        _one_hot_unknown(atom.GetAtomicNum() - 1, _ATOM_NUMBERS)
        + _one_hot_unknown(atom.GetTotalDegree(), _DEGREES)
        + _one_hot_unknown(atom.GetFormalCharge(), _FORMAL_CHARGES)
        + _one_hot_unknown(int(atom.GetChiralTag()), _CHIRAL_TAGS)
        + _one_hot_unknown(int(atom.GetTotalNumHs()), _TOTAL_HS)
        + _one_hot_unknown(atom.GetHybridization(), _HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.GetMass()) * 0.01]
    )


def _bond_features(
    bond_type: Chem.rdchem.BondType,
    *,
    conjugated: bool,
    in_ring: bool,
    stereo: int,
) -> list[float]:
    return (
        [0.0]
        + [float(bond_type == choice) for choice in _BOND_TYPES]
        + [float(conjugated), float(in_ring)]
        + _one_hot_unknown(int(stereo), _STEREO)
    )


def _assert_fallback_source_mapping(
    reference: Chem.Mol,
    source: Chem.Mol,
    wildcard_indices: Sequence[int],
    unit_size: int,
    left_endpoint: int,
    right_endpoint: int,
) -> None:
    """Fail if fallback sanitization changes mapped source chemistry/features.

    The center copy is the contextual template for every repeat unit, so its
    full model atom/internal-bond features must match the source repeat unit.
    Aromatic atom flags and internal bond type/aromatic flags are additionally
    checked in all three copies. No field is silently restored or zeroed.
    """
    wildcards = set(int(i) for i in wildcard_indices)
    local = {
        source_idx: source_idx - sum(w < source_idx for w in wildcards)
        for source_idx in range(source.GetNumAtoms())
        if source_idx not in wildcards
    }
    center_offset = unit_size
    for source_idx, local_idx in local.items():
        source_atom = source.GetAtomWithIdx(source_idx)
        center_atom = reference.GetAtomWithIdx(center_offset + local_idx)
        if _atom_features(source_atom) != _atom_features(center_atom):
            raise ValueError(
                f"fallback changed mapped center atom features at source atom {source_idx}"
            )
        source_extra = (
            source_atom.GetNumRadicalElectrons(),
            source_atom.GetTotalValence(),
            source_atom.IsInRing(),
        )
        center_extra = (
            center_atom.GetNumRadicalElectrons(),
            center_atom.GetTotalValence(),
            center_atom.IsInRing(),
        )
        if source_extra != center_extra:
            raise ValueError(
                f"fallback changed mapped center atom chemistry at source atom {source_idx}"
            )
        for copy_idx in range(3):
            copy_atom = reference.GetAtomWithIdx(copy_idx * unit_size + local_idx)
            if source_atom.GetIsAromatic() != copy_atom.GetIsAromatic():
                raise ValueError(
                    f"fallback changed mapped aromatic atom flag at copy {copy_idx}, source atom {source_idx}"
                )

    for source_bond in source.GetBonds():
        source_i, source_j = source_bond.GetBeginAtomIdx(), source_bond.GetEndAtomIdx()
        if source_i not in local or source_j not in local:
            continue
        for copy_idx in range(3):
            mapped = reference.GetBondBetweenAtoms(
                copy_idx * unit_size + local[source_i],
                copy_idx * unit_size + local[source_j],
            )
            if mapped is None or (
                source_bond.GetBondType(), source_bond.GetIsAromatic()
            ) != (mapped.GetBondType(), mapped.GetIsAromatic()):
                raise ValueError(
                    f"fallback changed mapped internal bond type/aromaticity at source bond {source_i}-{source_j}"
                )
        center_bond = reference.GetBondBetweenAtoms(
            center_offset + local[source_i], center_offset + local[source_j]
        )
        source_features = _bond_features(
            source_bond.GetBondType(),
            conjugated=source_bond.GetIsConjugated(),
            in_ring=source_bond.IsInRing(),
            stereo=int(source_bond.GetStereo()),
        )
        center_features = _bond_features(
            center_bond.GetBondType(),
            conjugated=center_bond.GetIsConjugated(),
            in_ring=center_bond.IsInRing(),
            stereo=int(center_bond.GetStereo()),
        )
        if source_features != center_features:
            raise ValueError(
                f"fallback changed mapped center internal-bond features at source bond {source_i}-{source_j}"
            )

    boundary = reference.GetBondBetweenAtoms(
        center_offset + right_endpoint,
        2 * unit_size + left_endpoint,
    )
    if boundary is None or boundary.GetBondType() != Chem.rdchem.BondType.SINGLE:
        raise ValueError("fallback boundary bond is missing or not a single bond")
    if boundary.GetIsAromatic() or boundary.IsInRing():
        raise ValueError("fallback boundary edge must be non-aromatic and non-ring")


def _reference_polymer_features(
    mol: Chem.Mol,
    wildcard_indices: Sequence[int],
    endpoint_indices: Sequence[int],
) -> tuple[np.ndarray, dict[tuple[int, int], list[float]], list[float], bool]:
    """Derive atom and bond features from the center of a connected open trimer.

    Wildcard atoms are removed, three repeat units are connected by ordinary
    single bonds, and the resulting RDKit graph is sanitized. If RDKit's
    Kekulize step alone fails, the builder retries every other sanitization step
    from the untouched graph and explicitly recalculates conjugation while
    retaining the aromatic flags from the valid source repeat-unit SMILES. The
    middle copy has
    polymer neighbors at both endpoints, so its atom and internal-bond features
    represent the connected repeat unit rather than a chain-end-truncated unit.
    These features are then reused for every copy in both ablation arms. The
    boundary-edge feature is read from a connected inter-copy bond; its ring bit
    is explicitly false because the polymer chain is not a physical ring.
    """
    if len(wildcard_indices) != 2 or len(endpoint_indices) != 2:
        raise ValueError("two wildcard and endpoint indices are required")
    wildcards = sorted(int(i) for i in wildcard_indices)
    endpoints = [int(i) for i in endpoint_indices]
    source_atom_count = mol.GetNumAtoms()
    unit_size = source_atom_count - len(wildcards)
    if unit_size < 1:
        raise ValueError("repeat unit has no non-wildcard atoms")

    def remap_after_removal(old_index: int) -> int:
        return old_index - sum(wildcard_idx < old_index for wildcard_idx in wildcards)

    left = remap_after_removal(endpoints[0])
    right = remap_after_removal(endpoints[1])
    combined = Chem.CombineMols(Chem.CombineMols(mol, mol), mol)
    reference_rw = Chem.RWMol(combined)
    all_wildcards = sorted(
        copy_idx * source_atom_count + wildcard_idx
        for copy_idx in range(3)
        for wildcard_idx in wildcards
    )
    for idx in reversed(all_wildcards):
        reference_rw.RemoveAtom(idx)

    # Removing two atoms per preceding copy shifts each copied endpoint by
    # exactly copy_idx * unit_size. The middle unit has both chain neighbors.
    for copy_idx in range(2):
        reference_rw.AddBond(
            copy_idx * unit_size + right,
            (copy_idx + 1) * unit_size + left,
            Chem.rdchem.BondType.SINGLE,
        )
    reference = reference_rw.GetMol()
    raw_reference = Chem.Mol(reference)
    kekulize_fallback = False
    try:
        Chem.SanitizeMol(reference)
    except Chem.KekulizeException:
        # Some fused heteroaromatic polymer graphs cannot be assigned a Kekule
        # form by RDKit even though parsing the source repeat unit and all other
        # valence/property sanitization steps succeed. Retry from the untouched
        # graph, excluding only Kekulize; retain aromatic bond flags and derive
        # conjugation explicitly. All other sanitization failures still stop.
        reference = raw_reference
        sanitize_without_kekulize = (
            Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        Chem.SanitizeMol(reference, sanitizeOps=sanitize_without_kekulize)
        Chem.SetConjugation(reference)
        _assert_fallback_source_mapping(
            reference,
            mol,
            wildcards,
            unit_size,
            left,
            right,
        )
        kekulize_fallback = True
    if len(Chem.GetMolFrags(reference)) != 1:
        raise ValueError("connected reference trimer is disconnected after wildcard removal")

    center_offset = unit_size
    atom_features = np.asarray(
        [_atom_features(reference.GetAtomWithIdx(center_offset + i)) for i in range(unit_size)],
        dtype=np.float32,
    )
    internal_bond_features: dict[tuple[int, int], list[float]] = {}
    for bond in reference.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if center_offset <= i < center_offset + unit_size and center_offset <= j < center_offset + unit_size:
            key = tuple(sorted((i - center_offset, j - center_offset)))
            internal_bond_features[key] = _bond_features(
                bond.GetBondType(),
                conjugated=bond.GetIsConjugated(),
                in_ring=bond.IsInRing(),
                stereo=int(bond.GetStereo()),
            )

    boundary = reference.GetBondBetweenAtoms(
        center_offset + right,
        2 * unit_size + left,
    )
    if boundary is None:
        raise ValueError("failed to construct the connected-trimer boundary bond")
    boundary_features = _bond_features(
        boundary.GetBondType(),
        conjugated=boundary.GetIsConjugated(),
        in_ring=False,
        stereo=int(boundary.GetStereo()),
    )
    return atom_features, internal_bond_features, boundary_features, kekulize_fallback


def build_trimer_graph(psmiles: str, *, periodic_closure: bool) -> PolymerGraph:
    """Build three repeat-unit copies, optionally closing the periodic edge.

    The two wildcard atoms are placeholders, not graph vertices. Atom and
    internal-bond features come from the middle copy of a sanitized open
    reference trimer, whose two endpoints both have polymer continuation bonds.
    Three copies keep the new closure edge between distinct atoms, including
    repeat units whose two endpoints share one neighboring atom.
    """
    mol = Chem.MolFromSmiles(psmiles)
    if mol is None:
        raise ValueError("invalid repeat-unit SMILES")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("repeat unit must be a single connected molecule")

    wildcard_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(wildcard_atoms) != 2 or any(atom.GetDegree() != 1 for atom in wildcard_atoms):
        raise ValueError("expected exactly two degree-one polymerization wildcards")
    if any(atom.GetIsotope() or atom.GetAtomMapNum() for atom in wildcard_atoms):
        raise ValueError("isotopic or mapped wildcard endpoints need an explicit mapping rule")

    # A deterministic endpoint order; swapping endpoints only reverses the
    # cycle's construction and leaves the undirected graph unchanged.
    wildcard_atoms = sorted(wildcard_atoms, key=lambda atom: atom.GetIdx())
    endpoint_atoms: list[int] = []
    endpoint_bonds: list[Chem.rdchem.BondType] = []
    for wildcard in wildcard_atoms:
        neighbor = wildcard.GetNeighbors()[0]
        if neighbor.GetAtomicNum() == 0:
            raise ValueError("polymerization wildcard may not connect to another wildcard")
        endpoint_atoms.append(neighbor.GetIdx())
        endpoint_bonds.append(mol.GetBondBetweenAtoms(wildcard.GetIdx(), neighbor.GetIdx()).GetBondType())
    if endpoint_bonds[0] != endpoint_bonds[1] or endpoint_bonds[0] != Chem.rdchem.BondType.SINGLE:
        raise ValueError("the two polymerization endpoints must both specify single bonds")

    unit_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 0]
    if not unit_atoms:
        raise ValueError("repeat unit has no non-wildcard atoms")
    original_to_local = {old: new for new, old in enumerate(unit_atoms)}
    left = original_to_local[endpoint_atoms[0]]
    right = original_to_local[endpoint_atoms[1]]
    unit_size = len(unit_atoms)

    # Derive all repeat-unit atom and internal-bond features after both polymer
    # connections are present in the middle reference copy.
    (
        unit_atom_features,
        internal_bond_features,
        boundary_features,
        kekulize_fallback,
    ) = _reference_polymer_features(
        mol,
        [atom.GetIdx() for atom in wildcard_atoms],
        endpoint_atoms,
    )
    atom_features = np.concatenate([unit_atom_features] * 3, axis=0)

    sources: list[int] = []
    destinations: list[int] = []
    edge_features: list[list[float]] = []
    reverse_edge: list[int] = []

    def add_undirected_bond(i: int, j: int, features: list[float]) -> None:
        forward = len(sources)
        sources.extend((i, j))
        destinations.extend((j, i))
        edge_features.extend((features, features))
        reverse_edge.extend((forward + 1, forward))

    # Copy within-repeat-unit bonds and their connected-reference features.
    # Wildcard bonds are replaced by inter-copy bonds below.
    for copy_idx in range(3):
        offset = copy_idx * unit_size
        for bond in mol.GetBonds():
            i_old, j_old = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i_old not in original_to_local or j_old not in original_to_local:
                continue
            local_i, local_j = original_to_local[i_old], original_to_local[j_old]
            key = tuple(sorted((local_i, local_j)))
            if key not in internal_bond_features:
                raise ValueError("reference trimer is missing an internal bond feature")
            add_undirected_bond(
                offset + local_i, offset + local_j, internal_bond_features[key]
            )

    # Join adjacent repeat units. The final bond is the periodic closure
    # ablated in the paired open-trimer representation.
    for copy_idx in range(2):
        add_undirected_bond(
            copy_idx * unit_size + right,
            (copy_idx + 1) * unit_size + left,
            boundary_features,
        )
    if periodic_closure:
        add_undirected_bond(2 * unit_size + right, left, boundary_features)

    return PolymerGraph(
        atom_features=atom_features,
        edge_index=np.asarray([sources, destinations], dtype=np.int64),
        edge_features=np.asarray(edge_features, dtype=np.float32),
        reverse_edge=np.asarray(reverse_edge, dtype=np.int64),
        repeat_unit_atoms=unit_size,
        periodic_closure=periodic_closure,
        reference_kekulize_fallback=kekulize_fallback,
    )


def batch_graphs(graphs: Iterable[PolymerGraph], *, device: str = "cpu") -> GraphBatch:
    """Concatenate small graph tensors without PyG or a persistent graph cache."""
    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("cannot batch an empty graph list")
    atom_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    bond_parts: list[np.ndarray] = []
    reverse_parts: list[np.ndarray] = []
    batch_parts: list[np.ndarray] = []
    atom_offset = 0
    edge_offset = 0
    for graph_idx, graph in enumerate(graph_list):
        atom_count = graph.atom_features.shape[0]
        edge_count = graph.edge_index.shape[1]
        if graph.edge_features.shape[0] != edge_count or graph.reverse_edge.shape[0] != edge_count:
            raise ValueError("edge index, bond features, and reverse-edge map disagree")
        if edge_count and (graph.reverse_edge.min() < 0 or graph.reverse_edge.max() >= edge_count):
            raise ValueError("reverse-edge index is out of range")
        if edge_count and not np.array_equal(graph.reverse_edge[graph.reverse_edge], np.arange(edge_count)):
            raise ValueError("reverse-edge map is not an involution")
        atom_parts.append(graph.atom_features)
        edge_parts.append(graph.edge_index + atom_offset)
        bond_parts.append(graph.edge_features)
        reverse_parts.append(graph.reverse_edge + edge_offset)
        batch_parts.append(np.full(atom_count, graph_idx, dtype=np.int64))
        atom_offset += atom_count
        edge_offset += edge_count

    return GraphBatch(
        atom_features=torch.as_tensor(np.concatenate(atom_parts), dtype=torch.float32, device=device),
        edge_index=torch.as_tensor(np.concatenate(edge_parts, axis=1), dtype=torch.long, device=device),
        edge_features=torch.as_tensor(np.concatenate(bond_parts), dtype=torch.float32, device=device),
        reverse_edge=torch.as_tensor(np.concatenate(reverse_parts), dtype=torch.long, device=device),
        graph_index=torch.as_tensor(np.concatenate(batch_parts), dtype=torch.long, device=device),
        graph_count=len(graph_list),
    )


class PeriodicDMPNN(nn.Module):
    """Compact Chemprop-style directed bond-message network for polymer graphs."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        *,
        hidden_size: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.input_bond = nn.Linear(atom_dim + bond_dim, hidden_size, bias=False)
        self.message_update = nn.Linear(hidden_size, hidden_size, bias=False)
        self.atom_update = nn.Linear(atom_dim + hidden_size, hidden_size)
        self.ffn_hidden = nn.Linear(hidden_size, hidden_size)
        self.ffn_output = nn.Linear(hidden_size, 1)
        # Match Chemprop's Xavier-normal weight and zero-bias initialization.
        for parameter in self.parameters():
            if parameter.dim() == 1:
                nn.init.constant_(parameter, 0.0)
            else:
                nn.init.xavier_normal_(parameter)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        atom_x = batch.atom_features
        src, dst = batch.edge_index
        edge_x = torch.cat((atom_x[src], batch.edge_features), dim=1)
        input_projection = self.input_bond(edge_x)
        message = F.relu(input_projection)

        # Directed edge u->v aggregates messages w->u, excluding v->u.
        for _ in range(self.depth - 1):
            incoming = atom_x.new_zeros((atom_x.shape[0], message.shape[1]))
            incoming.index_add_(0, dst, message)
            non_backtracking = incoming[src] - message[batch.reverse_edge]
            # Chemprop residual is applied to the pre-activation input
            # projection, while only the initial message is activated.
            message = F.relu(input_projection + self.message_update(non_backtracking))
            message = self.dropout(message)

        incoming = atom_x.new_zeros((atom_x.shape[0], message.shape[1]))
        incoming.index_add_(0, dst, message)
        atom_state = F.relu(self.atom_update(torch.cat((atom_x, incoming), dim=1)))
        atom_state = self.dropout(atom_state)

        pooled = atom_state.new_zeros((batch.graph_count, atom_state.shape[1]))
        pooled.index_add_(0, batch.graph_index, atom_state)
        counts = atom_state.new_zeros((batch.graph_count, 1))
        counts.index_add_(0, batch.graph_index, atom_state.new_ones((atom_state.shape[0], 1)))
        pooled = pooled / counts.clamp_min(1.0)

        hidden = F.relu(self.ffn_hidden(pooled))
        hidden = self.dropout(hidden)
        return self.ffn_output(hidden).squeeze(-1)

"""Portable inference using the three saved L1000 periodic-D-MPNN models."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from importlib import resources

import numpy as np
import torch
from rdkit import rdBase

from .graph import PeriodicDMPNN, batch_graphs, build_trimer_graph


class PolymerThermalPredictor:
    """Predict simulated isotropic amorphous-polymer conductivity at 300 K.

    The three models were independently fitted to the frozen L1000 design.
    Their mean is an ensemble prediction; their dispersion is descriptive and
    is not a calibrated prediction interval.
    """

    def __init__(self) -> None:
        weights_dir = resources.files("polymer_dmpnn").joinpath("weights")
        self.manifest = json.loads(weights_dir.joinpath("manifest.json").read_text(encoding="utf-8"))
        tested = self.manifest["training_rdkit"]
        if rdBase.rdkitVersion != tested:
            warnings.warn(
                f"RDKit {rdBase.rdkitVersion} is in use; the graph builder was "
                f"validated with {tested}. Check predictions before comparing results.",
                RuntimeWarning,
                stacklevel=2,
            )

        reference = build_trimer_graph("*CC*", periodic_closure=True)
        self.models = []
        for spec in self.manifest["weights"]:
            blob = weights_dir.joinpath(spec["file"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
                raise RuntimeError(f"Weight checksum mismatch: {spec['file']}")
            state = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
            model = PeriodicDMPNN(reference.atom_features.shape[1], reference.edge_features.shape[1])
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models.append((model, spec))

    def predict_one(self, smiles: str) -> dict:
        """Return a prediction or an error for a two-wildcard repeat unit."""
        if not isinstance(smiles, str) or not smiles.strip():
            return {"smiles": smiles, "success": False, "error": "SMILES is empty"}
        smiles = smiles.strip()
        try:
            graph = build_trimer_graph(smiles, periodic_closure=True)
            batch = batch_graphs([graph], device="cpu")
        except (ValueError, RuntimeError, TypeError) as exc:
            return {"smiles": smiles, "success": False, "error": str(exc)}

        values = []
        with torch.inference_mode():
            for model, spec in self.models:
                scaled = float(model(batch).item())
                values.append(scaled * spec["target_sd_ddof0"] + spec["target_mean"])
        if not np.isfinite(values).all():
            return {"smiles": smiles, "success": False, "error": "model produced a non-finite value"}
        return {
            "smiles": smiles,
            "success": True,
            "predicted_kappa_W_mK": float(np.mean(values)),
            "seed_std_W_mK": float(np.std(values, ddof=0)),
            "per_seed_predictions_W_mK": {
                str(spec["seed"]): float(value) for (_, spec), value in zip(self.models, values)
            },
            "seed_std_note": "Population SD across three fitted models; not a calibrated uncertainty interval.",
        }

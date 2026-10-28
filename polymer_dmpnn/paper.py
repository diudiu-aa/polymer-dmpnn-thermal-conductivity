"""Individual saved manuscript models; no ensemble or fitting at load time."""
from __future__ import annotations
import hashlib
import io
import json
from importlib import resources

import numpy as np
import torch
from torch import nn

from .graph import PeriodicDMPNN, batch_graphs, build_trimer_graph


class MultitaskDMPNN(nn.Module):
    """Regression backbone plus the original 128-dimensional classifier head."""
    def __init__(self):
        super().__init__()
        self.backbone = PeriodicDMPNN(133, 14, hidden_size=128, depth=3, dropout=.1)
        self.classifier = nn.Linear(128, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.backbone.ffn_hidden.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.features = output

    def forward(self, batch):
        prediction = self.backbone(batch)
        return prediction, self.classifier(torch.relu(self.features)).squeeze(-1)


def paper_registry():
    root = resources.files('polymer_dmpnn')
    return json.loads(root.joinpath('paper_weights/manifest.json').read_text(encoding='utf-8'))


class PaperPredictor:
    """One exact seed/model from Tables 3-5, selected by manifest key.

    Classification outputs are raw logits used only as ranking scores; they are
    not calibrated probabilities. Predictions are in W/(m K).
    """
    def __init__(self, model_id):
        registry = paper_registry()
        if model_id not in registry['models']:
            raise ValueError(f'Unknown paper model {model_id!r}. Use --list-models.')
        self.model_id = model_id
        self.spec = registry['models'][model_id]
        blob = resources.files('polymer_dmpnn').joinpath(self.spec['file']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != self.spec['sha256']:
            raise RuntimeError(f'Weight checksum mismatch: {model_id}')
        checkpoint = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
        for key in ('seed', 'arm'):
            if checkpoint[key] != self.spec[key]:
                raise RuntimeError(f'Checkpoint identity mismatch: {key}')
        if (checkpoint['target_mean'] != self.spec['target_mean'] or
                checkpoint['target_sd'] != self.spec['target_sd_ddof0']):
            raise RuntimeError('Checkpoint target scaling mismatch')
        self.model = (MultitaskDMPNN() if self.spec['arm'] == 'multitask'
                      else PeriodicDMPNN(133, 14, hidden_size=128, depth=3, dropout=.1))
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.model.eval()
        self.periodic = self.spec['arm'] != 'open'

    def predict_graphs(self, graphs):
        """Preserve the original float32 de-standardization and batch behavior."""
        with torch.inference_mode():
            out = self.model(batch_graphs(graphs, device='cpu'))
        if self.spec['arm'] == 'multitask':
            predictions, logits = out
            scores = logits.numpy().astype(np.float64)
        else:
            predictions, scores = out, None
        values = (predictions.numpy() * self.spec['target_sd_ddof0'] + self.spec['target_mean']).astype(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError('Non-finite model prediction')
        return values, scores

    def predict_one(self, smiles):
        if not isinstance(smiles, str) or not smiles.strip():
            return {'smiles': smiles, 'success': False, 'error': 'SMILES is empty'}
        smiles = smiles.strip()
        try:
            graph = build_trimer_graph(smiles, periodic_closure=self.periodic)
            values, scores = self.predict_graphs([graph])
        except (ValueError, TypeError, RuntimeError) as exc:
            return {'smiles': smiles, 'success': False, 'error': str(exc)}
        result = {'smiles': smiles, 'success': True, 'model_id': self.model_id,
                  'seed': self.spec['seed'], 'predicted_kappa_W_mK': float(values[0])}
        if scores is not None:
            result['high_kappa_classification_logit'] = float(scores[0])
        return result

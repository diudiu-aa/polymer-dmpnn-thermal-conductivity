# Attribution and release scope

The three weights in this package were trained on a fixed snapshot of the PolyOmics/RadonPy computational dataset (DOI 10.57967/hf/7475; snapshot `041e5834ea1a48682fae12dc39ccd723bcd4f771`). The current PolyOmics dataset card identifies its dataset license as CC BY 4.0. The dataset itself is not redistributed here; users should cite the dataset and review the applicable terms for their use.

The graph builder is the R04 project implementation, with the exact RDKit version rejection replaced by a compatibility warning and a connected-input check added. Its original source SHA-256 was `f2234546e083d6793784feb0dc206cf8bfab483f6bdef854d3e3e6b5a5e2ffdd`. It implements directed message passing in the style of Chemprop and a periodic repeat-unit graph, following the publications cited in the README. Chemprop is a separate project and is not bundled as a runtime dependency; its license is available at https://github.com/chemprop/chemprop/blob/main/LICENSE.txt.

Repository code and trained project weights are supplied under the accompanying MIT license. The source dataset is not redistributed. This repository provides an inference package and selected model-comparison utilities; it does not include the complete raw-to-curated dataset construction workflow or prepared input tables required for the training reproduction scripts.

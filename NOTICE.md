# Attribution and release scope

Code and project-trained weights are MIT licensed. Source datasets retain
separate terms; see THIRD_PARTY_DATA.md. The repository includes selected
fixed-role PolyOmics records (CC BY 4.0), saved derived predictions, and a
previously evaluated OPC excerpt (MIT). It does not redistribute the full
source databases or restricted experimental data.

The graph builder follows the directed-message-passing family and periodic
polymer representations cited in README.md. Its original source SHA256 was
`f2234546e083d6793784feb0dc206cf8bfab483f6bdef854d3e3e6b5a5e2ffdd`.
The hard RDKit-version rejection was replaced with compatibility handling;
connected-input validation was added. Chemprop is a separate project, not a
runtime dependency; its license is at
https://github.com/chemprop/chemprop/blob/main/LICENSE.txt.

Manifests distinguish the default original three-seed inference ensemble,
65 individual main-study checkpoints, fixed training inputs, and saved
prediction replay for Tables 1–5. Providing those materials does not claim a
complete reconstruction of historical raw-to-curated preparation or new MD
simulations. See reproduce/README.md for supported reproduction levels.

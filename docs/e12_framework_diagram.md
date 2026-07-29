# E12 Framework Diagram

Diagram file:

- [e12_framework_diagram.svg](C:/Users/ljr-w/Desktop/gnn/docs/e12_framework_diagram.svg)

What to tell your senior:

- `E12` keeps the dual-slot ALIGNN forward path, but removes explicit size features from the forward gate.
- The gate is predicted only from learned graph features in the forward pass.
- `num_atoms` enters only through an auxiliary supervision loss on `slot2_gate_logit`.
- So `E12` is cleaner than `E10`: size does not directly control the model at inference time.

Code anchors:

- E12 config: [E12_alignn_sizesupervised_virtualnode.yaml](C:/Users/ljr-w/Desktop/gnn/configs/E12_alignn_sizesupervised_virtualnode.yaml)
- Gate and dual-slot forward: [alignn_variants.py](C:/Users/ljr-w/Desktop/gnn/faithful_gnn/alignn_variants.py#L502)
- Size supervision loss: [alignn_runner.py](C:/Users/ljr-w/Desktop/gnn/faithful_gnn/alignn_runner.py#L190)

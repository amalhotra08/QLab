# A Hybrid Quantum-Classical Attention Mechanism for Efficient Large Language Models

**Authors:** Ansh Malhotra and Smaran Kudapa

## Abstract

Large language models rely on self-attention, but query-key similarity scales quadratically with sequence length. This project implements a compact hybrid quantum-classical attention block that replaces only the query-key dot product with a simulated four-qubit quantum kernel. On AG News, the classical, hybrid, and matched low-dimensional ablation models reached 76.9%, 70.6%, and 77.8% test accuracy. The hybrid simulation was 2.99x the classical runtime at sequence length 32, so the result does not support a practical classical-simulation speedup. Instead, the project demonstrates a reproducible framework for isolating quantum similarity inside attention and measuring expressivity, trainability, efficiency, and noise robustness.

## Introduction

Transformer models use self-attention to compare every token with every other token. This design is powerful, but the query-key similarity matrix requires O(n^2) pairwise interactions as sequence length grows. Quantum machine learning suggests a possible alternative: encode low-dimensional query and key features into Hilbert space and compute similarity through state overlap. Prior quantum attention work, including QSANN and QMSAN, shows that quantum self-attention can be applied to text classification, but those models do not fully isolate the query-key similarity step from other architectural changes.

## Research Question

Under controlled conditions, does replacing the query-key dot product with a simulated quantum kernel improve attention expressivity, preserve trainability, and reduce empirical cost?

## Methods

The experiment uses AG News classification with a simple local tokenizer, fixed vocabulary, and deterministic train/validation/test subsets. Three matched PyTorch models are compared. The classical model uses scaled dot-product attention. The hybrid model projects query and key vectors into four dimensions, encodes them into a four-qubit state-vector circuit using repeated RY rotations and nearest-neighbor CZ gates, then computes similarity as |<psi(q)|psi(k)>|^2. The ablation model uses the same four-dimensional query/key bottleneck but computes dot-product similarity classically.

Evaluation includes classification accuracy and macro-F1, forward-pass runtime by sequence length, memory deltas, centered kernel alignment (CKA) between attention maps, gradient variance across circuit depth, and simulated quantum noise. The noise test applies angle perturbation and depolarizing-style overlap mixing to the hybrid circuit.

## Results

The best compact-run classifier was **classical_ablation** with 77.8% test accuracy. The classical baseline reached 76.9%, the hybrid model reached 70.6%, and the ablation reached 77.8%. At sequence length 32, the hybrid forward pass was 2.99x the classical runtime, which is expected because a state-vector circuit is being simulated on classical hardware. The classical-hybrid attention CKA was 0.037. The gradient variance diagnostic increased from 0.000015 to 0.000035 over the tested circuit depths. At the highest simulated noise level, hybrid accuracy changed by +0.6 percentage points (70.6% to 71.2%).

## Discussion

The main finding is not that the simulated quantum layer is faster; it is not. The useful result is that the project isolates the quantum kernel inside an otherwise classical attention block and measures the consequences directly. If the hybrid model performs similarly to the bottlenecked ablation, then most of the behavior comes from dimensionality reduction rather than quantum geometry. If the hybrid model differs in attention alignment, noise response, or gradient behavior, those diagnostics identify where quantum kernels may be worth studying further.

## Limitations

This project uses a small classifier and a state-vector simulator, not quantum hardware. The reported runtime is therefore a measure of classical simulation overhead rather than a claim about future hardware advantage. The dataset is AG News only, and the model is intentionally small enough for reproducible senior-project experiments.

## Conclusion

A hybrid quantum-classical attention block can be implemented end-to-end and evaluated with real metrics. In this compact study, it did not demonstrate practical efficiency gains under classical simulation, but it produced a reproducible framework for testing attention expressivity, trainability, and robustness while keeping the quantum intervention isolated to the query-key similarity operation.

## References

- Vaswani et al. (2017). Attention Is All You Need.
- Li, Zhao, and Wang (2023). Quantum Self-Attention Neural Networks for Text Classification. arXiv:2205.05625.
- Chen et al. (2025). Quantum Mixed-State Self-Attention Network. Neural Networks, 185, 107123.
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. ICML.
- McClean et al. (2018). Barren Plateaus in Quantum Neural Network Training Landscapes. Nature Communications.
- Shen et al. (2025). Squat: Quant Small Language Models on the Edge. arXiv:2402.10787.

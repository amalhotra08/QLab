#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = ROOT / "figures"
OUT = PAPER / "comprehensive_report.pdf"


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitlePageTitle",
            parent=styles["Title"],
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TitlePageMeta",
            parent=styles["BodyText"],
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Justified",
            parent=styles["BodyText"],
            fontSize=10.2,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["BodyText"],
            fontSize=8.8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10.5,
        )
    )
    return styles


def p(text: str, styles, style: str = "Justified") -> Paragraph:
    return Paragraph(text, styles[style])


def bullets(items: list[str], styles) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles), leftIndent=12) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )


def table(data, widths):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e7ec")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.7),
                ("LEADING", (0, 0), (-1, -1), 10.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2c7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fa")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def fig(name: str, caption: str, width: float, styles, max_height: float = 5.35):
    path = FIGURES / name
    with PILImage.open(path) as image:
        image_width, image_height = image.size
    target_width = width * inch
    target_height = target_width * image_height / image_width
    if target_height > max_height * inch:
        target_height = max_height * inch
        target_width = target_height * image_width / image_height
    return KeepTogether(
        [
            Image(str(path), width=target_width, height=target_height),
            p(caption, styles, "Caption"),
        ]
    )


def build_story():
    styles = make_styles()
    story = []

    story.extend(
        [
            Spacer(1, 0.7 * inch),
            p("A Hybrid Quantum-Classical Attention Mechanism for Efficient Large Language Models", styles, "TitlePageTitle"),
            p("Ansh Malhotra and Smaran Kudapa", styles, "TitlePageMeta"),
            p("Quantum Information &amp; Optics Senior Research, 2025-2026", styles, "TitlePageMeta"),
            p("Thomas Jefferson High School for Science and Technology", styles, "TitlePageMeta"),
            Spacer(1, 0.65 * inch),
            table(
                [
                    ["Project Type", "Hybrid quantum-classical machine learning experiment"],
                    ["Dataset", "AG News text classification"],
                    ["Main Claim", "Quantum attention is feasible, but not faster under classical simulation"],
                    ["Artifacts", "Code, metrics, figures, LaTeX source, poster, and final PDF"],
                ],
                [1.65 * inch, 4.7 * inch],
            ),
            Spacer(1, 2.7 * inch),
            p("Final Comprehensive Research Report", styles, "TitlePageMeta"),
            PageBreak(),
        ]
    )

    story.append(p("Abstract", styles, "Heading1"))
    story.append(
        p(
            "Large language models rely on self-attention, but the query-key similarity step scales quadratically with sequence length. This project implements and evaluates a compact hybrid quantum-classical attention block that replaces only the classical query-key dot product with a simulated four-qubit parameterized quantum circuit. Value aggregation, residual structure, and classification layers remain classical, allowing the experiment to isolate the quantum kernel rather than replacing the entire Transformer block. Three matched AG News classifiers were implemented: a standard scaled dot-product attention baseline, a hybrid quantum-kernel model, and a low-dimensional classical ablation that uses the same query/key bottleneck as the quantum model. In the compact final run, the classical, hybrid, and ablation models reached 76.9%, 70.6%, and 77.8% test accuracy, respectively. At sequence length 32, the hybrid model required 4.16 ms per forward pass compared with 1.39 ms for the classical model, making the simulated quantum approach 2.99 times slower on classical hardware. Overall, the experiment demonstrates that quantum similarity can be isolated inside attention and evaluated reproducibly, but it does not support a practical efficiency advantage under state-vector simulation.",
            styles,
        )
    )

    story.append(p("Introduction", styles, "Heading1"))
    story.append(
        p(
            "Transformer architectures have become the dominant foundation for modern large language models. Their key mechanism is self-attention, which allows each token in a sequence to compare itself with every other token and assign context-dependent weights before information is aggregated. This design is powerful because it lets a model represent long-range dependencies without recurrence. However, the same all-to-all comparison creates a major efficiency challenge: if a sequence contains n tokens, the query-key similarity matrix contains n^2 pairwise scores.",
            styles,
        )
    )
    story.append(
        p(
            "Quantum machine learning offers a different way to represent similarity. Instead of comparing two vectors through a Euclidean dot product, a quantum model can encode classical features into a high-dimensional Hilbert space and compare the resulting quantum states through overlap or fidelity. This motivates a focused question: can the expensive similarity computation inside self-attention be replaced by a quantum kernel while leaving the rest of the Transformer block classical?",
            styles,
        )
    )

    story.append(p("Background Research", styles, "Heading1"))
    story.append(p("Classical Self-Attention", styles, "Heading2"))
    story.append(
        p(
            "In a standard Transformer, an input embedding matrix X is projected into query, key, and value matrices Q, K, and V. Attention is computed as softmax(QK^T / sqrt(d))V. The term QK^T contains every pairwise query-key interaction and is therefore the computational target of this study.",
            styles,
        )
    )
    story.append(p("Quantum Kernels and Trainability", styles, "Heading2"))
    story.append(
        p(
            "A parameterized quantum circuit can map a classical vector x to a quantum state |psi(x)>. Similarity can then be measured as |<psi(x_i)|psi(x_j)>|^2. This quantity is a quantum kernel. Quantum circuits also introduce optimization risks, especially barren plateaus, so this project includes a gradient-variance diagnostic across circuit depths.",
            styles,
        )
    )
    story.append(p("Prior Quantum Attention Work", styles, "Heading2"))
    story.append(
        p(
            "Quantum attention models such as QSANN and QMSAN show that quantum attention can be applied to text classification and related sequence tasks. The present project differs by isolating only the query-key similarity operation, which makes the quantum contribution easier to interpret.",
            styles,
        )
    )

    story.append(p("Research Question and Hypotheses", styles, "Heading1"))
    story.append(
        p(
            "Research question: under controlled conditions, does replacing the query-key dot product with a simulated quantum kernel improve attention expressivity, preserve trainability, and reduce empirical cost?",
            styles,
        )
    )
    story.append(
        bullets(
            [
                "The hybrid model should run end-to-end as an attention classifier and achieve non-trivial AG News performance.",
                "The hybrid kernel should produce attention maps that differ measurably from classical attention maps.",
                "Under classical state-vector simulation, the hybrid model is expected to be slower than the classical baseline.",
            ],
            styles,
        )
    )

    story.append(p("Methods", styles, "Heading1"))
    story.append(p("Dataset and Preprocessing", styles, "Heading2"))
    story.append(
        table(
            [
                ["Setting", "Value"],
                ["Dataset", "AG News"],
                ["Classes", "World, Sports, Business, Sci/Tech"],
                ["Training examples", "5,000"],
                ["Validation examples", "1,000"],
                ["Test examples", "1,000"],
                ["Vocabulary size", "8,000"],
                ["Maximum sequence length", "32 tokens"],
                ["Random seed", "42"],
            ],
            [2.5 * inch, 3.9 * inch],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(p("Model Families", styles, "Heading2"))
    story.append(
        bullets(
            [
                "Classical baseline: standard scaled dot-product self-attention.",
                "Hybrid quantum model: four-dimensional query/key projection, four-qubit simulated circuit, and fidelity-style state-overlap similarity.",
                "Classical ablation: the same four-dimensional query/key bottleneck as the hybrid model, but with a classical dot product.",
            ],
            styles,
        )
    )
    story.append(fig("architecture_diagram.png", "Figure 1. Hybrid quantum-classical attention block. Queries and keys are projected into a four-dimensional feature space, encoded into a simulated four-qubit circuit, and compared through quantum-state overlap.", 6.4, styles))

    story.append(p("Results", styles, "Heading1"))
    story.append(p("Classification Performance", styles, "Heading2"))
    story.append(
        table(
            [
                ["Model", "Accuracy", "Macro-F1", "Test Loss", "Best Val. Acc.", "Train Time (s)"],
                ["Classical", "0.769", "0.768", "0.734", "0.752", "5.65"],
                ["Classical ablation", "0.778", "0.779", "0.742", "0.770", "5.57"],
                ["Hybrid quantum", "0.706", "0.705", "0.815", "0.723", "13.84"],
            ],
            [1.55 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 1.05 * inch, 1.0 * inch],
        )
    )
    story.append(fig("accuracy_f1_summary.png", "Figure 2. Classification accuracy and macro-F1. The low-dimensional classical ablation slightly outperformed the standard classical baseline, while the hybrid quantum model trailed both.", 5.9, styles))

    story.append(p("Runtime Scaling", styles, "Heading2"))
    story.append(
        table(
            [
                ["Model", "Length 8", "Length 16", "Length 32", "Length 64"],
                ["Classical", "0.53", "0.62", "1.39", "2.29"],
                ["Classical ablation", "0.49", "0.75", "1.26", "1.84"],
                ["Hybrid quantum", "1.40", "3.09", "4.16", "6.17"],
            ],
            [1.65 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch],
        )
    )
    story.append(fig("benchmark_runtime.png", "Figure 3. Inference runtime scaling. Under classical state-vector simulation, the hybrid quantum attention block adds overhead rather than reducing empirical runtime.", 5.9, styles))

    story.append(p("Attention-Map Alignment", styles, "Heading2"))
    story.append(
        table(
            [
                ["Model Pair", "Linear CKA", "Mean Absolute Difference"],
                ["Classical vs. hybrid quantum", "0.037", "0.0386"],
                ["Classical vs. classical ablation", "0.100", "0.0361"],
                ["Classical ablation vs. hybrid quantum", "0.036", "0.0317"],
            ],
            [2.7 * inch, 1.2 * inch, 1.9 * inch],
        )
    )
    story.append(fig("attention_alignment.png", "Figure 4. Attention-map alignment. All alignments are low, but the classical baseline aligns more strongly with the classical ablation than with the hybrid model.", 5.9, styles))

    story.append(p("Noise Robustness and Trainability", styles, "Heading2"))
    story.append(
        p(
            "The simulated noise sweep applied angle perturbation and depolarizing-style mixing to the quantum attention component. The hybrid model's accuracy changed from 70.6% at zero noise to 71.2% at the highest tested noise level. Gradient variance peaked at circuit depth 3 and then declined at depths 4 and 5.",
            styles,
        )
    )
    story.append(fig("noise_robustness.png", "Figure 5. Noise robustness sweep. Classical models are repeated as references because they do not contain a quantum module.", 5.9, styles))
    story.append(fig("gradient_variance.png", "Figure 6. Trainability diagnostic. Gradient variance peaks at circuit depth 3 and then declines, showing that circuit depth changes optimization behavior.", 5.9, styles))

    story.append(p("Discussion", styles, "Heading1"))
    story.append(
        p(
            "The experiment supports the feasibility of a hybrid quantum-classical attention block. The model runs end-to-end, trains on a real text dataset, produces attention maps, and can be evaluated with the same metrics as classical baselines. However, the results do not support a near-term practical efficiency advantage under classical simulation. The hybrid model was slower than the classical baseline at every tested sequence length and achieved lower classification accuracy.",
            styles,
        )
    )
    story.append(
        p(
            "The ablation result is especially important. The classical ablation used the same four-dimensional query/key bottleneck as the hybrid model but did not use quantum state overlap. It achieved the highest accuracy in the final run. This implies that dimensionality reduction by itself can be competitive and that the quantum kernel did not provide an advantage on this task.",
            styles,
        )
    )

    story.append(p("Limitations", styles, "Heading1"))
    story.append(
        bullets(
            [
                "The quantum circuit is simulated on classical hardware, so runtime reflects simulation overhead rather than real quantum-device performance.",
                "The classifier is intentionally compact and should not be treated as a production-scale language model.",
                "The dataset is AG News only, so the findings should not be generalized to all NLP tasks.",
                "The circuit uses four qubits and a simple ansatz; other encodings may behave differently.",
            ],
            styles,
        )
    )

    story.append(p("Conclusion", styles, "Heading1"))
    story.append(
        p(
            "This project implemented a reproducible hybrid quantum-classical attention mechanism that isolates the query-key similarity computation inside an otherwise classical attention block. The hybrid model successfully trained and produced measurable attention behavior, demonstrating feasibility. However, it did not outperform the classical baselines and was approximately 2.99 times slower than classical attention at sequence length 32 under state-vector simulation. The strongest contribution is therefore methodological: the project provides a controlled framework for comparing quantum kernels, classical attention, and matched ablation models using accuracy, runtime, alignment, gradient variance, and noise robustness.",
            styles,
        )
    )

    story.append(p("Future Work", styles, "Heading1"))
    story.append(
        p(
            "Future work should test the hybrid attention block on longer sequences, larger datasets, and alternative circuit ansatzes. More realistic quantum-noise simulations would help determine whether the observed noise stability persists under hardware-like conditions. The most important next step is to evaluate whether actual quantum hardware can compute the kernel more efficiently than classical state-vector simulation.",
            styles,
        )
    )

    story.append(p("References", styles, "Heading1"))
    refs = [
        "Vaswani et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.",
        "Schuld and Killoran (2019). Quantum Machine Learning in Feature Hilbert Spaces. Physical Review Letters.",
        "Havlicek et al. (2019). Supervised Learning with Quantum-Enhanced Feature Spaces. Nature.",
        "Li, Zhao, and Wang (2023). Quantum Self-Attention Neural Networks for Text Classification. arXiv:2205.05625.",
        "Chen et al. (2025). Quantum Mixed-State Self-Attention Network. Neural Networks.",
        "McClean et al. (2018). Barren Plateaus in Quantum Neural Network Training Landscapes. Nature Communications.",
        "Cerezo et al. (2021). Cost Function Dependent Barren Plateaus in Shallow Parametrized Quantum Circuits. Nature Communications.",
        "Holmes et al. (2022). Connecting Ansatz Expressibility to Gradient Magnitudes and Barren Plateaus. PRX Quantum.",
        "Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. ICML.",
        "Choromanski et al. (2021). Rethinking Attention with Performers. ICLR.",
        "Beltagy, Peters, and Cohan (2020). Longformer: The Long-Document Transformer. arXiv:2004.05150.",
        "Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences. NeurIPS.",
        "Wang et al. (2020). Linformer: Self-Attention with Linear Complexity. arXiv:2006.04768.",
    ]
    story.append(bullets(refs, styles))

    story.append(PageBreak())
    story.append(p("Appendix: Reproducibility", styles, "Heading1"))
    story.append(
        p(
            "The full experiment can be reproduced from the repository using scripts/run_all.py. Metrics are written to results/metrics/, figures are written to figures/, and final paper/poster artifacts are written to paper/ and poster/. The repository excludes the virtual environment, cache folders, temporary QA renders, and training checkpoints.",
            styles,
        )
    )
    story.append(
        table(
            [
                ["Artifact", "Path"],
                ["LaTeX source", "paper/comprehensive_report.tex"],
                ["BibTeX references", "paper/references.bib"],
                ["Final PDF", "paper/comprehensive_report.pdf"],
                ["Metrics", "results/metrics/*.csv"],
                ["Figures", "figures/*.png"],
            ],
            [1.7 * inch, 4.4 * inch],
        )
    )

    return story


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="A Hybrid Quantum-Classical Attention Mechanism for Efficient Large Language Models",
        author="Ansh Malhotra and Smaran Kudapa",
    )
    doc.build(build_story())
    print(OUT)


if __name__ == "__main__":
    main()

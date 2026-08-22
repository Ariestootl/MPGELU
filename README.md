# Modified Parameterized Gaussian Error Linear Unit (MP-GELU)

🚧 **WORK IN PROGRESS:** *This research project and its accompanying codebase are currently under active development. Mathematical proofs, PyTorch implementations, and comparative benchmarking models are subject to ongoing refinement and validation.*

*   **School:** Parañaque Science High School
*   **Category:** Mathematics and Computational Science  
*   **Level:** Senior High School (STEM)

## 👥 About the Researchers
This research project is being developed by:
*   **Marion Dominic Reginales**
*   **Carl Jaeron Valmeo**
*   **Enzo Bautista**

*The team operates under my technical guidance as their qualified scientist and technical adviser.* To ensure the students build a robust foundational skillset in computational science and deep learning mathematics, **I, Jose Aries E. De Los Santos, provide hands-on mentorship in mathematical analysis and software engineering**. This includes:
*   Teaching and providing learning materials in standard calculus, which they use to assist in their mathematical analysis of the proposed activation function.
*   Providing hands-on mentoring in the PyTorch framework, NumPy, and other deep learning libraries to assist them in creating their proposed activation function.
*   Developing and coding the baseline Parameterized GELU (PGELU) from scratch. To the best of my knowledge, there is no implementation of it available online, and no code was provided by its original authors.

Through this guidance, I am equipping these students to bridge the gap between theoretical calculus and applied deep learning, ensuring their experiments are validated and mathematically sound.

## 🧠 Project Overview
Activation functions play a crucial role in deep neural networks because their mathematical properties directly influence gradient propagation, training stability, and model convergence. Smooth activation functions allow gradients to change continuously, reducing abrupt changes in parameter updates. 

This project introduces a Modified Parameterized Gaussian Error Linear Unit (MP-GELU), positioning it as an adaptive extension of P-GELU rather than a fundamentally different activation paradigm. The modification aims to preserve the smoothness and continuous differentiability of P-GELU while introducing a formulation that provides more controlled gradient behavior. By characterizing these properties theoretically, the proposed function can be evaluated not only on empirical performance but also on whether its gradients remain within a desirable and controllable range during neural network training.

## 🗂️ Repository Structure
```text
├── Analysis/                 # Contains the PDF of the mathematical analysis of MP-GELU and comprehensive math background of GELU
├── TeX/MP_GELU/              # LaTeX source files for the mathematical formulation and paper
├── data/                     # Directory for downloading and storing datasets
├── DataTransforms.py         # Data augmentation and preprocessing pipelines
├── Experiment.ipynb          # Main Jupyter Notebook for running and tracking experiments
├── MPGELU.py                 # Core PyTorch module for the MP-GELU activation function
├── MyCNN.py                  # Custom Convolutional Neural Network (CNN) architecture
├── Trainer.py                # Training loops, optimization steps, and evaluation metrics
└── README.md                 # Project documentation and methodology
```

## 🚀 Getting Started
To replicate this environment and run the predictive models locally, ensure you have Python 3.8+ installed, then clone the repository:

Bash
``git clone [https://github.com/Ariestootl/MPGELU.git](https://github.com/Ariestootl/MPGELU.git)``

## 📐 Mathematical Formulation

The standard Gaussian Error Linear Unit (GELU) relies on the fixed variance of the standard normal distribution. To address this limitation, MP-GELU introduces a learnable scaling parameter $\lambda \ge 1$ and formulates the activation function by scaling the input within the cumulative distribution function:
$$f(x) = \frac{x}{2}\left[1 + \text{erf}\left(\frac{\lambda x}{\sqrt{2}}\right)\right]$$

To strictly enforce the $\lambda \ge 1$ lower bound during unconstrained gradient-based optimization, we introduce a parameter $s \in \mathbb{R}$ mapped through the Softplus function:
$$\lambda = 1 + \ln(1 + e^s)$$

This ensures that the proposed activation function retains the core signal-preserving and noise-filtering properties of standard rectified units, effectively filtering out significant negative noise while preserving strong positive signals.  

## 📈 Methodology and Model Evaluation

The data gathered in this study will be analyzed by examining both the mathematical behavior and experimental performance of the proposed MP-GELU activation function. The analysis focuses on several key factors:  
* **Continuous Differentiability**: Standard calculus techniques are used to derive the first-order derivative and verify whether the function and its derivative remain continuous across the input domain.

* **Gradient Stability**: Gradient stability is assessed through the analysis of the derivative to identify possible occurrences of vanishing gradients, exploding gradients, or inactive region.

* **Learning Performance & Convergence**: Neural network simulations are conducted to compare MP-GELU against standard GELU, ReLU, and the newly implemented PGELU baselines[cite: 2]. The rate at which the network reaches stable learning states is examined using training and validation loss curves.

## 📝 How to Cite
If you utilize this codebase, mathematical derivation, or PyTorch implementation in your own research or projects, please cite this repository:

**APA Format:**
> De Los Santos, J. A. E. (2026). Modified Parameterized Gaussian Error Linear Unit (MP-GELU). GitHub. https://github.com/Ariestootl/MPGELUBibTeX:

**BibTex:**

```bibtex
@software{delossantos2026mpgelu,
  author = {De Los Santos, Jose Aries E.},
  title = {Modified Parameterized Gaussian Error Linear Unit (MP-GELU)},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/Ariestootl/MPGELU](https://github.com/Ariestootl/MPGELU)}}
}
```
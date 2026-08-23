# Modified Parameterized Gaussian Error Linear Unit (MPGELU)

🚧 **WORK IN PROGRESS:** *This research project and its accompanying codebase are currently under active development. Mathematical proofs, PyTorch implementations, and comparative benchmarking models are subject to ongoing refinement and validation.*

*   **School:** Parañaque Science High School
*   **Category:** Mathematics and Computational Science  
*   **Level:** Junior High School (STEM)

## 👥 About the Researchers
This research project is being developed by:

**Student Researchers:**
*   **Marion Dominic Reginales**
*   **Carl Jaeron Valmeo**
*   **Enzo Bautista**

**Qualified Scientist & Technical Adviser:**
*   **Jose Aries E. De Los Santos**

*The team operates under my technical guidance as their qualified scientist and technical adviser.* To ensure the students build a robust foundational skillset in computational science and deep learning mathematics, **I, Jose Aries E. De Los Santos, provide hands-on mentorship in mathematical analysis and software engineering**. This includes:
*   Teaching and providing learning materials in standard calculus, which they use to assist in their mathematical analysis of the proposed activation function.
*   Providing hands-on mentoring in the PyTorch framework, NumPy, and other deep learning libraries to assist them in creating their proposed activation function.
*   Developing and coding the baseline Parameterized GELU (PGELU) from scratch. To the best of my knowledge, there is no implementation of it available online, and no code was provided by its original authors.

Through this guidance, I am equipping these students to bridge the gap between theoretical calculus and applied deep learning, ensuring their experiments are validated and mathematically sound.

## 🧠 Project Overview
Activation functions play a crucial role in deep neural networks because their mathematical properties directly influence gradient propagation, training stability, and model convergence. Smooth activation functions allow gradients to change continuously, reducing abrupt changes in parameter updates. 

This project investigates a Modified Parameterized Gaussian Error Linear Unit (MPGELU), positioning it as an adaptive extension of P-GELU that independently aligns with the recently proposed $\lambda$-GELU (Pérez-Corral et al., 2026), rather than introducing a fundamentally different activation paradigm. 

Our investigation focuses on the rigorous mathematical foundations of this formulation. The modification aims to preserve the smoothness and continuous differentiability of standard GELU while introducing a controlled gating mechanism. By characterizing these properties using single-variable calculus, specifically analyzing asymptotic behavior and bounding the derivative via the Extreme Value Theorem—MPGELU is evaluated not only on empirical performance but on mathematically guaranteeing that its gradients remain within a strictly stable range to prevent exploding or vanishing signals during neural network training.

## 🗂️ Repository Structure
```text
├── Analysis/                 # Contains the PDF of the mathematical analysis of MPGELU and comprehensive math background of GELU
├── TeX/MP_GELU/              # LaTeX source files for the mathematical formulation and paper
├── data/                     # Directory for downloading and storing datasets
├── DataTransforms.py         # Data augmentation and preprocessing pipelines
├── Experiment.ipynb          # Main Jupyter Notebook for running and tracking experiments
├── MPGELU.py                 # Core PyTorch module for the MPGELU activation function
├── MyCNN.py                  # Custom Convolutional Neural Network (CNN) architecture
├── Trainer.py                # Training loops, optimization steps, and evaluation metrics
└── README.md                 # Project documentation and methodology
```

<!-- ## 🚀 Getting Started
To replicate this environment and run the predictive models locally, ensure you have Python 3.8+ installed, then clone the repository:

Bash
``git clone https://github.com/Ariestootl/MPGELU.git](https://github.com/Ariestootl/MPGELU.git)`` -->

## 📐 Mathematical Formulation

The standard Gaussian Error Linear Unit (GELU) relies on the fixed variance of the standard normal distribution. To address this limitation, MPGELU introduces a learnable scaling parameter $\lambda \ge 1$ and formulates the activation function by scaling the input within the cumulative distribution function:

$$f(x) = \frac{x}{2}\left[1 + \text{erf}\left(\frac{\lambda x}{\sqrt{2}}\right)\right]$$


To strictly enforce the $\lambda \ge 1$ lower bound during unconstrained gradient-based optimization, the parameter $s \in \mathbb{R}$ is introduced and mapped through the Softplus function:

$$\lambda = 1 + \ln(1 + e^s)$$

This ensures that the formulated activation function retains the core signal-preserving and noise-filtering properties of standard rectified units, effectively filtering out significant negative noise while preserving strong positive signals.  

## 📈 Methodology and Model Evaluation
The data gathered in this study will be analyzed by examining both the mathematical behavior and experimental performance of the MPGELU activation function. The analysis focuses on several key factors:

*   **Continuous Differentiability:** Standard calculus techniques are used to derive the first-order derivative and verify whether the function and its derivative remain continuous across the input domain.

*   **Asymptotic Behavior Analysis:** The asymptotic behavior of the MPGELU activation function is examined at its extremes to demonstrate that it retains the core signal-preserving and noise-filtering properties of standard rectified units. This mathematical analysis proves that the function maintains an asymptotically linear mapping for large positive inputs, while providing a soft-gating collapse for large negative inputs to induce network sparsity without causing dead neurons.

* **Gradient Stability:** First derivative analysis is applied to rigorously evaluate backpropagation stability. By computing asymptotic limits and invoking the Extreme Value Theorem, the derivative is proven to be strictly bounded ($\lvert f'(x) \rvert \le K$ for a constant $K >0$) to prevent the exploding gradient problem. Additionally, evaluating the non-zero derivative at the origin ($f'(0) > 0$) guarantees active signal flow, preventing vanishing gradients and the dying neuron pathology.

*   **Learning Performance & Convergence:** Neural network simulations are conducted to compare MPGELU against standard GELU, ReLU, and the newly implemented PGELU baselines. The rate at which the network reaches stable learning states is examined using training and validation loss curves.

## 📝 How to Cite
If you utilize this codebase, mathematical derivation, or PyTorch implementation in your own research or projects, please cite this repository:

**APA Format:**
> De Los Santos, J. A. E., Reginales, M. D., Valmeo, C. J., & Bautista, E. (2026). Modified Parameterized Gaussian Error Linear Unit (MPGELU) Demonstration. GitHub. https://github.com/Ariestootl/MPGELU

**BibTex:**

```bibtex
@software{delossantos2026mpgelu,
  author = {De Los Santos, Jose Aries E. and Reginales, Marion Dominic and Valmeo, Carl Jaeron and Bautista, Enzo},
  title = {Modified Parameterized Gaussian Error Linear Unit (MPGELU) Demonstration},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Ariestootl/MPGELU}}
}
```
## 🤝 Acknowledgments & Related Work
The core mathematical formulation of the MP-GELU activation function explored in this repository independently aligns with the $\lambda$-GELU function recently proposed by Pérez-Corral et al. (2026). We highly encourage readers to review their foundational paper for further insights into controlled ReLU-ization.

**Original Formulation:**
> Pérez-Corral, C., Fernández-Hernández, A., Mestre, J. I., Dolz, M. F., & Quintana-Ortí, E. S. (2026). *$\lambda$-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks*. arXiv preprint arXiv:2603.21991. https://arxiv.org/abs/2603.21991

**BibTeX:**
```bibtex
@misc{perezcorral2026lambdagelu,
      title={$\lambda$-GELU: Learning Gating Hardness for Controlled ReLU-ization in Deep Networks}, 
      author={Cristian P{\'e}rez-Corral and Alberto Fern{\'a}ndez-Hern{\'a}ndez and Jose I. Mestre and Manuel F. Dolz and Enrique S. Quintana-Ort{\'i}},
      year={2026},
      eprint={2603.21991},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.21991}
}
```
## 📚 References

*   Agarap, A. F. (2018). Deep learning using rectified linear units (ReLU). *arXiv*. https://arxiv.org/abs/1803.08375
*   Alkhouly, A. A., Mohammed, A., & Hefny, H. A. (2021). Improving the performance of deep neural networks using two proposed activation functions. *IEEE Access*. https://doi.org/10.1109/access.2021.3085855
*   Apicella, A., Donnarumma, F., Isgrò, F., & Prevete, R. (2021). A survey on modern trainable activation functions. *Neural Networks*, 138, 14-32. https://doi.org/10.1016/j.neunet.2021.01.026
*   Baheti, P. (2021). *Activation functions in neural networks: 12 types & use cases*. V7 Labs. https://www.v7labs.com/blog/neural-networks-activation-functions
*   Basirat, M., & Roth, P. M. (2018). The quest for the golden activation function. *arXiv*. https://arxiv.org/abs/1808.00783
*   Chandra, P., & Singh, Y. (2004). An activation function adapting training algorithm for sigmoidal feedforward networks. *Neurocomputing*. https://www.sciencedirect.com/science/article/abs/pii/S092523120400236X
*   Ciuparu, A., Nagy-Dabacan, A., & Muresan, R. C. (2019). Soft++, a multi-parametric non-saturating non-linearity that improves convergence in deep neural architectures. *Neurocomputing*, 381, 189-200. https://doi.org/10.1016/j.neucom.2019.12.014
*   Dai, S., Mahloujifar, S., & Mittal, P. (2022). Parameterizing activation functions for adversarial robustness. *arXiv*. https://arxiv.org/abs/2110.05626
*   Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). The hitchhiker's guide to testing statistical significance in natural language processing. *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics*, 1383-1392. https://doi.org/10.18653/v1/p18-1128
*   Dror, R., Shlomov, S., & Reichart, R. (2019). Deep dominance - How to properly compare deep neural models. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, 2773-2784. https://doi.org/10.18653/v1/p19-1266
*   Dubey, S. R., Singh, S. K., & Chaudhuri, B. B. (2022). Activation functions in deep learning: A comprehensive survey and benchmark. *Neurocomputing*. https://arxiv.org/pdf/2109.14545
*   Fernández, J. G., Keemink, S., & van Gerven, M. (2024). Gradient-free training of recurrent neural networks using random perturbations. *Frontiers in Neuroscience*, 18, 1439155. https://doi.org/10.3389/fnins.2024.1439155
*   Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, 15, 315-323. PMLR. https://proceedings.mlr.press/v15/glorot11a.html
*   Hayou, S., Clerico, E., & Doucet, A. (2020). Stable ResNet. *arXiv*. https://arxiv.org/abs/2010.12859
*   Hayou, S., Doucet, A., & Rousseau, J. (2019). On the Impact of the Activation Function on Deep Neural Networks Training. *arXiv*. https://arxiv.org/abs/1902.06853
*   Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). *arXiv*. https://arxiv.org/abs/1606.08415[cite: 1, 2]
*   Islam, M., Chen, G., & Jin, S. (2019). An overview of neural network. *American Journal of Neural Networks and Applications*, 5(1), 5. https://doi.org/10.11648/j.ajnna.20190501.12
*   Krithivasan, S., Sen, S., Venkataramani, S., & Raghunathan, A. (2022). Accelerating DNN training through selective localized learning. *Frontiers in Neuroscience*, 15, 759807. https://doi.org/10.3389/fnins.2021.759807
*   Labied, M., Belangour, A., & Banane, M. (2025). P-GELU: A novel activation function to optimize Whisper for Darija speech translation. *IEEE Access*, 13, 100198-100218. https://ieeexplore.ieee.org/document/11016691
*   Lau, M. M., & Lim, K. H. (2018). Review of Adaptive Activation Function in Deep Neural Network. *IEEE-ICBES*. https://ieeexplore.ieee.org/document/8626714
*   Lee, M. (2023). Mathematical analysis and performance evaluation of the GELU activation function in deep learning. *Journal of Mathematics*, 2023, 4229924. https://doi.org/10.1155/2023/4229924
*   Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In *Proceedings of the 27th International Conference on Machine Learning (ICML-10)*, 807-814.
*   Parhi, R., & Nowak, R. D. (2021). Banach space representer theorems for neural networks and ridge splines. *Journal of Machine Learning Research*, 22, 1-40. https://sparsity.ucsd.edu/publications/parhi2021banach.pdf
*   Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., & Lerer, A. (2017). Automatic differentiation in pytorch.
*   Tan, H. H., & Lim, K. H. (2019). Vanishing gradient mitigation with deep learning neural network optimization. *Proceedings of the 2019 7th International Conference on Smart Computing & Communications*, 1-4. https://doi.org/10.1109/icscc.2019.8843652
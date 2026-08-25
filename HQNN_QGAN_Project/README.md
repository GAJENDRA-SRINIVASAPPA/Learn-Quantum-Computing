# Hybrid Quantum-Classical Neural Network (HQNN) & Quantum GAN Project

## Overview
This project explores the integration of quantum computing with classical machine learning through two key approaches:

1. **Hybrid Quantum-Classical Neural Network (HQNN)**: Combines quantum circuits with classical neural networks
2. **Quantum Generative Adversarial Network (QGAN)**: A quantum-classical hybrid approach to generative modeling

## Project Structure
```
HQNN_QGAN_Project/
├── README.md
├── requirements.txt
├── hqnn/
│   ├── __init__.py
│   ├── quantum_layer.py
│   ├── hybrid_model.py
│   └── training.py
├── qgan/
│   ├── __init__.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── qgan_model.py
│   └── training.py
├── examples/
│   ├── hqnn_example.py
│   └── qgan_example.py
├── notebooks/
│   ├── HQNN_Tutorial.ipynb
│   └── QGAN_Tutorial.ipynb
└── utils/
    ├── __init__.py
    ├── data_loader.py
    └── visualization.py
```

## Key Features

### HQNN
- Integration of quantum circuits as layers in neural networks
- Variational quantum algorithms for parameter optimization
- Hybrid loss functions combining quantum and classical metrics

### QGAN
- Quantum generator architecture
- Classical discriminator
- Training loop for adversarial learning
- Support for multiple quantum backends

## Prerequisites
- Python 3.8+
- Qiskit
- PyTorch or TensorFlow
- NumPy, Matplotlib

## Installation
```bash
pip install -r requirements.txt
```

## Getting Started
See the examples and notebooks directory for detailed tutorials and code samples.

## References
- [Quantum Machine Learning with Qiskit](https://qiskit.org/documentation/machine-learning/)
- [Variational Quantum Algorithms](https://arxiv.org/abs/2012.09265)
- [Quantum GANs](https://arxiv.org/abs/1804.08641)

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
MIT License
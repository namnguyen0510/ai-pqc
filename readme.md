This code repository is associated with article entitled: "Benchmarking AI-Enabled Attacks on Post-Quantum Cryptography: A Case study on Genetic Data Protection"

# Introduction
This study investigates whether state-of-the-art deep learning models can compromise post-quantum cryptography (PQC) methods designed to secure sensitive genetic data, such as gRNA sequences. By benchmarking five advanced neural architectures—FNN, RNN, GRN, LSTM, and Transformer models—across six gRNA datasets, our two-phase evaluation first identifies the most effective neural decoder for encrypted genomic inputs and then scales the model to assess its generalization capabilities. Our findings reveal that DL-driven attacks pose a tangible threat to PQC-protected data, highlighting critical vulnerabilities in current cryptographic frameworks and emphasizing the need for more robust security measures in biotechnology.

# Code usage:
## Phase 1
```
python main.py
```
## Phase 2
```
python main-transformer.py
```

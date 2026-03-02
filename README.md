# Deep Learning-Based Voice Spoof Detection

## Project Overview

This project uses the ASVspoof 2019 Logical Access (LA) dataset to construct and assess various deep learning architectures for voice spoof detection.

The main goal is to use deeplearning architecture-based classification models to differentiate between real and fake speech samples.

The implemented models include:

* Convolutional Neural Network (CNN)
* Long Short-Term Memory (LSTM)
* Gated Recurrent Unit (GRU)
* Bidirectional GRU (BiGRU)

A comparative analysis is performed using both classification and threshold-independent evaluation metrics.

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Equal Error Rate (EER)
* Minimum normalized tandem Detection Cost Function (min-tDCF)

These metrics allow the both performance comparison and the robustness evaluation.

---

## Dataset

This project uses the publicly available:

**ASVspoof 2019 Logical Access (LA) Dataset**

Dataset URL:
[https://datashare.ed.ac.uk/handle/10283/3336](https://datashare.ed.ac.uk/handle/10283/3336)

Note: The dataset is not included in this repository due to its large size. Please download it from the official source and place it in the appropriate directory before running the code.

---

## Requirements

Python 3.9+

Required libraries:

* numpy
* json
* tensorflow 
* scikit-learn
* matplotlib
* librosa

Install dependencies using:

```
pip install -r requirements.txt
```

---

## How to Run

1. Download the ASVspoof 2019 LA dataset.
2. Update the dataset path in the configuration file.
3. Run the desired model script

## Results Summary

The BiGRU model achieved the highest overall test performance in terms of accuracy, F1-score, and robustness metrics. GRU demonstrated strong convergence and stable generalization, while LSTM and CNN also produced competitive results.

The detailed performance analysis is available in the thesis documentation.

---

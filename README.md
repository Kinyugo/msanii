# Msanii: High Fidelity Music Synthesis on a Shoestring Budget

A novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently.

<iframe src="https://drive.google.com/file/d/1Bz8XyyyQxBdtbzKBYOeJ5g9fGIcflr79/preview" width="640" height="480" allow="autoplay"></iframe>

## Abstract

> In this paper, we present Msanii, a novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently. Our model combines the expressiveness of mel spectrograms, the generative capabilities of diffusion models, and the vocoding capabilities of neural vocoders. We demonstrate the effectiveness of Msanii by synthesizing tens of seconds _190 seconds_ of _stereo_ music at high sample rates _44.1 kHz_ without the use of concatenative synthesis, cascading architectures, or compression techniques. To the best of our knowledge, this is the first work to successfully employ a diffusion-based model for synthesizing such long music samples at high sample rates. Our code and demo can be found [here](https://github.com/Kinyugo/msanii)

## Disclaimer

This is a work in progress and has not been finalized. The results and approach presented are subject to change and should not be considered final.

## Setup

Setup your virtual environment using conda or venv.

### Install package from git

```bash
    pip install -q git+https://github.com/Kinyugo/msanii@main
```

### Install package in edit mode

```bash
    git clone https://github.com/Kinyugo/msanii.git
    cd msanii
    pip install -q -r requirements.txt
    pip install -e .
```

## Training

### Notebook

<a target="_blank" href="https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### CLI

To train via CLI you need to define a config file. Check for sample config files within the `conf` directory.

```bash
    wandb login
    python -m msanii.scripts.training <path-to-your-config.yml-file>
```

## Inference

### Notebook

<a target="_blank" href="https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_inference.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### CLI

Msanii supports the following inference tasks:

- sampling
- audio2audio
- interpolation
- inpainting
- outpainting

Each task requires a different config file. Check `conf` directory for samples.

```bash
    python -m msanii.scripts.inference <task> <path-to-your-config.yml-file>
```

## Demo

### Notebook

<a target="_blank" href="https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### CLI

To run the demo via CLI you need to define a config file. Check for sample config files within the `conf` directory.

```bash
    python -m msanii.demo.demo <path-to-your-config.yml-file>
```

# Msanii: High Fidelity Music Synthesis on a Shoestring Budget

A novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently.

## Abstract

In this paper, we present Msanii, a novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently. Our model combines the expressiveness of mel spectrograms, the generative capabilities of diffusion models, and the vocoding capabilities of neural vocoders. We demonstrate the effectiveness of Msanii by synthesizing tens of seconds _190 seconds_ of _stereo_ music at high sample rates _44.1 kHz_ without the use of concatenative synthesis, cascading architectures, or compression techniques. To the best of our knowledge, this is the first work to successfully employ a diffusion-based model for synthesizing such long music samples at high sample rates. Our code and demo can be found [here](https://github.com/Kinyugo/msanii)

## Disclaimer

This paper is a work in progress and has not been finalized. The results presented in this paper are subject to change and should not be considered final.

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

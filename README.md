# Msanii: High Fidelity Music Synthesis on a Shoestring Budget

[![arXiv](https://img.shields.io/badge/arXiv-2301.06468-<COLOR>.svg)](https://arxiv.org/abs/2301.06468) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kinyugo/msanii) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_demo.ipynb) [![GitHub Repo stars](https://img.shields.io/github/stars/Kinyugo/msanii?style=social) ](https://github.com/Kinyugo/msanii)

A novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently.

## Abstract

> In this paper, we present Msanii, a novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently. Our model combines the expressiveness of mel spectrograms, the generative capabilities of diffusion models, and the vocoding capabilities of neural vocoders. We demonstrate the effectiveness of Msanii by synthesizing tens of seconds (_190 seconds_) of _stereo_ music at high sample rates (_44.1 kHz_) without the use of concatenative synthesis, cascading architectures, or compression techniques. To the best of our knowledge, this is the first work to successfully employ a diffusion-based model for synthesizing such long music samples at high sample rates. Our demo can be found [here](https://kinyugo.github.io/msanii-demo) and our code [here](https://github.com/Kinyugo/msanii).

## Disclaimer

This is a work in progress and has not been finalized. The results and approach presented are subject to change and should not be considered final.

## Samples

See more [here](https://kinyugo.github.io/msanii-demo/).

|                                                              **Midnight Melodies**                                                              |                                                           **Echoes of Yesterday**                                                           |
| :---------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
|     [ ![ Midnight Melodies ](http://img.youtube.com/vi/cFrpR0wc_A4/0.jpg) ](http://www.youtube.com/watch?v=cFrpR0wc_A4 "Midnight Melodies")     | [ ![ Echoes of Yesterday ](http://img.youtube.com/vi/tWlEqkRxZSU/0.jpg) ](http://www.youtube.com/watch?v=tWlEqkRxZSU "Echoes of Yesterday") |
|                                                            **Rainy Day Reflections**                                                            |                                                            **Starlight Sonatas**                                                            |
| [ ![ Rainy Day Reflections ](http://img.youtube.com/vi/-ZikAJxNomM/0.jpg) ](http://www.youtube.com/watch?v=-ZikAJxNomM "Rainy Day Reflections") |   [ ![ Starlight Sonatas ](http://img.youtube.com/vi/3adYlNVZSxA/0.jpg) ](http://www.youtube.com/watch?v=3adYlNVZSxA "Starlight Sonatas")   |

## Setup

Setup your virtual environment using conda or venv.

### Install package from git

```bash
    pip install -q git+https://github.com/Kinyugo/msanii.git
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
    gdown 1G9kF0r5vxYXPSdSuv4t3GR-sBO8xGFCe # model checkpoint
    python -m msanii.scripts.inference <task> <path-to-your-config.yml-file>
```

## Demo

### HF Spaces & Notebook

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kinyugo/msanii) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kinyugo/msanii/blob/main/notebooks/msanii_demo.ipynb)

### CLI

To run the demo via CLI you need to define a config file. Check for sample config files within the `conf` directory.

```bash
    gdown 1G9kF0r5vxYXPSdSuv4t3GR-sBO8xGFCe # model checkpoint
    python -m msanii.demo.demo <path-to-your-config.yml-file>
```

## Contribute to the Project

We are always looking for ways to improve and expand our project, and we welcome contributions from the community. Here are a few ways you can get involved:

- **Bug Fixes and Feature Requests:** If you find any issues with the project, please open a GitHub issue or submit a pull request with a fix.
- **Data Collection:** We are always in need of more data to improve the performance of our models. If you have any relevant data that you would like to share, please let us know.
- **Feedback:** We value feedback from our users and would love to hear your thoughts on the project. Please feel free to reach out to us with any suggestions or comments.
- **Funding:** If you find our project helpful, consider supporting us through GitHub Sponsors. Your support will help us continue to maintain and improve the project.
- **Computational Resources:** If you have access to computational resources such as GPU clusters, you can help us by providing access to these resources to run experiments and improve the project.
- **Code Contributions:** If you are a developer and want to contribute to the codebase, feel free to open a pull request.
- **Documentation:** If you have experience with documentation and want to help improve the project's documentation please let us know.
- **Promotion:** Help increase the visibility and attract more contributors by sharing the project with your friends, colleagues, and on social media.
- **Educational Material:** If you are an educator or content creator you can help by creating tutorials, guides or educational material that can help others understand the project better.
- **Discussing and Sharing Ideas:** Even if you don't have the time or technical skills to contribute directly to the code or documentation, you can still help by sharing and discussing ideas with the community. This can help identify new features or use cases, or find ways to improve existing ones.
- **Ethical Review:** Help us ensure that the project follows ethical standards by reviewing data and models for potential infringements. Additionally, please do not use the project or its models to train or generate copyrighted works without proper authorization.

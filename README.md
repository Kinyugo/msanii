# Msanii: High Fidelity Music Synthesis on a Shoestring Budget

A novel diffusion-based model for synthesizing long-context, high-fidelity music efficiently.

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

## Contribute to the Project

We are always looking for ways to improve and expand our project, and we welcome contributions from the community. Here are a few ways you can get involved:

- **Bug fixes and feature requests:** If you find any issues with the project, please open a GitHub issue or submit a pull request with a fix.
- **Data collection:** We are always in need of more data to improve the performance of our models. If you have any relevant data that you would like to share, please let us know.
- **Feedback:** We value feedback from our users and would love to hear your thoughts on the project. Please feel free to reach out to us with any suggestions or comments.
- **Funding:** If you find our project helpful, consider supporting us through GitHub Sponsors. Your support will help us continue to maintain and improve the project.
- **Computational resources:** If you have access to computational resources such as GPU clusters, you can help us by providing access to these resources to run experiments and improve the project.
- **Code contributions:** If you are a developer and want to contribute to the codebase, feel free to open a pull request.
- **Documentation:** If you have experience with documentation and want to help improve the project's documentation please let us know.
- **Marketing:** Share the project with your friends, colleagues, and on social media to help increase its visibility and attract more contributors.
- **Educational Material:** If you are an educator or content creator you can help by creating tutorials, guides or educational material that can help others understand the project better.
- **Discussing and sharing ideas:** Even if you don't have the time or technical skills to contribute directly to the code or documentation, you can still help by sharing and discussing ideas with the community. This can help identify new features or use cases, or find ways to improve existing ones.

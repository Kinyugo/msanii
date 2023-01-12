from setuptools import find_packages, setup

# Metadata
NAME = "msanii"
DESCRIPTION = "Msanii: High Fidelity Music Synthesis on a Shoestring Budget"
URL = "https://github.com/Kinyugo/msanii"
EMAIL = "kinyugomaina@gmail.com"
AUTHOR = "Kinyugo Maina"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = "0.0.1"

# Required packages
REQUIRED = [
    "torch>=1.6",
    "torchaudio",
    "lightning",
    "diffusers",
    "tqdm",
    "numpy",
    "einops>=0.4",
    "gradio",
    "matplotlib",
    "typing_extensions",
]

# Extra packages
EXTRAS = {}

# Load long description or fallback to short description
try:
    with open("README.md", "r", encoding="Utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Setup package
setup(
    name=NAME,
    version=VERSION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    keywords=[
        "artificial intelligence",
        "deep learning",
        "audio synthesis",
        "music synthesis",
    ],
)

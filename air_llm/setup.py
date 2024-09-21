import sys
import setuptools
from setuptools.command.install import install
import subprocess

# upgrade transformers to latest version to avoid "`rope_scaling` must be a dictionary with two fields" error
class PostInstallCommand(install):
    def run(self):
        install.run(self)
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
        except subprocess.CalledProcessError:
            print("Warning: Unable to upgrade transformers package. Please upgrade manually.")

# Windows uses a different default encoding (use a consistent encoding)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="airllm",
    version="2.11.0",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="AirLLM allows single 4GB GPU card to run 70B large language models without quantization, distillation or pruning. 8GB vmem to run 405B Llama3.1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyogavin/airllm",
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm',
        'torch',
        'transformers',
        'accelerate',
        'safetensors',
        'optimum',
        'huggingface-hub',
        'scipy',
        #'bitsandbytes' set it to optional to support fallback when not installable
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

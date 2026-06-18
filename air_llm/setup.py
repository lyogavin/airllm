import setuptools

# Windows uses a different default encoding (use a consistent encoding)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="airllm",
    version="3.0.0",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="AirLLM allows single 4GB GPU card to run 70B large language models without quantization, distillation or pruning. 8GB vmem to run 405B Llama3.1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyogavin/airllm",
    packages=setuptools.find_packages(),
    # Keep the dependency surface small and pinned to a tested range so a plain
    # `pip install airllm` gives users a known-good stack with no manual upgrades.
    install_requires=[
        'tqdm',
        'torch>=2.4',
        'transformers>=4.49,<5.13',
        'accelerate>=1.0',
        'safetensors',
        'huggingface-hub',
        'scipy',
        # 'bitsandbytes' is optional (used only for --compression); we fall back gracefully when absent.
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)

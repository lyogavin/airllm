import setuptools

# Windows uses a different default encoding (use a consistent encoding)
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="airllm",
    version="2.11.1",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="AirLLM allows single 4GB GPU card to run 70B large language models without quantization, distillation or pruning. 8GB vmem to run 405B Llama3.1.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lyogavin/airllm",
    packages=setuptools.find_packages(),
    install_requires=[
        "tqdm",
        "torch",
        "transformers>=4.38,<4.49",  # rope_scaling-safe range
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "scipy",
        # optimum intentionally NOT required by default
    ],
    extras_require={
        "bettertransformer": [
            "optimum>=1.27.0,<2.2.0",
            "transformers<4.49",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

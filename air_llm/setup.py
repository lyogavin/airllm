import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# The single source of truth for the README is the repository's top-level README.md. The release
# workflow copies it next to this setup.py before building, so it ships in both the sdist and the
# wheel; for a plain local checkout we fall back to the copy one directory up.
long_description = ""
for _readme in (os.path.join(here, "README.md"), os.path.join(here, os.pardir, "README.md")):
    if os.path.exists(_readme):
        with open(_readme, "r", encoding="utf-8") as fh:
            long_description = fh.read()
        break

setuptools.setup(
    name="airllm",
    version="3.0.0",
    author="Gavin Li",
    author_email="gavinli@animaai.cloud",
    description="AirLLM runs 70B large language models on a single 4GB GPU without quantization, "
                "distillation or pruning. 405B Llama 3.1 on 8GB, DeepSeek-V3 671B on ~12GB.",
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

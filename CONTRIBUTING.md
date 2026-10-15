# Contributing to AirLLM

First off, thank you for considering contributing to AirLLM! Because of community contributions, we're able to continuously optimize LLM inference on low-end commodity GPUs.

## 1. Where do I go from here?

If you've noticed a bug, want to request support for a new model (like QWen, Baichuan, Mistral, etc.), or have a feature request, please [open an issue](../../issues) on GitHub.

## 2. Fork & create a branch

If this is something you think you can fix or build, then [fork AirLLM](../../fork) and create a branch with a descriptive name.

A good branch name:

```sh
# For adding model support
git checkout -b feat/support-llama4

# For a bug fix
git checkout -b fix/safetensors-loading-error
```

## 3. Local Development

1. Install the dependencies locally:
   ```sh
   pip install -r requirements.txt
   ```
2. We recommend testing your changes on constrained hardware (like a 4GB or 8GB VRAM GPU) since AirLLM's entire goal is memory optimization.
3. If you integrate a new model, ensure it's evaluated against standard prompts to verify no major accuracy loss occurred during block-wise quantization/compression.

## 4. Get the style right

Your patch should follow the standard Python coding conventions used across the project. 
- Format your code (consider using tools like `black` or `flake8` if enforced).
- Do not check in large `.safetensors` or `.bin` model files. Always download them via Python scripts or point to `huggingface` handles.

## 5. Make a Pull Request

Make sure your fork is up-to-date with upstream before submitting your PR:

```sh
git remote add upstream https://github.com/lyogavin/airllm.git
git checkout main
git pull upstream main
```

Rebase or merge your feature branch:

```sh
git checkout feat/support-llama4
git rebase main
git push --set-upstream origin feat/support-llama4
```

Finally, go to GitHub and create a Pull Request on the main repository comparing against `lyogavin/airllm`.

Thank you for contributing!

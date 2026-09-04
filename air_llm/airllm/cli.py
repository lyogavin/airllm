import argparse
import sys
import os
from typing import Optional

from .models import (
    resolve_model_name,
    list_local_models,
    remove_model_cache,
    get_model_info,
    MODEL_ALIASES,
)
from .chat import InteractiveChatSession, Colors


def print_logo():
    print(f"""{Colors.CYAN}{Colors.BOLD}
     _    _      _     _     __  __ 
    / \  (_)_ __| |   | |   |  \/  |
   / _ \ | | '__| |   | |   | |\/| |
  / ___ \| | |  | |___| |___| |  | |
 /_/   \_\_|_|  |_____|_____|_|  |_|
{Colors.RESET}{Colors.DIM} Run 70B+ LLMs without massive VRAM | Easy CLI & Chat{Colors.RESET}
""")


def cmd_run(args):
    """Load model and launch interactive terminal chat session."""
    print_logo()
    model_name_or_path = resolve_model_name(args.model)
    display_name = args.model

    print(f"[*] Initializing model: {Colors.BOLD}{display_name}{Colors.RESET}")
    if model_name_or_path != display_name:
        print(f"[*] Resolved to HuggingFace repo: {Colors.DIM}{model_name_or_path}{Colors.RESET}")

    from .auto_model import AutoModel

    kwargs = {
        "device": args.device,
        "max_seq_len": args.max_seq_len,
    }
    if args.compression:
        kwargs["compression"] = args.compression
    if args.hf_token:
        kwargs["hf_token"] = args.hf_token

    try:
        print(f"[*] Loading layers (streaming from disk on demand)...")
        model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
    except Exception as e:
        print(f"\n{Colors.RED}[!] Failed to load model '{model_name_or_path}': {e}{Colors.RESET}")
        sys.exit(1)

    session = InteractiveChatSession(
        model=model,
        model_name=display_name,
        system_prompt=args.system or "You are a helpful, respectful, and honest AI assistant.",
        max_new_tokens=args.max_new_tokens,
        show_stats=not args.no_stats,
    )
    session.start_loop()


def cmd_pull(args):
    """Download and split a model into layer shards ahead of time."""
    print_logo()
    model_name_or_path = resolve_model_name(args.model)
    display_name = args.model

    print(f"[*] Pulling and sharding model: {Colors.BOLD}{display_name}{Colors.RESET}")
    if model_name_or_path != display_name:
        print(f"[*] Resolved to HuggingFace repo: {Colors.DIM}{model_name_or_path}{Colors.RESET}")

    from .utils import find_or_create_local_splitted_path

    try:
        model_path, split_path = find_or_create_local_splitted_path(
            model_name_or_path,
            compression=args.compression,
            hf_token=args.hf_token,
        )
        print(f"\n{Colors.GREEN}[+] Model successfully pulled and sharded!{Colors.RESET}")
        print(f"    Base checkpoint: {model_path}")
        print(f"    Shards cache:    {split_path}")
        print(f"\nYou can now chat with it anytime using:")
        print(f"    {Colors.CYAN}airllm run {display_name}{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}[!] Failed to pull model '{display_name}': {e}{Colors.RESET}")
        sys.exit(1)


def cmd_list(args):
    """List local cached and sharded models."""
    models = list_local_models()
    if not models:
        print("\nNo cached models found in HuggingFace/AirLLM storage.")
        print(f"To run or download a model, use: {Colors.CYAN}airllm run <model_name>{Colors.RESET}\n")
        return

    print(f"\n{Colors.BOLD}{'NAME':<24} {'REPO ID':<38} {'SHARDS':<10} {'SIZE':<10}{Colors.RESET}")
    print("-" * 86)
    for m in models:
        shards_info = "Ready" if m["sharded"] else "Not split"
        shards_color = Colors.GREEN if m["sharded"] else Colors.YELLOW
        print(f"{Colors.CYAN}{m['name']:<24}{Colors.RESET} {m['repo_id']:<38} {shards_color}{shards_info:<10}{Colors.RESET} {m['size_formatted']:<10}")
    print()


def cmd_show(args):
    """Show detailed model configuration and architecture information."""
    info = get_model_info(args.model, hf_token=args.hf_token)
    if "error" in info:
        print(f"\n{Colors.RED}[!] Could not retrieve model info: {info['error']}{Colors.RESET}\n")
        return

    print(f"\n{Colors.BOLD}Model Information:{Colors.RESET}")
    print(f"  Name:           {Colors.CYAN}{info['name']}{Colors.RESET}")
    print(f"  Repo ID:        {info['repo_id']}")
    print(f"  Architecture:   {info['architecture']}")
    print(f"  Layers:         {info['layers']}")
    print(f"  Hidden Size:    {info['hidden_size']}")
    print(f"  Attention Heads:{info['attention_heads']}")
    print(f"  Context Window: {info['context_length']}")
    print(f"  Precision:      {info['dtype']}\n")


def cmd_rm(args):
    """Delete model cache and splitted shards to free disk space."""
    resolved = resolve_model_name(args.model)
    if not args.yes:
        confirm = input(f"Are you sure you want to delete cache for '{args.model}' ({resolved})? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Deletion cancelled.")
            return

    success = remove_model_cache(args.model)
    if success:
        print(f"{Colors.GREEN}[+] Model cache for '{args.model}' deleted successfully.{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}[!] Model cache for '{args.model}' not found or already deleted.{Colors.RESET}")


def cmd_aliases(args):
    """List available friendly model aliases."""
    print(f"\n{Colors.BOLD}{'ALIAS':<22} {'HUGGINGFACE REPO':<50}{Colors.RESET}")
    print("-" * 74)
    for alias, repo in sorted(MODEL_ALIASES.items()):
        print(f"{Colors.CYAN}{alias:<22}{Colors.RESET} {repo:<50}")
    print(f"\nYou can run any of these with: {Colors.CYAN}airllm run <alias>{Colors.RESET}")
    print(f"Or pass any arbitrary HuggingFace repo ID: {Colors.CYAN}airllm run <org/repo>{Colors.RESET}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airllm",
        description="AirLLM: Run 70B+ LLMs without needing massive GPU memory. Interactive CLI & Chat like Ollama.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to execute")

    # run
    p_run = subparsers.add_parser("run", help="Run a model and enter interactive chat session")
    p_run.add_argument("model", help="Model alias (e.g. llama3:70b, deepseek-r1:70b, qwen2.5:72b) or HuggingFace repo ID")
    p_run.add_argument("-c", "--compression", choices=["4bit", "8bit"], default=None, help="Block-wise quantization for on-disk shards (requires bitsandbytes)")
    p_run.add_argument("-d", "--device", default="cuda:0", help="Execution device (default: cuda:0)")
    p_run.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length (default: 512)")
    p_run.add_argument("--max-new-tokens", type=int, default=512, help="Max new tokens generated per turn (default: 512)")
    p_run.add_argument("--system", type=str, default=None, help="Custom system prompt")
    p_run.add_argument("--no-stats", action="store_true", help="Hide generation speed stats (tokens/s)")
    p_run.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token for gated models")
    p_run.set_defaults(func=cmd_run)

    # pull
    p_pull = subparsers.add_parser("pull", help="Download and shard a model in advance")
    p_pull.add_argument("model", help="Model alias or HuggingFace repo ID")
    p_pull.add_argument("-c", "--compression", choices=["4bit", "8bit"], default=None, help="Compression type (4bit or 8bit)")
    p_pull.add_argument("--hf-token", type=str, default=None, help="Hugging Face token")
    p_pull.set_defaults(func=cmd_pull)

    # list / ls
    p_list = subparsers.add_parser("list", aliases=["ls"], help="List local cached models and shard readiness")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show architecture and context info for a model")
    p_show.add_argument("model", help="Model alias or repo ID")
    p_show.add_argument("--hf-token", type=str, default=None, help="Hugging Face token")
    p_show.set_defaults(func=cmd_show)

    # rm
    p_rm = subparsers.add_parser("rm", help="Remove cached model layers to free disk space")
    p_rm.add_argument("model", help="Model alias or repo ID to remove")
    p_rm.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_rm.set_defaults(func=cmd_rm)

    # aliases
    p_aliases = subparsers.add_parser("aliases", help="List available friendly model aliases")
    p_aliases.set_defaults(func=cmd_aliases)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        print_logo()
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

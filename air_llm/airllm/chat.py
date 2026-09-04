import sys
import time
import threading
from typing import List, Dict, Optional, Any
from transformers import TextIteratorStreamer

# ANSI Terminal Color Helpers
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

    @classmethod
    def enable_windows_ansi(cls):
        """Enable ANSI escape codes in Windows cmd/PowerShell console."""
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
                mode = ctypes.c_ulong()
                kernel32.GetConsoleMode(handle, ctypes.byref(mode))
                mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                kernel32.SetConsoleMode(handle, mode)
            except Exception:
                pass


Colors.enable_windows_ansi()


DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful, and honest AI assistant."


class InteractiveChatSession:
    """Manages an interactive multi-turn terminal chat session for an AirLLM model."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 512,
        show_stats: bool = True,
    ):
        self.model = model
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.show_stats = show_stats
        self.messages: List[Dict[str, str]] = []
        self._init_history()

    def _init_history(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def print_banner(self):
        print(f"\n{Colors.CYAN}{Colors.BOLD}======================================================{Colors.RESET}")
        print(f"  {Colors.BOLD}AirLLM Interactive Chat{Colors.RESET} - {Colors.GREEN}{self.model_name}{Colors.RESET}")
        print(f"  Type your message and press {Colors.BOLD}Enter{Colors.RESET}.")
        print(f"  Commands: {Colors.YELLOW}/help{Colors.RESET}, {Colors.YELLOW}/clear{Colors.RESET}, {Colors.YELLOW}/history{Colors.RESET}, {Colors.YELLOW}/stats{Colors.RESET}, {Colors.YELLOW}/bye{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}======================================================{Colors.RESET}\n")

    def print_help(self):
        print(f"\n{Colors.BOLD}Available Commands:{Colors.RESET}")
        print(f"  {Colors.YELLOW}/bye{Colors.RESET} or {Colors.YELLOW}/exit{Colors.RESET}   : Exit the chat session")
        print(f"  {Colors.YELLOW}/clear{Colors.RESET}          : Reset the conversation history")
        print(f"  {Colors.YELLOW}/history{Colors.RESET}        : Display past turns in this conversation")
        print(f"  {Colors.YELLOW}/system <msg>{Colors.RESET}   : Set a new system prompt")
        print(f"  {Colors.YELLOW}/stats{Colors.RESET}          : Toggle generation speed statistics")
        print(f"  {Colors.YELLOW}/help{Colors.RESET}           : Show this help message\n")

    def print_history(self):
        print(f"\n{Colors.BOLD}--- Conversation History ---{Colors.RESET}")
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]
            if role == "SYSTEM":
                print(f"{Colors.YELLOW}[SYSTEM]{Colors.RESET} {content}")
            elif role == "USER":
                print(f"{Colors.CYAN}[USER]{Colors.RESET} {content}")
            else:
                print(f"{Colors.GREEN}[ASSISTANT]{Colors.RESET} {content}")
        print(f"{Colors.BOLD}----------------------------{Colors.RESET}\n")

    def format_prompt(self) -> str:
        """Format the message history using the tokenizer chat template or standard fallback."""
        tokenizer = self.model.tokenizer

        # Try tokenizer's built-in chat template
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                return tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback format
        lines = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lines.append(f"System: {content}\n")
            elif role == "user":
                lines.append(f"User: {content}\n")
            elif role == "assistant":
                lines.append(f"Assistant: {content}\n")
        lines.append("Assistant: ")
        return "".join(lines)

    def stream_response(self, prompt_text: str) -> str:
        """Stream model generation tokens in real time to stdout and return full response."""
        tokenizer = self.model.tokenizer
        max_len = getattr(self.model, "max_seq_len", 512)

        input_tokens = tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=max_len,
        )

        input_ids = input_tokens["input_ids"]
        # Move to model's execution device
        running_device = getattr(self.model, "running_device", "cuda:0")
        try:
            input_ids = input_ids.to(running_device)
        except Exception:
            pass

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )

        # Launch generation in background thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        start_time = time.time()
        thread.start()

        full_response = []
        token_count = 0

        # Stream tokens live
        for new_text in streamer:
            sys.stdout.write(new_text)
            sys.stdout.flush()
            full_response.append(new_text)
            token_count += 1

        thread.join()
        elapsed = time.time() - start_time
        print()  # Newline after stream ends

        response_str = "".join(full_response).strip()

        if self.show_stats and elapsed > 0 and token_count > 0:
            speed = token_count / elapsed
            print(f"{Colors.DIM}({token_count} tokens, {elapsed:.2f}s, {speed:.2f} tokens/s){Colors.RESET}")

        return response_str

    def step(self, user_input: str) -> bool:
        """Process a single turn. Returns False if the session should exit."""
        text = user_input.strip()
        if not text:
            return True

        # Check slash commands
        if text.lower() in ("/bye", "/exit", "/quit"):
            print(f"\n{Colors.CYAN}Goodbye!{Colors.RESET}\n")
            return False

        if text.lower() == "/clear":
            self._init_history()
            print(f"{Colors.YELLOW}Conversation history cleared.{Colors.RESET}")
            return True

        if text.lower() == "/history":
            self.print_history()
            return True

        if text.lower() == "/stats":
            self.show_stats = not self.show_stats
            state = "enabled" if self.show_stats else "disabled"
            print(f"{Colors.YELLOW}Generation stats {state}.{Colors.RESET}")
            return True

        if text.lower() == "/help":
            self.print_help()
            return True

        if text.lower().startswith("/system"):
            new_sys = text[len("/system"):].strip()
            if new_sys:
                self.system_prompt = new_sys
                self._init_history()
                print(f"{Colors.YELLOW}System prompt updated and conversation reset.{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}Current system prompt:{Colors.RESET} {self.system_prompt}")
            return True

        # Process user message
        self.messages.append({"role": "user", "content": text})
        prompt = self.format_prompt()

        try:
            response = self.stream_response(prompt)
            self.messages.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}[Generation interrupted by user]{Colors.RESET}")
        except Exception as e:
            print(f"\n{Colors.RED}Generation Error: {e}{Colors.RESET}")

        return True

    def start_loop(self):
        """Start the interactive REPL prompt loop."""
        self.print_banner()
        while True:
            try:
                prompt_prefix = f"{Colors.BOLD}{Colors.CYAN}>>> {Colors.RESET}"
                user_input = input(prompt_prefix)
                if not self.step(user_input):
                    break
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{Colors.CYAN}Exiting AirLLM chat.{Colors.RESET}")
                break

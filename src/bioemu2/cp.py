"""Context-parallel CLI entry point for ``torchrun --module bioemu2.cp``."""

from bioemu2.cli import context_parallel_commands

if __name__ == "__main__":
    context_parallel_commands()

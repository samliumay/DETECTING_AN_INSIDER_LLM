"""Repository-level entry point for the command-line interface."""

#Calls the main 'main' from the cli.
from src.detecting_an_insider_llm.cli import main


#Runs the main 'main' from the cli.
if __name__ == "__main__":
    raise SystemExit(main())

"""Top-level entrypoint wrapper — avoids Windows console-script package resolution issues."""
from aao_cli.main import app

if __name__ == "__main__":
    app()

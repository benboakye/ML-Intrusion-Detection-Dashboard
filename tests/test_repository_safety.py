from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sensitive_and_generated_files_are_ignored() -> None:
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    required_rules = {
        ".venv/",
        ".env",
        "data/raw/**",
        "data/processed/**",
        "data/replay/**",
        "*.joblib",
        "*.pcap",
    }
    assert required_rules <= set(rules)

from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08b_run_behaviour_modelling_no_roberta.py"
SCRIPT_TEXT = SCRIPT_PATH.read_text(encoding="utf-8")


def test_resolve_preprocessor_path_uses_none_fallback():
    assert "_resolve_preprocessor_path" in SCRIPT_TEXT
    assert 'Path("")' not in SCRIPT_TEXT
    assert "return None" in SCRIPT_TEXT


def test_no_roberta_script_does_not_load_missing_config():
    assert "pre_path is not None and pre_path.exists() and pre_path.is_file()" in SCRIPT_TEXT
    assert "_load_preprocessor_from_config(pre_path)" in SCRIPT_TEXT
    assert "return None" in SCRIPT_TEXT

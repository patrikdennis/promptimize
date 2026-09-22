"""Tests for lib/config.py: the dependency-free YAML-subset parser and the
defaults/deep-merge fallback behavior."""
import config as config_mod


def test_parses_nested_block_mapping_and_scalars(tmp_path):
    text = """
scoring:
  specificity:
    vague_weight: 0.55
    ideal_length_min: 6
team_mode:
  anonymize_labels: true
  min_users_for_aggregate: 2
"""
    parsed = config_mod._parse_yaml_subset(text)
    assert parsed["scoring"]["specificity"]["vague_weight"] == 0.55
    assert parsed["scoring"]["specificity"]["ideal_length_min"] == 6
    assert parsed["team_mode"]["anonymize_labels"] is True
    assert parsed["team_mode"]["min_users_for_aggregate"] == 2


def test_coerces_bool_int_float_and_quoted_string():
    assert config_mod._coerce_scalar("true") is True
    assert config_mod._coerce_scalar("false") is False
    assert config_mod._coerce_scalar("yes") is True
    assert config_mod._coerce_scalar("no") is False
    assert config_mod._coerce_scalar("42") == 42
    assert config_mod._coerce_scalar("0.25") == 0.25
    assert config_mod._coerce_scalar("'hello'") == "hello"
    assert config_mod._coerce_scalar('"world"') == "world"
    assert config_mod._coerce_scalar("bareword") == "bareword"


def test_strips_comments():
    text = """
scoring:  # a top-level comment
  specificity:
    vague_weight: 0.55  # trailing comment
"""
    parsed = config_mod._parse_yaml_subset(text)
    assert parsed["scoring"]["specificity"]["vague_weight"] == 0.55


def test_load_config_falls_back_to_defaults_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    cfg = config_mod.load_config(path=missing)
    assert cfg == config_mod.DEFAULTS


def test_load_config_deep_merges_partial_override(tmp_path):
    custom = tmp_path / "config.yaml"
    custom.write_text("""
scoring:
  specificity:
    vague_weight: 0.9
""")
    cfg = config_mod.load_config(path=custom)
    # Overridden leaf value takes effect...
    assert cfg["scoring"]["specificity"]["vague_weight"] == 0.9
    # ...while sibling keys not present in the override fall back to defaults.
    assert cfg["scoring"]["specificity"]["length_weight"] == config_mod.DEFAULTS["scoring"]["specificity"]["length_weight"]
    # ...and entirely untouched top-level sections are preserved verbatim.
    assert cfg["team_mode"] == config_mod.DEFAULTS["team_mode"]


def test_load_config_ignores_unparseable_file_and_uses_defaults(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_bytes(b"\xff\xfe\x00bad-bytes-not-utf8")
    cfg = config_mod.load_config(path=bad)
    assert cfg["scoring"]["specificity"]["vague_weight"] == config_mod.DEFAULTS["scoring"]["specificity"]["vague_weight"]


def test_load_config_default_path_is_cached():
    cfg1 = config_mod.load_config()
    cfg2 = config_mod.load_config()
    assert cfg1 is cfg2

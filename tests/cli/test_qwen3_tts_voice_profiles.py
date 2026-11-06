"""Tests for the named voice profiles.

A profile pins one description (and optionally a language, temperature and
seed) to a name, so a project can keep the same narrator across runs.
"""

import json
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from mflux.models.qwen3_tts_voice import profiles
from mflux.models.qwen3_tts_voice.cli import voice_design


def _args(**kwargs):
    base = {
        "voice": None,
        "voice_file": None,
        "voice_profile": None,
        "voice_profile_file": None,
        "language": None,
        "temperature": None,
        "seed": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_builtin_profiles_are_available():
    available = profiles.load_profiles()
    assert "narrator-male" in available
    assert available["narrator-male"].description
    assert available["narrator-male"].source == "builtin"


def test_a_user_file_overrides_a_builtin_and_adds_a_profile(tmp_path, monkeypatch):
    path = tmp_path / "voices.json"
    path.write_text(
        json.dumps(
            {
                "narrator-male": {"description": "一位沉稳低沉的男性旁白", "language": "chinese", "seed": 7},
                "episode-3": "a bright young female voice",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv(profiles.PROFILE_ENV_VAR, raising=False)
    available = profiles.load_profiles(str(path))
    assert available["narrator-male"].description == "一位沉稳低沉的男性旁白"
    assert available["narrator-male"].language == "chinese"
    assert available["narrator-male"].seed == 7
    assert available["narrator-male"].source == str(path)
    # A plain string is accepted as a description-only profile.
    assert available["episode-3"].description == "a bright young female voice"
    # Untouched built-ins survive.
    assert "news-anchor" in available


def test_the_profile_file_is_found_through_the_environment(tmp_path, monkeypatch):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"env-profile": "a voice from the environment"}), encoding="utf-8")
    monkeypatch.setenv(profiles.PROFILE_ENV_VAR, str(path))
    assert profiles.load_profiles()["env-profile"].description == "a voice from the environment"


def test_an_unknown_profile_lists_the_known_ones(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(json.dumps({"only-one": "a voice"}), encoding="utf-8")
    with pytest.raises(ValueError) as excinfo:
        profiles.resolve_profile("missing", str(path))
    message = str(excinfo.value)
    assert "missing" in message
    assert "only-one" in message


def test_command_line_values_win_over_the_profile(tmp_path):
    path = tmp_path / "voices.json"
    path.write_text(
        json.dumps(
            {
                "p": {
                    "description": "a profile voice",
                    "language": "chinese",
                    "temperature": 0.8,
                    "seed": 11,
                }
            }
        ),
        encoding="utf-8",
    )
    parser = ArgumentParser()
    selection = profiles.resolve_selection(
        _args(voice_profile="p", voice_profile_file=str(path)),
        parser,
    )
    assert selection.description == "a profile voice"
    assert selection.language == "chinese"
    assert selection.temperature == 0.8
    assert selection.seed == 11
    assert selection.profile.name == "p"

    overridden = profiles.resolve_selection(
        _args(
            voice_profile="p",
            voice_profile_file=str(path),
            language="english",
            temperature=1.2,
            seed=99,
        ),
        parser,
    )
    assert overridden.language == "english"
    assert overridden.temperature == 1.2
    assert overridden.seed == 99


def test_a_profile_cannot_be_combined_with_a_description():
    parser = ArgumentParser()
    with pytest.raises(SystemExit):
        profiles.resolve_selection(_args(voice="a voice", voice_profile="p"), parser)


def test_the_voice_design_cli_prints_the_profile_list(capsys, monkeypatch):
    from unittest.mock import patch

    monkeypatch.delenv(profiles.PROFILE_ENV_VAR, raising=False)
    with patch("sys.argv", ["mlxgen", "voice-design", "--list-voice-profiles"]):
        voice_design.main()
    out = capsys.readouterr().out
    assert "narrator-male" in out
    assert "Voice profiles:" in out

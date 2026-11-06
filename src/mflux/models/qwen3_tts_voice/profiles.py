"""Named voice profiles: pin one description to one narrator across projects.

A voice description is a long string that a project has to keep consistent —
"the narrator of episode 3" has to sound like the narrator of episode 1. Passing
the same sentence on every call works, but it is easy to drift: a comma moves
and the timbre moves with it.

A profile gives that sentence a name. Profiles live in a small JSON file:

    {
      "narrator-male": {
        "description": "a deep, warm male narrator voice, steady and unhurried",
        "language": "chinese",
        "seed": 42
      }
    }

`language`, `temperature` and `seed` are optional; a value given on the command
line always wins over the profile's.

Profile files are searched in this order (the first file that exists is used,
and its entries override the built-in starters):

    1. --voice-profile-file PATH
    2. $MLXGEN_VOICE_PROFILES
    3. ./mlxgen-voice-profiles.json
    4. ~/.config/mlxgen/voice-profiles.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from mflux.cli.runtime_events import cli_print
from mflux.models.qwen3_tts_voice import constants

PROFILE_FILE_NAME = "mlxgen-voice-profiles.json"
PROFILE_ENV_VAR = "MLXGEN_VOICE_PROFILES"

# Starter profiles. They are deliberately plain: the point of a starter is that
# it sounds acceptable without editing, not that it is the only right voice.
BUILTIN_PROFILES: dict[str, dict[str, object]] = {
    "narrator-male": {
        "description": "a deep, warm male narrator voice, steady and unhurried",
    },
    "narrator-female": {
        "description": "a warm female narrator voice in her thirties, calm and clear",
    },
    "documentary": {
        "description": "a calm male documentary voice, low and measured, with a hint of gravel",
    },
    "news-anchor": {
        "description": "a clear female news-anchor voice, neutral and professional",
    },
    "warm-female": {
        "description": "a gentle female voice with a warm, slightly breathy timbre, smiling",
    },
    "energetic-young": {
        "description": "a bright, energetic young voice, quick and enthusiastic",
    },
}


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    description: str
    language: str | None = None
    temperature: float | None = None
    seed: int | None = None
    source: str = "builtin"

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "source": self.source,
        }
        if self.language is not None:
            payload["language"] = self.language
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload


@dataclass
class VoiceSelection:
    """What the CLI should use after merging the profile with the command line."""

    description: str | None = None
    language: str | None = None
    temperature: float | None = None
    seed: int | None = None
    profile: VoiceProfile | None = field(default=None)


def profile_search_paths(explicit: str | None = None) -> list[Path]:
    """The candidate profile files, most specific first."""
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    env_value = os.environ.get(PROFILE_ENV_VAR)
    if env_value:
        paths.append(Path(env_value).expanduser())
    paths.append(Path.cwd() / PROFILE_FILE_NAME)
    paths.append(Path.home() / ".config" / "mlxgen" / PROFILE_FILE_NAME)
    return paths


def load_profiles(explicit: str | None = None) -> dict[str, VoiceProfile]:
    """Built-in starters merged with the first profile file that exists."""
    profiles: dict[str, VoiceProfile] = {
        name: VoiceProfile(name=name, description=str(entry["description"]))
        for name, entry in BUILTIN_PROFILES.items()
    }

    for path in profile_search_paths(explicit):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read the voice profile file {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"The voice profile file {path} must contain a JSON object of profiles.")
        for name, entry in raw.items():
            profiles[str(name)] = _profile_from_entry(str(name), entry, str(path))
        break  # the first file that exists wins

    return profiles


def _profile_from_entry(name: str, entry: object, source: str) -> VoiceProfile:
    if isinstance(entry, str):
        entry = {"description": entry}
    if not isinstance(entry, dict):
        raise ValueError(f"The voice profile {name!r} in {source} must be a string or an object.")
    description = str(entry.get("description", "")).strip()
    if not description:
        raise ValueError(f"The voice profile {name!r} in {source} has no description.")
    language = entry.get("language")
    temperature = entry.get("temperature")
    seed = entry.get("seed")
    return VoiceProfile(
        name=name,
        description=description,
        language=str(language) if language is not None else None,
        temperature=float(temperature) if temperature is not None else None,
        seed=int(seed) if seed is not None else None,
        source=source,
    )


def resolve_profile(name: str, explicit: str | None = None) -> VoiceProfile:
    profiles = load_profiles(explicit)
    if name not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown voice profile {name!r}. Available profiles: {known}")
    return profiles[name]


def format_profile_list(profiles: dict[str, VoiceProfile]) -> str:
    lines = ["Voice profiles:"]
    for name in sorted(profiles):
        profile = profiles[name]
        extras = []
        if profile.language:
            extras.append(f"language={profile.language}")
        if profile.temperature is not None:
            extras.append(f"temperature={profile.temperature}")
        if profile.seed is not None:
            extras.append(f"seed={profile.seed}")
        suffix = f"  [{', '.join(extras)}]" if extras else ""
        lines.append(f"  {name}{suffix}\n    {profile.description}")
        lines.append(f"    source: {profile.source}")
    return "\n".join(lines)


def print_profile_list(explicit: str | None = None, *, json_events: bool = False) -> None:
    cli_print(format_profile_list(load_profiles(explicit)), json_events=json_events)


def resolve_selection(
    args,
    parser,
    *,
    required: bool = True,
    voice_attr: str = "voice",
    file_attr: str = "voice_file",
) -> VoiceSelection:
    """Merge the description options and --voice-profile into one selection.

    Command-line values always win over the profile's, so a profile can set the
    voice and the language while a single call still overrides the language.
    `voice_attr`/`file_attr` name the arguments that carry the description
    (`mlxgen dub` calls it `--voice-design`).
    """
    profile_name = getattr(args, "voice_profile", None)
    has_description = getattr(args, voice_attr, None) is not None or getattr(args, file_attr, None) is not None
    if profile_name is not None and has_description:
        parser.error("--voice-profile cannot be combined with --voice or --voice-file.")

    selection = VoiceSelection()
    profile: VoiceProfile | None = None
    if profile_name is not None:
        try:
            profile = resolve_profile(profile_name, getattr(args, "voice_profile_file", None))
        except ValueError as exc:
            parser.error(str(exc))
        selection.description = profile.description
        selection.language = profile.language
        selection.temperature = profile.temperature
        selection.seed = profile.seed
        selection.profile = profile
    elif has_description:
        selection.description = _read_description(args, parser, voice_attr, file_attr)

    if selection.description is None:
        if required:
            parser.error(
                "Provide --voice <description> (or --voice-file) describing the voice to design, "
                'e.g. --voice "a warm female voice in her thirties, speaking slowly", '
                "or --voice-profile <name>."
            )
        return selection

    # Command-line values win over the profile's.
    if getattr(args, "language", None) is not None:
        selection.language = args.language
    if getattr(args, "temperature", None) is not None:
        selection.temperature = args.temperature
    if getattr(args, "seed", None) is not None:
        selection.seed = args.seed
    return selection


def _read_description(args, parser, voice_attr: str = "voice", file_attr: str = "voice_file") -> str:
    voice_file = getattr(args, file_attr, None)
    if voice_file is not None:
        try:
            description = Path(voice_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            parser.error(f"Could not read --{file_attr.replace('_', '-')} {voice_file}: {exc}")
    else:
        description = (getattr(args, voice_attr, None) or "").strip()
    if not description:
        parser.error("Voice description is empty.")
    return description


def normalize_profile_language(language: str | None) -> str:
    """Profile languages go through the same validation as command-line ones."""
    if language is None:
        return "auto"
    return constants.normalize_language(language)

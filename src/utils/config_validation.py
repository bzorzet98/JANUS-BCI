"""
Config validation before a stage runs (R14, STAGE_CONTRACTS.md section 0.4).

This module holds the mechanism only. The rules of each stage are data
(ConfigSpec), written next to the domain code and passed in by the command.
validate_config collects every problem and raises one error, so the command
can stop before anything is written.
"""
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Tuple


@dataclass(frozen=True)
class ConfigSpec:
    """Rules for the config of one script_type.

    required: dotted paths (`project_config.label`) that must exist and not
        be null or empty text.
    old_keys: old-format key name -> how to replace it. Matched at any depth.
    old_module_names: old `module_name` value -> where the code is now. Also
        matches its submodules. Matched at any depth, lists included.
    same_value: pairs of dotted paths that must hold the same value when both
        are given. The message asks to change the first one.
    """
    required: Tuple[str, ...] = ()
    old_keys: Mapping[str, str] = field(default_factory=dict)
    old_module_names: Mapping[str, str] = field(default_factory=dict)
    same_value: Tuple[Tuple[str, str], ...] = ()


_MISSING = object()


def _get(config: dict, dotted_path: str) -> Any:
    """Value at `a.b.c`, or _MISSING if any level is absent."""
    node = config
    for key in dotted_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _walk(node: Any, path: str = "") -> Iterator[Tuple[str, str, Any]]:
    """Yields (path, key, value) for every key of every dict, inside nested
    dicts and lists. Paths look like `callbacks[0].module_name`."""
    if isinstance(node, dict):
        for key, value in node.items():
            key_path = f"{path}.{key}" if path else key
            yield key_path, key, value
            yield from _walk(value, key_path)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from _walk(item, f"{path}[{i}]")


def validate_config(config: dict, specs: Mapping[str, ConfigSpec]) -> None:
    """Raises one ValueError that lists every problem of `config`, or returns
    None if there is none. `specs` maps each script_type the command accepts
    to its rules."""
    script_type = config.get("script_type")
    if script_type not in specs:
        expected = " or ".join(f"'{name}'" for name in specs)
        raise ValueError(
            f"script_type is {script_type!r}, but this command expects {expected}. "
            f"Use the command for that script_type, or fix script_type in the config."
        )
    spec = specs[script_type]

    problems = []
    for path, key, value in _walk(config):
        if key in spec.old_keys:
            problems.append(f"{path}: old-format key. {spec.old_keys[key]}")
        if key == "module_name" and isinstance(value, str):
            for old_module, fix in spec.old_module_names.items():
                if value == old_module or value.startswith(old_module + "."):
                    problems.append(f"{path} is '{value}', an old module. {fix}")

    for path in spec.required:
        value = _get(config, path)
        if value is _MISSING or value is None or (isinstance(value, str) and not value.strip()):
            problems.append(f"{path} is missing or empty. Add it.")

    for path_to_change, other_path in spec.same_value:
        value, other = _get(config, path_to_change), _get(config, other_path)
        both_given = all(v is not _MISSING and v is not None for v in (value, other))
        if both_given and value != other:
            problems.append(
                f"{path_to_change} is {value!r} but {other_path} is {other!r}. "
                f"Change {path_to_change} to {other!r}."
            )

    if problems:
        listed = "\n".join(f"  - {problem}" for problem in problems)
        raise ValueError(
            f"The config is not valid for script_type '{script_type}' "
            f"({len(problems)} problem(s)). Fix them and run again:\n{listed}"
        )

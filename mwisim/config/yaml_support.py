"""Safe YAML loading with an offline fallback for the Phoenix schema subset."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path


def _strip_comment(line: str) -> str:
    quote = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote is not None:
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _split_key(text: str) -> tuple[str, str]:
    quote = None
    depth = 0
    for index, char in enumerate(text):
        if char in ("'", '"'):
            quote = char if quote is None else (None if quote == char else quote)
        elif quote is None:
            if char in "[{(":
                depth += 1
            elif char in "]})":
                depth -= 1
            elif char == ":" and depth == 0:
                key = text[:index].strip()
                if not key:
                    raise ValueError("empty YAML mapping key")
                return key, text[index + 1 :].strip()
    raise ValueError(f"expected a YAML key:value mapping, got {text!r}")


def _parse_scalar(text: str):
    text = text.strip()
    lower = text.lower()
    if lower in ("null", "none", "~"):
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    if text.startswith(("[", "{", "(", "'", '"')):
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"unsupported inline YAML scalar: {text!r}") from exc
    try:
        return int(text, 10)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


class _SubsetParser:
    def __init__(self, text: str):
        self.tokens = []
        for line_number, raw in enumerate(text.splitlines(), start=1):
            if "\t" in raw[: len(raw) - len(raw.lstrip())]:
                raise ValueError(f"tabs are not allowed for YAML indentation (line {line_number})")
            clean = _strip_comment(raw).rstrip()
            if not clean.strip() or clean.lstrip().startswith("---"):
                continue
            indent = len(clean) - len(clean.lstrip(" "))
            self.tokens.append((indent, clean.strip(), line_number))

    def parse(self):
        if not self.tokens:
            return {}
        value, index = self._block(0, self.tokens[0][0])
        if index != len(self.tokens):
            _, _, line_number = self.tokens[index]
            raise ValueError(f"could not parse YAML near line {line_number}")
        return value

    def _block(self, index: int, indent: int):
        current_indent, content, line_number = self.tokens[index]
        if current_indent != indent:
            raise ValueError(f"unexpected indentation at YAML line {line_number}")
        if content.startswith("-"):
            return self._list(index, indent)
        return self._mapping(index, indent)

    def _mapping(self, index: int, indent: int):
        result = {}
        while index < len(self.tokens):
            current_indent, content, line_number = self.tokens[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"unexpected indentation at YAML line {line_number}")
            if content.startswith("-"):
                break
            key, rest = _split_key(content)
            index += 1
            if rest:
                result[key] = _parse_scalar(rest)
            elif index < len(self.tokens) and self.tokens[index][0] > indent:
                result[key], index = self._block(index, self.tokens[index][0])
            else:
                result[key] = {}
        return result, index

    def _list(self, index: int, indent: int):
        result = []
        while index < len(self.tokens):
            current_indent, content, line_number = self.tokens[index]
            if current_indent < indent:
                break
            if current_indent != indent or not content.startswith("-"):
                break
            rest = content[1:].strip()
            index += 1
            if not rest:
                if index >= len(self.tokens) or self.tokens[index][0] <= indent:
                    result.append(None)
                else:
                    value, index = self._block(index, self.tokens[index][0])
                    result.append(value)
                continue
            if ":" not in rest:
                result.append(_parse_scalar(rest))
                continue

            key, value_text = _split_key(rest)
            item = {}
            if value_text:
                item[key] = _parse_scalar(value_text)
            elif index < len(self.tokens) and self.tokens[index][0] > indent:
                item[key], index = self._block(index, self.tokens[index][0])
            else:
                item[key] = {}
            if index < len(self.tokens) and self.tokens[index][0] > indent:
                extra, index = self._mapping(index, self.tokens[index][0])
                item.update(extra)
            result.append(item)
        return result, index


def safe_load_text(text: str):
    """Load YAML with PyYAML when available, otherwise parse Phoenix's safe subset."""
    try:
        import yaml
    except ModuleNotFoundError:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = _SubsetParser(text).parse()
    else:
        value = yaml.safe_load(text)
    return {} if value is None else value


def load_yaml_source(source):
    """Load a dict, pathlib path, path string, or YAML text into a detached dict."""
    if isinstance(source, dict):
        return deepcopy(source)
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str):
        if "\n" not in source and "\r" not in source:
            candidate = Path(source)
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
            elif candidate.suffix.lower() in (".yaml", ".yml"):
                raise FileNotFoundError(candidate)
            else:
                text = source
        else:
            text = source
    else:
        raise TypeError("YAML source must be a dict, path, or text string")
    value = safe_load_text(text)
    if not isinstance(value, dict):
        raise ValueError("Phoenix YAML root must be a mapping")
    return value


__all__ = ["safe_load_text", "load_yaml_source"]

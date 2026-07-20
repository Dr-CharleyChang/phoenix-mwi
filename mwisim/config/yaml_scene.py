"""YAML SceneBuilder for the Phase-1 composite-circle schema."""
from __future__ import annotations

from collections.abc import Mapping

from ..core.interfaces import SceneBuilder
from ..core.registry import register
from ..phantoms.composite import CircularInclusion, CompositeCirclePhantom
from .yaml_support import load_yaml_source


def _complex_value(value, field_name: str) -> complex:
    if isinstance(value, Mapping):
        unknown = set(value).difference({"real", "imag"})
        if unknown:
            raise ValueError(f"{field_name} has unknown complex fields: {sorted(unknown)}")
        value = complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    else:
        value = complex(value)
    return value


def load_yaml_spec(source) -> dict:
    """Load and minimally validate a Phoenix schema-version-1 configuration."""
    spec = load_yaml_source(source)
    version = int(spec.get("schema_version", 1))
    if version != 1:
        raise ValueError(f"unsupported schema_version {version}; expected 1")
    if "scene" not in spec or not isinstance(spec["scene"], Mapping):
        raise ValueError("configuration requires a scene mapping")
    return spec


@register("scene_builder", "yaml")
class YamlSceneBuilder(SceneBuilder):
    """Build CompositeCirclePhantom from a path, YAML text, or dictionary."""

    def load(self, spec) -> dict:
        return load_yaml_spec(spec)

    def build(self, spec, **kwargs):
        root = self.load(spec)
        scene = dict(root["scene"])
        scene_type = scene.get("type", "composite_circles")
        if scene_type != "composite_circles":
            raise ValueError(
                f"unsupported scene.type {scene_type!r}; Phase 1 supports 'composite_circles'"
            )
        required = ("domain_size_m", "cell_size_m", "inclusions")
        missing = [key for key in required if key not in scene]
        if missing:
            raise ValueError(f"scene is missing required fields: {missing}")
        inclusions = []
        for index, item in enumerate(scene["inclusions"]):
            if not isinstance(item, Mapping):
                raise ValueError(f"scene.inclusions[{index}] must be a mapping")
            missing_item = [
                key for key in ("center_m", "radius_m", "eps_r") if key not in item
            ]
            if missing_item:
                raise ValueError(
                    f"scene.inclusions[{index}] is missing fields: {missing_item}"
                )
            center = tuple(item["center_m"])
            inclusions.append(
                CircularInclusion(
                    center=center,
                    radius=float(item["radius_m"]),
                    eps_r=_complex_value(item["eps_r"], f"inclusions[{index}].eps_r"),
                    label=str(item.get("label", f"inclusion_{index}")),
                )
            )
        return CompositeCirclePhantom(
            domain_size=float(scene["domain_size_m"]),
            d=float(scene["cell_size_m"]),
            inclusions=inclusions,
            eps_b=_complex_value(scene.get("background_eps_r", 1.0), "background_eps_r"),
            overlap_policy=str(scene.get("overlap_policy", "last_wins")),
            name=str(root.get("name", scene.get("name", "yaml_scene"))),
        )


__all__ = ["load_yaml_spec", "YamlSceneBuilder"]

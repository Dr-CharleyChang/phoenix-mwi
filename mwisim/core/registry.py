"""A tiny name -> class registry so implementations can be selected by string.

This is **optional sugar** (PROJECT_PLAN §11.3). The primary way to plug in your own
algorithm is to subclass an interface and pass the instance directly:

    pipeline.run(inverter=MyInverter())

The registry only matters when you want to choose an implementation by *name* from a
config file or command line (gprMax-style ``inverter: dbim``):

    @register("inverter", "dbim")          # the class registers itself by name
    class DBIM(Inverter): ...

    inv = build("inverter", "dbim", tol=1e-3)   # look up + instantiate

See CODE_GUIDE Appendix D.4 for the decorator mechanics from zero.
"""
from __future__ import annotations

# kind -> { name -> class }.  New kinds are created on first use (see register()).
REGISTRY: dict[str, dict[str, type]] = {}


def register(kind: str, name: str):
    """Decorator factory: returns a decorator that files a class under (kind, name).

    Usage::

        @register("forward", "mom2d")
        class MoM2D(ForwardSolver): ...

    ``register("forward","mom2d")`` runs first and returns ``deco``; the ``@`` then applies
    ``deco`` to the class below it, which stores it and returns it unchanged.
    """
    def deco(cls: type) -> type:
        REGISTRY.setdefault(kind, {})          # create the inner dict if first of its kind
        if name in REGISTRY[kind]:
            raise KeyError(f"{kind!r} already has an implementation named {name!r}")
        REGISTRY[kind][name] = cls
        return cls                             # return the class untouched
    return deco


def build(kind: str, name: str, **cfg):
    """Look up a registered class by (kind, name) and instantiate it with ``**cfg``.

    ``build("forward","mom2d", method="dense")`` == ``MoM2D(method="dense")``.
    """
    try:
        cls = REGISTRY[kind][name]
    except KeyError:
        raise KeyError(
            f"no {kind!r} named {name!r}; available: {available(kind)}"
        ) from None
    return cls(**cfg)


def available(kind: str) -> list[str]:
    """List the registered names for a kind (e.g. ``available('forward')``)."""
    return sorted(REGISTRY.get(kind, {}).keys())

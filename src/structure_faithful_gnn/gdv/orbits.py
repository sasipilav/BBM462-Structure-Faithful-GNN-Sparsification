from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Final, Iterable, Mapping


ORBIT_REGISTRY_VERSION: Final[str] = "orca-node-orbits-2to4-v1"
ORBIT_DIM: Final[int] = 15


class OrbitId(IntEnum):
    """Canonical ORCA node-orbit identifiers for connected 2--4 node graphlets."""

    EDGE_ENDPOINT = 0
    PATH3_ENDPOINT = 1
    PATH3_CENTER = 2
    TRIANGLE_NODE = 3
    PATH4_ENDPOINT = 4
    PATH4_INTERNAL = 5
    STAR4_LEAF = 6
    STAR4_CENTER = 7
    CYCLE4_NODE = 8
    TAILED_TRIANGLE_TAIL = 9
    TAILED_TRIANGLE_TRIANGLE_NODE = 10
    TAILED_TRIANGLE_ATTACHMENT = 11
    DIAMOND_DEGREE2 = 12
    DIAMOND_DEGREE3 = 13
    CLIQUE4_NODE = 14


@dataclass(frozen=True, slots=True)
class OrbitSpec:
    """Semantic description of one canonical node orbit.

    ``role_multiplicity`` is the number of vertices occupying the role in one
    graphlet instance. ``role_degree`` is the induced degree of such a vertex.
    ``orca_index`` is kept explicit so accidental reordering cannot silently
    change the external ORCA contract.
    """

    orbit_id: int
    enum_name: str
    graphlet_name: str
    graphlet_size: int
    graphlet_edge_count: int
    role_name: str
    role_degree: int
    role_multiplicity: int
    family: str
    orca_index: int

    @property
    def label(self) -> str:
        return f"{self.graphlet_name}:{self.role_name}"

    def as_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["label"] = self.label
        return payload


def _spec(
    orbit: OrbitId,
    *,
    graphlet_name: str,
    graphlet_size: int,
    graphlet_edge_count: int,
    role_name: str,
    role_degree: int,
    role_multiplicity: int,
    family: str,
) -> OrbitSpec:
    return OrbitSpec(
        orbit_id=int(orbit),
        enum_name=orbit.name,
        graphlet_name=graphlet_name,
        graphlet_size=graphlet_size,
        graphlet_edge_count=graphlet_edge_count,
        role_name=role_name,
        role_degree=role_degree,
        role_multiplicity=role_multiplicity,
        family=family,
        orca_index=int(orbit),
    )


ORBIT_REGISTRY: Final[tuple[OrbitSpec, ...]] = (
    _spec(
        OrbitId.EDGE_ENDPOINT,
        graphlet_name="edge",
        graphlet_size=2,
        graphlet_edge_count=1,
        role_name="endpoint",
        role_degree=1,
        role_multiplicity=2,
        family="edge",
    ),
    _spec(
        OrbitId.PATH3_ENDPOINT,
        graphlet_name="path3",
        graphlet_size=3,
        graphlet_edge_count=2,
        role_name="endpoint",
        role_degree=1,
        role_multiplicity=2,
        family="three_node_path",
    ),
    _spec(
        OrbitId.PATH3_CENTER,
        graphlet_name="path3",
        graphlet_size=3,
        graphlet_edge_count=2,
        role_name="center",
        role_degree=2,
        role_multiplicity=1,
        family="three_node_path",
    ),
    _spec(
        OrbitId.TRIANGLE_NODE,
        graphlet_name="triangle",
        graphlet_size=3,
        graphlet_edge_count=3,
        role_name="node",
        role_degree=2,
        role_multiplicity=3,
        family="triangle",
    ),
    _spec(
        OrbitId.PATH4_ENDPOINT,
        graphlet_name="path4",
        graphlet_size=4,
        graphlet_edge_count=3,
        role_name="endpoint",
        role_degree=1,
        role_multiplicity=2,
        family="four_node_path",
    ),
    _spec(
        OrbitId.PATH4_INTERNAL,
        graphlet_name="path4",
        graphlet_size=4,
        graphlet_edge_count=3,
        role_name="internal",
        role_degree=2,
        role_multiplicity=2,
        family="four_node_path",
    ),
    _spec(
        OrbitId.STAR4_LEAF,
        graphlet_name="star4",
        graphlet_size=4,
        graphlet_edge_count=3,
        role_name="leaf",
        role_degree=1,
        role_multiplicity=3,
        family="star",
    ),
    _spec(
        OrbitId.STAR4_CENTER,
        graphlet_name="star4",
        graphlet_size=4,
        graphlet_edge_count=3,
        role_name="center",
        role_degree=3,
        role_multiplicity=1,
        family="star",
    ),
    _spec(
        OrbitId.CYCLE4_NODE,
        graphlet_name="cycle4",
        graphlet_size=4,
        graphlet_edge_count=4,
        role_name="node",
        role_degree=2,
        role_multiplicity=4,
        family="cycle",
    ),
    _spec(
        OrbitId.TAILED_TRIANGLE_TAIL,
        graphlet_name="tailed_triangle",
        graphlet_size=4,
        graphlet_edge_count=4,
        role_name="tail",
        role_degree=1,
        role_multiplicity=1,
        family="tailed_triangle",
    ),
    _spec(
        OrbitId.TAILED_TRIANGLE_TRIANGLE_NODE,
        graphlet_name="tailed_triangle",
        graphlet_size=4,
        graphlet_edge_count=4,
        role_name="triangle_non_attachment",
        role_degree=2,
        role_multiplicity=2,
        family="tailed_triangle",
    ),
    _spec(
        OrbitId.TAILED_TRIANGLE_ATTACHMENT,
        graphlet_name="tailed_triangle",
        graphlet_size=4,
        graphlet_edge_count=4,
        role_name="attachment",
        role_degree=3,
        role_multiplicity=1,
        family="tailed_triangle",
    ),
    _spec(
        OrbitId.DIAMOND_DEGREE2,
        graphlet_name="diamond",
        graphlet_size=4,
        graphlet_edge_count=5,
        role_name="degree2",
        role_degree=2,
        role_multiplicity=2,
        family="diamond",
    ),
    _spec(
        OrbitId.DIAMOND_DEGREE3,
        graphlet_name="diamond",
        graphlet_size=4,
        graphlet_edge_count=5,
        role_name="degree3",
        role_degree=3,
        role_multiplicity=2,
        family="diamond",
    ),
    _spec(
        OrbitId.CLIQUE4_NODE,
        graphlet_name="clique4",
        graphlet_size=4,
        graphlet_edge_count=6,
        role_name="node",
        role_degree=3,
        role_multiplicity=4,
        family="clique",
    ),
)

ORBIT_BY_ID: Final[Mapping[int, OrbitSpec]] = MappingProxyType(
    {spec.orbit_id: spec for spec in ORBIT_REGISTRY}
)
ORBIT_BY_ENUM_NAME: Final[Mapping[str, OrbitSpec]] = MappingProxyType(
    {spec.enum_name: spec for spec in ORBIT_REGISTRY}
)

GRAPHLET_ORBIT_GROUPS: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {
        name: tuple(spec.orbit_id for spec in ORBIT_REGISTRY if spec.family == name)
        for name in (
            "edge",
            "three_node_path",
            "triangle",
            "four_node_path",
            "star",
            "cycle",
            "tailed_triangle",
            "diamond",
            "clique",
        )
    }
)

# Predeclared groups used by Phase-2 leave-group-out experiments. Groups are
# intentionally non-overlapping so their aggregate contributions are auditable.
ANALYSIS_ORBIT_GROUPS: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {
        "low_order": (0, 1, 2, 3),
        "path4": (4, 5),
        "star4": (6, 7),
        "cycle4": (8,),
        "tailed_triangle": (9, 10, 11),
        "dense4": (12, 13, 14),
    }
)


def get_orbit_spec(orbit: int | OrbitId) -> OrbitSpec:
    orbit_id = int(orbit)
    try:
        return ORBIT_BY_ID[orbit_id]
    except KeyError as exc:
        raise ValueError(f"Unknown orbit id {orbit_id}; expected 0..{ORBIT_DIM - 1}.") from exc


def orbit_ids_for_graphlet(graphlet_name: str) -> tuple[int, ...]:
    ids = tuple(
        spec.orbit_id for spec in ORBIT_REGISTRY if spec.graphlet_name == graphlet_name
    )
    if not ids:
        raise ValueError(f"Unknown graphlet name: {graphlet_name!r}.")
    return ids


def orbit_registry_payload() -> dict[str, object]:
    """Return a deterministic JSON-serializable registry payload."""

    return {
        "schema_version": ORBIT_REGISTRY_VERSION,
        "orbit_dim": ORBIT_DIM,
        "orbits": [spec.as_dict() for spec in ORBIT_REGISTRY],
        "graphlet_groups": {name: list(ids) for name, ids in GRAPHLET_ORBIT_GROUPS.items()},
        "analysis_groups": {name: list(ids) for name, ids in ANALYSIS_ORBIT_GROUPS.items()},
    }


def _flatten(groups: Iterable[tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(orbit for group in groups for orbit in group)


def _validate_registry() -> None:
    expected_ids = tuple(range(ORBIT_DIM))
    observed_ids = tuple(spec.orbit_id for spec in ORBIT_REGISTRY)
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"Canonical orbit registry must be ordered as {expected_ids}, found {observed_ids}."
        )
    if tuple(spec.orca_index for spec in ORBIT_REGISTRY) != expected_ids:
        raise RuntimeError("ORCA indices must match canonical orbit IDs exactly.")
    if tuple(int(orbit) for orbit in OrbitId) != expected_ids:
        raise RuntimeError("OrbitId values must be contiguous from zero to ORBIT_DIM - 1.")
    if len(ORBIT_BY_ID) != ORBIT_DIM or len(ORBIT_BY_ENUM_NAME) != ORBIT_DIM:
        raise RuntimeError("Canonical orbit registry contains duplicate IDs or enum names.")

    by_graphlet: dict[str, list[OrbitSpec]] = {}
    for spec in ORBIT_REGISTRY:
        if spec.graphlet_size not in (2, 3, 4):
            raise RuntimeError(f"Invalid graphlet size in {spec}.")
        if not 0 <= spec.role_degree < spec.graphlet_size:
            raise RuntimeError(f"Invalid role degree in {spec}.")
        if spec.role_multiplicity <= 0:
            raise RuntimeError(f"Invalid role multiplicity in {spec}.")
        by_graphlet.setdefault(spec.graphlet_name, []).append(spec)

    for graphlet_name, specs in by_graphlet.items():
        sizes = {spec.graphlet_size for spec in specs}
        edge_counts = {spec.graphlet_edge_count for spec in specs}
        if len(sizes) != 1 or len(edge_counts) != 1:
            raise RuntimeError(f"Inconsistent graphlet metadata for {graphlet_name}.")
        graphlet_size = specs[0].graphlet_size
        if sum(spec.role_multiplicity for spec in specs) != graphlet_size:
            raise RuntimeError(f"Role multiplicities do not cover {graphlet_name}.")
        degree_sum = sum(spec.role_degree * spec.role_multiplicity for spec in specs)
        if degree_sum != 2 * specs[0].graphlet_edge_count:
            raise RuntimeError(f"Role degrees violate the handshake lemma for {graphlet_name}.")

    if tuple(sorted(_flatten(GRAPHLET_ORBIT_GROUPS.values()))) != expected_ids:
        raise RuntimeError("Graphlet orbit groups must partition all canonical orbit IDs.")
    if tuple(sorted(_flatten(ANALYSIS_ORBIT_GROUPS.values()))) != expected_ids:
        raise RuntimeError("Analysis orbit groups must partition all canonical orbit IDs.")


_validate_registry()


__all__ = [
    "ANALYSIS_ORBIT_GROUPS",
    "GRAPHLET_ORBIT_GROUPS",
    "ORBIT_BY_ENUM_NAME",
    "ORBIT_BY_ID",
    "ORBIT_DIM",
    "ORBIT_REGISTRY",
    "ORBIT_REGISTRY_VERSION",
    "OrbitId",
    "OrbitSpec",
    "get_orbit_spec",
    "orbit_ids_for_graphlet",
    "orbit_registry_payload",
]

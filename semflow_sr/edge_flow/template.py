"""Register-operator edge-choice templates for complete expression DAGs."""
from __future__ import annotations

from dataclasses import dataclass

from ..sr.ops import NAME_TO_ID, get_op


@dataclass(frozen=True)
class EdgeGroup:
    group_id: str
    layer_id: int
    mode_id: int | None
    group_type: str
    candidate_ids: tuple[int, ...]
    arity_slot: int | None = None
    target_node: str = ""
    primitive: str | None = None

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_ids)


@dataclass(frozen=True)
class RegisterOperatorTemplate:
    num_vars: int
    num_registers: int
    num_layers: int
    primitives: tuple[str, ...]
    mixture_modes: int = 1
    output_terms: int = 1

    def __post_init__(self) -> None:
        if self.num_registers < self.num_vars + 1:
            raise ValueError("num_registers must hold variables plus constant register")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.mixture_modes <= 0:
            raise ValueError("mixture_modes must be positive")
        if self.output_terms <= 0:
            raise ValueError("output_terms must be positive")
        if self.output_terms > 1 and self.num_registers < self.num_vars + 2:
            raise ValueError("multi-term output requires a zero register")
        for name in self.primitives:
            if name not in NAME_TO_ID:
                raise ValueError(f"unknown primitive: {name}")

    @property
    def groups(self) -> list[EdgeGroup]:
        groups: list[EdgeGroup] = []
        primitive_count = len(self.primitives)
        for layer in range(self.num_layers):
            for op_index, primitive in enumerate(self.primitives):
                arity = get_op(NAME_TO_ID[primitive]).arity
                for slot in range(arity):
                    groups.append(EdgeGroup(
                        group_id=f"L{layer}:OP{op_index}:{primitive}:ARG{slot}",
                        layer_id=layer,
                        mode_id=None,
                        group_type="ARG_SELECT",
                        candidate_ids=tuple(range(self.num_registers)),
                        arity_slot=slot,
                        target_node=f"op_{layer}_{op_index}",
                        primitive=primitive,
                    ))
            update_candidates = tuple(range(self.num_registers + primitive_count))
            for reg in range(self.num_registers):
                groups.append(EdgeGroup(
                    group_id=f"L{layer}:REG{reg}:UPDATE",
                    layer_id=layer,
                    mode_id=None,
                    group_type="REG_UPDATE",
                    candidate_ids=update_candidates,
                    target_node=f"reg_{layer + 1}_{reg}",
                ))
        if int(self.output_terms) <= 1:
            groups.append(EdgeGroup(
                group_id="OUT:SELECT",
                layer_id=self.num_layers,
                mode_id=None,
                group_type="OUTPUT_SELECT",
                candidate_ids=tuple(range(self.num_registers)),
                target_node="output",
            ))
        else:
            for term_idx in range(int(self.output_terms)):
                groups.append(EdgeGroup(
                    group_id=f"OUT:TERM{term_idx}:SELECT",
                    layer_id=self.num_layers,
                    mode_id=None,
                    group_type="OUTPUT_SELECT",
                    candidate_ids=tuple(range(self.num_registers + 1)),
                    target_node=f"output_term_{term_idx}",
                ))
        return groups

    @property
    def group_ids(self) -> list[str]:
        return [group.group_id for group in self.groups]

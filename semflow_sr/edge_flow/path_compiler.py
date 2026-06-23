"""Compile ground-truth formulas into trainable CSEF paths when representable."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ..sr.ast import Expr, eval_expr
from ..sr.ops import NAME_TO_ID, get_op
from ..sr.parser import parse_formula
from .circuit_sampler import CircuitSample
from .conditional import (
    ConditionalEdgeFlowModel,
    _active_variable_count,
    _eval_op_semantics,
    _initial_registers_and_semantics,
    _is_carry_write_mode,
    _operator_output_semantics,
    _source_pool,
    _source_output_semantics,
    _write_targets,
)
from .semantic_teacher import DecisionTrace
from .template import RegisterOperatorTemplate


@dataclass(frozen=True)
class CompileResult:
    sample: CircuitSample
    decision_count: int


def compile_formula_to_csef_sample(
    formula: str,
    *,
    variable_count: int,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = "policy",
    flow_steps: int = 1,
    flow_time: float | None = None,
) -> CircuitSample | None:
    """Return a differentiable CSEF sample for simple representable formulas.

    The compiler is intentionally conservative. It supports formulas whose op
    tree depth fits the template and whose constants are already represented by
    the initial constant registers. Unsupported formulas return ``None`` rather
    than introducing fake supervision.
    """

    try:
        expr = parse_formula(str(formula), [f"x{i}" for i in range(int(variable_count))])
        result = _compile_expr(
            expr,
            variable_count=int(variable_count),
            template=template,
            model=model,
            x=x.float(),
            y=y.float(),
            method=str(method),
            flow_steps=int(flow_steps),
            flow_time=flow_time,
        )
    except Exception:
        return None
    return result.sample if result is not None else None


def compile_expr_to_csef_sample(
    expr: Expr,
    *,
    variable_count: int,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = "policy",
    flow_steps: int = 1,
    flow_time: float | None = None,
) -> CircuitSample | None:
    """Return a differentiable CSEF sample for an already parsed expression."""

    try:
        result = _compile_expr(
            expr,
            variable_count=int(variable_count),
            template=template,
            model=model,
            x=x.float(),
            y=y.float(),
            method=str(method),
            flow_steps=int(flow_steps),
            flow_time=flow_time,
        )
    except Exception:
        return None
    return result.sample if result is not None else None


def _compile_expr(
    expr: Expr,
    *,
    variable_count: int,
    template: RegisterOperatorTemplate,
    model: ConditionalEdgeFlowModel,
    x: torch.Tensor,
    y: torch.Tensor,
    method: str,
    flow_steps: int,
    flow_time: float | None,
) -> CompileResult | None:
    if expr.depth > int(template.num_layers):
        return None
    primitive_ids = [NAME_TO_ID[name] for name in template.primitives]
    if not _ops_supported(expr, set(primitive_ids)):
        return None
    active_vars = _active_variable_count(x, template_num_vars=template.num_vars, explicit=variable_count)
    regs, reg_sem, source_active = _initial_registers_and_semantics(
        x,
        template.num_vars,
        template.num_registers,
        active_vars,
    )
    base_regs = list(regs)
    base_sem = reg_sem.clone()
    base_active = source_active.clone()
    base_expr_ids = {
        id(item)
        for item, active in zip(base_regs, base_active.tolist())
        if bool(active)
    }
    head_pool: list[Expr] = [reg for idx, reg in enumerate(regs) if bool(source_active[idx])]
    head_semantics = [reg_sem[:, idx] for idx in range(reg_sem.shape[1]) if bool(source_active[idx])]
    head_active: list[bool] = [True for _ in head_pool]
    head_base_flags: list[bool] = [True for _ in head_pool]
    choices: dict[str, int] = {}
    log_terms: list[torch.Tensor] = []
    decision_count = 0
    decision_traces: list[DecisionTrace] = []
    ancestry_trace_ids: dict[int, list[int]] = {id(item): [] for item in regs}
    nodes_by_depth = _nodes_by_depth(expr)
    for layer in range(template.num_layers):
        target_nodes = list(nodes_by_depth.get(layer + 1, []))
        write_targets = sorted(_write_targets(
            template.num_registers,
            layer_id=layer,
            configured=int(model.cfg.write_registers_per_layer),
            update_mode=str(model.cfg.update_mode),
        ))
        if len(target_nodes) > len(write_targets):
            return None
        assignments = {target: node for target, node in zip(write_targets, target_nodes)}
        reg_tokens = model.register_tokens(x, y, regs, reg_sem, layer_id=layer)
        source_exprs, source_sem, source_mask = _source_pool(
            regs,
            reg_sem,
            source_active,
            base_regs=base_regs,
            base_sem=base_sem,
            base_active=base_active,
            layer_id=layer,
            include_base_source_pool=bool(model.cfg.include_base_source_pool),
        )
        source_tokens = model.register_tokens(x, y, source_exprs, source_sem, layer_id=layer)
        next_regs: list[Expr] = []
        next_semantics: list[torch.Tensor] = []
        next_active: list[bool] = []
        layer_branches: list[Expr] = []
        layer_branch_semantics: list[torch.Tensor] = []
        layer_branch_trace_ids: list[list[int]] = []
        for target in range(template.num_registers):
            node = assignments.get(target)
            if node is None:
                if bool(model.cfg.enable_keep_option) and target in write_targets:
                    probs = model.update_action_probs(
                        target_token=reg_tokens[target],
                        layer_id=layer,
                        target_register=target,
                        method=method,
                        flow_steps=flow_steps,
                        flow_time=flow_time,
                    )
                    choices[f"L{layer}:TARGET{target}:UPDATE_ACTION"] = 0
                    log_terms.append(probs[0].clamp_min(1e-12).log())
                    decision_count += 1
                next_regs.append(regs[target])
                next_semantics.append(reg_sem[:, target])
                next_active.append(bool(source_active[target]))
                ancestry_trace_ids[id(regs[target])] = ancestry_trace_ids.get(id(regs[target]), [])
                continue
            if bool(model.cfg.enable_keep_option):
                probs = model.update_action_probs(
                    target_token=reg_tokens[target],
                    layer_id=layer,
                    target_register=target,
                    method=method,
                    flow_steps=flow_steps,
                    flow_time=flow_time,
                )
                choices[f"L{layer}:TARGET{target}:UPDATE_ACTION"] = 1
                log_terms.append(probs[1].clamp_min(1e-12).log())
                decision_count += 1
            op_pos = _primitive_position(primitive_ids, int(node.op_id))
            if op_pos is None:
                return None
            op_dist = model.operator_probs(
                target_token=reg_tokens[target],
                primitive_ids=primitive_ids,
                layer_id=layer,
                target_register=target,
                branch_id=0,
                method=method,
                flow_steps=flow_steps,
                return_details=True,
                flow_time=flow_time,
            )
            op_probs = op_dist["probs"]
            choices[f"L{layer}:TARGET{target}:BRANCH0:OP"] = int(op_pos)
            log_terms.append(op_probs[int(op_pos)].clamp_min(1e-12).log())
            decision_count += 1
            child_semantics: list[torch.Tensor] = []
            op_trace_id = len(decision_traces)
            decision_traces.append(DecisionTrace(
                group_id=f"L{layer}:TARGET{target}:BRANCH0:OP",
                choice=int(op_pos),
                current_probs=op_dist["current_probs"],
                candidate_semantics=_operator_output_semantics(
                    primitive_ids,
                    reg_sem[:, target],
                    fallback=y,
                ).detach(),
                predicted_sqrt_velocity=op_dist["predicted_sqrt_velocity"],
                initial_probs=op_dist["initial_probs"],
                velocity_fn=op_dist["velocity_fn"],
                flow_time=float(op_dist["flow_time"]),
                candidate_keys=tuple(get_op(op_id).name for op_id in primitive_ids),
            ))
            node_trace_ids: list[int] = [op_trace_id]
            for slot, child in enumerate(node.children):
                src_idx = _find_expr_index(source_exprs, child)
                if src_idx is None or not bool(source_mask[int(src_idx)].item()):
                    return None
                dist = model.source_probs(
                    target_token=reg_tokens[target],
                    source_tokens=source_tokens,
                    layer_id=layer,
                    target_register=target,
                    branch_id=0,
                    arity_slot=slot,
                    primitive_id=int(node.op_id),
                    method=method,
                    flow_steps=flow_steps,
                    source_mask=source_mask.to(source_tokens.device),
                    return_details=True,
                    flow_time=flow_time,
                )
                src_probs = dist["probs"]
                choices[f"L{layer}:TARGET{target}:BRANCH0:ARG{slot}:SRC"] = int(src_idx)
                log_terms.append(src_probs[int(src_idx)].clamp_min(1e-12).log())
                decision_count += 1
                trace_id = len(decision_traces)
                decision_traces.append(DecisionTrace(
                    group_id=f"L{layer}:TARGET{target}:BRANCH0:ARG{slot}:SRC",
                    choice=int(src_idx),
                    current_probs=dist["current_probs"],
                    candidate_semantics=_source_output_semantics(
                        int(node.op_id),
                        slot,
                        source_sem,
                        child_semantics,
                        fallback=y,
                    ).detach(),
                    predicted_sqrt_velocity=dist["predicted_sqrt_velocity"],
                    initial_probs=dist["initial_probs"],
                    velocity_fn=dist["velocity_fn"],
                    flow_time=float(dist["flow_time"]),
                    candidate_keys=_expr_keys(source_exprs),
                ))
                node_trace_ids.append(trace_id)
                node_trace_ids.extend(ancestry_trace_ids.get(id(child), []))
                child_semantics.append(source_sem[:, int(src_idx)])
            sem = _eval_op_semantics(int(node.op_id), child_semantics, fallback=y)
            next_regs.append(node)
            next_semantics.append(sem)
            next_active.append(True)
            ancestry_trace_ids[id(node)] = list(node_trace_ids)
            layer_branches.append(node)
            layer_branch_semantics.append(sem)
            layer_branch_trace_ids.append(list(node_trace_ids))
        regs = next_regs
        reg_sem = torch.stack(next_semantics, dim=1)
        source_active = torch.tensor(next_active, dtype=torch.bool, device=reg_sem.device)
        head_pool.extend(layer_branches)
        head_pool.extend(regs)
        head_semantics.extend(layer_branch_semantics)
        head_semantics.extend(next_semantics)
        head_active.extend(True for _ in layer_branches)
        head_active.extend(bool(flag) for flag in next_active)
        head_base_flags.extend(False for _ in layer_branches)
        head_base_flags.extend(id(item) in base_expr_ids for item in regs)
        for branch, trace_ids in zip(layer_branches, layer_branch_trace_ids):
            ancestry_trace_ids[id(branch)] = list(trace_ids)
    final_tokens = model.register_tokens(x, y, regs, reg_sem, layer_id=template.num_layers)
    final_query = final_tokens.mean(dim=0)
    head_matrix = torch.stack(head_semantics, dim=1)
    head_tokens = model.register_tokens(x, y, head_pool, head_matrix, layer_id=template.num_layers)
    head_mask = torch.tensor(head_active, dtype=torch.bool, device=head_tokens.device)
    if bool(model.cfg.exclude_base_head_candidates):
        base_mask = torch.tensor(head_base_flags, dtype=torch.bool, device=head_tokens.device)
        masked = head_mask & ~base_mask
        if bool(masked.any()):
            head_mask = masked
    head_idx = _find_expr_index(head_pool, expr)
    if head_idx is None or not bool(head_mask[int(head_idx)].item()):
        return None
    dist = model.source_probs(
        target_token=final_query,
        source_tokens=head_tokens,
        layer_id=template.num_layers,
        target_register=-1,
        branch_id=0,
        arity_slot=0,
        primitive_id=-1,
        method=method,
        flow_steps=flow_steps,
        source_mask=head_mask,
        return_details=True,
        flow_time=flow_time,
    )
    probs = dist["probs"]
    choices["HEAD:TERM0:SRC"] = int(head_idx)
    log_terms.append(probs[int(head_idx)].clamp_min(1e-12).log())
    decision_count += 1
    head_trace_id = len(decision_traces)
    decision_traces.append(DecisionTrace(
        group_id="HEAD:TERM0:SRC",
        choice=int(head_idx),
        current_probs=dist["current_probs"],
        candidate_semantics=head_matrix.detach(),
        predicted_sqrt_velocity=dist["predicted_sqrt_velocity"],
        initial_probs=dist["initial_probs"],
        velocity_fn=dist["velocity_fn"],
        flow_time=float(dist["flow_time"]),
        candidate_keys=_expr_keys(head_pool),
    ))
    active_trace_ids = [head_trace_id] + ancestry_trace_ids.get(id(expr), [])
    for trace_id in set(active_trace_ids):
        if 0 <= int(trace_id) < len(decision_traces):
            decision_traces[int(trace_id)].active = True
    log_prob_tensor = torch.stack(log_terms).sum() if log_terms else torch.zeros((), dtype=x.dtype, device=x.device)
    sample = CircuitSample(
        sample_id=-1,
        mode=0,
        edge_choices=choices,
        expression=expr,
        log_prob=float(log_prob_tensor.detach().cpu().item()),
        complexity=int(expr.complexity),
        head_terms=(expr,),
        log_prob_tensor=log_prob_tensor,
        active_log_prob_tensor=log_prob_tensor,
        decision_traces=tuple(decision_traces),
        diagnostics={
            "decision_count": int(decision_count),
            "active_decision_count": int(decision_count),
            "is_gt_elite": True,
            "gt_compile_success": True,
            "active_variable_count": int(variable_count),
        },
    )
    return CompileResult(sample=sample, decision_count=int(decision_count))


def _nodes_by_depth(expr: Expr) -> dict[int, list[Expr]]:
    out: dict[int, list[Expr]] = {}
    seen: set[Expr] = set()

    def visit(node: Expr) -> None:
        for child in node.children:
            visit(child)
        if node.kind == "op" and node not in seen:
            out.setdefault(int(node.depth), []).append(node)
            seen.add(node)

    visit(expr)
    return out


def _ops_supported(expr: Expr, primitive_ids: set[int]) -> bool:
    if expr.kind != "op":
        return _leaf_supported(expr)
    if int(expr.op_id) not in primitive_ids:
        return False
    return all(_ops_supported(child, primitive_ids) for child in expr.children)


def _leaf_supported(expr: Expr) -> bool:
    if expr.kind == "var":
        return True
    if expr.kind == "const":
        return float(expr.value) in {0.0, 1.0}
    return False


def _primitive_position(primitive_ids: list[int], op_id: int) -> int | None:
    for idx, value in enumerate(primitive_ids):
        if int(value) == int(op_id):
            return int(idx)
    return None


def _find_expr_index(exprs: list[Expr], target: Expr) -> int | None:
    for idx, item in enumerate(exprs):
        if item == target:
            return int(idx)
    return None


def _expr_keys(exprs: list[Expr]) -> tuple[str, ...]:
    return tuple(str(expr) for expr in exprs)

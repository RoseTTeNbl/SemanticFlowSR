import torch
from semflow_sr.actions.action_space import ActionSpace, ActionSpec
from semflow_sr.registers.state import init_register_state


def test_encode_decode_bijection():
    space = ActionSpace(K=6)
    from semflow_sr.sr.ops import get_op, N_OPS
    # decode is canonical (unary -> read_2=0); check decode(encode(spec))==spec on the
    # canonical action set, and that encode(decode(aid)) is idempotent for every id.
    for aid in range(space.size):
        spec = space.decode(aid)
        assert space.decode(space.encode(spec)) == spec               # canonical roundtrip
        assert space.encode(space.decode(aid)) == space.encode(spec)  # idempotent projection
    # explicit canonical bijection over (op, r1, r2*, w)
    seen = set()
    for op_id in range(N_OPS):
        arity = get_op(op_id).arity
        r2_range = [0] if arity == 1 else range(6)
        for r1 in range(6):
            for r2 in r2_range:
                for w in range(6):
                    aid = space.encode(ActionSpec(op_id, r1, r2, w))
                    assert aid not in seen
                    seen.add(aid)


def test_unary_ignores_read2():
    space = ActionSpace(K=5)
    # op_id 5 = sin (unary); read_2 should canonicalize to 0
    aid = space.encode(ActionSpec(op_id=5, read_1=2, read_2=3, write=1))
    spec = space.decode(aid)
    assert spec.read_2 == 0 and spec.op_id == 5 and spec.read_1 == 2 and spec.write == 1


def test_valid_mask_respects_active():
    state = init_register_state(num_vars=1, K=5)   # reg0=var, reg1=const1, rest inactive
    space = ActionSpace(K=5, allowed_ops=[2])      # mul (binary)
    mask = space.valid_mask(state)
    ids = mask.nonzero(as_tuple=False).squeeze(-1)
    for aid in ids.tolist():
        spec = space.decode(aid)
        assert state.active[spec.read_1] and state.active[spec.read_2]

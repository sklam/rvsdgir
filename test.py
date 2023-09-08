import itertools
from pprint import pprint

from rvsdgir import Region, RegionOp

# -----------------------------------------------------------------------------
# Test utils


def check_overlap_refs(*regions: Region):
    # Gather all _Refs
    refsets: list[set[_Ref]] = []
    for reg in regions:
        myrefs = set()
        for op in reg.body.iter_ops_unordered():
            myrefs.add(op._ref)
        refsets.append(myrefs)

    for a, b in itertools.permutations(refsets, r=2):
        if a.intersection(b):
            return True
    return False


def make_func():
    func_reg = Region.make(
        "func", name="foo", ins=["env", "arg"], outs=["env", "ret"]
    )

    binadd_op = func_reg.add_simple_op(
        "BinAdd",
        ins=["lhs", "rhs"],
        outs=["out"],
    )

    binadd_op.ins(
        lhs=func_reg.args.arg,
        rhs=func_reg.args.arg,
    )

    func_reg.results(
        env=func_reg.args.env,
        ret=binadd_op.outs.out,
    )
    return func_reg


# -----------------------------------------------------------------------------
# Tests


def test_use():
    first = Region.make(
        "func",
        name="foo",
        ins=["env", "arg"],
        outs=["env", "ret"],
    )
    inner = first.add_subregion(
        "block", ins=["env", "arg"], outs=["env", "ret"]
    )
    inner.ins(env=first.args.env, arg=first.args.arg)
    first.results(env=inner.outs.env, ret=inner.outs.ret)

    with inner as cur:
        binadd_op = cur.add_simple_op(
            "BinAdd1",
            ins=["lhs", "rhs"],
            outs=["out"],
        )

        binadd_op2 = cur.add_simple_op(
            "BinAdd2",
            ins=["lhs", "rhs"],
            outs=["out"],
        )
        binadd_op2.ins(
            lhs=cur.args.arg,
            rhs=cur.args.arg,
        )

        binadd_op.ins(
            lhs=cur.args.arg,
            rhs=binadd_op2.outs.out,
        )

        binadd_op3 = cur.add_simple_op(
            "BinAdd3",
            ins=["lhs", "rhs"],
            outs=["out"],
        )
        binadd_op3.ins(
            lhs=binadd_op.outs.out,
            rhs=binadd_op2.outs.out,
        )

        ret_op = cur.add_simple_op(
            "Ret",
            ins=["env", "retval"],
            outs=["env", "retval"],
        )
        ret_op.ins(
            env=cur.args.env,
            retval=binadd_op3.outs.out,
        )

        cur.results(
            env=ret_op.outs.env,
            ret=ret_op.outs.retval,
        )

    got = first.prettyformat().strip()
    expect = """
 Region[func] (env=$2, arg=$1) -> (env=$3, ret=$4) {
   (env=$2, arg=$1) -> (env=$3, ret=$4) =>
     Region[block] (env=$6, arg=$5) -> (env=$10, ret=$11) {
       Op BinAdd2 (lhs=$5, rhs=$5) -> (out=$8)
       Op BinAdd1 (lhs=$5, rhs=$8) -> (out=$7)
       Op BinAdd3 (lhs=$7, rhs=$8) -> (out=$9)
       Op Ret (env=$6, retval=$9) -> (env=$10, retval=$11)
     }
 }
""".strip()
    assert got == expect

    # Test that the inner region does not contains edges to the outer region
    inner_edges = inner.subregion.iter_edges()
    outer_ports = set(first.iter_edges().iter_ports())
    assert not set(inner_edges.filter_by_ports(outer_ports))


def test_assert_no_overlap_refs():
    func1 = make_func()
    func2 = make_func()
    assert check_overlap_refs(func1, func1)
    assert not check_overlap_refs(func1, func2)


def test_copy_in():
    func1 = make_func()
    func2 = make_func()
    print("dump")
    func1.prettyprint()
    func2.prettyprint()

    module_reg = Region.make("module")
    module_reg.add_region_clone(func1)
    module_reg.add_region_clone(func2)
    print("module dump")
    module_reg.prettyprint()

    assert all(isinstance(x, RegionOp) for x in module_reg.body)
    assert not check_overlap_refs(module_reg, func1, func2)

def test_split():
    module_reg = Region.make("module")
    func_reg = module_reg.add_subregion(
        "func", name="foo", ins=["env", "arg"], outs=["env", "ret"]
    )
    binadd_op1 = func_reg.subregion.add_simple_op(
        opname="BinAdd1",
        ins=["lhs", "rhs"],
        outs=["out"],
    )
    binadd_op1.ins(
        lhs=func_reg.subregion.args.arg,
        rhs=func_reg.subregion.args.arg,
    )

    binmul_op1 = func_reg.subregion.add_simple_op(
        opname="BinMul1",
        ins=["lhs", "rhs"],
        outs=["out"],
    )
    binmul_op1.ins(
        lhs=binadd_op1.outs.out,
        rhs=func_reg.subregion.args.arg,
    )

    binadd_op2 = func_reg.subregion.add_simple_op(
        opname="BinAdd2",
        ins=["lhs", "rhs"],
        outs=["out"],
    )
    binadd_op2.ins(
        lhs=func_reg.subregion.args.arg,
        rhs=binmul_op1.outs.out,
    )
    binmul_op2 = func_reg.subregion.add_simple_op(
        opname="BinMul2",
        ins=["lhs", "rhs"],
        outs=["out"],
    )
    binmul_op2.ins(
        lhs=binadd_op1.outs.out,
        rhs=binadd_op2.outs.out,
    )

    func_reg.subregion.results(
        env=func_reg.subregion.args.env,
        ret=binmul_op2.outs.out,
    )
    module_reg.prettyprint()

    # Split region
    ops = list(func_reg.subregion.body.toposorted_ops())
    splitted = func_reg.subregion.split_after(ops[1])
    func_reg.outs.bridge(splitted.ins)

    module_reg.prettyprint()

    assert False

if __name__ == "__main__":
    test_split()

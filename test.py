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
    print(got)
    expect = """
 Region[func] (env=$1, arg=$2) -> (env=$3, ret=$4) {
   (env=$1, arg=$2) -> (env=$3, ret=$4) =>
     Region[block] (env=$5, arg=$6) -> (env=$10, ret=$11) {
       Op BinAdd2 (lhs=$6, rhs=$6) -> (out=$7)
       Op BinAdd1 (lhs=$6, rhs=$7) -> (out=$8)
       Op BinAdd3 (lhs=$8, rhs=$7) -> (out=$9)
       Op Ret (env=$5, retval=$9) -> (env=$10, retval=$11)
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
    parent_reg = Region.make("func")
    func_reg = parent_reg.add_subregion(
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
    # add source and sink that uses the above function op
    source_op = parent_reg.add_simple_op(
        "Source",
        ins=(),
        outs=("env", "val"),
    )
    source_op.outs.chain(func_reg.ins)
    sink_op = parent_reg.add_simple_op(
        "Sink",
        ins=("env", "val"),
        outs=(),
    )
    func_reg.outs.chain(sink_op.ins)
    parent_reg.prettyprint()

    # Keep copy of the graph
    orig_str = parent_reg.prettyformat()
    orig_copy = parent_reg.clone()

    print("pre-split clone".center(80, "-"))
    print(orig_copy.prettyformat())
    print(orig_str)
    assert orig_copy.prettyformat() == orig_str

    # Split region
    ops = list(func_reg.subregion.body.toposorted_ops())
    splitted = func_reg.subregion.split_after(ops[1])

    assert isinstance(splitted, RegionOp)

    print("splitted region".center(80, "-"))
    got = parent_reg.prettyformat().strip()
    print(got)
    expected = """
 Region[func] () -> () {
   Op Source () -> (env=$1, val=$2)
   (env=$1, arg=$2) -> (_redir_env=$3, _redir_arg=$4, _redir_0=$5, _redir_1=$6) =>
     Region[func] (env=$9, arg=$10) -> (_redir_env=$9, _redir_arg=$10, _redir_0=$12, _redir_1=$11) {
       Op BinAdd1 (lhs=$10, rhs=$10) -> (out=$11)
       Op BinMul1 (lhs=$11, rhs=$10) -> (out=$12)
     }
   (_redir_env=$3, _redir_arg=$4, _redir_0=$5, _redir_1=$6) -> (env=$7, ret=$8) =>
     Region[func] (_redir_env=$13, _redir_arg=$14, _redir_0=$15, _redir_1=$16) -> (env=$13, ret=$18) {
       Op BinAdd2 (lhs=$14, rhs=$15) -> (out=$17)
       Op BinMul2 (lhs=$16, rhs=$17) -> (out=$18)
     }
   Op Sink (env=$7, val=$8) -> ()
 }
""".strip()
    assert got == expected


    print("post-split clone".center(80, "-"))
    print(orig_copy.prettyformat())
    print(orig_str)
    assert orig_copy.prettyformat() == orig_str


def test_clone():
    parent_reg = Region.make("func", ins=("a", "b"), outs=["c"])
    add = parent_reg.add_simple_op("Add", ins=["a", "b"], outs=["c"])
    add.ins(a=parent_reg.args.a, b=parent_reg.args.b)
    parent_reg.results(c=add.outs.c)

    orig_str = parent_reg.prettyformat()

    cloned_reg = parent_reg.clone()

    add = parent_reg.add_simple_op("Sub", ins=["a", "b"], outs=["c"])
    add.ins(a=parent_reg.args.a, b=add.outs.c)

    changed_str = parent_reg.prettyformat()
    cloned_str = cloned_reg.prettyformat()
    assert cloned_str == orig_str
    assert changed_str != cloned_str

def test_clone_nested():
    module_reg = Region.make("func", ins=(), outs=())
    parent_regop = module_reg.add_subregion("func", ins=("a", "b"), outs=["c"])
    parent_reg = parent_regop.subregion
    add = parent_reg.add_simple_op("Add", ins=["a", "b"], outs=["c"])
    add.ins(a=parent_reg.args.a, b=parent_reg.args.b)
    parent_reg.results(c=add.outs.c)

    orig_str = module_reg.prettyformat()
    cloned_reg = module_reg.clone()

    sub = parent_reg.add_simple_op("Sub", ins=["a", "b"], outs=["c"])
    sub.ins(a=parent_reg.args.a, b=add.outs.c)
    parent_reg.results(c=sub.outs.c)

    changed_str = module_reg.prettyformat()
    cloned_str = cloned_reg.prettyformat()

    assert cloned_str == orig_str
    assert changed_str != cloned_str
    assert changed_str.strip() == """
 Region[func] () -> () {
   (a=?, b=?) -> (c=?) =>
     Region[func] (a=$1, b=$2) -> (c=$4) {
       Op Add (a=$1, b=$2) -> (c=$3)
       Op Sub (a=$1, b=$3) -> (c=$4)
     }
 }
 """.strip()


def test_pretty_print():
    module_reg = Region.make("func", ins=(), outs=())
    parent_regop = module_reg.add_subregion("func", ins=("a", "b"), outs=["c"])
    parent_reg = parent_regop.subregion
    add = parent_reg.add_simple_op("Add", ins=["a", "b"], outs=["c"])
    add.ins(a=parent_reg.args.a, b=parent_reg.args.b)
    # intentionally not connect the output to test for dead ports
    orig_str = module_reg.prettyformat().strip()
    print(orig_str)
    assert orig_str == """
Region[func] () -> () {
   (a=?, b=?) -> (c=?) =>
     Region[func] (a=$1, b=$2) -> (c=?) {
       Op Add (a=$1, b=$2) -> (c=?)
     }
 }""".strip()


if __name__ == "__main__":
    test_pretty_print()
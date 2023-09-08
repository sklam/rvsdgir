import io
from copy import deepcopy
import weakref
from pprint import pprint
from typing import (
    Any,
    NamedTuple,
    Sequence,
    TypeVar,
    Type,
    Optional,
    Iterable,
    Self,
    # Mapping,
    Tuple,

)
from collections.abc import Sequence, Set, Iterator, Mapping
from collections import defaultdict
from abc import abstractmethod, ABC

from dataclasses import dataclass, field, replace


T = TypeVar("T")


def _type_expect(obj: Any, _expected: Type[T]) -> T:
    assert isinstance(obj, _expected)
    return obj


@dataclass(frozen=True, order=True)
class _Ref:
    storage: weakref.ReferenceType["GraphStorage"] = field(
        hash=False, repr=False
    )  # for GraphStorage
    ident: int

    @classmethod
    def make(cls, storage: "GraphStorage", ident: int) -> "_Ref":
        return _Ref(weakref.ref(storage), ident)

    def __repr__(self) -> str:
        gs = self.storage()
        try:
            wrapper = gs.get_wrapper(self)
        except KeyError:
            typename = "?"
        else:
            typename = type(wrapper).__name__
        return f"Ref[{typename} ident={self.ident}]@0x{id(gs):x}"


@dataclass(frozen=True, order=True)
class Port:
    ref: _Ref
    portname: str
    kind: str

    def __post_init__(self):
        assert self.kind in {"in", "out", "arg", "result"}, self.kind

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class PortMap(Mapping, ABC):
    __slots__ = ()
    _kind: str

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            self.connect(k, v)

    def __getitem__(self, name):
        return self._get_port(name)

    def __getattr__(self, name: str) -> Port:
        return self._get_port(name)

    def connect(self, portname: str, incoming: Port):
        target = self._get_port(portname)
        self._get_storage().connect(incoming, target)

    def list_ports(self) -> list[Port]:
        return [
            Port(self._get_ref(), k, self._kind) for k in self._get_attr_ports()
        ]

    def __len__(self):
        return len(self._get_attr_ports())

    def __iter__(self):
        return iter(self._get_attr_ports())

    @abstractmethod
    def _get_attr_ports(self) -> tuple[str]:
        raise NotImplementedError

    @abstractmethod
    def _get_ref(self) -> _Ref:
        raise NotImplementedError

    @abstractmethod
    def _get_storage(self) -> "GraphStorage":
        raise NotImplementedError

    @abstractmethod
    def _get_port(self, name: str) -> Port:
        raise NotImplementedError


class InputPorts(PortMap):
    __slots__ = ("_ref",)
    _kind = "in"

    _ref: _Ref

    def __init__(self, ref: _Ref):
        self._ref = ref

    def _get_ref(self) -> _Ref:
        return self._ref

    def _get_storage(self) -> "GraphStorage":
        return self._ref.storage()

    def _get_port(self, name) -> Port:
        return self._get_storage().get_input_port(self._ref, name)

    def _get_attr_ports(self) -> tuple[str]:
        attrs = self._get_storage().get_wrapper(self._ref).attrs
        return attrs.ins


class OutputPorts(PortMap):
    __slots__ = ("_ref",)
    _kind = "out"

    _ref: _Ref

    def __init__(self, ref: _Ref):
        self._ref = ref

    def _get_ref(self) -> _Ref:
        return self._ref

    def _get_storage(self) -> "GraphStorage":
        return self._ref.storage()

    def _get_port(self, name) -> Port:
        return self._get_storage().get_output_port(self._ref, name)

    def _get_attr_ports(self) -> tuple[str]:
        attrs = self._get_storage().get_wrapper(self._ref).attrs
        return attrs.outs

    def bridge(self, ports: InputPorts):
        for k, inp in zip(self.keys(), ports.values(), strict=True):
            self.connect(k, inp)


def _forever_counter():
    n = 0
    while True:
        n += 1
        yield n


class VariableNamer:
    def __init__(self):
        self._names = {}
        self._counter = iter(_forever_counter())

    def _fresh_name(self) -> str:
        n = next(self._counter)
        name = f"${n}"
        return name

    def load_edges(self, edges: Sequence["Edge"]):
        # Bundle edges so they all have the same name
        bundles: defaultdict[Port, set[Port]] = defaultdict(set)
        for edge in sorted(edges):
            src = edge.source
            dst = edge.target
            bundles[src].add(dst)
            if dst in bundles:
                bundles[src] |= bundles.pop(dst)

        # Assign names
        for first, remains in bundles.items():
            if first not in self._names:
                self._names[first] = self._fresh_name()
            name = self._names[first]
            for v in remains:
                self._names[v] = name

    def get(self, port: Port) -> str:
        return self._names.get(port, "?")


class PrettyPrinter:
    @classmethod
    def make(
        cls, pp: Optional["PrettyPrinter"], storage: "GraphStorage"
    ) -> "PrettyPrinter":
        if pp is None:
            pp = PrettyPrinter(pp)
            storage.assign_port_names(pp._varnamer)
        return pp

    def __init__(
        self, file=None, indent="", _varnamer: VariableNamer | None = None
    ):
        self.file = file
        self.indent = indent
        if _varnamer is None:
            varnamer = VariableNamer()
        else:
            varnamer = _varnamer
        self._varnamer = varnamer

    def assign_port_names(self, storage: "GraphStorage"):
        storage.assign_port_names(self._varnamer)

    def println(self, *args):
        print(self.indent, *args, file=self.file)

    def get_ports(self, portmap: PortMap) -> tuple[str]:
        ports = portmap.list_ports()
        names = [f"{v.portname}={self._varnamer.get(v)}" for v in ports]
        combined = ", ".join(names)
        return f"({combined})"

    def nest(self) -> "PrettyPrinter":
        return PrettyPrinter(
            file=self.file, indent=self.indent + "  ", _varnamer=self._varnamer
        )


@dataclass(frozen=True)
class OpNode:
    _ref: _Ref

    def prettyprint(self, pp: PrettyPrinter):
        raise NotImplementedError

    @property
    def attrs(self):
        gs = self._ref.storage()
        return gs._nodes[self._ref]

    def get_consuming_ports(self) -> set[Port]:
        out = set()
        ports = self.ins.list_ports()
        gs = self._ref.storage()
        for port in ports:
            sources = gs.iter_edges().filter_by_target({port}).iter_sources()
            out |= set(sources)
        return out

    def get_producing_ports(self) -> set[Port]:
        out = set()
        ports = self.outs.list_ports()
        gs = self._ref.storage()
        for port in ports:
            targets = gs.iter_edges().filter_by_source({port}).iter_targets()
            out |= set(targets)
        return out


@dataclass(frozen=True)
class CopyContext:
    target_storage: "GraphStorage"
    ref_replacements: dict[_Ref, _Ref] = field(default_factory=dict)


# def find_consumed_external_ports(body: Sequence[OpNode]) -> set[Port]:
#     used_after = set()
#     produced_after = set()
#     for op in body:
#         used_after |= op.get_consuming_ports()
#         produced_after |= op.get_producing_ports()
#     results = used_after - produced_after
#     return results


class Region:
    __slots__ = ["_storage", "_ref"]

    _storage: "GraphStorage"
    _ref: _Ref

    @classmethod
    def make(
        cls,
        opname: str,
        ins: Sequence[str] = (),
        outs: Sequence[str] = (),
        parent: Optional["GraphStorage"] = None,
        **kwargs,
    ):
        gs = GraphStorage(parent=parent)
        ref = gs.add_root_region(opname, ins, outs, op_type=RegionOp, **kwargs)
        return Region(gs, ref)

    def __init__(self, storage: "GraphStorage", ref: _Ref) -> None:
        self._storage = storage
        self._ref = ref

    def __deepcopy__(self, memo) -> "Region":
        cls = self.__class__
        ret = cls.__new__(cls)
        memo[id(self)] = ret
        cloned = deepcopy(self._storage, memo)
        ref = cloned.get_root_region()
        ret._storage = cloned
        ret._ref = ref
        return ret


    @property
    def attrs(self):
        gs = self._ref.storage()
        return gs._nodes[self._ref]

    @property
    def args(self) -> "RegionArguments":
        return RegionArguments(self)

    @property
    def results(self) -> "RegionResults":
        return RegionResults(self)

    @property
    def body(self) -> "RegionBody":
        return RegionBody(self)

    def get_parent(self) -> "Region":
        parent = self._storage.parent
        if parent is None:
            raise MalformOperationError("Region has no parent")
        return Region(parent, parent.get_root_region())

    def add_subregion(
        self, opname: str, ins: Sequence[str], outs: Sequence[str], **kwargs
    ) -> "RegionOp":
        ref = self._storage.add_subregion(opname, ins, outs, **kwargs)
        return RegionOp(ref)

    def add_simple_op(
        self, opname: str, ins: Sequence[str], outs: Sequence[str], **kwargs
    ) -> "RegionOp":
        ref = self._storage.add_node(
            opname, ins, outs, op_type=SimpleOp, **kwargs
        )
        return SimpleOp(ref)

    def add_region_clone(self, region: "Region") -> "RegionOp":
        cloned = deepcopy(region)
        ref = self._storage.add_cloned_subregion(cloned)
        return RegionOp(ref)

    def split_after(self, op: OpNode) -> "RegionOp":
        """Split the region after the given ``OpNode`` returning the new
        second RegionOp.

        The resulting graph does not contain interconnections between the
        splitted regions. The original output edges will be orphaned.

        """
        gs = self._storage
        # Split the ops
        topo_ops = self.body.toposorted_ops()
        idx = topo_ops.index(op)
        before_ops = topo_ops[:idx + 1]
        after_ops = topo_ops[idx + 1:]
        parent = self.get_parent()

        second = parent.add_subregion(
            opname=self.attrs.opname, ins=(), outs=(),
            )

        first_half = {x._ref for x in before_ops}
        second_half = {x._ref for x in after_ops}

        gs.split(first_half, second_half, second.subregion)
        return second

    def prettyformat(self) -> str:
        with io.StringIO() as buf:
            pp = PrettyPrinter(file=buf)
            self.prettyprint(pp)
            return buf.getvalue()

    def prettyprint(self, pp: PrettyPrinter | None = None):
        if pp is None:
            pp = PrettyPrinter.make(pp, self._storage)
        else:
            pp.assign_port_names(self._storage)
        ins = pp.get_ports(self.args)
        outs = pp.get_ports(self.results)
        attrs = self.attrs.opname
        pp.println(f"Region[{attrs}] {ins} -> {outs} {{")
        for op in self.body.toposorted_ops():
            op.prettyprint(pp.nest())
        pp.println("}")

    def iter_edges(self) -> "EdgeIterator":
        return self._storage.iter_edges()


class RegionArguments(PortMap):
    _region: Region
    _kind = "arg"

    def __init__(self, region: Region):
        self._region = region

    def _get_ref(self) -> _Ref:
        return self._region._ref

    def _get_storage(self) -> "GraphStorage":
        return self._region._storage

    def _get_port(self, name: str) -> Port:
        port = self._region._storage.get_input_port(self._region._ref, name)
        return replace(port, kind="arg")

    def _get_attr_ports(self) -> tuple[str]:
        attrs = self._get_storage().get_wrapper(self._region._ref).attrs
        return attrs.ins


class RegionResults(PortMap):
    _region: Region
    _kind = "result"

    def __init__(self, region: Region):
        self._region = region

    def _get_ref(self) -> _Ref:
        return self._region._ref

    def _get_storage(self) -> "GraphStorage":
        return self._region._storage

    def _get_port(self, name: str) -> Port:
        port = self._region._storage.get_output_port(self._region._ref, name)
        return replace(port, kind="result")

    def _get_attr_ports(self) -> tuple[str]:
        attrs = self._get_storage().get_wrapper(self._region._ref).attrs
        return attrs.outs


class RegionBody:
    # friend with Region and GraphStorage
    _region: Region

    def __init__(self, region: Region):
        self._region = region

    def __iter__(self) -> Iterable[OpNode]:
        gs = self._region._storage
        return iter((gs.get_wrapper(ref) for ref in gs.iter_nodes()))

    def iter_ops_unordered(self) -> Iterable[OpNode]:
        return iter(self)

    def toposorted_ops(self) -> list[OpNode]:
        avail_ports: set[Port] = set()
        pending = list(self.iter_ops_unordered())
        results = []

        avail_ports |= set(self._region.args.list_ports())

        while pending:
            putback = []
            for op in pending:
                ports = op.get_consuming_ports()
                missing = ports - avail_ports
                if missing:
                    # op not ready, goes to putback
                    putback.append(op)
                else:
                    # op ready, add to results and make outports available
                    outports = set(op.outs.list_ports())
                    avail_ports |= outports
                    results.append(op)
            # Check if toposort is making progress
            if putback and len(putback) == len(pending):
                # No progress
                results += pending
                break
            pending = putback
        return results


@dataclass(frozen=True)
class RegionOp(OpNode):
    @property
    def ins(self):
        return InputPorts(self._ref)

    @property
    def outs(self):
        return OutputPorts(self._ref)

    @property
    def subregion(self) -> "Region":
        gs = self._ref.storage()
        return gs.get_subregion(self._ref)

    def __enter__(self) -> Region:
        return self.subregion

    def __exit__(self, exc_val, exc_type, tb):
        pass

    def prettyprint(self, pp: PrettyPrinter | None = None):
        pp = PrettyPrinter.make(pp, self._ref.storage())
        ins = pp.get_ports(self.ins)
        outs = pp.get_ports(self.outs)
        pp.println(f"{ins} -> {outs} =>")
        with self as region:
            region.prettyprint(pp.nest())


@dataclass(frozen=True)
class SimpleOp(OpNode):
    @property
    def ins(self):
        return InputPorts(self._ref)

    @property
    def outs(self):
        return OutputPorts(self._ref)

    def prettyprint(self, pp: PrettyPrinter | None = None):
        pp = PrettyPrinter.make(pp, self._ref.storage())
        ins = pp.get_ports(self.ins)
        outs = pp.get_ports(self.outs)
        attrs = self.attrs.opname
        pp.println(f"Op {attrs} {ins} -> {outs}")


@dataclass(frozen=True)
class NodeAttrs:
    opname: str
    ins: list[str]
    outs: list[str]
    op_type: Type[OpNode]
    extras: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, k):
        try:
            return self.extras[k]
        except KeyError:
            raise AttributeError(k)

    def __deepcopy__(self, memo) -> "NodeAttrs":
        return self.__class__(
            opname=deepcopy(self.opname, memo),
            ins=deepcopy(self.ins, memo),
            outs=deepcopy(self.outs, memo),
            op_type=self.op_type,
            extras=deepcopy(self.extras, memo),
        )



@dataclass(frozen=True, order=True)
class Edge:
    source: Port
    target: Port

    def is_connecting_with(self, port: Port):
        return self.source == port or self.target == port

    def move(self, repl: Mapping[_Ref, _Ref]) -> "Edge":
        return Edge(self.source.move(repl), self.target.move(repl))

    def replace(self, **kwargs):
        return replace(self, **kwargs)



class EdgeIterator:
    _edges: Iterable[Edge]

    def __init__(self, edges: Iterable[Edge]):
        self._edges = edges

    def __iter__(self) -> Iterator[Edge]:
        return iter(self._edges)

    def iter_ports(self) -> Iterable[Port]:
        for e in self._edges:
            yield e.source
            yield e.target

    def iter_sources(self) -> Iterable[Port]:
        for e in self._edges:
            yield e.source

    def iter_targets(self) -> Iterable[Port]:
        for e in self._edges:
            yield e.target

    def filter_by_ports(
        self,
        ports: Iterable[Port],
    ):
        def test(edge):
            return any(map(edge.is_connecting_with, ports))

        filtered = filter(test, self._edges)
        return EdgeIterator(filtered)

    def filter_by_target(self, ports: Set[Port]) -> "EdgeIterator":
        return EdgeIterator(filter(lambda e: e.target in ports, self._edges))

    def filter_by_source(self, ports: Set[Port]) -> "EdgeIterator":
        return EdgeIterator(filter(lambda e: e.source in ports, self._edges))


class RefHeap:
    _heap: list[_Ref]

    def __init__(self):
        self._heap = []

    def alloc(self, storage: "GraphStorage") -> _Ref:
        ref = _Ref.make(storage, len(self._heap))
        self._heap.append(ref)
        return ref


class _DanglingEdges(NamedTuple):
    dead_source: set[Edge]
    dead_target: set[Edge]


@dataclass(frozen=True)
class GraphStorage:
    parent: Optional["GraphStorage"] = None
    _refheap: RefHeap = field(default_factory=RefHeap)
    _sentinels: dict[str, _Ref] = field(default_factory=dict)
    _nodes: dict[_Ref, NodeAttrs] = field(default_factory=dict)
    _edges: list[Edge] = field(default_factory=list)

    def __deepcopy__(self, memo) -> "GraphStorage":
        cls = self.__class__
        ret = cls.__new__(cls)
        memo[id(self)] = ret

        gs = self.__class__()
        repl = {}
        for ref, attrs in self._nodes.items():
            newref = gs._refheap.alloc(self)
            gs._nodes[newref] = deepcopy(attrs, memo)
            repl[ref] = newref
        for name, ref in self._sentinels.items():
            gs._sentinels[name] = repl[ref]
        for edge in self._edges:
            gs._edges.append(edge)
        return gs

    def add_node(self, opname, ins, outs, op_type, **kwargs) -> _Ref:
        ref = self._refheap.alloc(self)
        attrs = NodeAttrs(
            opname=opname, ins=list(ins), outs=list(outs), op_type=op_type
        )
        attrs.extras.update(kwargs)
        self._nodes[ref] = attrs
        return ref

    def add_root_region(self, opname, ins, outs, op_type, **kwargs) -> _Ref:
        assert "root" not in self._sentinels
        ref = self.add_node(opname, ins, outs, op_type, **kwargs)
        self._sentinels["root"] = ref
        return ref

    def add_subregion(self, opname, ins, outs, **kwargs) -> _Ref:
        subregion = Region.make(opname, ins, outs, parent=self, **kwargs)
        ref = self._refheap.alloc(self)
        # share the same NodeAttrs
        self._nodes[ref] = attrs = subregion._storage._nodes[subregion._ref]
        attrs.extras["region"] = subregion
        return ref

    def add_cloned_subregion(self, subregion) -> _Ref:
        ref = self._refheap.alloc(self)
        # share the same NodeAttrs
        self._nodes[ref] = attrs = subregion._storage._nodes[subregion._ref]
        attrs.extras["region"] = subregion
        return ref

    def get_root_region(self) -> _Ref:
        return self._sentinels["root"]

    def get_attrs(self, ref) -> NodeAttrs:
        return self._nodes[ref]

    def get_subregion(self, ref: _Ref) -> Region:
        attrs = self._nodes[ref]
        return attrs.region

    def get_input_port(self, ref: _Ref, portname: str) -> Port:
        if portname in self._nodes[ref].ins:
            return Port(ref, portname, kind="in")
        else:
            raise InvalidPortNameError(portname)

    def get_output_port(self, ref: _Ref, portname: str) -> Port:
        if portname in self._nodes[ref].outs:
            return Port(ref, portname, kind="out")
        else:
            raise InvalidPortNameError(portname)

    def split(self, first: Set[_Ref], second: Set[_Ref], newregion: "Region"):
        newgs = newregion._storage

        first_nodes = [*self._sentinels.values(), *(ref for ref in self._nodes if ref in first)]
        second_nodes = [ref for ref in self._nodes if ref in second]
        diff = set(first_nodes).union(second_nodes).difference(self._nodes)
        if diff:
            raise MalformOperationError(
                f"Given nodes do not match all of contained nodes: diff={diff}"
            )

        # Fix up the other's nodes
        refmap = newgs.copy_nodes(self, second_nodes)

        # Fix up this GraphStorage nodes
        for k in list(self._nodes.keys()):
            if k not in first_nodes:
                self._nodes.pop(k)

        # Split edges
        first_edges: list[Edge] = []
        second_edges: list[Edge] = []
        first_in_edges: list[Edge] = []
        second_in_edges: list[Edge] = []
        first_out_edges: list[Edge] = []
        second_out_edges: list[Edge] = []
        sentinel_only_edges: list[Edge] = []
        for edge in self._edges:
            assert self.check_edge(edge)
            source_in_first = edge.source.ref in first
            source_in_second = edge.source.ref in second
            target_in_first = edge.target.ref in first
            target_in_second = edge.target.ref in second
            if source_in_first and target_in_first:
                first_edges.append(edge)
            elif source_in_second and target_in_second:
                second_edges.append(edge)
            else:
                if not any((source_in_first, source_in_second, target_in_first, target_in_second)):
                    sentinel_only_edges.append(edge)
                else:
                    if source_in_first:
                        first_out_edges.append(edge)
                    elif source_in_second:
                        second_out_edges.append(edge)

                    if target_in_first:
                        first_in_edges.append(edge)
                    elif target_in_second:
                        second_in_edges.append(edge)

        # --- Fixup first partition's edges ---
        self._edges.clear()
        self._edges.extend(first_edges)
        self._edges.extend(first_in_edges) # use these unchanged

        this_ref = self.get_root_region()
        this_attrs = self.get_attrs(this_ref)

        old_result_ports = tuple(Port(this_ref, x, kind="result") for x in this_attrs.outs)

        this_attrs.outs.clear()

        first_newports: dict[Port, Port] = {}
        redir_args: list[tuple[Port, Port]] = []
        # connect any args to the results
        for portname in this_attrs.ins:
            port = Port(this_ref, portname, kind="arg")
            # forward this port
            newport = self.append_output_port(this_ref, f"_redir_{portname}")
            self.connect(port, newport.replace(kind="result"))

            redir_args.append((port, newport))

        # update target on out edges
        for edge in first_out_edges:
            if edge.target not in first_newports:
                # allocate new port
                first_newports[edge.target] = self.append_output_port(this_ref, f"_redir_{len(first_newports)}")
            newport = first_newports[edge.target].replace(kind="result")
            self.connect(edge.source, newport)
            redir_args.append((edge.source, newport))

        # --- Fixup second partition's edges ---
        portmap: dict[Port, Port] = {}
        other_ref = newgs.get_root_region()

        for i, portname in enumerate(this_attrs.outs):
            newport = newgs.append_input_port(other_ref, portname).replace(kind="arg")
            inport, _ = redir_args[i]
            portmap[inport] = newport

        for port in old_result_ports:
            newport = newgs.append_output_port(other_ref, port.portname).replace(kind="result")
            portmap[port] = newport

        for edge in second_edges:
            newgs.connect(
                edge.source.replace(ref=refmap[edge.source.ref]),
                edge.target.replace(ref=refmap[edge.target.ref]),
            )

        for edge in second_in_edges:
            newgs.connect(
                portmap[edge.source],
                edge.target.replace(ref=refmap[edge.target.ref]),
            )

        for edge in second_out_edges:
            newgs.connect(
                edge.source.replace(ref=refmap[edge.source.ref]),
                portmap[edge.target],
            )

        for edge in sentinel_only_edges:
            newgs.connect(portmap[edge.source], portmap[edge.target])

    def copy_nodes(self, source_graph: Self, nodes: Iterable[_Ref]) -> dict[_Ref, _Ref]:
        """Copy nodes from another storage and return a ref mapping.
        Edges are not copied nor fixed up.
        """
        refmap = {}
        for ref in nodes:
            # Copy node
            newref = self._refheap.alloc(self)
            attrs = deepcopy(source_graph._nodes[ref])
            self._nodes[newref] = attrs
            refmap[ref] = newref
        return refmap

    def erase_nodes(self, deadset: Set[_Ref]) -> _DanglingEdges:
        for ref in deadset:
            self._nodes.pop(ref)
        dead_target = set()
        dead_source = set()
        remain_edges = []
        for edge in self._edges:
            source_is_dead = edge.source.ref in deadset
            target_is_dead = edge.target.ref in deadset
            if source_is_dead and target_is_dead:
                pass
            elif not (source_is_dead or target_is_dead):
                remain_edges.append(edge)
            elif source_is_dead:
                dead_source.add(edge)
            elif target_is_dead:
                dead_target.add(edge)
        self._edges.clear()
        self._edges.extend(remain_edges)
        return _DanglingEdges(dead_target=dead_target, dead_source=dead_source)

    def check_edge(self, edge: Edge) -> bool:
        return all((
            edge.source.ref.storage() is self,
            edge.source.ref.storage() is self,
        ))


    def connect(self, source: Port, target: Port):
        if source.ref.storage() is not target.ref.storage():
            raise MalformOperationError("Ports must be from same GraphStorage")
        if source.ref.storage() is not self:
            raise MalformOperationError("Ports do not belong to this GraphStorage")
        self._edges.append(Edge(source, target))

    def append_input_port(self, ref: _Ref, portname: str) -> Port:
        attrs = self._nodes[ref]
        assert portname not in attrs.ins
        attrs.ins.append(portname)
        return Port(ref, portname, "in")

    def append_output_port(self, ref: _Ref, portname: str) -> Port:
        attrs = self._nodes[ref]
        assert portname not in attrs.outs
        attrs.outs.append(portname)
        return Port(ref, portname, "out")

    def assign_port_names(self, varnamer: VariableNamer):
        varnamer.load_edges(self._edges)

    def get_wrapper(self, ref: _Ref) -> "OpNode":
        attrs = self._nodes[ref]
        return attrs.op_type(ref)

    def iter_nodes(self) -> Iterable[_Ref]:
        # Skip sentinels
        return filter(
            lambda x: x not in self._sentinels.values(), self._nodes.keys()
        )

    def iter_edges(self) -> EdgeIterator:
        return EdgeIterator(self._edges)


class RVSDGError(Exception):
    """Any error related to the RVSDGraph container and operations
    """


class MalformOperationError(RVSDGError):
    pass

class MalformContainerError(RVSDGError):
    pass

class InvalidPortNameError(RVSDGError):
    pass

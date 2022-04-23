from dataclasses import dataclass
from copy import deepcopy
from typing import Iterable, List, Tuple, Dict, Optional, Callable, TypeVar
import math
import numpy as np
import string

Shape = Tuple[int,...]
Idxs = List[int]
IdxsMap = Dict[int,int]
Tensor = np.ndarray


def einsum_tensor(x) -> Tensor:
    return np.asarray(x)

def einsum_empty_tensor(shape: Shape) -> Tensor:
    return np.empty(shape)

def einsum_broadcast_to(t: Tensor, shape: Shape) -> Tensor:
    return np.broadcast_to(t, shape)

# returns tensor of given shape with value fn(i1,...,iN) at pos (i1...iN)
def einsum_tensor_frompos(fn: Callable[..., float], shape: Shape) -> Tensor:
    if math.prod(shape) == 0:
        return einsum_empty_tensor(shape)

    def recurse(pos: List[int], shape: Shape):
        if len(shape) == 0:
            return fn(*pos)
        else:
            remainder = shape[1:]
            return [ recurse(pos + [i], remainder) for i in range(shape[0]) ]

    return einsum_tensor(recurse([], shape))


@dataclass
class EinsumOutputSpec:

    # tuple of dimensions (non-negative integers)
    shape: Shape

    # idxs is a list of integers in [0,52) representing letters and
    # of consecutive negative integers from -1 and down representing an ellipsis
    idxs: Idxs

@dataclass
class EinsumInputSpec:

    # tuple of dimensions
    shape: Shape

    # idxs is a list of integers in [0,52) representing letters and
    # of consecutive negative integers from -1 and down representing an ellipsis
    idxs: Idxs

@dataclass
class EinsumSpec:

    # idxs_map maps indices to dimensions, where an index
    #   is an integer with [0,52) ranging over upper and lower case letters
    #   (e.g. A and Z are 0 and 25, a and z are 26 and 51) and with
    #   negative integers ranging over the indices of the ellipsis, if any
    #   (-1 is the last index of the ellipsis, -2 is the second last, etc)
    idxs_map: IdxsMap

    # inputs is a non-empty list of instances of EinsumInputSpec
    inputs: List[EinsumInputSpec]

    # output is an instance of EinsumOutputSpec
    output: EinsumOutputSpec

EINSUM_LETTERS_UPPER = string.ascii_uppercase # A-Z
EINSUM_LETTERS_LOWER = string.ascii_lowercase # a-z
EINSUM_LETTERS = EINSUM_LETTERS_UPPER + EINSUM_LETTERS_LOWER # A-Za-z

def einsum_index(letter: str) -> int:
    assert letter in EINSUM_LETTERS, f"index '{letter}' ({ord(letter)}) is not a valid einsum letter"
    return EINSUM_LETTERS.index(letter)

def einsum_letter(idx: int) -> str:
    assert 0 <= idx < len(EINSUM_LETTERS)
    return EINSUM_LETTERS[idx]

def einsum_idxs(subscripts: str) -> Idxs:
    return [ einsum_index(letter) for letter in subscripts ]

def einsum_infer_output_subscripts(
        idxs_map: IdxsMap,
        ispecs: List[EinsumInputSpec]) -> str:
    # count occurrences of letter indices in inputs
    idxs_count = [0] * len(EINSUM_LETTERS)
    for spec in ispecs:
        for idx in spec.idxs:
            if idx >= 0:
                idxs_count[idx] += 1
    subscripts = "..."
    for idx in idxs_map:
        if idxs_count[idx] == 1:
            subscripts += einsum_letter(idx)
    return subscripts

E = TypeVar('E')
def einsum_count_elements(elements: Iterable[E]) -> Dict[E,int]:
    counts : Dict[E,int] = {}
    for x in elements:
        counts[x] = counts.get(x, 0) + 1
    return counts

def einsum_find_duplicate(letters: str) -> Optional[str]:
    counts = einsum_count_elements(letters)
    return next((x for x, y in counts.items() if y > 1), None)

def einsum_ellipsis_idxs(idxs_map: IdxsMap) -> Idxs:
    return sorted([ idx for idx in idxs_map if idx < 0 ])

def einsum_output(
        idxs_map: IdxsMap,
        ispecs: List[EinsumInputSpec],
        subscripts: Optional[str]) \
        -> EinsumOutputSpec:
    if subscripts is None:
        subscripts = einsum_infer_output_subscripts(idxs_map, ispecs)
        assert subscripts.startswith("...")
        trailing = subscripts[len("..."):]
        assert trailing.count(".") == trailing.count(" ") == 0
        lst = ["", trailing]
    else:
        lst = [ s.replace(" ", "") for s in subscripts.split("...") ]
        assert 1 <= len(lst) <= 2, f"multiple ellipses in '{subscripts}'"

        # following torch and onnx, we don't require that the output has
        # an ellipsis even if any appears in the inputs, whereas to follow
        # numpy we'd require:
        #
        #   if einsum_ellipsis_idxs(idxs_map) != []: assert len(lst) == 2

        duplicate = einsum_find_duplicate("".join(lst))
        assert duplicate is None, f"duplicate index {duplicate} in '{subscripts}'"

    idxs = [ einsum_index(letter) for letter in lst[0] ]
    if len(lst) == 2:
        idxs += einsum_ellipsis_idxs(idxs_map)
        idxs += [ einsum_index(letter) for letter in lst[1] ]
    assert [] == [ idx for idx in idxs if idx not in idxs_map ]

    shape = tuple( idxs_map[idx] for idx in idxs )

    return EinsumOutputSpec(shape, idxs)

def einsum_input(subscripts: str, shape: Shape) -> EinsumInputSpec:
    lst = [ s.replace(" ", "") for s in subscripts.split("...") ]
    if len(lst) == 1:
        assert len(lst[0]) == len(shape), \
            f"# indices in '{subscripts}' != length of shape {list(shape)}"
        lst.append("") # treat this case the same as an empty ellipsis at end
    else:
        assert len(lst) == 2, f"multiple ellipses in '{subscripts}'"
        assert len(lst[0]) + len(lst[1]) <= len(shape), \
            f"# indices in '{subscripts}' > length of shape {list(shape)}"
    leading_idxs = einsum_idxs(lst[0])
    trailing_idxs = einsum_idxs(lst[1])
    letters_len = len(leading_idxs) + len(trailing_idxs)
    ellipsis_len = len(shape) - len(leading_idxs) - len(trailing_idxs)
    ellipsis_idxs = list(range(-ellipsis_len, 0))
    idxs = leading_idxs + ellipsis_idxs + trailing_idxs
    # following numpy and torch, don't broadcast-match the shapes of multiple
    # occurrences of a subscript index within the same operand
    for idx, dim in zip(idxs, shape):
        assert dim == shape[idxs.index(idx)], \
                f"operand has repeated subscript {einsum_letter(idx)} with different shape sizes"
    return EinsumInputSpec(shape, idxs)

def einsum_extend_idxs_map(idxs_map: IdxsMap, idx: int, n: int) -> IdxsMap:
    old = idxs_map.get(idx)
    if old is None or old == 1:
        idxs_map[idx] = n
    else:
        assert n == 1 or n == old, f"cannot unify dimensions {old}, {n}"
    return idxs_map

def einsum_idxs_map(ispecs: List[EinsumInputSpec]) -> IdxsMap:
    idxs_map: IdxsMap = {}
    for spec in ispecs:
        for idx, n in zip(spec.idxs, spec.shape):
            einsum_extend_idxs_map(idxs_map, idx, n)
    return idxs_map

def einsum_spec(equation: str, ishapes: List[Shape]) -> EinsumSpec:
    io = equation.split("->")
    assert 1 <= len(io) <= 2, f"multiple arrows in '{equation}'"
    osubscripts = io[1] if len(io) == 2 else None
    isubscripts = io[0].split(",")
    assert len(isubscripts) == len(ishapes), "# equation inputs != # input shapes"
    ispecs = [ einsum_input(*p) for p in zip(isubscripts, ishapes) ]
    idxs_map = einsum_idxs_map(ispecs)
    ospec = einsum_output(idxs_map, ispecs, osubscripts)
    return EinsumSpec(idxs_map, ispecs, ospec)

def einsum_squeeze_input_spec(input_spec: EinsumInputSpec) -> EinsumInputSpec:
    shape = list(input_spec.shape)
    idxs = list(input_spec.idxs)
    while 1 in shape:
        p = shape.index(1)
        del shape[p]
        del idxs[p]
    return EinsumInputSpec(tuple(shape), idxs)

def einsum_execute(spec: EinsumSpec, tensors: List[Tensor]) -> Tensor:
    idxs_map = spec.idxs_map
    inputs = spec.inputs
    output = spec.output
    assert len(inputs) == len(tensors)
    for input_spec, tensor in zip(inputs, tensors):
        assert input_spec.shape == tensor.shape

    # squeeze inputs to avoid multiplying over axes of length 1 which
    # which may, through broadcast, have different length in idxs_map
    inputs = list(map(einsum_squeeze_input_spec, inputs))
    tensors = [tensor.squeeze() for tensor in tensors]
    for input_spec, tensor in zip(inputs, tensors):
        assert input_spec.shape == tensor.shape

    out_idxs = output.idxs
    in_only_idxs = list(set(idxs_map).difference(out_idxs))
    def fn(*opos) -> float:
        assert len(opos) == len(out_idxs)
        pos_map = dict(zip(out_idxs, opos))

        def recurse(remaining_idxs: Idxs) -> float:
            if len(remaining_idxs) == 0:
                prod = 1.
                for input_spec, tensor in zip(inputs, tensors):
                    pos = [ pos_map[idx] for idx in input_spec.idxs ]
                    prod *= tensor.item(*pos)
                return prod
            else:
                head = remaining_idxs[0]
                tail = remaining_idxs[1:]
                acc = 0.
                for p in range(idxs_map[head]):
                    pos_map[head] = p
                    acc += recurse(tail)
                return acc

        return recurse(in_only_idxs)

    return einsum_tensor_frompos(fn, output.shape)

def einsum(equation: str, *tensors: Tensor) -> Tensor:
    ishapes = [ tensor.shape for tensor in tensors ]
    spec = einsum_spec(equation, ishapes)
    return einsum_execute(spec, list(tensors))


def einsum_is_identity_spec(spec: EinsumSpec) -> bool:
    return len(spec.inputs) == 1 and spec.inputs[0].idxs == spec.output.idxs

# an einsum rewrite takes src and dest EinsumSpec and a transform function
# that maps a list of src tensors to a list of dest tensors, where the
# src and dest tensor lists match the src and dest input specs
# TODO: change EinsumRewrite to a dataclass
Tensors = List[Tensor]
EinsumTransform = Callable[[Tensors],Tensors]
EinsumRewrite = Tuple[EinsumSpec,EinsumSpec,EinsumTransform]

def einsum_rewrite_diagonal(src_spec: EinsumSpec,
        arg: int, axis1: int, axis2: int) -> EinsumRewrite:
    assert 0 <= arg < len(src_spec.inputs)
    src_shape = src_spec.inputs[arg].shape
    src_idxs = src_spec.inputs[arg].idxs
    assert 0 <= axis1 <= axis2 < len(src_shape)
    dim = src_shape[axis1]
    idx = src_idxs[axis1]
    assert dim == src_shape[axis2]
    assert idx == src_idxs[axis2]
    dest_spec = deepcopy(src_spec)
    dest_ispec = dest_spec.inputs[arg]
    # dest_ispec.shape is an immutable tuple, so we make it a list to
    # mutate it and then write it back as a tuple
    shape = list(dest_ispec.shape)
    shape.pop(axis2)
    shape.pop(axis1)
    shape.append(dim)
    dest_ispec.shape = tuple(shape)
    # dest_ispec.idxs is a list and we mutate it in place
    idxs = dest_ispec.idxs
    idxs.pop(axis2)
    idxs.pop(axis1)
    idxs.append(idx)
    def transform(ts : Tensors) -> Tensors:
        ts[arg] = ts[arg].diagonal(axis1=axis1, axis2=axis2)
        return ts
    return (src_spec, dest_spec, transform)

def einsum_rewrites_diagonals(src_spec: EinsumSpec, arg: int) -> List[EinsumRewrite]:
    assert 0 <= arg < len(src_spec.inputs)
    rewrites = []
    idxs = list(src_spec.inputs[arg].idxs)
    for idx, c in einsum_count_elements(idxs).items():
        while c > 1:
            axis1 = idxs.index(idx)
            axis2 = idxs.index(idx, axis1 + 1)
            rewrite = einsum_rewrite_diagonal(src_spec, arg, axis1, axis2) 
            rewrites.append(rewrite)
            prev_spec, next_spec, transform = rewrite
            assert prev_spec == src_spec
            src_spec = next_spec
            idxs = list(src_spec.inputs[arg].idxs)
            c -= 1
    return rewrites

def einsum_rewrites_multiply(src_spec: EinsumSpec, arg1: int, arg2: int) -> List[EinsumRewrite]:
    # TODO
    return []

def einsum_rewrites_sum(src_spec: EinsumSpec, arg: int) -> List[EinsumRewrite]:
    # TODO
    return []

def einsum_rewrites(spec: EinsumSpec) -> List[EinsumRewrite]:
    rewrites : List[EinsumRewrite] = []

    def last_spec() -> EinsumSpec:
        assert len(rewrites) > 0
        penultimate, last, transform =  rewrites[-1]
        return last

    for arg in range(len(spec.inputs)):
        rewrites += einsum_rewrites_diagonals(spec, arg)
        if len(rewrites) > 0:
            spec = last_spec()
    # TODO: append bunch of other rewrites
    if len(spec.inputs) >= 2:
        for arg in reversed(range(len(spec.inputs) - 1)):
            rewrites += einsum_rewrites_multiply(spec, arg, arg + 1)
            if len(rewrites) > 0:
                spec = last_spec()
    assert len(spec.inputs) == 1
    rewrites += einsum_rewrites_sum(spec, 0)
    if len(rewrites) > 0:
        spec = last_spec()
    assert len(rewrites) == 0 or spec == last_spec()
    assert einsum_is_identity_spec(spec) 
    return rewrites

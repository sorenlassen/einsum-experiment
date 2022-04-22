from typing import List, Tuple, Dict, Optional, Callable, NamedTuple
import math
import numpy as np
import string
import functools

Shape = Tuple[int, ...]
Idxs = Tuple[int, ...]
IdxsMap = Dict[int, int]
Tensor = np.ndarray

memoize = functools.lru_cache(maxsize=None)


def einsum_tensor(x) -> Tensor:
    return np.asarray(x)

def einsum_empty_tensor(shape: Shape) -> Tensor:
    return np.empty(shape)

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


class EinsumOutputSpec(NamedTuple):

    # idxs is a tuple of integers in [0,52) representing letters and
    # of consecutive negative integers from -1 and down representing an ellipsis
    idxs: Idxs

class EinsumInputSpec(NamedTuple):

    # tuple of dimensions
    shape: Shape

    # idxs is a tuple of integers in [0,52) representing letters and
    # of consecutive negative integers from -1 and down representing an ellipsis
    idxs: Idxs

class EinsumSpec(NamedTuple):

    # inputs is a non-empty tuple of instances of EinsumInputSpec
    inputs: Tuple[EinsumInputSpec, ...]

    # output is an instance of EinsumOutputSpec
    output: EinsumOutputSpec

    # idxs_map maps indices to dimensions, where an index
    #   is an integer with [0,52) ranging over upper and lower case letters
    #   (e.g. A and Z are 0 and 25, a and z are 26 and 51) and with
    #   negative integers ranging over the indices of the ellipsis, if any
    #   (-1 is the last index of the ellipsis, -2 is the second last, etc)
    @property
    def idxs_map(self):
        return einsum_idxs_map(self.inputs)

    @property
    def output_shape(self):
        return tuple( self.idxs_map[idx] for idx in self.output.idxs )

EINSUM_LETTERS = string.ascii_uppercase + string.ascii_lowercase # A-Za-z

def einsum_index(letter: str) -> int:
    assert letter in EINSUM_LETTERS, f"index '{letter}' ({ord(letter)}) is not a valid einsum letter"
    return EINSUM_LETTERS.index(letter)

def einsum_letter(idx: int) -> str:
    assert 0 <= idx < len(EINSUM_LETTERS)
    return EINSUM_LETTERS[idx]

def einsum_idxs(subscripts: str) -> Idxs:
    return tuple([ einsum_index(letter) for letter in subscripts ])

def einsum_infer_output_subscripts(
        ispecs: Tuple[EinsumInputSpec, ...]) -> str:
    # count occurrences of letter indices in inputs
    idxs_count = [0] * len(EINSUM_LETTERS)
    for spec in ispecs:
        for idx in spec.idxs:
            if idx >= 0:
                idxs_count[idx] += 1
    subscripts = "..."
    for idx in einsum_idxs_map(ispecs):
        if idxs_count[idx] == 1:
            subscripts += einsum_letter(idx)
    return subscripts

def einsum_find_duplicate(l):
    for x in l:
        if l.count(x) > 1:
            return x

def einsum_ellipsis_idxs(idxs_map: IdxsMap) -> Idxs:
    return tuple(sorted([ idx for idx in idxs_map if idx < 0 ]))

def einsum_output(
        ispecs: Tuple[EinsumInputSpec, ...],
        subscripts: Optional[str]) \
        -> EinsumOutputSpec:
    if subscripts is None:
        subscripts = einsum_infer_output_subscripts(ispecs)
    leading, ellipsis, trailing = [s.replace(" ", "") for s in subscripts.partition("...")]
    assert "." not in leading + trailing
    assert " " not in leading + trailing
    assert (duplicate := einsum_find_duplicate(leading + trailing)) is None, \
        f"duplicate index {duplicate} in '{subscripts}'"

    # following torch and onnx, we don't require that the output has
    # an ellipsis even if any appears in the inputs, whereas to follow
    # numpy we'd require:
    #
    #   if einsum_ellipsis_idxs(idxs_map): assert ellipsis

    idxs = [ einsum_index(letter) for letter in leading ]
    if ellipsis:
        idxs += einsum_ellipsis_idxs(einsum_idxs_map(ispecs))
        idxs += [ einsum_index(letter) for letter in trailing ]
    assert all([idx in einsum_idxs_map(ispecs) for idx in idxs])

    return EinsumOutputSpec(tuple(idxs))

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
    ellipsis_len = len(shape) - len(leading_idxs) - len(trailing_idxs)
    ellipsis_idxs = tuple(range(-ellipsis_len, 0))
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

@memoize
def einsum_idxs_map(ispecs: Tuple[EinsumInputSpec, ...]) -> IdxsMap:
    idxs_map: IdxsMap = {}
    for spec in ispecs:
        for idx, n in zip(spec.idxs, spec.shape):
            einsum_extend_idxs_map(idxs_map, idx, n)
    return idxs_map

def einsum_spec(equation: str, ishapes: Tuple[Shape, ...]) -> EinsumSpec:
    assert equation.count('->') <= 1, f"multiple arrows in '{equation}'"
    left, arrow, right = equation.partition('->')
    osubscripts = right if arrow else None
    isubscripts = left.split(",")
    assert len(isubscripts) == len(ishapes), "# equation inputs != # input shapes"
    ispecs = tuple([ einsum_input(*p) for p in zip(isubscripts, ishapes) ])
    ospec = einsum_output(ispecs, osubscripts)
    return EinsumSpec(ispecs, ospec)

def einsum_execute(spec: EinsumSpec, tensors: Tuple[Tensor, ...]) -> Tensor:
    assert len(spec.inputs) == len(tensors)
    for input_spec, tensor in zip(spec.inputs, tensors):
        assert input_spec.shape == tensor.shape

    in_only_idxs = tuple(set(spec.idxs_map).difference(spec.output.idxs))
    def fn(*opos) -> float:
        assert len(opos) == len(spec.output.idxs)
        pos_map = dict(zip(spec.output.idxs, opos))

        def recurse(remaining_idxs: Idxs) -> float:
            if len(remaining_idxs) == 0:
                prod = 1.
                for input_spec, tensor in zip(spec.inputs, tensors):
                    pos = [ pos_map[idx] for idx in input_spec.idxs ]
                    prod *= tensor.item(*pos)
                return prod
            else:
                head, *tail = remaining_idxs
                acc = 0.
                for p in range(spec.idxs_map[head]):
                    pos_map[head] = p
                    acc += recurse(tail)
                return acc

        return recurse(in_only_idxs)

    return einsum_tensor_frompos(fn, spec.output_shape)

def einsum_compile(equation: str, *tensors: Tensor) -> EinsumSpec:
    ishapes = tuple([ tensor.shape for tensor in tensors ])
    return einsum_spec(equation, ishapes)

def einsum(equation: str, *tensors: Tensor) -> Tensor:
    spec = einsum_compile(equation, *tensors)
    return einsum_execute(spec, tensors)

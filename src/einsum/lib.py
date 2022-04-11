from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable
import math
import numpy as np

Shape = Tuple[int,...]
Idxs = List[int]
IdxsMap = Dict[int,int]
Tensor = np.ndarray


def einsum_tensor(x) -> Tensor:
    return np.array(x)

def einsum_empty_tensor(shape: Shape) -> Tensor:
    return np.broadcast_to(0., shape)

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

    # tuple of magnitudes (non-negative integers)
    shape: Shape

    # idxs is a list of integers in the output with [0,52) referring to
    # regular indices (representing upper and lower case letters) and
    # consecutive negative indices representing ellipsis
    idxs: Idxs

@dataclass
class EinsumInputSpec:

    # tuple of magnitudes
    shape: Shape

    # leading_idxs is a list of integers in [0,52), namely all indices at the
    # beginning of the input up to the ellipsis, if any, otherwise to the end of
    # the input
    leading_idxs: Idxs

    # trailing_idxs is a list of integers in [0,52) in the input after any
    # ellipsis
    trailing_idxs: Idxs

    def ellipsis_shape(self) -> Shape:
        begin = len(self.leading_idxs)
        end = len(self.shape) - len(self.trailing_idxs)
        return self.shape[begin:end]

@dataclass
class EinsumSpec:

    # idxs_map maps indices to magnitudes, where an index
    #   is an integer with [0,52) ranging over upper and lower case letters
    #   (e.g. A and Z are 0 and 25, a and z are 26 and 51) and with
    #   negative integers ranging over the indices of the ellipsis, if any
    #   (-1 is the last index of the ellipsis, -2 is the second last, etc)
    idxs_map: IdxsMap

    # inputs is a non-empty list of instances of EinsumInputSpec
    inputs: List[EinsumInputSpec]

    # output is an instance of EinsumOutputSpec
    output: EinsumOutputSpec

EINSUM_LETTERS_UPPER = 26 # oounts upper case letters A-Z
EINSUM_LETTERS_LOWER = 26 # counts lower case letters a-z
EINSUM_LETTERS = EINSUM_LETTERS_UPPER + EINSUM_LETTERS_LOWER

def einsum_index(letter: str) -> int:
    if "A" <= letter and letter <= "Z":
        x = ord(letter) - ord("A")
        assert 0 <= x < EINSUM_LETTERS_UPPER
        return x
    elif "a" <= letter and letter <= "z":
        x = ord(letter) - ord("a")
        assert 0 <= x < EINSUM_LETTERS_LOWER
        return EINSUM_LETTERS_UPPER + x
    assert False, f"index '{letter}' ({ord(letter)}) is not a letter"

def einsum_letter(idx: int) -> str:
    assert 0 <= idx < EINSUM_LETTERS
    if idx < EINSUM_LETTERS_UPPER:
        return chr(ord("A") + idx)
    else:
        return chr(ord("a") + idx - EINSUM_LETTERS_UPPER)

def einsum_idxs(formula: str) -> Idxs:
    return [ einsum_index(letter) for letter in formula ]

def einsum_infer_output_formula( \
        idxs_map: IdxsMap, \
        inputs: List[EinsumInputSpec]) -> str:
    # count occurrences of letter indices in inputs
    idxs_count = [0] * EINSUM_LETTERS
    for spec in inputs:
        for idxs in (spec.leading_idxs, spec.trailing_idxs):
            for idx in idxs:
                idxs_count[idx] += 1
    formula = "..."
    for idx in idxs_map:
        if idxs_count[idx] == 1:
            formula += einsum_letter(idx)
    return formula

def einsum_find_duplicate(letters: str) -> Optional[str]:
    if len(letters) > 1:
        s = sorted(letters)
        for x in range(len(s) - 1):
            if s[x] == s[x + 1]:
                return s[x]
    return None

def einsum_ellipsis_idxs(idxs_map: IdxsMap) -> Idxs:
    return sorted([ idx for idx in idxs_map if idx < 0 ])

def einsum_output( \
        idxs_map: IdxsMap, \
        inputs: List[EinsumInputSpec], \
        formula: Optional[str], \
        shape: Shape) -> EinsumOutputSpec:
    lst = None
    if formula is None:
        formula = einsum_infer_output_formula(idxs_map, inputs)
        assert formula.startswith("...")
        trailing = formula[len("..."):]
        assert trailing.count(".") == trailing.count(" ") == 0
        lst = ["", trailing]
    else:
        lst = [ s.replace(" ", "") for s in formula.split("...") ]
        assert 1 <= len(lst) <= 2, f"multiple ellipses in '{formula}'"
        duplicate = einsum_find_duplicate("".join(lst))
        assert duplicate is None, f"duplicate index {duplicate} in '{formula}'"

    idxs = [ einsum_index(letter) for letter in lst[0] ]
    if len(lst) == 2:
        idxs += einsum_ellipsis_idxs(idxs_map)
        idxs += [ einsum_index(letter) for letter in lst[1] ]
    assert [] == [ idx for idx in idxs if idx not in idxs_map ]

    # validate idxs match shape
    assert len(idxs) == len(shape)
    for x in range(len(idxs)):
        idx = idxs[x]
        n = idxs_map[idx]
        assert n == shape[x]

    return EinsumOutputSpec(shape, idxs)

def einsum_input(formula: str, shape: Shape) -> EinsumInputSpec:
    lst = [ s.replace(" ", "") for s in formula.split("...") ]
    if len(lst) == 1:
        assert len(lst[0]) == len(shape), \
            f"# indices in '{formula}' != length of shape {list(shape)}"
        lst.append("") # treat this case the same as an empty ellipsis at end
    else:
        assert len(lst) == 2, f"multiple ellipses in '{formula}'"
        assert len(lst[0]) + len(lst[1]) <= len(shape), \
            f"# indices in '{formula}' > length of shape {list(shape)}"
    leading_idxs = einsum_idxs(lst[0])
    trailing_idxs = einsum_idxs(lst[1])
    return EinsumInputSpec(shape, leading_idxs, trailing_idxs)

def einsum_extend_idxs_map(idxs_map: IdxsMap, idx: int, n: int) -> IdxsMap:
    old = idxs_map.get(idx)
    if old == None or old == 1:
        idxs_map[idx] = n
    else:
        assert n == 1 or n == old, f"cannot unify magnitudes {old}, {n}"
    return idxs_map

def einsum_idxs_map(inputs: List[EinsumInputSpec]) -> IdxsMap:
    idxs_map: IdxsMap = {}

    for spec in inputs:
        # process leading indices
        leading = spec.leading_idxs
        for x in range(len(leading)):
            idx = leading[x]
            n = spec.shape[x]
            einsum_extend_idxs_map(idxs_map, idx, n)
        # process trailing indices
        trailing = spec.trailing_idxs
        offset = len(spec.shape) - len(trailing)
        for x in range(len(trailing)):
            idx = trailing[x]
            n = spec.shape[offset + x]
            einsum_extend_idxs_map(idxs_map, idx, n)
        # process ellipsis
        eshape = spec.ellipsis_shape()
        for x in range(len(eshape)):
            idx = -1 - x
            einsum_extend_idxs_map(idxs_map, idx, eshape[idx])

    return idxs_map

def einsum_spec(equation: str, input_shapes: List[Shape], output_shape: Shape) -> EinsumSpec:
    io = equation.split("->")
    assert 1 <= len(io) <= 2, f"multiple arrows in '{equation}'"
    output_formula = io[1] if len(io) == 2 else None
    input_formulas = io[0].split(",")
    assert len(input_formulas) == len(input_shapes), "# equation inputs != # input shapes"
    inputs = [ einsum_input(*p) for p in zip(input_formulas, input_shapes) ]
    idxs_map = einsum_idxs_map(inputs)
    output = einsum_output(idxs_map, inputs, output_formula, output_shape)
    return EinsumSpec(idxs_map, inputs, output)

def einsum_input_idxs(input_spec: EinsumInputSpec):
    count = len(input_spec.leading_idxs) + len(input_spec.trailing_idxs)
    assert count <= len(input_spec.shape)
    ellipsis = list(range(count - len(input_spec.shape), 0))
    idxs = input_spec.leading_idxs + ellipsis + input_spec.trailing_idxs
    assert len(idxs) == len(input_spec.shape)
    return idxs


def einsum_execute(spec: EinsumSpec, tensors: List[Tensor]) -> Tensor:
    assert len(spec.inputs) == len(tensors)
    for input_spec, tensor in zip(spec.inputs, tensors):
        assert input_spec.shape == tensor.shape

    in_idxs = list(map(einsum_input_idxs, spec.inputs))
    out_idxs = spec.output.idxs
    in_only_idxs = list(set(spec.idxs_map).difference(out_idxs))
    def fn(*opos) -> float:
        assert len(opos) == len(out_idxs)
        pos_map = dict(zip(out_idxs, opos))

        def recurse(remaining_idxs: Idxs) -> float:
            if len(remaining_idxs) == 0:
                prod = 1.
                for idxs, tensor in zip(in_idxs, tensors):
                    pos = [ pos_map[idx] for idx in idxs ]
                    prod *= tensor.item(*pos)
                return prod
            else:
                head = remaining_idxs[0]
                tail = remaining_idxs[1:]
                acc = 0.
                for p in range(spec.idxs_map[head]):
                    pos_map[head] = p
                    acc += recurse(tail)
                return acc

        return recurse(in_only_idxs)

    return einsum_tensor_frompos(fn, spec.output.shape)



def einsum_test():
    print("einsum_test() start")

    def asserts(fn, *args):
        try:
            fn(*args)
            assert False, "didn't assert"
        except AssertionError as msg:
            assert str(msg) != "didn't assert"

    EIS = EinsumInputSpec
    EOS = EinsumOutputSpec
    ES = EinsumSpec

    assert "B" == einsum_letter(1)
    asserts(lambda: einsum_letter(52))

    assert 1 == einsum_index("B")
    asserts(lambda: einsum_index("-"))

    assert [0,26,51] == einsum_idxs("Aaz")
    assert [] == einsum_idxs("")

    assert None == einsum_find_duplicate("")
    assert None == einsum_find_duplicate("Unique")
    assert "U" == einsum_find_duplicate("UNIQUE")
    assert "x" == einsum_find_duplicate("xx")

    assert [] == einsum_ellipsis_idxs({})
    assert [] == einsum_ellipsis_idxs({34:2, 35:3})
    assert [-2, -1] == einsum_ellipsis_idxs({34:2, -1:4, -2:5})

    assert {34:0} == einsum_extend_idxs_map({}, 34, 0)
    assert {34:1} == einsum_extend_idxs_map({}, 34, 1)
    assert {34:2} == einsum_extend_idxs_map({}, 34, 2)
    assert {34:1} == einsum_extend_idxs_map({34:1}, 34, 1)
    assert {34:2} == einsum_extend_idxs_map({34:1}, 34, 2)
    assert {34:2} == einsum_extend_idxs_map({34:2}, 34, 1)
    assert {34:2} == einsum_extend_idxs_map({34:2}, 34, 2)
    assert {34:2, 35:3} == einsum_extend_idxs_map({34:2}, 35, 3)
    asserts(lambda: einsum_extend_idxs_map({34:2}, 34, 0))

    assert EIS((),[],[]) == einsum_input("", ())
    assert EIS((),[],[]) == einsum_input(" ", ())
    asserts(lambda: einsum_input("", (2,)))
    asserts(lambda: einsum_input("i", ()))
    assert EIS((),[],[]) == einsum_input("...", ())
    assert EIS((),[],[]) == einsum_input(" ... ", ())
    asserts(lambda: einsum_input(". ..", ()))
    asserts(lambda: einsum_input("..", ()))
    assert EIS((2,),[],[]) == einsum_input("...", (2,))
    assert EIS((2,),[34],[]) == einsum_input("i", (2,))
    assert EIS((2,),[34],[]) == einsum_input(" i ", (2,))
    assert EIS((2,),[34],[]) == einsum_input("i...", (2,))
    assert EIS((2,),[34],[]) == einsum_input("i ...", (2,))
    assert EIS((2,),[],[34]) == einsum_input("...i", (2,))
    assert EIS((2,3),[34,35],[]) == einsum_input("ij", (2,3))
    assert EIS((2,3),[34,35],[]) == einsum_input("i j", (2,3))
    assert EIS((2,3,4),[34],[35,36]) == einsum_input("i...jk", (2,3,4))
    assert EIS((2,3,4),[34],[8])== einsum_input("i...I", (2,3,4))
    assert EIS((2,3,4),[34],[]) == einsum_input("i...", (2,3,4))

    eI1 = einsum_input("I",(1,))
    eI2 = einsum_input("I",(2,))
    eI3 = einsum_input("I",(3,))
    eI_245 = einsum_input("I...",(2,4,5))
    e_J456 = einsum_input("...J",(4,5,6))
    e_315 = einsum_input("...",(3,1,5))
    e_ = einsum_input("...",())
    e_2 = einsum_input("...",(2,))
    assert {} == einsum_idxs_map([])
    assert {8:2} == einsum_idxs_map([eI1,eI2])
    assert {8:2} == einsum_idxs_map([eI1,eI2,eI2])
    asserts(lambda: einsum_idxs_map([eI2,eI3]))
    assert {-3:3,-2:4,-1:5,8:2,9:6} == einsum_idxs_map([eI_245,e_J456,e_315,e_])
    asserts(lambda: einsum_idxs_map([e_315,e_2]))

    assert "..." == einsum_infer_output_formula({}, [])
    assert "...I" == einsum_infer_output_formula({8:2}, [EIS((2,),[8],[])])
    assert "...I" == einsum_infer_output_formula({-1:3,8:2}, [EIS((2,3),[8],[])])
    assert "..." == einsum_infer_output_formula({8:2}, [EIS((2,2),[8,8],[])])
    assert "..." == einsum_infer_output_formula({8:2}, [EIS((2,),[8],[]), EIS((2,),[8],[])])

    assert EOS((),[]) == einsum_output({}, [], "", ())
    assert EOS((),[]) == einsum_output({}, [], " ", ())
    assert EOS((),[]) == einsum_output({}, [], "...", ())
    asserts(lambda: einsum_output({}, [], ". ..", ()))
    asserts(lambda: einsum_output({}, [], "..", ()))
    asserts(lambda: einsum_output({}, [], "......", ()))
    assert EOS((),[]) == einsum_output({8:2}, [EIS((2,),[8],[])], "", ())
    assert EOS((2,),[8]) == einsum_output({8:2}, [EIS((2,),[8],[])], "I", (2,))
    assert EOS((2,),[8]) == einsum_output({8:2}, [EIS((2,),[8],[])], None, (2,))
    assert EOS((2,4),[8,10]) == \
        einsum_output( \
            {8:2,9:3,10:4}, [EIS((2,3),[8,9],[]), EIS((3,4),[9,10],[])], None, (2,4))
    assert EOS((2,4),[8,10]) == \
        einsum_output( \
            {8:2,9:3,10:4}, [EIS((2,3),[8,9],[]), EIS((3,4),[9,10],[])], "IK", (2,4))
    asserts(lambda: einsum_output({}, [], "......", ()))
    asserts(lambda: einsum_output({8:2}, [EIS((2,),[8],[])], "II", (2,2)))

    assert ES({},[EIS((),[],[])],EOS((),[])) == einsum_spec("", [()], ())
    assert ES({},[EIS((),[],[])],EOS((),[])) == einsum_spec("->", [()], ())
    assert ES({},[EIS((),[],[])],EOS((),[])) == einsum_spec("...->...", [()], ())
    asserts(lambda: einsum_spec("->->", [()], ()))

    eqT = np.array_equal
    t_0 = einsum_tensor(0.)
    t0_0 = einsum_tensor([])
    t1_0 = einsum_tensor([0.])
    assert eqT(t_0, einsum_execute(ES({},[EIS((),[],[])],EOS((),[])),[t_0]))
    assert eqT(t_0, einsum_execute(ES({8:0},[EIS((0,),[8],[])],EOS((),[])),[t0_0]))
    assert eqT(t0_0, einsum_execute(ES({8:0},[EIS((),[],[])],EOS((0,),[8])),[t_0]))
    assert eqT(t_0, einsum_execute(ES({8:1},[EIS((1,),[8],[])],EOS((),[])),[t1_0]))
    assert eqT(t1_0, einsum_execute(ES({8:1},[EIS((1,),[8],[])],EOS((1,),[8])),[t1_0]))

    print("einsum_test() end")

if __name__ == "__main__":
   einsum_test()


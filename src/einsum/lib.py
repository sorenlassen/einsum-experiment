from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class EinsumOutputSpec:

    # tuple of magnitudes (non-negative integers)
    shape: Tuple[int,...]

    # indices is a list of integers in the output with [0,52) referring to
    # regular indices (representing upper and lower case letters) and
    # consecutive negative indices representing ellipsis
    indices: List[int]

@dataclass
class EinsumInputSpec:

    # tuple of magnitudes
    shape: Tuple[int,...]

    # leading_indices is a list of integers in [0,52), namely all indices at the
    # beginning of the input up to the ellipsis, if any, otherwise to the end of
    # the input
    leading_indices: List[int]

    # trailing_indices is a list of integers in [0,52) in the input after any
    # ellipsis
    trailing_indices: List[int]

    def ellipsis_shape(self):
        begin = len(self.leading_indices)
        end = len(self.shape) - len(self.trailing_indices)
        return self.shape[begin:end]

def einsum_indices(formula):
    return [ einsum_index(letter) for letter in formula ]

EINSUM_LETTERS_UPPER = 26 # oounts upper case letters A-Z
EINSUM_LETTERS_LOWER = 26 # counts lower case letters a-z
EINSUM_LETTERS = EINSUM_LETTERS_UPPER + EINSUM_LETTERS_LOWER

def einsum_index(letter):
    if "A" <= letter and letter <= "Z":
        x = ord(letter) - ord("A")
        assert 0 <= x < EINSUM_LETTERS_UPPER
        return x
    elif "a" <= letter and letter <= "z":
        x = ord(letter) - ord("a")
        assert 0 <= x < EINSUM_LETTERS_LOWER
        return EINSUM_LETTERS_UPPER + x
    assert False, f"index '{letter}' ({ord(letter)}) is not a letter"

def einsum_letter(index):
    assert 0 <= index < EINSUM_LETTERS
    if index < EINSUM_LETTERS_UPPER:
        return chr(ord("A") + index)
    else:
        return chr(ord("a") + index - EINSUM_LETTERS_UPPER)

def einsum_infer_output_formula(indices_dict, input_specs):
    # count occurrences of letter indices in inputs
    indices_count = [0] * EINSUM_LETTERS
    for spec in input_specs:
        for indices in (spec.leading_indices, spec.trailing_indices):
            for index in indices:
                indices_count[index] += 1
    formula = "..."
    for index in indices_dict:
        if indices_count[index] == 1:
            formula += einsum_letter(index)
    return formula

def einsum_find_duplicate(letters):
    if len(letters) > 1:
        s = sorted(letters)
        for x in range(len(s) - 1):
            if s[x] == s[x + 1]:
                return s[x]
    return None

def einsum_ellipsis_indices(indices_dict):
    return sorted([ index for index in indices_dict if index < 0 ])

def einsum_output_spec(indices_dict, input_specs, formula, shape):
    lst = None
    if formula == None:
        formula = einsum_infer_output_formula(indices_dict, input_specs)
        assert formula.startswith("...")
        trailing = formula[len("..."):]
        assert trailing.count(".") == trailing.count(" ") == 0
        lst = ["", trailing]
    else:
        lst = [ s.replace(" ", "") for s in formula.split("...") ]
        assert 1 <= len(lst) <= 2, f"multiple ellipses in '{formula}'"
        duplicate = einsum_find_duplicate("".join(lst))
        assert duplicate == None, f"duplicate index {duplicate} in '{formula}'"

    indices = [ einsum_index(letter) for letter in lst[0] ]
    if len(lst) == 2:
        indices += einsum_ellipsis_indices(indices_dict)
        indices += [ einsum_index(letter) for letter in lst[1] ]
    assert [] == [ index for index in indices if index not in indices_dict ]

    # validate indices match shape
    assert len(indices) == len(shape)
    for x in range(len(indices)):
        index = indices[x]
        n = indices_dict[index]
        assert n == shape[x]

    return EinsumOutputSpec(shape, indices)

def einsum_input_spec(formula, shape):
    lst = [ s.replace(" ", "") for s in formula.split("...") ]
    if len(lst) == 1:
        assert len(lst[0]) == len(shape), \
            f"# indices in '{formula}' != length of shape {list(shape)}"
        lst.append("") # treat this case the same as an empty ellipsis at end
    else:
        assert len(lst) == 2, f"multiple ellipses in '{formula}'"
        assert len(lst[0]) + len(lst[1]) <= len(shape), \
            f"# indices in '{formula}' > length of shape {list(shape)}"
    leading_indices = einsum_indices(lst[0])
    trailing_indices = einsum_indices(lst[1])
    return EinsumInputSpec(shape, leading_indices, trailing_indices)

def einsum_extend_indices_dict(indices_dict, index, n):
    old = indices_dict.get(index)
    if old == None or old == 1:
        indices_dict[index] = n
    else:
        assert n == 1 or n == old, \
            f"cannot unify magnitudes {old}, {n}"
    return indices_dict

def einsum_indices_dict(input_specs):
    indices_dict = {}

    for spec in input_specs:
        # process leading indices
        leading = spec.leading_indices
        for x in range(len(leading)):
            index = leading[x]
            n = spec.shape[x]
            einsum_extend_indices_dict(indices_dict, index, n)
        # process trailing indices
        trailing = spec.trailing_indices
        offset = len(spec.shape) - len(trailing)
        for x in range(len(trailing)):
            index = trailing[x]
            n = spec.shape[offset + x]
            einsum_extend_indices_dict(indices_dict, index, n)
        # process ellipsis
        eshape = spec.ellipsis_shape()
        for x in range(len(eshape)):
            index = -1 - x
            einsum_extend_indices_dict(indices_dict, index, eshape[index])

    return indices_dict

def einsum_spec(equation, input_shapes, output_shape):
    io = equation.split("->")
    assert 1 <= len(io) <= 2, f"multiple arrows in '{equation}'"
    output_formula = io[1] if len(io) == 2 else None
    input_formulas = io[0].split(",")
    assert len(input_formulas) == len(input_shapes), "# equation inputs != # input shapes"
    input_specs = [ einsum_input_spec(*p) for p in zip(input_formulas, input_shapes) ]
    indices_dict = einsum_indices_dict(input_specs)
    output_spec = einsum_output_spec(indices_dict, input_specs, output_formula, output_shape)
    return EinsumSpec(indices_dict, input_specs, output_spec)

@dataclass
class EinsumSpec:

    # indices_dict maps indices to magnitudes, where an index
    #   is an integer with [0,52) ranging over upper and lower case letters
    #   (e.g. A and Z are 0 and 25, a and z are 26 and 51) and with
    #   negative integers ranging over the indices of the ellipsis, if any
    #   (-1 is the last index of the ellipsis, -2 is the second last, etc)
    indices_dict: Dict[int,int]
    
    # input_specs is a non-empty list of instances of EinsumInputSpec
    input_specs: List[int]

    # output_spec is an instance of EinsumOutputSpec
    output_spec: EinsumOutputSpec


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

    assert [0,26,51] == einsum_indices("Aaz")
    assert [] == einsum_indices("")

    assert None == einsum_find_duplicate("")
    assert None == einsum_find_duplicate("Unique")
    assert "U" == einsum_find_duplicate("UNIQUE")
    assert "x" == einsum_find_duplicate("xx")

    assert [] == einsum_ellipsis_indices({})
    assert [] == einsum_ellipsis_indices({34:2, 35:3})
    assert [-2, -1] == einsum_ellipsis_indices({34:2, -1:4, -2:5})

    assert {34:0} == einsum_extend_indices_dict({}, 34, 0)
    assert {34:1} == einsum_extend_indices_dict({}, 34, 1)
    assert {34:2} == einsum_extend_indices_dict({}, 34, 2)
    assert {34:1} == einsum_extend_indices_dict({34:1}, 34, 1)
    assert {34:2} == einsum_extend_indices_dict({34:1}, 34, 2)
    assert {34:2} == einsum_extend_indices_dict({34:2}, 34, 1)
    assert {34:2} == einsum_extend_indices_dict({34:2}, 34, 2)
    assert {34:2, 35:3} == einsum_extend_indices_dict({34:2}, 35, 3)
    asserts(lambda: einsum_extend_indices_dict({34:2}, 34, 0))

    assert EIS((),[],[]) == einsum_input_spec("", ())
    assert EIS((),[],[]) == einsum_input_spec(" ", ())
    asserts(lambda: einsum_input_spec("", (2,)))
    asserts(lambda: einsum_input_spec("i", ()))
    assert EIS((),[],[]) == einsum_input_spec("...", ())
    assert EIS((),[],[]) == einsum_input_spec(" ... ", ())
    asserts(lambda: einsum_input_spec(". ..", ()))
    asserts(lambda: einsum_input_spec("..", ()))
    assert EIS((2,),[],[]) == einsum_input_spec("...", (2,))
    assert EIS((2,),[34],[]) == einsum_input_spec("i", (2,))
    assert EIS((2,),[34],[]) == einsum_input_spec(" i ", (2,))
    assert EIS((2,),[34],[]) == einsum_input_spec("i...", (2,))
    assert EIS((2,),[34],[]) == einsum_input_spec("i ...", (2,))
    assert EIS((2,),[],[34]) == einsum_input_spec("...i", (2,))
    assert EIS((2,3),[34,35],[]) == einsum_input_spec("ij", (2,3))
    assert EIS((2,3),[34,35],[]) == einsum_input_spec("i j", (2,3))
    assert EIS((2,3,4),[34],[35,36]) == einsum_input_spec("i...jk", (2,3,4))
    assert EIS((2,3,4),[34],[8])== einsum_input_spec("i...I", (2,3,4))
    assert EIS((2,3,4),[34],[]) == einsum_input_spec("i...", (2,3,4))

    eI1 = einsum_input_spec("I",(1,))
    eI2 = einsum_input_spec("I",(2,))
    eI3 = einsum_input_spec("I",(3,))
    eI_245 = einsum_input_spec("I...",(2,4,5))
    e_J456 = einsum_input_spec("...J",(4,5,6))
    e_315 = einsum_input_spec("...",(3,1,5))
    e_ = einsum_input_spec("...",())
    e_2 = einsum_input_spec("...",(2,))
    assert {} == einsum_indices_dict([])
    assert {8:2} == einsum_indices_dict([eI1,eI2])
    assert {8:2} == einsum_indices_dict([eI1,eI2,eI2])
    asserts(lambda: einsum_indices_dict([eI2,eI3]))
    assert {-3:3,-2:4,-1:5,8:2,9:6} == einsum_indices_dict([eI_245,e_J456,e_315,e_])
    asserts(lambda: einsum_indices_dict([e_315,e_2]))

    assert "..." == einsum_infer_output_formula({}, [])
    assert "...I" == einsum_infer_output_formula({8:2}, [EIS((2,),[8],[])])
    assert "...I" == einsum_infer_output_formula({-1:3,8:2}, [EIS((2,3),[8],[])])
    assert "..." == einsum_infer_output_formula({8:2}, [EIS((2,2),[8,8],[])])
    assert "..." == einsum_infer_output_formula({8:2}, [EIS((2,),[8],[]), EIS((2,),[8],[])])

    assert EOS((),[]) == einsum_output_spec({}, [], "", ())
    assert EOS((),[]) == einsum_output_spec({}, [], " ", ())
    assert EOS((),[]) == einsum_output_spec({}, [], "...", ())
    asserts(lambda: einsum_output_spec({}, [], ". ..", ()))
    asserts(lambda: einsum_output_spec({}, [], "..", ()))
    asserts(lambda: einsum_output_spec({}, [], "......", ()))
    assert EOS((),[]) == einsum_output_spec({8:2}, [EIS((2,),[8],[])], "", ())
    assert EOS((2,),[8]) == einsum_output_spec({8:2}, [EIS((2,),[8],[])], "I", (2,))
    assert EOS((2,),[8]) == einsum_output_spec({8:2}, [EIS((2,),[8],[])], None, (2,))
    assert EOS((2,4),[8,10]) == \
        einsum_output_spec( \
            {8:2,9:3,10:4}, [EIS((2,3),[8,9],[]), EIS((3,4),[9,10],[])], None, (2,4))
    assert EOS((2,4),[8,10]) == \
        einsum_output_spec( \
            {8:2,9:3,10:4}, [EIS((2,3),[8,9],[]), EIS((3,4),[9,10],[])], "IK", (2,4))
    asserts(lambda: einsum_output_spec({}, [], "......", ()))
    asserts(lambda: einsum_output_spec({8:2}, [EIS((2,),[8],[])], "II", (2,2)))

    assert ES({},[EIS((),[],[])],EOS((),[])) == einsum_spec("", [()], ())
    assert ES({},[EIS((),[],[])],EOS((),[])) == einsum_spec("->", [()], ())
    assert ES({},[EIS((),[],[])],EOS((),[])) == einsum_spec("...->...", [()], ())
    asserts(lambda: einsum_spec("->->", [()], ()))

    print("einsum_test() end")

if __name__ == "__main__":
   einsum_test()


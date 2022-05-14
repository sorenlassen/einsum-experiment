import unittest
from typing import Any

from einsum import *

class TestCase(unittest.TestCase):
    def test_einsum(self):

        def fails(fn, *args, **kwargs):
            self.assertRaises(AssertionError, fn, *args, **kwargs)

        def expect(expected, result: Any = True, msg: Optional[str] = None):
            if isinstance(expected, list):
                expected = tuple(expected)
            if isinstance(result, list):
                result = tuple(result)
            self.assertEqual(expected, result, msg=msg)

        EIS = EinsumInputSpec
        EOS = EinsumOutputSpec
        ES = EinsumSpec

        expect("B", einsum_letter(1))
        fails(lambda: einsum_letter(52))

        expect(1, einsum_index("B"))
        fails(lambda: einsum_index("-"))

        expect([0,26,51], einsum_idxs("Aaz"))
        expect([], einsum_idxs(""))

        expect(None, einsum_find_duplicate(""))
        expect(None, einsum_find_duplicate("Unique"))
        expect("U", einsum_find_duplicate("UNIQUE"))
        expect("x", einsum_find_duplicate("xx"))

        expect([], einsum_ellipsis_idxs({}))
        expect([], einsum_ellipsis_idxs({34:2, 35:3}))
        expect([-2, -1], einsum_ellipsis_idxs({34:2, -1:4, -2:5}))

        expect({34:0}, einsum_extend_idxs_map({}, 34, 0))
        expect({34:1}, einsum_extend_idxs_map({}, 34, 1))
        expect({34:2}, einsum_extend_idxs_map({}, 34, 2))
        expect({34:1}, einsum_extend_idxs_map({34:1}, 34, 1))
        expect({34:2}, einsum_extend_idxs_map({34:1}, 34, 2))
        expect({34:2}, einsum_extend_idxs_map({34:2}, 34, 1))
        expect({34:2}, einsum_extend_idxs_map({34:2}, 34, 2))
        expect({34:2, 35:3}, einsum_extend_idxs_map({34:2}, 35, 3))
        fails(lambda: einsum_extend_idxs_map({34:2}, 34, 0))

        expect(EIS((),[],[]), einsum_input("", ()))
        expect(EIS((),[],[]), einsum_input(" ", ()))
        fails(lambda: einsum_input("", (2,)))
        fails(lambda: einsum_input("i", ()))
        expect(EIS((),[],[]), einsum_input("...", ()))
        expect(EIS((),[],[]), einsum_input(" ... ", ()))
        fails(lambda: einsum_input(". ..", ()))
        fails(lambda: einsum_input("..", ()))
        expect(EIS((2,),[],[]), einsum_input("...", (2,)))
        expect(EIS((2,),[34],[]), einsum_input("i", (2,)))
        expect(EIS((2,),[34],[]), einsum_input(" i ", (2,)))
        expect(EIS((2,),[34],[]), einsum_input("i...", (2,)))
        expect(EIS((2,),[34],[]), einsum_input("i ...", (2,)))
        expect(EIS((2,),[],[34]), einsum_input("...i", (2,)))
        expect(EIS((2,3),[34,35],[]), einsum_input("ij", (2,3)))
        expect(EIS((2,3),[34,35],[]), einsum_input("i j", (2,3)))
        expect(EIS((2,3,4),[34],[35,36]), einsum_input("i...jk", (2,3,4)))
        expect(EIS((2,3,4),[34],[8]), einsum_input("i...I", (2,3,4)))
        expect(EIS((2,3,4),[34],[]), einsum_input("i...", (2,3,4)))

        eI1 = einsum_input("I",(1,))
        eI2 = einsum_input("I",(2,))
        eI3 = einsum_input("I",(3,))
        eI_245 = einsum_input("I...",(2,4,5))
        e_J456 = einsum_input("...J",(4,5,6))
        e_315 = einsum_input("...",(3,1,5))
        e_ = einsum_input("...",())
        e_2 = einsum_input("...",(2,))
        expect({}, einsum_idxs_map([]))
        expect({8:2}, einsum_idxs_map([eI1,eI2]))
        expect({8:2}, einsum_idxs_map([eI1,eI2,eI2]))
        fails(lambda: einsum_idxs_map([eI2,eI3]))
        expect({-3:3,-2:4,-1:5,8:2,9:6}, einsum_idxs_map([eI_245,e_J456,e_315,e_]))
        fails(lambda: einsum_idxs_map([e_315,e_2]))

        expect("...", einsum_infer_output_formula({}, []))
        expect("...I", einsum_infer_output_formula({8:2}, [EIS((2,),[8],[])]))
        expect("...I", einsum_infer_output_formula({-1:3,8:2}, [EIS((2,3),[8],[])]))
        expect("...", einsum_infer_output_formula({8:2}, [EIS((2,2),[8,8],[])]))
        expect("...", einsum_infer_output_formula({8:2}, [EIS((2,),[8],[]), EIS((2,),[8],[])]))

        expect(EOS((),[]), einsum_output({}, [], ""))
        expect(EOS((),[]), einsum_output({}, [], " "))
        expect(EOS((),[]), einsum_output({}, [], "..."))
        fails(lambda: einsum_output({}, [], ". .."))
        fails(lambda: einsum_output({}, [], ".."))
        fails(lambda: einsum_output({}, [], "......"))
        expect(EOS((),[]), einsum_output({8:2}, [EIS((2,),[8],[])], ""))
        expect(EOS((2,),[8]), einsum_output({8:2}, [EIS((2,),[8],[])], "I"))
        expect(EOS((2,),[8]), einsum_output({8:2}, [EIS((2,),[8],[])], None))
        expect(EOS((2,4),[8,10]),
            einsum_output(
                {8:2,9:3,10:4}, [EIS((2,3),[8,9],[]), EIS((3,4),[9,10],[])], None))
        expect(EOS((2,4),[8,10]),
            einsum_output(
                {8:2,9:3,10:4}, [EIS((2,3),[8,9],[]), EIS((3,4),[9,10],[])], "IK"))
        fails(lambda: einsum_output({}, [], "......"))
        fails(lambda: einsum_output({8:2}, [EIS((2,),[8],[])], "II"))

        expect(ES([EIS((),[],[])],EOS((),[])), einsum_spec("", [()]))
        expect(ES([EIS((),[],[])],EOS((),[])), einsum_spec("->", [()]))
        expect(ES([EIS((),[],[])],EOS((),[])), einsum_spec("...->...", [()]))
        fails(lambda: einsum_spec("->->", [()]))

        eqT = np.array_equal
        t_0 = einsum_tensor(0.)
        t0_0 = einsum_tensor([])
        t1_0 = einsum_tensor([0.])
        expect(eqT(t_0, einsum_execute(ES([EIS((),[],[])],EOS((),[])),[t_0])))
        expect(eqT(t_0, einsum_execute(ES([EIS((0,),[8],[])],EOS((),[])),[t0_0])))
        expect(eqT(t0_0, einsum_execute(ES([EIS((0,),[8],[])],EOS((0,),[8])),[t0_0])))
        expect(eqT(t_0, einsum_execute(ES([EIS((1,),[8],[])],EOS((),[])),[t1_0])))
        expect(eqT(t1_0, einsum_execute(ES([EIS((1,),[8],[])],EOS((1,),[8])),[t1_0])))

        expect(eqT(t_0, einsum("->",[t_0])))
        expect(eqT(t_0, einsum("I->",[t0_0])))
        expect(eqT(t0_0, einsum("I->I",[t0_0])))
        expect(eqT(t_0, einsum("I->",[t1_0])))
        expect(eqT(t1_0, einsum("I",[t1_0])))

if __name__ == '__main__':
    unittest.main()


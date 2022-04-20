import unittest
from typing import Any, Optional

from einsum import *

class TestCase(unittest.TestCase):
    def test_einsum(self):

        def fails(fn, *args, **kwargs):
            self.assertRaises(AssertionError, fn, *args, **kwargs)

        def eq(expected, result, msg: Optional[str] = None):
            if isinstance(expected, Tensor):
                self.assertTrue(np.array_equal(expected, result), msg=msg)
            else:
                self.assertEqual(expected, result, msg=msg)

        EIS = EinsumInputSpec
        EOS = EinsumOutputSpec
        ES = EinsumSpec

        eq("B", einsum_letter(1))
        fails(lambda: einsum_letter(52))

        eq(1, einsum_index("B"))
        fails(lambda: einsum_index("-"))

        eq([0,26,51], einsum_idxs("Aaz"))
        eq([], einsum_idxs(""))

        eq(None, einsum_find_duplicate(""))
        eq(None, einsum_find_duplicate("Unique"))
        eq("U", einsum_find_duplicate("UNIQUE"))
        eq("x", einsum_find_duplicate("xx"))

        eq([], einsum_ellipsis_idxs({}))
        eq([], einsum_ellipsis_idxs({34:2, 35:3}))
        eq([-2, -1], einsum_ellipsis_idxs({34:2, -1:4, -2:5}))

        eq({34:0}, einsum_extend_idxs_map({}, 34, 0))
        eq({34:1}, einsum_extend_idxs_map({}, 34, 1))
        eq({34:2}, einsum_extend_idxs_map({}, 34, 2))
        eq({34:1}, einsum_extend_idxs_map({34:1}, 34, 1))
        eq({34:2}, einsum_extend_idxs_map({34:1}, 34, 2))
        eq({34:2}, einsum_extend_idxs_map({34:2}, 34, 1))
        eq({34:2}, einsum_extend_idxs_map({34:2}, 34, 2))
        eq({34:2, 35:3}, einsum_extend_idxs_map({34:2}, 35, 3))
        fails(lambda: einsum_extend_idxs_map({34:2}, 34, 0))

        eq(EIS((),[]), einsum_input("", ()))
        eq(EIS((),[]), einsum_input(" ", ()))
        fails(lambda: einsum_input("", (2,)))
        fails(lambda: einsum_input("i", ()))
        eq(EIS((),[]), einsum_input("...", ()))
        eq(EIS((),[]), einsum_input(" ... ", ()))
        fails(lambda: einsum_input(". ..", ()))
        fails(lambda: einsum_input("..", ()))
        eq(EIS((2,),[-1]), einsum_input("...", (2,)))
        eq(EIS((2,),[34]), einsum_input("i", (2,)))
        eq(EIS((2,),[34]), einsum_input(" i ", (2,)))
        eq(EIS((2,),[34]), einsum_input("i...", (2,)))
        eq(EIS((2,),[34]), einsum_input("i ...", (2,)))
        eq(EIS((2,),[34]), einsum_input("...i", (2,)))
        eq(EIS((2,3),[34,35]), einsum_input("ij", (2,3)))
        eq(EIS((2,3),[34,35]), einsum_input("i j", (2,3)))
        eq(EIS((2,3,4),[34,35,36]), einsum_input("i...jk", (2,3,4)))
        eq(EIS((2,3,4),[34,-1,8]), einsum_input("i...I", (2,3,4)))
        eq(EIS((2,3,4),[34,-2,-1]), einsum_input("i...", (2,3,4)))

        eI1 = einsum_input("I",(1,))
        eI2 = einsum_input("I",(2,))
        eI3 = einsum_input("I",(3,))
        eI_245 = einsum_input("I...",(2,4,5))
        e_J456 = einsum_input("...J",(4,5,6))
        e_315 = einsum_input("...",(3,1,5))
        e_ = einsum_input("...",())
        e_2 = einsum_input("...",(2,))
        eq({}, einsum_idxs_map([]))
        eq({8:2}, einsum_idxs_map([eI1,eI2]))
        eq({8:2}, einsum_idxs_map([eI1,eI2,eI2]))
        fails(lambda: einsum_idxs_map([eI2,eI3]))
        eq({-3:3,-2:4,-1:5,8:2,9:6}, einsum_idxs_map([eI_245,e_J456,e_315,e_]))
        fails(lambda: einsum_idxs_map([e_315,e_2]))

        eq("...", einsum_infer_output_formula({}, []))
        eq("...I", einsum_infer_output_formula({8:2}, [EIS((2,),[8])]))
        eq("...I", einsum_infer_output_formula({-1:3,8:2}, [EIS((2,3),[8,-1])]))
        eq("...", einsum_infer_output_formula({8:2}, [EIS((2,2),[8,8])]))
        eq("...", einsum_infer_output_formula({8:2}, [EIS((2,),[8]), EIS((2,),[8])]))

        eq(EOS((),[]), einsum_output({}, [], ""))
        eq(EOS((),[]), einsum_output({}, [], " "))
        eq(EOS((),[]), einsum_output({}, [], "..."))
        fails(lambda: einsum_output({}, [], ". .."))
        fails(lambda: einsum_output({}, [], ".."))
        fails(lambda: einsum_output({}, [], "......"))
        eq(EOS((),[]), einsum_output({8:2}, [EIS((2,),[8])], ""))
        eq(EOS((2,),[8]), einsum_output({8:2}, [EIS((2,),[8])], "I"))
        eq(EOS((2,),[8]), einsum_output({8:2}, [EIS((2,),[8])], None))
        eq(EOS((2,4),[8,10]),
            einsum_output(
                {8:2,9:3,10:4}, [EIS((2,3),[8,9]), EIS((3,4),[9,10])], None))
        eq(EOS((2,4),[8,10]),
            einsum_output(
                {8:2,9:3,10:4}, [EIS((2,3),[8,9]), EIS((3,4),[9,10])], "IK"))
        fails(lambda: einsum_output({}, [], "......"))
        fails(lambda: einsum_output({8:2}, [EIS((2,),[8])], "II"))

        eq(ES({},[EIS((),[])],EOS((),[])), einsum_spec("", [()]))
        eq(ES({},[EIS((),[])],EOS((),[])), einsum_spec("->", [()]))
        eq(ES({},[EIS((),[])],EOS((),[])), einsum_spec("...->...", [()]))
        fails(lambda: einsum_spec("->->", [()]))

        t_0 = einsum_tensor(0.)
        t0_0 = einsum_tensor([])
        t1_0 = einsum_tensor([0.])
        eq(t_0, einsum_execute(ES({},[EIS((),[])],EOS((),[])),[t_0]))
        eq(t_0, einsum_execute(ES({8:0},[EIS((0,),[8])],EOS((),[])),[t0_0]))
        eq(t0_0, einsum_execute(ES({8:0},[EIS((0,),[8])],EOS((0,),[8])),[t0_0]))
        eq(t_0, einsum_execute(ES({8:1},[EIS((1,),[8])],EOS((),[])),[t1_0]))
        eq(t1_0, einsum_execute(ES({8:1},[EIS((1,),[8])],EOS((1,),[8])),[t1_0]))

        eq(t_0, einsum("->",[t_0]))
        eq(t_0, einsum("I->",[t0_0]))
        eq(t0_0, einsum("I->I",[t0_0]))
        eq(t_0, einsum("I->",[t1_0]))
        eq(t1_0, einsum("I",[t1_0]))

if __name__ == '__main__':
    unittest.main()


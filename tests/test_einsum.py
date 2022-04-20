import unittest
from typing import Any, Optional

from einsum import *

EIS = EinsumInputSpec
EOS = EinsumOutputSpec
ES = EinsumSpec

class TestCase(unittest.TestCase):

    def fails(self, fn, *args, **kwargs):
        self.assertRaises(AssertionError, fn, *args, **kwargs)

    def eq(self, expected, result, msg: Optional[str] = None):
        if isinstance(expected, Tensor):
            msg = f"expected={expected}, result={result}" if msg is None else msg
            self.assertTrue(np.array_equal(expected, result), msg=msg)
        else:
            self.assertEqual(expected, result, msg=msg)

    def test_einsum_letter(self):
        self.eq("B", einsum_letter(1))
        self.fails(lambda: einsum_letter(52))

    def test_einsum_index(self):
        self.eq(1, einsum_index("B"))
        self.fails(lambda: einsum_index("-"))

    def test_einsum_idxs(self):
        self.eq([0,26,51], einsum_idxs("Aaz"))
        self.eq([], einsum_idxs(""))

    def test_einsum_find_duplicate(self):
        self.eq(None, einsum_find_duplicate(""))
        self.eq(None, einsum_find_duplicate("Unique"))
        self.eq("U", einsum_find_duplicate("UNIQUE"))
        self.eq("x", einsum_find_duplicate("xx"))

    def test_einsum_ellipsis_idxs(self):
        self.eq([], einsum_ellipsis_idxs({}))
        self.eq([], einsum_ellipsis_idxs({34:2, 35:3}))
        self.eq([-2, -1], einsum_ellipsis_idxs({34:2, -1:4, -2:5}))

    def test_einsum_extend_idxs_map(self):
        self.eq({34:0}, einsum_extend_idxs_map({}, 34, 0))
        self.eq({34:1}, einsum_extend_idxs_map({}, 34, 1))
        self.eq({34:2}, einsum_extend_idxs_map({}, 34, 2))
        self.eq({34:1}, einsum_extend_idxs_map({34:1}, 34, 1))
        self.eq({34:2}, einsum_extend_idxs_map({34:1}, 34, 2))
        self.eq({34:2}, einsum_extend_idxs_map({34:2}, 34, 1))
        self.eq({34:2}, einsum_extend_idxs_map({34:2}, 34, 2))
        self.eq({34:2, 35:3}, einsum_extend_idxs_map({34:2}, 35, 3))
        self.fails(lambda: einsum_extend_idxs_map({34:2}, 34, 0))

    def test_einsum_input(self):
        self.eq(EIS((),[]), einsum_input("", ()))
        self.eq(EIS((),[]), einsum_input(" ", ()))
        self.fails(lambda: einsum_input("", (2,)))
        self.fails(lambda: einsum_input("i", ()))
        self.eq(EIS((),[]), einsum_input("...", ()))
        self.eq(EIS((),[]), einsum_input(" ... ", ()))
        self.fails(lambda: einsum_input(". ..", ()))
        self.fails(lambda: einsum_input("..", ()))
        self.eq(EIS((2,),[-1]), einsum_input("...", (2,)))
        self.eq(EIS((2,),[34]), einsum_input("i", (2,)))
        self.eq(EIS((2,),[34]), einsum_input(" i ", (2,)))
        self.eq(EIS((2,),[34]), einsum_input("i...", (2,)))
        self.eq(EIS((2,),[34]), einsum_input("i ...", (2,)))
        self.eq(EIS((2,),[34]), einsum_input("...i", (2,)))
        self.eq(EIS((2,3),[34,35]), einsum_input("ij", (2,3)))
        self.eq(EIS((2,3),[34,35]), einsum_input("i j", (2,3)))
        self.eq(EIS((2,3,4),[34,35,36]), einsum_input("i...jk", (2,3,4)))
        self.eq(EIS((2,3,4),[34,-1,8]), einsum_input("i...I", (2,3,4)))
        self.eq(EIS((2,3,4),[34,-2,-1]), einsum_input("i...", (2,3,4)))

    def test_einsum_input(self):
        eI1 = einsum_input("I",(1,))
        eI2 = einsum_input("I",(2,))
        eI3 = einsum_input("I",(3,))
        eI_245 = einsum_input("I...",(2,4,5))
        e_J456 = einsum_input("...J",(4,5,6))
        e_315 = einsum_input("...",(3,1,5))
        e_ = einsum_input("...",())
        e_2 = einsum_input("...",(2,))
        self.eq({}, einsum_idxs_map([]))
        self.eq({8:2}, einsum_idxs_map([eI1,eI2]))
        self.eq({8:2}, einsum_idxs_map([eI1,eI2,eI2]))
        self.fails(lambda: einsum_idxs_map([eI2,eI3]))
        self.eq({-3:3,-2:4,-1:5,8:2,9:6}, einsum_idxs_map([eI_245,e_J456,e_315,e_]))
        self.fails(lambda: einsum_idxs_map([e_315,e_2]))

    def test_einsum_infer_output_formula(self):
        self.eq("...", einsum_infer_output_formula({}, []))
        self.eq("...I", einsum_infer_output_formula({8:2}, [EIS((2,),[8])]))
        self.eq("...I", einsum_infer_output_formula({-1:3,8:2}, [EIS((2,3),[8,-1])]))
        self.eq("...", einsum_infer_output_formula({8:2}, [EIS((2,2),[8,8])]))
        self.eq("...", einsum_infer_output_formula({8:2}, [EIS((2,),[8]), EIS((2,),[8])]))

    def test_einsum_output(self):
        self.eq(EOS((),[]), einsum_output({}, [], ""))
        self.eq(EOS((),[]), einsum_output({}, [], " "))
        self.eq(EOS((),[]), einsum_output({}, [], "..."))
        self.fails(lambda: einsum_output({}, [], ". .."))
        self.fails(lambda: einsum_output({}, [], ".."))
        self.fails(lambda: einsum_output({}, [], "......"))
        self.eq(EOS((),[]), einsum_output({8:2}, [EIS((2,),[8])], ""))
        self.eq(EOS((2,),[8]), einsum_output({8:2}, [EIS((2,),[8])], "I"))
        self.eq(EOS((2,),[8]), einsum_output({8:2}, [EIS((2,),[8])], None))
        self.eq(EOS((2,4),[8,10]),
            einsum_output(
                {8:2,9:3,10:4}, [EIS((2,3),[8,9]), EIS((3,4),[9,10])], None))
        self.eq(EOS((2,4),[8,10]),
            einsum_output(
                {8:2,9:3,10:4}, [EIS((2,3),[8,9]), EIS((3,4),[9,10])], "IK"))
        self.fails(lambda: einsum_output({}, [], "......"))
        self.fails(lambda: einsum_output({8:2}, [EIS((2,),[8])], "II"))

    def test_einsum_spec(self):
        self.eq(ES({},[EIS((),[])],EOS((),[])), einsum_spec("", [()]))
        self.eq(ES({},[EIS((),[])],EOS((),[])), einsum_spec("->", [()]))
        self.eq(ES({},[EIS((),[])],EOS((),[])), einsum_spec("...->...", [()]))
        self.fails(lambda: einsum_spec("->->", [()]))

    def test_einsum_execute(self):
        t_0 = einsum_tensor(0.)
        t0_0 = einsum_tensor([])
        t1_0 = einsum_tensor([0.])
        self.eq(t_0, einsum_execute(ES({},[EIS((),[])],EOS((),[])),[t_0]))
        self.eq(t_0, einsum_execute(ES({8:0},[EIS((0,),[8])],EOS((),[])),[t0_0]))
        self.eq(t0_0, einsum_execute(ES({8:0},[EIS((0,),[8])],EOS((0,),[8])),[t0_0]))
        self.eq(t_0, einsum_execute(ES({8:1},[EIS((1,),[8])],EOS((),[])),[t1_0]))
        self.eq(t1_0, einsum_execute(ES({8:1},[EIS((1,),[8])],EOS((1,),[8])),[t1_0]))

    def test_einsum(self):
        t_0 = einsum_tensor(0.)
        t0_0 = einsum_tensor([])
        t1_0 = einsum_tensor([0.])
        self.eq(t_0, einsum("->",[t_0]))
        self.eq(t_0, einsum("I->",[t0_0]))
        self.eq(t0_0, einsum("I->I",[t0_0]))
        self.eq(t_0, einsum("I->",[t1_0]))
        self.eq(t1_0, einsum("I",[t1_0]))
        t12 = einsum_tensor([[1.1, 1.2]])
        t21 = einsum_tensor([[1.1], [2.1]])
        t22 = t21 @ t12
        self.eq(t12 @ t21, einsum("ij,jk", [t12, t21]))
        self.eq((t12 @ t21)[0,0], einsum("ij,jk->", [t12, t21]))
        self.eq(t22.sum(axis=0), einsum("ij->j", [t22]))
        self.eq(t22.diagonal(), einsum("ii->i", [t22]))
        self.eq(t22.trace(), einsum("ii", [t22]))

if __name__ == '__main__':
    unittest.main()


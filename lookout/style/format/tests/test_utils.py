import unittest

from lookout.style.format.utils import flatten_dict, merge_dicts


class RulesMergeDicts(unittest.TestCase):
    def test_merge_two_dicts(self):
        cases = [
            ({}, {}, {}),
            (dict(a=1), dict(b=2), dict(a=1, b=2)),
            (dict(a=1), dict(a=2, b=2), dict(a=2, b=2)),
            (dict(a=1, b=1), dict(b=2), dict(a=1, b=2)),
            (dict(a=1, b={"c": 1}), dict(b={"c": 2}), dict(a=1, b={"c": 2})),
            (dict(a=1), dict(b={"c": 2}), dict(a=1, b={"c": 2})),
            (dict(a=1, b={"c": 1}), dict(b={"c": 2}), dict(a=1, b={"c": 2})),
            (dict(a=dict(b=dict(c=dict(d=1)))),
             dict(a=dict(b=dict(c=dict(d=2)))),
             dict(a=dict(b=dict(c=dict(d=2))))),
            (dict(a=dict(b=dict(c=dict(d=1)))),
             dict(a=dict(b=dict(c=dict(d2=2)))),
             dict(a=dict(b=dict(c=dict(d=1, d2=2))))),
            (dict(a=dict(b=dict(c=dict(d=1)))),
             dict(a=dict(b=dict(c2=dict(d=2)))),
             dict(a=dict(b=dict(c=dict(d=1), c2=dict(d=2))))),
        ]
        for d1, d2, res in cases:
            self.assertEqual(merge_dicts(d1, d2), res)

    def test_merge_three_dicts(self):
        d1 = dict(a=1, b={"c": 1})
        d2 = dict(b={"c": 2})
        d3 = dict(b={"c": 3}, d=4)
        res = dict(a=1, b={"c": 3}, d=4)
        self.assertEqual(merge_dicts(d1, d2, d3), res)

    def test_flatten_dict(self):
        cases = [
            ({"1": 1, "2": 2}, {"1": 1, "2": 2}),
            ({"1": 1, "2": {"3": 3, "4": 4}}, {"1": 1, "2_3": 3, "2_4": 4}),
            ({"1": 1, "2": {"3": 3, "4": {"5": 5, "6": 6}}},
             {"1": 1, "2_3": 3, "2_4_5": 5, "2_4_6": 6}),
        ]

        for d, res in cases:
            self.assertEqual(flatten_dict(d), res)


if __name__ == "__main__":
    unittest.main()

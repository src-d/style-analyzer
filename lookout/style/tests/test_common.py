import unittest

from lookout.style.common import merge_dicts


class CommonTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

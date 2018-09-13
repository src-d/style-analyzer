import unittest

import numpy
from lookout.style.format.features import BucketsEncoder


class BucketEncoderTests(unittest.TestCase):
    def test_diff_1(self):
        be = BucketsEncoder(2, [0, 1], 6)
        X = numpy.array([
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ]).T
        new_X = be.fit_transform(X)
        correct_new_X = numpy.array([
            [1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3],
            [0, 0, 0, 3, 2, 1, 0, 0, 0, 3, 2, 1],
            [3, 2, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ]).T

        self.assertTrue(numpy.all(new_X == correct_new_X))

    def test_diff_2(self):
        be = BucketsEncoder(6, [0, 1], 6)
        X = numpy.array([
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ]).T
        new_X = be.fit_transform(X)
        correct_new_X = numpy.array(
            [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]).T

        self.assertTrue(numpy.all(new_X == correct_new_X))

    def test_diff_3(self):
        be = BucketsEncoder(1, [0, 1], 6)
        X = numpy.array([
            [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ]).T
        new_X = be.fit_transform(X)
        correct_new_X = numpy.array([
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            [6, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2, 1],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]).T
        self.assertTrue(numpy.all(new_X == correct_new_X))

    def test_diff_4(self):
        be = BucketsEncoder(4, [0, 1], 7)
        X = numpy.array([
            [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0],
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        ]).T
        new_X = be.fit_transform(X)
        correct_new_X = numpy.array(
            [[1, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
             [0, 0, 1, 2, 0, 0, 0, 0, 1, 2, 0, 0],
             [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 1],
             [0, 0, 2, 1, 0, 0, 0, 0, 2, 1, 0, 0],
             [2, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]]).T
        self.assertTrue(numpy.all(new_X == correct_new_X))


if __name__ == "__main__":
    unittest.main()

import os
import unittest

long_test = unittest.skipUnless(os.getenv("LONG_TESTS", False),
                                "Time-consuming tests are skipped by default.")

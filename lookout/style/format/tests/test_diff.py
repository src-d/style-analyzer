import unittest

from lookout.core.api.service_data_pb2 import File
from lookout.style.format.diff import find_new_lines


class DiffTests(unittest.TestCase):
    def test_diff(self):
        text_base = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Maecenas volutpat dui id ipsum cursus, sit amet accumsan nisl ornare.
        Vivamus euismod lorem viverra semper dictum.
        Nam consectetur enim eget elementum mattis.
        Ut condimentum metus vehicula tellus tempus, vel ultricies lectus dapibus.
        Etiam vitae nisi at ante pretium lacinia et eu massa."""
        # inserted lines: 3 and 6 (counting from 1 with a new line at the start)
        # modified line: 4
        text_head = """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Curabitur congue libero vitae quam venenatis, tristique commodo diam lacinia.
        Mecenas volutpat dui id ipsum cursus, sit amet accumsan nisl ornare.
        Vivamus euismod lorem viverra semper dictum.
        Praesent eu ipsum sit amet elit aliquam laoreet.
        Nam consectetur enim eget elementum mattis.
        Ut condimentum metus vehicula tellus tempus, vel ultricies lectus dapibus.
        Etiam vitae nisi at ante pretium lacinia et eu massa."""
        new_line_indices = find_new_lines(File(content=bytes(text_base, "utf-8")),
                                          File(content=bytes(text_head, "utf-8")))
        self.assertEqual(new_line_indices, [3, 4, 6])


if __name__ == "__main__":
    unittest.main()

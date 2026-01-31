import unittest

import numpy as np
from pathlib import Path


class TestProjections(unittest.TestCase):

    def test_npz_files(self):

        # cartella test
        root = Path(__file__).parent
        data_path = root / "projections"

        expected = np.load(data_path / "expected_projection_test_data.npz")
        output = np.load(data_path / "output_projection_test_data.npz")

        # controlliamo che abbiano le stesse chiavi
        self.assertEqual(expected.files, output.files)

        # controlliamo che i valori siano uguali
        for key in expected.files:
            np.testing.assert_allclose(expected[key], output[key])


if __name__ == '__main__':
    unittest.main()

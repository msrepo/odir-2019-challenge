from unittest import TestCase
import unittest

import torch
from transforms.transforms import pad_to_largest_square


class TestTransforms(TestCase):
    def test_padding(self):
        a = torch.zeros((3,128,200))
        a_padded = pad_to_largest_square(a)
        self.assertEqual(a_padded.shape,(3,200,200))

        a = torch.zeros((3,129,200))
        a_padded = pad_to_largest_square(a)
        self.assertEqual(a_padded.shape,(3,200,200))

if __name__ == '__main__':
    unittest.main()
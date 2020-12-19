import os
import unittest

from mllib import get_data


PATH = os.path.dirname(os.path.realpath(__file__))
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"


class TestGeneric(unittest.TestCase):
    def test(self):
        dat = get_data(ATTR_URL)
        print(dat)
        raise Exception

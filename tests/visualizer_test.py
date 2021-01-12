import io
import os
import unittest

import numpy as np
from PIL import Image

from vltk import SingleImageViz


PATH = os.path.dirname(os.path.realpath(__file__))

URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"


class TestVisaulizer(unittest.TestCase):

    url = URL

    def test_viz(self):
        viz = SingleImageViz(self.url)
        viz.show()

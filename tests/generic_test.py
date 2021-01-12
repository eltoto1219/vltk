import os
import unittest

from vltk import get_data

TEST_PATH = os.path.dirname(os.path.realpath(__file__))


class TestGeneric(unittest.TestCase):

    # setup rando things like schema, etc
    # for tests, we will want to test each new method, plus a test extraction. The test extraction will have to be first
    # in order of most general to most specific tests

    def test_extraction_single_dir(self):
        # okay so the extraction single dir will
        pass

    def test_extaction_multi_dir(self):
        pass

    def add_text_dataset(self):
        pass

    def test_create_column_text(self):
        pass

    def test_append_column_text(self):
        pass

    def test_remove_column_text(self):
        pass

    def test_create_labeled_column_text(self):
        pass

    def test_append_labeled_column_text(self):
        pass

    def test_remove_labeled_column_text(self):
        pass


'''
useful methods:
    save_to_disk
    datasets.concatenate_datasets
    load from disk
'''

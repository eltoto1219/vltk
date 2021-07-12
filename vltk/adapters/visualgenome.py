from vltk import adapters


class VisualGenome(adapters.VisnDataset):
    @staticmethod
    def schema():
        return {}

    @staticmethod
    def forward(json_files, splits):
        return {}

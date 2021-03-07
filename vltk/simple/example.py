from vltk.abc.simple import SimpleExperiment

__all__ = ["Data"]


class Data(SimpleExperiment):
    """
    iter_outputs contain combined batch dictionary and ouput dictionary from forward
    """

    name = "data"
    model_list = []

    def forward(self, batch) -> dict:
        print(batch.keys())
        print(batch["image"].shape)

        return {}

    def iter_tqdm(self, forward_outputs, train=True):
        return {}

    def iter_save(forward_outputs, train=True):
        return {}

    def epoch_logstr(self, loop_outputs, train=True):
        return {}

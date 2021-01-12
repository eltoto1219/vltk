from vltk.abc.loop import Loop


class Data(Loop):
    name: str = "data"
    is_train: bool = False

    def forward(self, batch):
        return super().forward(batch)

    def loop(self, batch, model_outputs):
        return super().loop(batch, model_outputs)

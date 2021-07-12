import datasets as ds


class Features:
    @staticmethod
    def Boxes():
        return ds.Sequence(
            length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
        )

    # legacy
    @staticmethod
    def Box():
        return ds.Sequence(
            length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
        )

    def Polygons():
        return ds.Sequence(
            length=-1,
            feature=ds.Sequence(
                length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
            ),
        )

    # def RLE():
    #   ds.Sequence(length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32")))

    def RLE():
        return ds.Sequence(length=-1, feature=ds.Value("float32"))

    def FloatList():
        return ds.Sequence(length=-1, feature=ds.Value("float32"))

    def Imgid():
        return ds.Value("string")

    def String():
        return ds.Value("string")

    def StringList():
        return ds.Sequence(length=-1, feature=ds.Value("string"))

    def NestedStringList():
        return ds.Sequence(ds.Sequence(length=-1, feature=ds.Value("string")))

    def Int():
        return ds.Value("int32")

    def IntList():
        return ds.Sequence(length=-1, feature=ds.Value("int32"))

    def NestedIntList():
        return ds.Sequence(
            length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("int32"))
        )

    def Span():
        return ds.Sequence(length=-1, feature=ds.Value("int32"))

    def Float():
        return ds.Value("float32")

    def Ids():
        return ds.Sequence(length=-1, feature=ds.Value("float32"))

    def Boxtensor(n):
        return ds.Array2D((n, 4), dtype="float32")

    # something doesnt look right here (between 2d and 3d features)
    def Features2D(d):
        return ds.Array2D((-1, d), dtype="float32")

    def Features3D(n, d):
        return ds.Array2D((n, d), dtype="float32")

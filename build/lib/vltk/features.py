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

    @staticmethod
    def BoolList():
        return ds.Sequence(length=-1, feature=ds.Value("bool"))

    @staticmethod
    def Polygons():
        return ds.Sequence(
            length=-1,
            feature=ds.Sequence(
                length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32"))
            ),
        )

    # def RLE():
    #   ds.Sequence(length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("float32")))

    @staticmethod
    def RLE():
        return ds.Sequence(length=-1, feature=ds.Value("float32"))

    @staticmethod
    def FloatList():
        return ds.Sequence(length=-1, feature=ds.Value("float32"))

    @staticmethod
    def Imgid():
        return ds.Value("string")

    @staticmethod
    def String():
        return ds.Value("string")

    @staticmethod
    def StringList():
        return ds.Sequence(length=-1, feature=ds.Value("string"))

    @staticmethod
    def NestedStringList():
        return ds.Sequence(ds.Sequence(length=-1, feature=ds.Value("string")))

    @staticmethod
    def Int():
        return ds.Value("int32")

    @staticmethod
    def IntList():
        return ds.Sequence(length=-1, feature=ds.Value("int32"))

    @staticmethod
    def NestedIntList():
        return ds.Sequence(
            length=-1, feature=ds.Sequence(length=-1, feature=ds.Value("int32"))
        )

    @staticmethod
    def Span():
        return ds.Sequence(length=-1, feature=ds.Value("int32"))

    @staticmethod
    def Float():
        return ds.Value("float32")

    @staticmethod
    def Ids():
        return ds.Sequence(length=-1, feature=ds.Value("float32"))

    @staticmethod
    def Boxtensor(n):
        return ds.Array2D((n, 4), dtype="float32")

    # something doesnt look right here (between 2d and 3d features)
    @staticmethod
    def Features2D(d):
        return ds.Array2D((-1, d), dtype="float32")

    @staticmethod
    def Features3D(n, d):
        return ds.Array2D((n, d), dtype="float32")

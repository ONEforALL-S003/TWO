# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class LocalResponseNormalizationOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LocalResponseNormalizationOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsLocalResponseNormalizationOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def LocalResponseNormalizationOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # LocalResponseNormalizationOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # LocalResponseNormalizationOptions
    def Radius(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # LocalResponseNormalizationOptions
    def Bias(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # LocalResponseNormalizationOptions
    def Alpha(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # LocalResponseNormalizationOptions
    def Beta(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

def Start(builder): builder.StartObject(4)
def LocalResponseNormalizationOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddRadius(builder, radius): builder.PrependInt32Slot(0, radius, 0)
def LocalResponseNormalizationOptionsAddRadius(builder, radius):
    """This method is deprecated. Please switch to AddRadius."""
    return AddRadius(builder, radius)
def AddBias(builder, bias): builder.PrependFloat32Slot(1, bias, 0.0)
def LocalResponseNormalizationOptionsAddBias(builder, bias):
    """This method is deprecated. Please switch to AddBias."""
    return AddBias(builder, bias)
def AddAlpha(builder, alpha): builder.PrependFloat32Slot(2, alpha, 0.0)
def LocalResponseNormalizationOptionsAddAlpha(builder, alpha):
    """This method is deprecated. Please switch to AddAlpha."""
    return AddAlpha(builder, alpha)
def AddBeta(builder, beta): builder.PrependFloat32Slot(3, beta, 0.0)
def LocalResponseNormalizationOptionsAddBeta(builder, beta):
    """This method is deprecated. Please switch to AddBeta."""
    return AddBeta(builder, beta)
def End(builder): return builder.EndObject()
def LocalResponseNormalizationOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
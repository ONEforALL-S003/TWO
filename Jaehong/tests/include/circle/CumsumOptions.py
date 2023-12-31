# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class CumsumOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CumsumOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsCumsumOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def CumsumOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # CumsumOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CumsumOptions
    def Exclusive(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # CumsumOptions
    def Reverse(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def Start(builder): builder.StartObject(2)
def CumsumOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddExclusive(builder, exclusive): builder.PrependBoolSlot(0, exclusive, 0)
def CumsumOptionsAddExclusive(builder, exclusive):
    """This method is deprecated. Please switch to AddExclusive."""
    return AddExclusive(builder, exclusive)
def AddReverse(builder, reverse): builder.PrependBoolSlot(1, reverse, 0)
def CumsumOptionsAddReverse(builder, reverse):
    """This method is deprecated. Please switch to AddReverse."""
    return AddReverse(builder, reverse)
def End(builder): return builder.EndObject()
def CumsumOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
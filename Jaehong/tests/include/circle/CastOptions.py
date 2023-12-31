# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class CastOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = CastOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsCastOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def CastOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # CastOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # CastOptions
    def InDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # CastOptions
    def OutDataType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(2)
def CastOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddInDataType(builder, inDataType): builder.PrependInt8Slot(0, inDataType, 0)
def CastOptionsAddInDataType(builder, inDataType):
    """This method is deprecated. Please switch to AddInDataType."""
    return AddInDataType(builder, inDataType)
def AddOutDataType(builder, outDataType): builder.PrependInt8Slot(1, outDataType, 0)
def CastOptionsAddOutDataType(builder, outDataType):
    """This method is deprecated. Please switch to AddOutDataType."""
    return AddOutDataType(builder, outDataType)
def End(builder): return builder.EndObject()
def CastOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
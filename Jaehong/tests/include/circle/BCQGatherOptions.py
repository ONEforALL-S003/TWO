# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class BCQGatherOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BCQGatherOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsBCQGatherOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def BCQGatherOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # BCQGatherOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BCQGatherOptions
    def InputHiddenSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # BCQGatherOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(2)
def BCQGatherOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddInputHiddenSize(builder, inputHiddenSize): builder.PrependInt32Slot(0, inputHiddenSize, 0)
def BCQGatherOptionsAddInputHiddenSize(builder, inputHiddenSize):
    """This method is deprecated. Please switch to AddInputHiddenSize."""
    return AddInputHiddenSize(builder, inputHiddenSize)
def AddAxis(builder, axis): builder.PrependInt32Slot(1, axis, 0)
def BCQGatherOptionsAddAxis(builder, axis):
    """This method is deprecated. Please switch to AddAxis."""
    return AddAxis(builder, axis)
def End(builder): return builder.EndObject()
def BCQGatherOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
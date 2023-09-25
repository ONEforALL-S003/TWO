# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class GatherOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GatherOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsGatherOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def GatherOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # GatherOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # GatherOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # GatherOptions
    def BatchDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(2)
def GatherOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddAxis(builder, axis): builder.PrependInt32Slot(0, axis, 0)
def GatherOptionsAddAxis(builder, axis):
    """This method is deprecated. Please switch to AddAxis."""
    return AddAxis(builder, axis)
def AddBatchDims(builder, batchDims): builder.PrependInt32Slot(1, batchDims, 0)
def GatherOptionsAddBatchDims(builder, batchDims):
    """This method is deprecated. Please switch to AddBatchDims."""
    return AddBatchDims(builder, batchDims)
def End(builder): return builder.EndObject()
def GatherOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
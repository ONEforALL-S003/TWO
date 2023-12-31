# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SqueezeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SqueezeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSqueezeOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SqueezeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # SqueezeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SqueezeOptions
    def SqueezeDims(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SqueezeOptions
    def SqueezeDimsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SqueezeOptions
    def SqueezeDimsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SqueezeOptions
    def SqueezeDimsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def Start(builder): builder.StartObject(1)
def SqueezeOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddSqueezeDims(builder, squeezeDims): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(squeezeDims), 0)
def SqueezeOptionsAddSqueezeDims(builder, squeezeDims):
    """This method is deprecated. Please switch to AddSqueezeDims."""
    return AddSqueezeDims(builder, squeezeDims)
def StartSqueezeDimsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SqueezeOptionsStartSqueezeDimsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartSqueezeDimsVector(builder, numElems)
def End(builder): return builder.EndObject()
def SqueezeOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class BucketizeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BucketizeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsBucketizeOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def BucketizeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # BucketizeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BucketizeOptions
    def Boundaries(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # BucketizeOptions
    def BoundariesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # BucketizeOptions
    def BoundariesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # BucketizeOptions
    def BoundariesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def Start(builder): builder.StartObject(1)
def BucketizeOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddBoundaries(builder, boundaries): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(boundaries), 0)
def BucketizeOptionsAddBoundaries(builder, boundaries):
    """This method is deprecated. Please switch to AddBoundaries."""
    return AddBoundaries(builder, boundaries)
def StartBoundariesVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def BucketizeOptionsStartBoundariesVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartBoundariesVector(builder, numElems)
def End(builder): return builder.EndObject()
def BucketizeOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
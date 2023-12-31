# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReshapeOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReshapeOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReshapeOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ReshapeOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # ReshapeOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReshapeOptions
    def NewShape(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # ReshapeOptions
    def NewShapeAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # ReshapeOptions
    def NewShapeLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ReshapeOptions
    def NewShapeIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def Start(builder): builder.StartObject(1)
def ReshapeOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddNewShape(builder, newShape): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(newShape), 0)
def ReshapeOptionsAddNewShape(builder, newShape):
    """This method is deprecated. Please switch to AddNewShape."""
    return AddNewShape(builder, newShape)
def StartNewShapeVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ReshapeOptionsStartNewShapeVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartNewShapeVector(builder, numElems)
def End(builder): return builder.EndObject()
def ReshapeOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
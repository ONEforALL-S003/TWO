# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ExpandDimsOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ExpandDimsOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsExpandDimsOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def ExpandDimsOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # ExpandDimsOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def Start(builder): builder.StartObject(0)
def ExpandDimsOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def End(builder): return builder.EndObject()
def ExpandDimsOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SpaceToDepthOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SpaceToDepthOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSpaceToDepthOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SpaceToDepthOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # SpaceToDepthOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SpaceToDepthOptions
    def BlockSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(1)
def SpaceToDepthOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddBlockSize(builder, blockSize): builder.PrependInt32Slot(0, blockSize, 0)
def SpaceToDepthOptionsAddBlockSize(builder, blockSize):
    """This method is deprecated. Please switch to AddBlockSize."""
    return AddBlockSize(builder, blockSize)
def End(builder): return builder.EndObject()
def SpaceToDepthOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
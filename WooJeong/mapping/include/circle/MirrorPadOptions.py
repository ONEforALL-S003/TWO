# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class MirrorPadOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MirrorPadOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMirrorPadOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def MirrorPadOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # MirrorPadOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MirrorPadOptions
    def Mode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(1)
def MirrorPadOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddMode(builder, mode): builder.PrependInt8Slot(0, mode, 0)
def MirrorPadOptionsAddMode(builder, mode):
    """This method is deprecated. Please switch to AddMode."""
    return AddMode(builder, mode)
def End(builder): return builder.EndObject()
def MirrorPadOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RNNOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RNNOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRNNOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def RNNOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # RNNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RNNOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # RNNOptions
    def AsymmetricQuantizeInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def Start(builder): builder.StartObject(2)
def RNNOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def RNNOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    """This method is deprecated. Please switch to AddFusedActivationFunction."""
    return AddFusedActivationFunction(builder, fusedActivationFunction)
def AddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs): builder.PrependBoolSlot(1, asymmetricQuantizeInputs, 0)
def RNNOptionsAddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs):
    """This method is deprecated. Please switch to AddAsymmetricQuantizeInputs."""
    return AddAsymmetricQuantizeInputs(builder, asymmetricQuantizeInputs)
def End(builder): return builder.EndObject()
def RNNOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
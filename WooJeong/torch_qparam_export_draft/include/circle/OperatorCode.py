# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class OperatorCode(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = OperatorCode()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsOperatorCode(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def OperatorCodeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # OperatorCode
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # OperatorCode
    def DeprecatedBuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # OperatorCode
    def CustomCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # OperatorCode
    def Version(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # OperatorCode
    def BuiltinCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(4)
def OperatorCodeStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode): builder.PrependInt8Slot(0, deprecatedBuiltinCode, 0)
def OperatorCodeAddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode):
    """This method is deprecated. Please switch to AddDeprecatedBuiltinCode."""
    return AddDeprecatedBuiltinCode(builder, deprecatedBuiltinCode)
def AddCustomCode(builder, customCode): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(customCode), 0)
def OperatorCodeAddCustomCode(builder, customCode):
    """This method is deprecated. Please switch to AddCustomCode."""
    return AddCustomCode(builder, customCode)
def AddVersion(builder, version): builder.PrependInt32Slot(2, version, 1)
def OperatorCodeAddVersion(builder, version):
    """This method is deprecated. Please switch to AddVersion."""
    return AddVersion(builder, version)
def AddBuiltinCode(builder, builtinCode): builder.PrependInt32Slot(3, builtinCode, 0)
def OperatorCodeAddBuiltinCode(builder, builtinCode):
    """This method is deprecated. Please switch to AddBuiltinCode."""
    return AddBuiltinCode(builder, builtinCode)
def End(builder): return builder.EndObject()
def OperatorCodeEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
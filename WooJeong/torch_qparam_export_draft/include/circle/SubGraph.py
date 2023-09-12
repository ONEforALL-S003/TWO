# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class SubGraph(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubGraph()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsSubGraph(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def SubGraphBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x43\x49\x52\x30", size_prefixed=size_prefixed)

    # SubGraph
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubGraph
    def Tensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from circle.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SubGraph
    def TensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def TensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # SubGraph
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SubGraph
    def InputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SubGraph
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        return o == 0

    # SubGraph
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # SubGraph
    def OutputsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # SubGraph
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # SubGraph
    def Operators(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from circle.Operator import Operator
            obj = Operator()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # SubGraph
    def OperatorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # SubGraph
    def OperatorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # SubGraph
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # SubGraph
    def DataFormat(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(6)
def SubGraphStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddTensors(builder, tensors): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(tensors), 0)
def SubGraphAddTensors(builder, tensors):
    """This method is deprecated. Please switch to AddTensors."""
    return AddTensors(builder, tensors)
def StartTensorsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SubGraphStartTensorsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartTensorsVector(builder, numElems)
def AddInputs(builder, inputs): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)
def SubGraphAddInputs(builder, inputs):
    """This method is deprecated. Please switch to AddInputs."""
    return AddInputs(builder, inputs)
def StartInputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SubGraphStartInputsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartInputsVector(builder, numElems)
def AddOutputs(builder, outputs): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)
def SubGraphAddOutputs(builder, outputs):
    """This method is deprecated. Please switch to AddOutputs."""
    return AddOutputs(builder, outputs)
def StartOutputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SubGraphStartOutputsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartOutputsVector(builder, numElems)
def AddOperators(builder, operators): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(operators), 0)
def SubGraphAddOperators(builder, operators):
    """This method is deprecated. Please switch to AddOperators."""
    return AddOperators(builder, operators)
def StartOperatorsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def SubGraphStartOperatorsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartOperatorsVector(builder, numElems)
def AddName(builder, name): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)
def SubGraphAddName(builder, name):
    """This method is deprecated. Please switch to AddName."""
    return AddName(builder, name)
def AddDataFormat(builder, dataFormat): builder.PrependInt8Slot(5, dataFormat, 0)
def SubGraphAddDataFormat(builder, dataFormat):
    """This method is deprecated. Please switch to AddDataFormat."""
    return AddDataFormat(builder, dataFormat)
def End(builder): return builder.EndObject()
def SubGraphEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow/core/framework/tensor_description.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2
from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
from tensorflow.core.framework import allocation_description_pb2 as tensorflow_dot_core_dot_framework_dot_allocation__description__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow/core/framework/tensor_description.proto',
  package='tensorflow',
  syntax='proto3',
  serialized_options=_b('\n\030org.tensorflow.frameworkB\027TensorDescriptionProtosP\001\370\001\001'),
  serialized_pb=_b('\n2tensorflow/core/framework/tensor_description.proto\x12\ntensorflow\x1a%tensorflow/core/framework/types.proto\x1a,tensorflow/core/framework/tensor_shape.proto\x1a\x36tensorflow/core/framework/allocation_description.proto\"\xa8\x01\n\x11TensorDescription\x12#\n\x05\x64type\x18\x01 \x01(\x0e\x32\x14.tensorflow.DataType\x12+\n\x05shape\x18\x02 \x01(\x0b\x32\x1c.tensorflow.TensorShapeProto\x12\x41\n\x16\x61llocation_description\x18\x04 \x01(\x0b\x32!.tensorflow.AllocationDescriptionB8\n\x18org.tensorflow.frameworkB\x17TensorDescriptionProtosP\x01\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow_dot_core_dot_framework_dot_types__pb2.DESCRIPTOR,tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2.DESCRIPTOR,tensorflow_dot_core_dot_framework_dot_allocation__description__pb2.DESCRIPTOR,])




_TENSORDESCRIPTION = _descriptor.Descriptor(
  name='TensorDescription',
  full_name='tensorflow.TensorDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dtype', full_name='tensorflow.TensorDescription.dtype', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorflow.TensorDescription.shape', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='allocation_description', full_name='tensorflow.TensorDescription.allocation_description', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=208,
  serialized_end=376,
)

_TENSORDESCRIPTION.fields_by_name['dtype'].enum_type = tensorflow_dot_core_dot_framework_dot_types__pb2._DATATYPE
_TENSORDESCRIPTION.fields_by_name['shape'].message_type = tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2._TENSORSHAPEPROTO
_TENSORDESCRIPTION.fields_by_name['allocation_description'].message_type = tensorflow_dot_core_dot_framework_dot_allocation__description__pb2._ALLOCATIONDESCRIPTION
DESCRIPTOR.message_types_by_name['TensorDescription'] = _TENSORDESCRIPTION
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TensorDescription = _reflection.GeneratedProtocolMessageType('TensorDescription', (_message.Message,), dict(
  DESCRIPTOR = _TENSORDESCRIPTION,
  __module__ = 'tensorflow.core.framework.tensor_description_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.TensorDescription)
  ))
_sym_db.RegisterMessage(TensorDescription)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)

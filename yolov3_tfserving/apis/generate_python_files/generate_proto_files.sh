#!/bin/bash

set -x
set -e

python -m grpc.tools.protoc -I./ --python_out=.. --grpc_python_out=.. ./*.proto

python -m grpc.tools.protoc -I./ --python_out=.. --grpc_python_out=.. ./tensorflow/core/framework/*.proto
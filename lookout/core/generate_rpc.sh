#!/bin/sh -ex

base_dir=$(dirname "$(readlink -f \"$0\")")
out_dir=$base_dir/api

mkdir -p $out_dir
python3 -m grpc_tools.protoc -I$base_dir/server/sdk \
    --python_out=$out_dir --grpc_python_out=$out_dir \
    $base_dir/server/sdk/*.proto
python3 -m grpc_tools.protoc -I$base_dir/server/sdk \
    --python_out=$out_dir \
    $base_dir/server/sdk/github.com/gogo/protobuf/gogoproto/gogo.proto
python3 -m grpc_tools.protoc -I$base_dir/server/sdk \
    --python_out=$out_dir \
    $base_dir/server/sdk/gopkg.in/bblfsh/sdk.v1/uast/generated.proto
touch $out_dir/__init__.py
# https://github.com/google/protobuf/issues/1491
find $out_dir -name '*.py' -exec sed -Ei 's/^(import [^ ]+_pb2 as)/from . \1/g' {} \;
find $out_dir -name '*.py' -exec sed -i 's/from github/from lookout.core.api.github/g' {} \;
find $out_dir -name '*.py' -exec sed -i 's/from gopkg/from lookout.core.api.gopkg/g' {} \;
find $out_dir -name '*.py' -exec sed -i "s/importlib.import_module('/importlib.import_module('lookout.core.api./g" {} \;

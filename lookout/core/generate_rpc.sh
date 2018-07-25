#!/bin/sh -ex

base_dir=$(dirname "$(readlink -f \"$0\")")
out_dir=$base_dir/api

mkdir -p $out_dir
python3 -m grpc_tools.protoc -I$base_dir/server/sdk \
    --python_out=$out_dir --grpc_python_out=$out_dir \
    $base_dir/server/sdk/*.proto
touch $out_dir/__init__.py

#!/bin/sh -ex

base_dir=$(dirname "$(readlink -f \"$0\")")
out_dir=$base_dir/../..
real_out_dir=$base_dir/api
tmp_dir=$base_dir/lookout/core/api

rm -rf $real_out_dir
rm -rf $tmp_dir
mkdir -p $tmp_dir
cp -r $base_dir/server/sdk/* $tmp_dir
find $tmp_dir -name '*.proto' -exec sed -i '/"google/! {/\//! s/import "/import "lookout\/core\/api\//g}' {} \;
python3 -m grpc_tools.protoc -I$base_dir -I$tmp_dir \
    --python_out=$out_dir --grpc_python_out=$out_dir \
    $tmp_dir/*.proto
find $real_out_dir -name '*.py' -exec sed -i 's/^from github/from bblfsh.github/g' {} \;
find $real_out_dir -name '*.py' -exec sed -i "s/import_module('gopkg.in/import_module('bblfsh.gopkg.in/g" {} \;
touch $real_out_dir/__init__.py
rm -r $base_dir/lookout

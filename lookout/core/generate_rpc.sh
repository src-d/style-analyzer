#!/bin/sh -ex

base_dir=$(dirname "$(readlink -f \"$0\")")
out_dir=$base_dir/../..
real_out_dir=$base_dir/api
tmp_dir=$base_dir/lookout/core/api
rm -rf $real_out_dir
mkdir -p $real_out_dir
rm -rf $tmp_dir
mkdir -p $tmp_dir

git -C $base_dir/server fetch --tags
version=$(git -C $base_dir/server tag | grep '^v[[:digit:]]' | sort -nr | head -n1)
git -C $base_dir/server checkout $version
echo "__version__ = \"$version\"\n" > $real_out_dir/version.py
sdk_hash=$(grep -A3 src-d/lookout-sdk $base_dir/server/Gopkg.lock | grep revision | cut -d'"' -f 2)
rm -rf /tmp/lookout-sdk
git clone https://github.com/src-d/lookout-sdk /tmp/lookout-sdk
git -C /tmp/lookout-sdk checkout $sdk_hash
cp -r /tmp/lookout-sdk/proto/* $tmp_dir
rm -rf /tmp/lookout-sdk
find $tmp_dir/lookout/sdk -name '*.proto' -exec sed -i 's/lookout\/sdk\///g' {} \;
mv $tmp_dir/lookout/sdk/*.proto $tmp_dir
rm -r $tmp_dir/lookout
find $tmp_dir -name '*.proto' -exec sed -i '/"google/! {/\//! s/import "/import "lookout\/core\/api\//g}' {} \;
python3 -m grpc_tools.protoc -I$base_dir -I$tmp_dir \
    --python_out=$out_dir --grpc_python_out=$out_dir \
    $tmp_dir/*.proto
find $real_out_dir -name '*.py' -exec sed -i 's/^from github/from bblfsh.github/g' {} \;
find $real_out_dir -name '*.py' -exec sed -i "s/import_module('gopkg.in/import_module('bblfsh.gopkg.in/g" {} \;
touch $real_out_dir/__init__.py
rm -r $base_dir/lookout

#!/usr/bin/env bash
# Build the ucar-napi native addon and copy it to ucar-napi/ucar-napi.node.
set -euo pipefail
cd "$(dirname "$0")"
cargo build --release -p ucar-napi
case "$(uname -s)" in
    Darwin) artifact="target/release/libucar_napi.dylib" ;;
    *)      artifact="target/release/libucar_napi.so" ;;
esac
cp -f "$artifact" ucar-napi/ucar-napi.node.tmp && mv -f ucar-napi/ucar-napi.node.tmp ucar-napi/ucar-napi.node
echo "ucar-napi.node updated"

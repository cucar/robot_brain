#!/usr/bin/env bash
# Build the brain-napi native addon and copy it to brain-napi/brain-napi.node.
# Linux/macOS equivalent of build.ps1.
set -euo pipefail

# Run from this script's directory so relative paths match build.ps1.
cd "$(dirname "$0")"

cargo build --release -p brain-napi

# cargo names the cdylib per-platform: libbrain_napi.so on Linux, libbrain_napi.dylib on macOS.
case "$(uname -s)" in
    Darwin) artifact="target/release/libbrain_napi.dylib" ;;
    *)      artifact="target/release/libbrain_napi.so" ;;
esac

cp -f "$artifact" brain-napi/brain-napi.node
echo "brain-napi.node updated"

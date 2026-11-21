#!/bin/bash
# The benchmarks of this build (see README.md beside this script). Builds the examples, runs every benchmark, strips
# the timings from what it printed and compares that with the output saved in output/: "same" or "different" per
# benchmark, and the run time. Exits non-zero where any benchmark differs.
# usage: benchmarks/run.sh [--record] [--only <name>] [mnist-data-dir]
#   mnist-data-dir  the four raw (gunzipped) MNIST idx files; default apps/mnist/data in this repository, where
#                   they are downloaded from the mirror apps/mnist/jobs/download.js uses if they are not there
#   --record        write each run's stripped output as the saved output instead of comparing
#   --only <name>   run that benchmark alone
set -u
here="$(cd "$(dirname "$0")" && pwd)"
core="$(cd "$here/.." && pwd)"
repo="$(cd "$core/../.." && pwd)"
record=0; only=""; mn="$repo/apps/mnist/data"
while [ $# -gt 0 ]; do
  case "$1" in
    --record) record=1;;
    --only) only="$2"; shift;;
    *) mn="$1";;
  esac
  shift
done
mkdir -p "$mn"
mn="$(cd "$mn" && pwd)"
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  if [ ! -s "$mn/$f" ]; then
    echo "downloading $f.gz into $mn"
    curl -fsSL "https://storage.googleapis.com/cvdf-datasets/mnist/$f.gz" | gunzip > "$mn/$f" || { echo "could not get $f"; exit 2; }
  fi
done
echo "building the examples"
(cd "$core" && cargo build --release --examples 2>&1 | grep -E "^error" -A8) && exit 2
bin="$core/../target/release/examples"
exp="$here/output"; got="$here/got"
mkdir -p "$exp" "$got"

# the timings a run prints, taken out: milliseconds, seconds at the end of a line, a line that is a time alone
strip() { sed -E 's/[0-9]+(\.[0-9]+)? ?ms(\/image)?//g; s/ ?\|? [0-9]+(\.[0-9]+)? ?s$//; /^ *[0-9]+(\.[0-9]+)? ?s$/d' "$1"; }

fails=0
printf "%-26s %-10s %s\n" "benchmark" "output" "time"
bench() {  # name, command...
  local name="$1"; shift
  [ -n "$only" ] && [ "$only" != "$name" ] && return
  local t0=$(date +%s.%N)
  (cd "$core" && "$@") > "$got/$name.raw.out" 2> "$got/$name.raw.err"
  local code=$?
  local t1=$(date +%s.%N)
  strip "$got/$name.raw.out" > "$got/$name.out"
  strip "$got/$name.raw.err" > "$got/$name.err"
  if [ $record -eq 1 ]; then cp "$got/$name.out" "$exp/$name.out"; cp "$got/$name.err" "$exp/$name.err"; fi
  local verdict="same"
  if [ $code -ne 0 ]; then verdict="failed($code)"; fi
  cmp -s "$exp/$name.out" "$got/$name.out" && cmp -s "$exp/$name.err" "$got/$name.err" || verdict="different"
  [ "$verdict" = "same" ] || fails=$((fails + 1))
  printf "%-26s %-10s %.1fs\n" "$name" "$verdict" "$(echo "$t1 - $t0" | bc)"
}

# 1 the alternation
bench alternation "$bin/xy" 33
# 2 stocks on laps
bench six-bar-lap "$bin/stocks" 30 256
bench kgc-gld "$bin/channels" two 10
bench two-six-bar-laps "$bin/channels" lap 30
bench six-and-five-bar-laps "$bin/channels" lap65 12
bench three-stocks "$bin/channels" three 100
bench four-stocks "$bin/channels" three4 100 --all4
# 3 real stocks
bench three-stocks-5-laps "$bin/channels" seq 5
bench ten-stocks "$bin/channels" seq 1 --symbols=SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --max-positions=3 --cost=0.02 --every=100
# 4 tap and text
bench tap "$bin/tap" 6 8
bench text "$bin/text" "the the the the the the the the the " 1 64
bench text-without-space "$bin/text" "theathebthecthedthefthegtheithejthek" 1 64
# 5 lines and squares
bench hline "$bin/shapes" hline 30
bench vline "$bin/shapes" vline 30
bench filled-square "$bin/shapes" square 30
bench hollow-square "$bin/shapes" hollow 30
bench two-and-three "$bin/shapes" teach23 30 0 -
# 6 MNIST
bench mnist-1x60 "$bin/mnist" "$mn" 1 0 2048 14 all 128 0 all 60
bench mnist-10x20 "$bin/mnist" "$mn" 10 0 2048 14 all 128 0 all 20
bench mnist-10x300 "$bin/mnist" "$mn" 10 0 2048 14 all 128 0 all 300
bench mnist-255 "$bin/mnist" "$mn" 255 0 2048 14 all 128 0 all 1
bench mnist-100 "$bin/mnist" "$mn" 100 0 2048 14 all 128 0 all 1
bench mnist-1000 "$bin/mnist" "$mn" 1000 0 2048 14 all 128 0 all 1

echo "different: $fails"
[ $fails -eq 0 ]

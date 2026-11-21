#!/bin/bash
# The benchmarks of this build (see README.md beside this script). Builds the examples, runs every benchmark, strips
# the timings from what it printed, keeps its result lines and compares them with the result lines of the output saved
# in output/: "same" or "different" per benchmark, and the run time. Exits non-zero where any benchmark differs.
# Only stdout is compared: stderr holds traces, which are no part of the results.
# usage: benchmarks/run.sh [--js] [--record] [--only <name>] [--regions <R>] [--columns <C>] [--flag <flag>]...
#        [--out <dir>] [mnist-data-dir]
#   --js            run each benchmark's JS job in apps/ (node, calling the Rust core) instead of its Rust example
#   mnist-data-dir  the four raw (gunzipped) MNIST idx files; default apps/mnist/data in this repository, where
#                   they are downloaded from the mirror apps/mnist/jobs/download.js uses if they are not there
#   --record        write each run's stripped output as the saved output instead of comparing
#   --only <name>   run that benchmark alone
#   --regions <R> --columns <C>   lay out each benchmark's dispatch so; one region of one column by default
#   --flag <flag>   pass that flag to every benchmark, as `--flag --end-check=join`
#   --out <dir>     write each run's output there instead of beside this script, in got/
set -u
here="$(cd "$(dirname "$0")" && pwd)"
core="$(cd "$here/.." && pwd)"
repo="$(cd "$core/../.." && pwd)"
record=0; only=""; layout=(); js_layout=(); use_js=0; out=""; mn="$repo/apps/mnist/data"
while [ $# -gt 0 ]; do
  case "$1" in
    --js) use_js=1;;
    --record) record=1;;
    --only) only="$2"; shift;;
    --regions) layout+=("--regions=$2"); js_layout+=(--regions "$2"); shift;;
    --columns) layout+=("--columns=$2"); js_layout+=(--columns "$2"); shift;;
    --flag) layout+=("$2"); js_layout+=("$2"); shift;;
    --out) out="$2"; shift;;
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
if [ $use_js -eq 1 ]; then
  echo "building the Node addon"
  (cd "$core/.." && cargo build --release -p brain-napi 2>&1 | grep -E "^error" -A8) && exit 2
  cp "$core/../target/release/brain_napi.dll" "$core/../brain-napi/brain-napi.node" 2>/dev/null     || cp "$core/../target/release/libbrain_napi.so" "$core/../brain-napi/brain-napi.node"
else
  echo "building the examples"
  (cd "$core" && cargo build --release --examples 2>&1 | grep -E "^error" -A8) && exit 2
fi
bin="$core/../target/release/examples"
exp="$here/output"; got="${out:-$here/got}"
mkdir -p "$exp" "$got"

# the timings a run prints, taken out: milliseconds, seconds at the end of a line, a line that is a time alone
strip() { tr -d '\r' < "$1" | sed -E 's/[0-9]+(\.[0-9]+)? ?ms(\/image)?//g; s/ ?\|? [0-9]+(\.[0-9]+)? ?s$//; /^ *[0-9]+(\.[0-9]+)? ?s$/d'; }

# the result lines of a benchmark's output: every line but those of the diagnostics the build no longer prints, which
# each benchmark names in `drop` as one extended regular expression
declare -A drop edit
drop[text]='^(CLOSING|UNPAID) '
drop[text-without-space]='^(CLOSING|UNPAID) '
drop[six-bar-lap]='^(LIFTED TO THE STANDING CALL|LIFT PASSED OVER) '
for b in hline vline filled-square hollow-square two-and-three; do
  drop[$b]='^(BARE|CLOSING|UNPAID|LIFTED TO THE STANDING CALL|LIFT PASSED OVER|TWICE|CYCLELAPS) |^dictionary: [0-9]+ functions, [0-9]+ edges'
done
for b in mnist-1x60 mnist-10x20 mnist-10x300 mnist-255 mnist-100 mnist-1000; do
  drop[$b]='^(CLOSING|UNPAID|LIFTED TO THE STANDING CALL|LIFT PASSED OVER|TWICE|CYCLELAPS|JOINSMOVES) '
  edit[$b]='s/^(CHANGE .*) \| joins [0-9]+ \| moves [0-9]+$/\1/'
done
for b in kgc-gld two-six-bar-laps six-and-five-bar-laps three-stocks four-stocks three-stocks-5-laps ten-stocks; do
  drop[$b]='^(CLOSING|UNPAID|LIFTED TO THE STANDING CALL|LIFT PASSED OVER|CYCLELAPS|JOINSMOVES) '
done
# and where a result line also carried such a diagnostic, the benchmark's `edit` (one sed expression) takes it out
results() {
  local d="${drop[$1]:-}" e="${edit[$1]:-}"
  if [ -n "$d" ]; then grep -Ev "$d" "$2"; else cat "$2"; fi | if [ -n "$e" ]; then sed -E "$e"; else cat; fi
}

# each benchmark's JS job, run from the repository's root with --js: the job file and its arguments
declare -A js
js[alternation]='apps/text/jobs/alternation.js 33'
js[six-bar-lap]='apps/stocks/jobs/six-bar-lap.js 30 256'
js[kgc-gld]='apps/stocks/jobs/channels.js two 10'
js[two-six-bar-laps]='apps/stocks/jobs/channels.js lap 30'
js[six-and-five-bar-laps]='apps/stocks/jobs/channels.js lap65 12'
js[three-stocks]='apps/stocks/jobs/channels.js three 100'
js[four-stocks]='apps/stocks/jobs/channels.js three4 100 --all4'
js[three-stocks-5-laps]='apps/stocks/jobs/channels.js seq 5'
js[ten-stocks]='apps/stocks/jobs/channels.js seq 1 --symbols SO,VALE,STLD,GOOGL,MU,PLTR,UUUU,PFE,CRM,HAL --max-positions 3 --cost 0.02 --every 100'
js[tap]='apps/tap/jobs/tap.js 6 8'
js[text]='apps/text/jobs/text.js "the the the the the the the the the " 1 64'
js[text-without-space]='apps/text/jobs/text.js theathebthecthedthefthegtheithejthek 1 64'
js[hline]='apps/shapes/jobs/shapes.js hline 30'
js[vline]='apps/shapes/jobs/shapes.js vline 30'
js[filled-square]='apps/shapes/jobs/shapes.js square 30'
js[hollow-square]='apps/shapes/jobs/shapes.js hollow 30'
js[two-and-three]='apps/shapes/jobs/shapes.js teach23 30 0 -'
# node reads a Windows path where bash holds /c/...: the MNIST folder is passed in the form node reads
mn_js="$(cygpath -m "$mn" 2>/dev/null || echo "$mn")"
js[mnist-1x60]='apps/mnist/jobs/mnist.js "$mn_js" 1 0 2048 14 all 128 0 all 60'
js[mnist-10x20]='apps/mnist/jobs/mnist.js "$mn_js" 10 0 2048 14 all 128 0 all 20'
js[mnist-10x300]='apps/mnist/jobs/mnist.js "$mn_js" 10 0 2048 14 all 128 0 all 300'
js[mnist-255]='apps/mnist/jobs/mnist.js "$mn_js" 255 0 2048 14 all 128 0 all 1'
js[mnist-100]='apps/mnist/jobs/mnist.js "$mn_js" 100 0 2048 14 all 128 0 all 1'
js[mnist-1000]='apps/mnist/jobs/mnist.js "$mn_js" 1000 0 2048 14 all 128 0 all 1'

fails=0
printf "%-26s %-10s %s\n" "benchmark" "output" "time"
bench() {  # name, command...
  local name="$1"; shift
  [ -n "$only" ] && [ "$only" != "$name" ] && return
  local t0=$(date +%s%N)
  if [ $use_js -eq 1 ]; then
    [ -n "${js[$name]:-}" ] || { printf "%-26s %s
" "$name" "no JS job"; fails=$((fails + 1)); return; }
    (cd "$repo" && eval "node ${js[$name]}" '"${js_layout[@]}"') > "$got/$name.raw.out" 2> "$got/$name.raw.err"
  else
    (cd "$core" && "$@" "${layout[@]}") > "$got/$name.raw.out" 2> "$got/$name.raw.err"
  fi
  local code=$?
  local t1=$(date +%s%N)
  strip "$got/$name.raw.out" > "$got/$name.out"
  strip "$got/$name.raw.err" > "$got/$name.err"
  if [ $record -eq 1 ]; then cp "$got/$name.out" "$exp/$name.out"; cp "$got/$name.err" "$exp/$name.err"; fi
  local verdict="same"
  if [ $code -ne 0 ]; then verdict="failed($code)"; fi
  cmp -s <(results "$name" <(strip "$exp/$name.out")) <(results "$name" "$got/$name.out") || verdict="different"
  [ "$verdict" = "same" ] || fails=$((fails + 1))
  local ms=$(( (t1 - t0) / 1000000 ))
  printf "%-26s %-10s %d.%01ds\n" "$name" "$verdict" $((ms / 1000)) $(( (ms % 1000) / 100 ))
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

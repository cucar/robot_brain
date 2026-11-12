#!/bin/bash
# The fixed test set (run after every change): alternation, stocks, tap, text with and without the recurring
# space, the 2 and the 3 taught in turn, MNIST 10. Each test's key results are compared with the recorded ones.
# usage: testset/run.sh <mnist-data-dir> <target-dir> [--s15-off] [--record]
#   --s15-off   run with switch 15 off (expected results kept apart for each setting)
#   --record    write the results as the expected ones instead of comparing
set -u
here="$(cd "$(dirname "$0")" && pwd)"
data="$1"; tgt="$2"; shift 2
flag=""; set_name="s15-on"; record=0
for a in "$@"; do
  case "$a" in
    --s15-off) flag="--s15-off"; set_name="s15-off";;
    --record) record=1;;
  esac
done
exp="$here/expected/$set_name"; got="$here/got/$set_name"
mkdir -p "$exp" "$got"
(cd "$here/.." && cargo build --release --examples --target-dir "$tgt" 2>&1 | grep -E "^error" -A6)
bin="$tgt/release/examples"
row() { printf "| %s | %s | %s | %s | %s |\n" "$1" "$2" "$3" "$4" "$5"; }
# the short form shown in the table: each piece (separated by ' ;; ') read from the file on its own
summary() {
  local spec="$1" file="$2" out="" piece
  while [ -n "$spec" ]; do
    piece="${spec%% ;; *}"; [ "$piece" = "$spec" ] && spec="" || spec="${spec#* ;; }"
    out="$out $(bash -c "$piece" < "$file" | tr '\n' ' ')"
  done
  echo "$out" | sed 's/  */ /g; s/^ //; s/ $//'
}
run_test() {  # name, key extractor, short extractor, command...
  local name="$1" key="$2" short="$3"; shift 3
  local t0=$(date +%s.%N)
  "$@" > "$got/$name.out" 2> "$got/$name.err"
  local t1=$(date +%s.%N)
  bash -c "$key" < "$got/$name.out" > "$got/$name.key"
  local secs=$(echo "$t1 - $t0" | bc)
  if [ $record -eq 1 ]; then cp "$got/$name.key" "$exp/$name.key"; fi
  local e="(none)"; [ -f "$exp/$name.key" ] && e=$(summary "$short" "$exp/$name.key")
  local g=$(summary "$short" "$got/$name.key")
  local verdict="fail"; [ -f "$exp/$name.key" ] && cmp -s "$exp/$name.key" "$got/$name.key" && verdict="pass"
  row "$name" "$e" "$g" "$verdict" "$(printf '%.1fs' $secs)"
}
row "test ($set_name)" "expected" "got" "" "time"
row "---" "---" "---" "---" "---"
run_test alternation "grep -E '^f +[0-9]+|^right'" "grep -oE '^right [0-9]+/[0-9]+'  ;; grep -oE 'functions [0-9]+$'" "$bin/xy" 9 9 $flag
run_test stocks "grep -vE 'ms|took'" "grep -oE 'overall right [0-9]+/[0-9]+ = [0-9.]+%' | head -1 ;; grep -oE 'functions [0-9]+' | tail -1" "$bin/stocks" 6 256 6 $flag
run_test tap "grep -v ms" "grep -oE '^right [0-9]+/[0-9]+' ;; grep -oE '^dictionary: [0-9]+ functions'" "$bin/tap" 6 8 $flag
run_test text "grep -E '^right|patterns'" "grep -oE '^right [0-9]+/[0-9]+' ;; grep -oE 'patterns [0-9]+' | sed 's/patterns/functions/'" "$bin/text" "the the the the the the the the the " $flag
run_test text-without-space "grep -E '^right|patterns'" "grep -oE '^right [0-9]+/[0-9]+' ;; grep -oE 'patterns [0-9]+' | sed 's/patterns/functions/'" "$bin/text" "theathebthecthedthefthegtheithejthek" $flag
run_test two-and-three "grep -E '^  round|last changed|test, the|^  on the apex|^    f[0-9]+:|^      '" "grep -oE 'last changed at round [0-9]+' ;; grep -oE 'the [23] without its digit: says [0-9a-z]+' | sed 's/ without its digit://'" "$bin/shapes" teach23 30 0 - $flag
run_test mnist-10 "grep -E 'episode +[0-9]+:' | sed -E 's/\| [0-9]+ ms\/image//'" "grep -oE 'right [0-9]+/10' | grep -oE '[0-9]+/' | tr -d '/' ;; tail -1 | grep -oE 'neurons [0-9]+ functions [0-9]+'" "$bin/mnist" "$data" 10 0 2048 14 all 128 0 all 20 $flag

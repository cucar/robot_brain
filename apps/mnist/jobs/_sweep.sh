#!/bin/bash
# Quick parameter sweep on N=20 to find best config.
# Output: one line per config with label / event accuracy.
cd "$(dirname "$0")/../../.."

run() {
	local desc=$1
	shift
	local out
	out=$(node apps/mnist/jobs/implant_test.js --max-images 20 "$@" 2>&1 | tail -1)
	printf '%-50s %s\n' "$desc" "$out"
}

echo "=== Baseline ==="
run "default"

echo "=== Merge thresholds ==="
for mt in 0.5 0.7 0.9 0.95; do run "merge=$mt" --merge-threshold $mt; done

echo "=== Error modes (threshold=0.5) ==="
for em in static conservative neutral aggressive; do run "errMode=$em" --error-mode $em; done

echo "=== Error thresholds (static) ==="
for et in 0.1 0.3 0.5 0.7; do run "static, errThresh=$et" --error-mode static --error-threshold $et; done

echo "=== Forget rates ==="
for fr in 0 0.0001 0.001 0.01; do run "forget=$fr" --forget-rate $fr; done

echo "=== Epochs ==="
for e in 1 2 3 4; do run "epochs=$e" --epochs $e; done

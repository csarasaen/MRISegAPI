#!/usr/bin/env bash
# health-check.sh — verify all pods in the MRI segmentation deployment are ready
set -euo pipefail

NAMESPACE="${1:-default}"
DEPLOYMENT="mri-seg"
RETRIES=10
SLEEP=6

echo "Checking deployment: $DEPLOYMENT in namespace: $NAMESPACE"

for i in $(seq 1 $RETRIES); do
  READY=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" \
    -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
  DESIRED=$(kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" \
    -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "?")

  echo "[$i/$RETRIES] Ready: $READY / $DESIRED"

  if [[ "$READY" == "$DESIRED" && "$READY" != "0" ]]; then
    echo "✓ Deployment healthy"
    exit 0
  fi

  sleep "$SLEEP"
done

echo "✗ Deployment not healthy after $((RETRIES * SLEEP))s"
exit 1

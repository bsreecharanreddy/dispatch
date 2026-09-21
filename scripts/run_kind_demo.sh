#!/usr/bin/env bash
# Brings up the K8s side of Phase 7's demo: a kind cluster with the
# router, Prometheus, and Grafana. Assumes something is already
# listening on the dev machine's port 50051 for the model-server
# ExternalName Service to reach -- the stub responder for this task's
# rehearsal (`uv run python scripts/run_model_server.py --responder stub
# --port 50051`, in another terminal), or the real SSH tunnel for
# Task 13's paid session. This script's own job is K8s bring-up only.
set -euo pipefail

CLUSTER_NAME="dispatch-demo"
NAMESPACE="dispatch-demo"

echo "[1/6] Creating kind cluster ${CLUSTER_NAME} (if not already up)"
if ! kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  kind create cluster --name "${CLUSTER_NAME}"
fi
kubectl config use-context "kind-${CLUSTER_NAME}"

echo "[2/6] Building and loading images into the kind cluster"
docker build -t dispatch-router:demo -f docker/router.Dockerfile .
docker build -t dispatch-model-server:demo -f docker/model-server.Dockerfile .
kind load docker-image dispatch-router:demo --name "${CLUSTER_NAME}"
kind load docker-image dispatch-model-server:demo --name "${CLUSTER_NAME}"

echo "[3/6] Applying manifests"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/router-deployment.yaml
kubectl apply -f k8s/model-server-external.yaml
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

echo "[4/6] Waiting for rollouts"
kubectl -n "${NAMESPACE}" rollout status deployment/router --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/prometheus --timeout=120s
kubectl -n "${NAMESPACE}" rollout status deployment/grafana --timeout=120s

echo "[5/6] Port-forwarding router (8080) and Grafana (3000) to localhost"
kubectl -n "${NAMESPACE}" port-forward svc/router 8080:8080 &
ROUTER_PF_PID=$!
kubectl -n "${NAMESPACE}" port-forward svc/grafana 3000:3000 &
GRAFANA_PF_PID=$!
trap 'kill ${ROUTER_PF_PID} ${GRAFANA_PF_PID} 2>/dev/null || true' EXIT
sleep 3

echo "[6/6] Ready."
echo "  Demo request: curl -s -X POST localhost:8080/generate -H 'content-type: application/json' -d '{\"prompt\":\"The quick brown fox\",\"max_new_tokens\":16}'"
echo "  Grafana: http://localhost:3000 (anonymous admin)"
echo "Press Ctrl-C to stop the port-forwards. Tear down the cluster separately with:"
echo "  kind delete cluster --name ${CLUSTER_NAME}"
wait

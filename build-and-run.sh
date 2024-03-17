#!/usr/bin/env bash

set -xe

kubectl delete -f job.yaml || true

export PROJECT_ID=$(gcloud config get project)
export TAG=${TAG:-latest}
export IMAGE=gcr.io/${PROJECT_ID}/gemma-finetune:${TAG}
export BUILD_IMAGE=${BUILD_IMAGE:-yes}

if [ "${BUILD_IMAGE}" = "yes" ]; then
  docker build --platform linux/amd64 -t ${IMAGE} .
  docker push ${IMAGE}
fi

envsubst < job.yaml | kubectl apply -f -
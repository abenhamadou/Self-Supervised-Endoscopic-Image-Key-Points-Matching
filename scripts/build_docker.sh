#!/bin/sh
cd ..
set -e

VERSION=$(awk '/__version__ =/{print $3}' _version.py | sed "s/['\"]//g")
IMAGE_NAME="self_supervised_matching"

docker build --no-cache -t ${IMAGE_NAME}:${VERSION} -f Dockerfile .
docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY_HOSTNAME}/${IMAGE_NAME}:${VERSION}

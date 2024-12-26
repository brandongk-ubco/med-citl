#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

python -m citl standardtrain PumaTissue efficientnet-b0 \
    "--augmentation-policy-path=./policies/noop.yaml" \
    "--lr-method=plateau"

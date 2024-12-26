#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

python -m citl train PumaTissue efficientnet-b0 \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--selectively-backpropagate" \
    "--alpha=0.10" \
    "--lr-method=plateau" \
    "--method=score"

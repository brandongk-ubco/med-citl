#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

python -m citl standardtrain PumaTissue efficientnet-b0 \
    "--augmentation-policy-path=./policies/pumatissue.yaml" \
    "--lr-method=plateau"


# python -m citl train PumaTissue efficientnet-b0 \
#     "--augmentation-policy-path=./policies/pumatissue.yaml" \
#     "--selectively-backpropagate" \
#     "--alpha=0.03" \
#     "--lr-method=plateau" \
#     "--method=score"
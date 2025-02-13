#!/usr/bin/env bash

set -eux

# rm -rf lightning_logs
# rm .*.ckpt || true

python -m citl train PumaTissue hf-hub:MahmoodLab/UNI2-h \
    "--augmentation-policy-path=./policies/pumatissue.yaml" \
    "--lr-method=plateau"

# numbers=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
# for alpha in "${numbers[@]}"
# do

#     python -m citl train PumaTissue mit_b4 \
#         "--augmentation-policy-path=./policies/pumatissue.yaml" \
#         "--selectively-backpropagate" \
#         "--alpha=${alpha}" \
#         "--lr-method=plateau" \
#         "--method=score"
# done
#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

python -m citl standardtrain PumaTissue mit_b4  \
    "--augmentation-policy-path=./policies/pumatissue.yaml" \
    "--loss-function=cross_entropy" \
    "--margin-weighting" \
    "--lr-method=plateau" \
    "--no-pretrained"

python -m citl standardtrain PumaTissue mit_b4 \
    "--augmentation-policy-path=./policies/pumatissue.yaml" \
    "--loss-function=cross_entropy" \
    "--no-margin-weighting" \
    "--lr-method=plateau" \
    "--no-pretrained"

python -m citl standardtrain PumaTissue mit_b4 \
    "--augmentation-policy-path=./policies/pumatissue.yaml" \
    "--loss-function=focal" \
    "--no-margin-weighting" \
    "--lr-method=plateau" \
    "--no-pretrained"

# METHOD ALPHA SWEEP
alphas=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

for alpha in "${alphas[@]}"
do
    for level in "${levels[@]}"
    do

        python -m citl train PumaTissue mit_b4 \
            "--augmentation-policy-path=./policies/pumatissue.yaml" \
            "--alpha=${alpha}" \
            "--loss-function=cross_entropy" \
            "--lr-method=plateau" \
            "--selectively-backpropagate" \
            "--no-pretrained"
    
    done
done



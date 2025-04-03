#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

python -m citl standardtrain CityscapesFine efficientnet-b0  \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--loss-function=cross_entropy" \
    "--margin-weighting" \
    "--lr-method=plateau" \
    "--pretrained"

python -m citl standardtrain CityscapesFine efficientnet-b0 \
    "--augmentation-policy-path=./policies/pumatcityscapesissue.yaml" \
    "--loss-function=cross_entropy" \
    "--no-margin-weighting" \
    "--lr-method=plateau" \
    "--pretrained"

python -m citl standardtrain CityscapesFine efficientnet-b0 \
    "--augmentation-policy-path=./policies/cityscapes.yaml" \
    "--loss-function=focal" \
    "--no-margin-weighting" \
    "--lr-method=plateau" \
    "--pretrained"

# METHOD ALPHA SWEEP
alphas=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

for alpha in "${alphas[@]}"
do
    for level in "${levels[@]}"
    do

        python -m citl train PumaTissuCityscapesFinee efficientnet-b0 \
            "--augmentation-policy-path=./policies/cityscapes.yaml" \
            "--alpha=${alpha}" \
            "--loss-function=cross_entropy" \
            "--lr-method=plateau" \
            "--selectively-backpropagate" \
            "--pretrained"
    
    done
done



#!/usr/bin/env bash

set -eux

rm -rf lightning_logs
rm .*.ckpt || true

# Train the three baselines
levels=(0.0 0.1 0.2 0.3 0.4 0.5)
for level in "${levels[@]}"
do

    python -m citl standardtrain CIFAR10UB mnasnet_small \
        "--augmentation-policy-path=./policies/cifar10.yaml" \
        "--noise-level=${level}" \
        "--loss-function=cross_entropy" \
        "--margin-weighting" \
        "--lr-method=plateau" \
        "--no-pretrained"

    python -m citl standardtrain CIFAR10UB mnasnet_small \
        "--augmentation-policy-path=./policies/cifar10.yaml" \
        "--noise-level=${level}" \
        "--loss-function=cross_entropy" \
        "--no-margin-weighting" \
        "--lr-method=plateau" \
        "--no-pretrained"

    python -m citl standardtrain CIFAR10UB mnasnet_small \
        "--augmentation-policy-path=./policies/cifar10.yaml" \
        "--noise-level=${level}" \
        "--loss-function=focal" \
        "--no-margin-weighting" \
        "--lr-method=plateau" \
        "--no-pretrained"

done

# METHOD ALPHA SWEEP
alphas=(0.10 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19)

for alpha in "${alphas[@]}"
do
    for level in "${levels[@]}"
    do

        python -m citl train CIFAR10UB mnasnet_small \
            "--augmentation-policy-path=./policies/cifar10.yaml" \
            "--noise-level=${level}" \
            "--alpha=${alpha}" \
            "--loss-function=cross_entropy" \
            "--lr-method=plateau" \
            "--selectively-backpropagate" \
            "--no-pretrained"
    
    done
done



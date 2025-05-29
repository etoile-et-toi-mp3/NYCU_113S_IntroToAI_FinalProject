#!/usr/bin/zsh

backbones=("mobilenet" "resnet18" "resnet50" "efficientnet_b0" "efficientnet_b2" "vit_tiny" "vit_small" "fashion_resnet")

for backbone in $backbones; do
    echo "Training with backbone: $backbone"
    python3 supervised_train.py --data ../cdb_dataset/ --config performance --backbone $backbone --platform cuda
done

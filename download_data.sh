#!/bin/bash
# Download a CycleGAN dataset from the official Berkeley mirror.
# Usage: ./download_data.sh horse2zebra
# Available: apple2orange, summer2winter_yosemite, horse2zebra,
#            monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo,
#            maps, cityscapes, facades, iphone2dslr_flower

set -e
NAME=${1:-horse2zebra}
URL="https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/${NAME}.zip"
mkdir -p datasets
ZIP="datasets/${NAME}.zip"

echo "Downloading $NAME from $URL"
curl -L -o "$ZIP" "$URL"
unzip -q -o "$ZIP" -d datasets/
rm "$ZIP"
echo "done. dataset at datasets/${NAME}/{trainA,trainB,testA,testB}"

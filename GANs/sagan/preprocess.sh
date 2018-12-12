#!/bin/bash
IMAGENET_PATH=$1
SAVE_PATH=$2
IMAGENET_PATH_VAL=$3
SAVE_PATH_VAL=$4

# Preprocess training data
mkdir -p $SAVE_PATH
for dir in `find $IMAGENET_PATH -type d -maxdepth 1 -mindepth 1`; do
   echo $dir
   mkdir -p ${SAVE_PATH}/${dir##*/} 
   for name in ${dir}/*.JPEG; do
      w=`identify -format "%w" $name`
      h=`identify -format "%h" $name`
      if [ $w -ge 256 ] && [ $h -ge 256 ]; then
          convert -resize 256x256^ -quality 95 -gravity center -extent 256x256 $name ${SAVE_PATH}/${dir##*/}/${name##*/}
      fi
   done
done

# Preprocess validation data
if [ -n "$IMAGENET_PATH_VAL" ] && [ -n "$SAVE_PATH_VAL" ]; then
    mkdir -p $SAVE_PATH_VAL
    for name in ${IMAGENET_PATH_VAL}/*.JPEG; do
       echo $name
       convert -resize 256x256^ -quality 95 -gravity center -extent 256x256 $name ${SAVE_PATH_VAL}/${name##*/}
    done
fi

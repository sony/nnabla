#!/bin/bash

image_files=
for i in $@;
  do image_files=${image_files}$(cat $i);
done

echo ${image_files} | md5sum | cut -d \  -f 1

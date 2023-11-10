#!/bin/bash

rm -rf $1/doc && mkdir -p $1/doc/doxygen
cd $2 && \
	cat build-tools/doxygen/config >Doxyfile && \
	echo OUTPUT_DIRECTORY  = $1/doc/doxygen >>Doxyfile && \
	doxygen && rm -f Doxyfile
mv $1/doc/doxygen/html $1/doc/html-Cpp && mv $1/doc/doxygen/xml $1/doc/xml-Cpp
rm -rf $1/doc/doxygen
if [ -d $3 ]; then \
	mkdir -p $1/doc/doxygen; \
	cd $3 && \
	cat $2/build-tools/doxygen/config_ext_cuda >Doxyfile && \
	echo OUTPUT_DIRECTORY  = $1/doc/doxygen >>Doxyfile && \
	doxygen && rm -f Doxyfile; \
	mv $1/doc/doxygen/html $1/doc/html-Ext-Cuda-Cpp && mv $1/doc/doxygen/xml $1/doc/xml-Ext-Cuda-Cpp; \
	rm -rf $1/doc/doxygen; \
fi
cd $2 && rm -rf $2/doc/cpp/Cpp $2/doc/cpp/Ext-Cuda-Cpp
if [ -d $3 ]; then \
	python $2/doc/dox_to_rst.py 1; else \
	python $2/doc/dox_to_rst.py 0; \
fi

#!/bin/bash

ORGDIR=$1
TARGET=$2

dirname=${ORGDIR##*/}

for filename in $ORGDIR/* ; do
	name=${filename##*/}
	newname="${dirname}_${name}"
	cp $ORGDIR/$name $TARGET/$newname
done

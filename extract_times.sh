#!/bin/bash
cat $1 | grep Attack | sed -r 's/^.*in ([0-9][0-9]*.[0-9][0-9][0-9][0-9]) seconds/\1/g'

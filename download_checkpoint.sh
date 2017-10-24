#!/bin/bash

test -d ckpts || mkdir ckpts
cd ckpts
wget http://download.tensorflow.org/models/speech_commands_v0.01.zip
unzip speech_commands_v0.01.zip
rm -f speech_commands_v0.01.zip

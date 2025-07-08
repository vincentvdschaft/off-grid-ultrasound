#!/bin/bash


wget https://zenodo.org/api/records/14925758/files-archive -O data/archive.zip
unzip data/archive.zip -d data
rm data/archive.zip
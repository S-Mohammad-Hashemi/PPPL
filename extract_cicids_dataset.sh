#!/bin/bash

cat datasets/CICIDS2017_packet-based/tuesday.zip_parta* > datasets/CICIDS2017_packet-based/tuesday.zip
cat datasets/CICIDS2017_packet-based/wednesday.zip_parta* > datasets/CICIDS2017_packet-based/wednesday.zip

unzip datasets/CICIDS2017_packet-based/tuesday.zip -d datasets/CICIDS2017_packet-based
unzip datasets/CICIDS2017_packet-based/wednesday.zip -d datasets/CICIDS2017_packet-based
unzip datasets/CICIDS2017_packet-based/thursday.zip -d datasets/CICIDS2017_packet-based


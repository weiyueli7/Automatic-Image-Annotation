#!/bin/bash
nohup python3 main.py
wait
nohup python3 main.py RNN
wait
nohup python3 main.py ARCH2
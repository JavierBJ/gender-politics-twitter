#!/bin/bash

code="t mr"

for n in $(seq 1 13);
do
    for c in ${code}
    do
        echo $n $c
        python3 dump_to_csv.py $n $c
    done
done

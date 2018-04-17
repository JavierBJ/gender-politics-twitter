#!/bin/bash

#num=1 2 3 4 5 6 7 8 9 10 11 12 13
code="t mr"
is_users="1 0"

for n in $(seq 1 13);
do
    for c in ${code}
    do
        for u in ${is_users}
        do
            echo $n $c $u
            python3 clean_csv.py $n $c $u
        done
    done
done

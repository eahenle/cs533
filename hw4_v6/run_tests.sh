#!/bin/bash

# make sure code is up-to-date
git pull

# run tests on {1, 2, 4, 8} collectors w/ 2 evaluators, {1, 2, 4, 8} evaluators w/ 4 collectors

for nb_collectors in 1, 2, 4, 8
do
    ./distributed_dqn.py $nb_collectors 2 > $nb_collectors-2.txt 2>&1 &
done

for nb_evaluators in 1, 4, 8
do
    ./distributed_dqn.py 4 $nb_evaluators > 4-$nb_evaluators.txt 2>&1 &
done

watch ps

#!/bin/bash

for i in {1..16}
do
  python map_reduce.py $i > output_$i.out
done

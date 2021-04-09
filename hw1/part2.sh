#!/bin/bash

for i in {1..16}
do
  python map_reduce $i > output_$i.out
done

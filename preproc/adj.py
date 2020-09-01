#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

adj_dict = {}

with open("data/adjacency.txt") as f:
    for line in f.readlines():
        line = line.strip("\n")
        data = line.split("\t")
        if data[0] != "":
            cur_key = data[3]
            if cur_key not in adj_dict.keys():
                adj_dict[cur_key] = []
            else:
                print(data)
        else:
            adj_dict[cur_key].append(data[3])

print(adj_dict)
            
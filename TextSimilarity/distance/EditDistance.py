#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
import numpy

def get_distance(source, target):
    if len(source) < len(target):
        return get_distance(target, source)

        # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

        # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = numpy.array(tuple(source))
    target = numpy.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = numpy.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = numpy.minimum(
            current_row[1:],
            numpy.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = numpy.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def get_sim(sent1,sent2):
    dis=get_distance(sent1,sent2)
    sim=1-(dis-1)/max(len(sent1),len(sent2))
    if sim<=0:
        sim=0
    return sim
if __name__ == '__main__':
    sent1="我们都是好孩子"
    sent2="好孩子我们都是"
    print(get_distance(sent1, sent2))
    print(get_sim(sent1, sent2))
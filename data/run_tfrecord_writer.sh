#!/bin/bash
name_1=Movies_Books
name_2=Books

cat ${name_1}_train.csv | python tfrecord_writer.py ${name_1}_train
cat ${name_1}_val.csv | python tfrecord_writer.py ${name_1}_val
cat ${name_1}_test.csv | python tfrecord_writer.py ${name_1}_test
cat ${name_2}_train.csv | python tfrecord_writer.py ${name_2}_train
cat ${name_2}_val.csv | python tfrecord_writer.py ${name_2}_val
cat ${name_2}_test.csv | python tfrecord_writer.py ${name_2}_test

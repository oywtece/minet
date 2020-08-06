#!/bin/bash
name_1=Movies_Books
name_2=Books

cat ${name_1}_train.csv | python tfrecord_writer.py ${name_1}_train 25
cat ${name_1}_val.csv | python tfrecord_writer.py ${name_1}_val 25
cat ${name_1}_test.csv | python tfrecord_writer.py ${name_1}_test 25
cat ${name_2}_train.csv | python tfrecord_writer.py ${name_2}_train 715
cat ${name_2}_val.csv | python tfrecord_writer.py ${name_2}_val 715
cat ${name_2}_test.csv | python tfrecord_writer.py ${name_2}_test 715


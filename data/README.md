Run the run_tfrecord_writer.sh script to convert csv data to tfrecord data.\
(Zipped tfrecord data files are too large to be uploaded.)

```
$ nohup bash run_tfrecord_writer.sh &
```

scripts run much faster with tfrecord data files [than with csv data files]

tfrecord_writer.py takes 2 args: 1-file_name, 2-num_of_csv_cols

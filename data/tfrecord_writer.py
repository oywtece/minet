import sys
import tensorflow as tf 

#########################
# required
# label_col_idx = 0
#########################
file_name = sys.argv[1]
num_csv_col = int(sys.argv[2])

# Converting the values into features
# _int64 is used for numeric values (e.g., label)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

#########################
# write to tfrecord
#########################
tfrecord_filename = file_name + '.tfrecord'

# Initiating the writer and creating the tfrecord file
writer = tf.python_io.TFRecordWriter(tfrecord_filename)

for line in sys.stdin:
    line = line.strip('\n')
    # to list
    line_split = line.split(',')
    int_line = [int(float(i)) for i in line_split]
    label = [int_line[0]]
    data = int_line[1:]
    feature = {'label':_int64_feature(label), 'data':_int64_feature(data)}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Writing the serialized example.
    writer.write(example.SerializeToString())

writer.close()


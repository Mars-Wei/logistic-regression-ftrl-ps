#!/bin/bash

#hdfs_dir=dmlc

#${HADOOP_HOME}/bin/hadoop fs -rm -r -f $hdfs_dir/data
#${HADOOP_HOME}/bin/hadoop fs -mkdir $hdfs_dir/data
#${HADOOP_HOME}/bin/hadoop fs -put ../data/agaricus.txt.train $hdfs_dir/data
#${HADOOP_HOME}/bin/hadoop fs -put ../data/agaricus.txt.test $hdfs_dir/data

#cat <<< "
#train_data = \"hdfs://${hdfs_dir}/data/agaricus.txt.train\"
#val_data = \"hdfs://${hdfs_dir}/data/agaricus.txt.test\"
#max_data_pass = 3
#" >guide/demo_hdfs.conf

#./dmlc_mpi.py -n 3 -s 1 -H hosts /home/worker/xiaoshu/logistic-regression-ftrl-ps/lr_ftrl_ps /home/worker/xiaoshu/logistic-regression-ftrl-ps/data/v2v_train /home/worker/xiaoshu/logistic-regression-ftrl-ps/data/v2v_test 
./dmlc_mpi.py -n 3 -s 1 -H hosts /home/worker/xiaoshu/logistic-regression-ftrl-ps/lr_ftrl_ps /home/worker/xiaoshu/logistic-regression-ftrl-ps/data/n2n_train /home/worker/xiaoshu/logistic-regression-ftrl-ps/data/n2n_test 

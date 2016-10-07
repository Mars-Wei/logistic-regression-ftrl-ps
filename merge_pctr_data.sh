cat pred_0.txt >> pred.txt
cat pred_1.txt >> pred.txt
cat pred_1.txt >> pred.txt
cp pred.txt ../AUC-caculate-mpi/data/
cd ../AUC-caculate-mpi/data/
sh run_split_data.sh pred.txt
cd ../
sh run.sh

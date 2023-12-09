for bs in 1 2 4 8 16 32 64
do
  mpiexec -n 4 python data_parallelism_train.py --nb-proc 4 --batch-size $bs
done
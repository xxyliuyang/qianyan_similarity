source activate allennlp

device=(1 2 3)
dataname=("bq_corpus" "lcqmc" "paws-x-zh")
exp_name=albert_base

for((i=0; i<${#device[@]}; i++))
do
  PYTHONPATH=./ python run.py ${device[i]} ${dataname[i]} ${exp_name}&
done

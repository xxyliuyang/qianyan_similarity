
device=(1 2 3)
dataname=("bq_corpus" "lcqmc" "paws-x-zh")

for((i=0; i<${#device[@]}; i++))
do
  PYTHONPATH=./ python run.py ${device[i]} ${dataname[i]}&
done

jobn=5

for var in $(seq $jobn)
do
  let idx=`expr $var - 1`
  echo  "$idx"
  nohup python train_my_grid.py  $idx  $jobn > log_MyGraphSearch-$idx-$jobn.log 2>&1 &
done
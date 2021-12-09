jobn=5

for var in $(seq $jobn)
do
  let idx=`expr $var - 1`
  echo  "$idx"
  nohup python lstm4stock.py  $idx  $jobn > log_MyLSTMSearch-$idx-$jobn.log 2>&1 &
done

array=(1 3 5 7 9)

for p in ${array[@]}
do
    q=0
    echo  "$p  $q"
    nohup python ARIMA4stock.py  $p  $q > log_ARIMA4stock-p=$p-q=$q.log 2>&1 &
    for q in $(seq $p)
    do
  #    let idx=`expr $var - 1`
      echo  "$p  $q"
      nohup python ARIMA4stock.py  $p  $q > log_ARIMA4stock-p=$p-q=$q.log 2>&1 &
    done
done
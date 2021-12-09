jobn=$1

for var in $(seq $jobn)
do
  let idx=`expr $var - 1`
#  echo  "$idx"
#  python entity_node2stock_node.py $idx $jobn
  nohup python entity_node2stock_node.py  $idx  $jobn > log_eNode2sNode-$idx-$jobn.log 2>&1 &
done

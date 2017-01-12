Seek()
{
  ISGPU=false
  ISMULTIGPU=false

  GET_GPU=`cat $FILE | grep "CUDA :" | cut -d ':' -f2 | sed -e 's/ //g'`
  GET_MULT=`cat $FILE | grep "MultiGPU :" | cut -d ':' -f2 | sed -e 's/ //g'`

  if [ $GET_GPU = "On" ]; then
    ISGPU=true
  fi

  if [ $GET_MULT = "On" ]; then
    ISMULTIGPU=true
  fi

  TOTAL=`cat $FILE | grep "time" | head -1 | sed -e 's/[^0-9.]//g'`
  CONV=`cat $FILE | grep "converge"`
  LOOP=`cat $FILE | grep "loop" | sed -e 's/[^0-9]//g'`
  CPUDOT=`cat $FILE | grep "Dot" | head -1 | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`
  CPUDOT=`cat $FILE | grep "Dot" | head -1 | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`
  CPUMV=`cat $FILE | grep "MV" | head -1 | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`

  if [ $ISGPU = true ]; then
    GPUDOT=`cat $FILE | grep "Dot" | tail -1 | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`
    GPUMV=`cat $FILE | grep "MV" | tail -1 | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`
    GPUH2D=`cat $FILE | grep "H2D time" | cut -d ',' -f1 | sed -e 's/[^0-9.]//g' | cut -c 2-`
    GPUD2H=`cat $FILE | grep "D2H time" | cut -d ',' -f1 | sed -e 's/[^0-9.]//g' | cut -c 2-`
    GPUMEM=`cat $FILE | grep "Memset" | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`
fi
  OTHER=`cat $FILE | grep "other" | cut -d ',' -f1 | sed -e 's/[^0-9.]//g'`

  echo "Total, $TOTAL"
  echo "Converge, $CONV"
  echo "Loop, $LOOP"
  echo "CPU-Dot, $CPUDOT"
  echo "CPU-MV, $CPUMV"

  if [ $ISGPU = true ]; then
    echo "GPU-Dot, $GPUDOT"
    echo "GPU-MV, $GPUMV"
    echo "GPU-H2D, $GPUH2D"
    echo "GPU-D2H, $GPUD2H"
    echo "GPU-Memset, $GPUMEM"
  fi
  echo "Other, $OTHER"
  echo ""

}

CSV()
{
  echo "Data to CSV"
  DIR=$1

  NAME_CPU=$2
  CPU_ID=$3
  
  NAME_GPU=$4
  GPU_ID=$5

  NAME_GPUX=$6
  GPUX_ID=$7

  OUTPUT=$8

  if [ ! $NAME_CPU = "NONE" ]; then
    echo "$NAME_CPU CSV"
    FILE=$DIR/$NAME_CPU-$CPU_ID-log.txt
    Seek >> $OUTPUT 
  fi

  if [ ! $NAME_GPU = "NONE" ]; then
    echo "$NAME_GPU CSV"
    FILE=$DIR/$NAME_GPU-$GPU_ID-log.txt
    Seek >> $OUTPUT 
  fi

  if [ ! $NAME_GPUX = "NONE" ]; then
    echo "$NAME_GPUX CSV"
    FILE=$DIR/$NAME_GPUX-$GPUX_ID-log.txt
    Seek >> $OUTPUT 
  fi

}

#CSV ./TEST cpu 1 NONE 1 NONE 1 ./test2.csv

#!bin/zsh

. ./data2csv.sh --source-only

#   -N, --L1_Dir_Name        Default Matrix files Name (string [=Matrix])
#   -p, --CRS_Path           CRS format files path (string [=../])
#   -d, --CRS_Dir_Name       Default CRS format files Name (string [=CRS])
#   -m, --CRS_Matrix_Name    CRS Matrix Name (string [=NONE])
#   -P, --MM_Path            MatrixMarket files path (string [=../])
#   -D, --MM_Dir_Name        Default MM format files Name (string [=MM])
#   -M, --MM_Matrix_Name     MM Matrix Name (string [=NONE])
#   -t, --Openmp_Thread      Threads use in OpenMP (int [=8])
#   -S, --OuterSolver        method use in outersolver (string)
#   -L, --OuterMaxLoop       max loops in outersolver (int [=10000])
#   -E, --OuterEps           eps in outersolver (double [=1e-08])
#   -R, --OuterRestart       Restart number in outersolver (int [=1000])
#   -K, --OuterKskip         kskip number in outersolver (int [=2])
#   -F, --OuterFix           fix bug in outersolver (int [=2])
#   -s, --InnerSolver        method use in innersolver (string [=NO])
#   -l, --InnerMaxLoop       max loops in innersolver (int [=50])
#   -e, --InnerEps           eps in innersolver (double [=0.1])
#   -r, --InnerRestart       Restart number in innersolver (int [=1000])
#   -k, --InnerKskip         kskip number in innersolver (int [=2])
# -f, --InnerFix           fix bug in innersolver (int [=2])
#   -v, --verbose            verbose mode will printout all detel
#   -c, --cuda               cuda
#   -x, --pinned             use pinned memory
#   -g, --multiGPU           use multiGPU in CUDA
#   -?, --help               print this message

Avg()
{
  A_DIR=$1
  A_FILE_in=$2
  A_FILE_out=$3
  cat $A_DIR/$A_FILE_in.txt | awk '{tmp+=$1} END{print tmp/NR;}' >> $A_DIR/$A_FILE_out.txt
}

PickUp()
{
  C_DIR=$1
  C_FILE_in=$2
  C_FILE_out=$3
  cat $C_DIR/$C_FILE_in-log.txt| grep "time" | head -1 | sed -e 's/[^0-9.]//g' >> $C_DIR/$C_FILE_out.txt
}

Onece()
{
  O_DIR=$1
  O_FILE=$2

  rm -f ./output/*

  ./Solver -N $o_N -p $o_p -d $o_d -m $o_m -p $o_p -D $o_D -M $o_M -t $o_t -S $o_S -L $o_L -E $o_E -R $o_R -K $o_K -F $o_F -s $o_s -l $o_l -e $o_e -r $o_r -k $o_k -f $o_f $option > $O_DIR/$O_FILE-log.txt

  mv ./output/${o_S^^}_his.txt $O_DIR/$O_FILE-his.txt

}

BeseOne()
{
  B_DIR=$1
  B_FILE=$2
  cat $B_DIR/$B_FILE.txt | awk '{if(m<$1){m=$1;l=NR}} END{print l}'
}

MakeDir()
{
  O_DIR=$1
  if [ ! -e $O_DIR ]; then
    mkdir -p $O_DIR
    echo "mkdir $O_DIR"
  fi
}

Loop()
{
  OUTPUTFILE=$1
  SUMFILE=$2
  option=$3
  
  echo "$OUTPUTFILE Start"
  for i in `seq 1 $LOOP`
  do
    Onece $DIR $OUTPUTFILE-$i
    if [ $LOOP -gt 1 ]; then
      PickUp $DIR $OUTPUTFILE-$i $SUMFILE
    fi
    echo "$i times"
  done
  if [ $LOOP -gt 1 ]; then
    Get_Best=`BeseOne $DIR $SUMFILE`
    echo "Best => $Get_Best"
  else
    Get_Best=1
  fi
}
#--------------------------------------------
o_N="Matrix"
o_p="../"
o_d="CRS"
o_m="consph"
o_P="../" #Dont use
o_D="MM" #Dont use
o_M="NONE" #Dont use
o_t=8

o_S="cg"
o_L=20000
o_E=1e-8
o_R=1000
o_K=2
o_F=2

#o_s="kskipbicg"
o_s=""
o_l=50
o_e=1e-1
o_r=50
o_k=2
o_f=2

option=""

RESULTFILE="result"
CSV_ROOT="./CSV"
MakeDir $CSV_ROOT


#============================================
o_m="bcsstk17"
ROOT_DIR="./$o_m-test-part1"
MakeDir $ROOT_DIR


#======================================================== vp gcr=================================================

#============================================
LOOP=2
#============================================
#========VPGCR_CG========
o_S="vpcg"
o_s="cg"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_CG========

#============================================
LOOP=2
#============================================
#========VPGCR_CR========
o_S="vpcg"
o_s="cr"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_CR========

#============================================
LOOP=2
#============================================
#========VPGCR_GCR========
o_S="vpcg"
o_s="gcr"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_GCR========

#============================================
LOOP=2
#============================================
#========VPGCR_BICG========
o_S="vpcg"
o_s="bicg"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_BICG========

#============================================
LOOP=2
#============================================
#========VPGCR_GMRES========
o_S="vpcg"
o_s="gmres"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_GMRES========

#============================================
LOOP=2
#============================================
#========VPGCR_KSKIPCG========
o_S="vpcg"
o_s="kskipcg"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_KSKIPCG========

#============================================
LOOP=2
#============================================
#========VPGCR_KSKIPBICG========
o_S="vpcg"
o_s="kskipbicg"

DIR="./${o_S^^}_${o_s^^}_TEST"
MakeDir $DIR
CSV_FILE="$CSV_ROOT/$o_m-${o_S^^}-${o_s^^}.csv"

#===========

Loop cpu sum_cpu ""
B_CPU=$Get_Best

Loop gpu sum_gpu "-c -x"
B_GPU=$Get_Best

Loop gpux sum_gpux "-c -x -g"
B_GPUX=$Get_Best

CSV $DIR cpu $B_CPU gpu $B_GPU gpux $B_GPUX $CSV_FILE
mv $DIR $ROOT_DIR/
#========VPGCR_KSKIPBICG========






echo "Done"
#Avg $DIR $SUMFILE $RESULTFILE

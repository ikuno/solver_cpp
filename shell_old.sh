#!bin/zsh

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
  C_DIR=$1
  C_FILE_1=$2
  C_FILE_2=$3
  cat $C_DIR/$C_FILE_1.txt | awk '{tmp+=$1} END{print tmp/NR;}' >> $C_DIR/$C_FILE_2.txt
}

Count()
{
  C_DIR=$1
  C_FILE_1=$2
  C_FILE_2=$3
  cat $C_DIR/$C_FILE_1-log.txt| grep "time" | head -1 | sed -e 's/.......\(.*\).*/\1/' >> $C_DIR/$C_FILE_2.txt
}

Onece()
{

  O_DIR=$1
  O_FILE=$2

  rm -f ./output/*

  ./Solver -N $o_N -p $o_p -d $o_d -m $o_m -p $o_p -D $o_D -M $o_M -t $o_t -S $o_S -L $o_L -E $o_E -R $o_R -K $o_K -F $o_F -s $o_s -l $o_l -e $o_e -r $o_r -k $o_k -f $o_f $option > $O_DIR/$O_FILE-log.txt

  # mv ./output/${o_S^^}_his.txt $O_DIR/$O_FILE-his.txt

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

o_S="vpgcr"
o_L=20000
o_E=1e-8
o_R=1000
o_K=2
o_F=2

o_s="kskipbicg"
o_l=50
o_e=0.5
o_r=50
o_k=2
o_f=2

# o_v=
# o_c=
# o_x=
# o_g=
option=""

#============================================
DIR="./TEST"
RESULTFILE="result"

OUTPUTFILE="cpu"
SUMFILE="sum_cpu"
LOOP=2
# LOOP=2

if [ ! -e $DIR ]; then
  mkdir -p $DIR
  echo "mkdir $DIR"
fi

option=""
echo "$OUTPUTFILE Start"

for i in `seq 1 $LOOP`
do
  Onece $DIR $OUTPUTFILE
  Count $DIR $OUTPUTFILE $SUMFILE
  echo "$i times"
done
Avg $DIR $SUMFILE $RESULTFILE

option="-c -x"
OUTPUTFILE="gpuX"
SUMFILE="sum_gpuX"
echo "$OUTPUTFILE Start"

for i in `seq 1 $LOOP`
do
  Onece $DIR $OUTPUTFILE
  Count $DIR $OUTPUTFILE $SUMFILE
  echo "$i times"
done
Avg $DIR $SUMFILE $RESULTFILE

option="-c -g -x"
OUTPUTFILE="gpuGX"
SUMFILE="sum_gpuGX"
echo "$OUTPUTFILE Start"

for i in `seq 1 $LOOP`
do
  Onece $DIR $OUTPUTFILE
  Count $DIR $OUTPUTFILE $SUMFILE
  echo "$i times"
done
Avg $DIR $SUMFILE $RESULTFILE

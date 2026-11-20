#!/bin/sh
# End-to-end pipeline: download data, run the search teacher, train the painter and offset heads,
# evaluate, and export the browser model on a single 32 GB GPU (~3 h teacher search + ~10 h training
# on an RTX 5090).
#   sh reproduce.sh --sl1 40000 --runs runs/long     override any setting below as --name value
#   sh reproduce.sh --data-only                      stop after preparing data
#   sh reproduce.sh --help                           list all settings and defaults
# Setting a stage to 0 skips it and continues from the last checkpoint (or --init).
# Completed stages are automatically skipped on reruns.
set -e
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data RUNS=runs INIT=
# teacher sets, images and elements per image: A short with four primitives, B long with all five
A_EMOJI=3000 A_PHOTO=3200 A_EMOJI_LEN=80 A_PHOTO_LEN=112
B_EMOJI=960 B_PHOTO=1152 B_LEN=192
# steps per stage, and RL episode lengths in elements
SL1=28232 RL1=2511 RL2=8137 SL2A=2434 SL2=7293 RL3=2084 RL4=4233 RL5=6492 ZOOM=3719 HEADS1=820 HEADS2=1880
RL12_LEN=96 RL34_LEN=256 RL5_LEN=512 ZOOM_LEN=256
KNOWN="DATA RUNS INIT A_EMOJI A_PHOTO A_EMOJI_LEN A_PHOTO_LEN B_EMOJI B_PHOTO B_LEN SL1 RL1 RL2 SL2A SL2
RL3 RL4 RL5 ZOOM HEADS1 HEADS2 RL12_LEN RL34_LEN RL5_LEN ZOOM_LEN"

ONLY_DATA=
while [ $# -gt 0 ]; do
  case $1 in
    --data-only) ONLY_DATA=1; shift; continue ;;
    --help|-h)
      for k in $KNOWN; do eval "printf '  --%-13s %s\n' \$(echo $k | tr A-Z_ a-z-) \"\$$k\""; done
      exit 0 ;;
    --*=*) f=${1%%=*} v=${1#*=}; shift ;;
    --*) [ $# -ge 2 ] || { echo "$1 needs a value" >&2; exit 1; }; f=$1 v=$2; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
  k=$(echo ${f#--} | tr a-z- A-Z_)
  case " $(echo $KNOWN) " in *" $k "*) ;; *) echo "unknown setting $f" >&2; exit 1 ;; esac
  case $k in
    DATA|RUNS|INIT) ;;
    *) case $v in ''|*[!0-9]*) echo "$f must be a whole number" >&2; exit 1 ;; esac ;;
  esac
  eval "$k=\$v"
done
D=$DATA R=$RUNS
mkdir -p $D/raw $D/done $R/done

once() {
  m=$1/done/$2; shift 2
  [ -e $m ] && return 0
  echo "+ $*"
  "$@"
  touch $m
}
last=$INIT
train() {
  n=$1 s=$2; shift 2
  [ "$s" -gt 0 ] || return 0
  once $R $n "$@" --steps $s
  last=$R/$n/ckpt.pt
}
heads=
head_stage() {
  n=$1 s=$2
  [ "$s" -gt 0 ] || return 0
  once $R $n python train_offset.py --out $R/$n --init $last $P2 ${heads:+--heads $heads} --steps $s
  heads=$R/$n/heads.pt
}

# images
once $D get_test wget -c -q -O $D/raw/test2017.zip http://images.cocodataset.org/zips/test2017.zip
once $D get_val wget -c -q -O $D/raw/val2017.zip http://images.cocodataset.org/zips/val2017.zip
once $D get_twemoji sh -c "wget -q -O - https://github.com/jdecked/twemoji/archive/refs/tags/v15.1.0.tar.gz | tar xz -C $D/raw"
once $D img_coco_train python prep_images.py coco $D/raw/test2017.zip $D/img/coco_train 6000
once $D img_coco_val python prep_images.py coco $D/raw/val2017.zip $D/img/coco_val
once $D img_twemoji python prep_images.py svg $D/raw/twemoji-15.1.0/assets/svg $D/img/twemoji

# teacher: the held-out set is the first 48 of each; training sets skip the first 100 emoji.
# The search keeps what it already found, so raising a count only adds images.
T="python make_teacher.py"
once $D t_eval_emoji $T $D/img/twemoji $D/teacher_eval/twemoji --shapes 256 --count 48
once $D t_eval_photo $T $D/img/coco_val $D/teacher_eval/coco --shapes 256 --count 48
once $D pack_eval python pack.py $D/pack_eval $D/img/twemoji:$D/teacher_eval/twemoji $D/img/coco_val:$D/teacher_eval/coco
TA=$D/teacher_a TB=$D/teacher_b$B_LEN FOUR="--types rect,ellipse,tri,curve --grad_hi 0"
if [ $A_EMOJI -gt 0 ]; then $T $D/img/twemoji $TA$A_EMOJI_LEN/twemoji --shapes $A_EMOJI_LEN --first 100 --count $A_EMOJI $FOUR; fi
if [ $A_PHOTO -gt 0 ]; then $T $D/img/coco_train $TA$A_PHOTO_LEN/coco --shapes $A_PHOTO_LEN --count $A_PHOTO $FOUR; fi
if [ $B_EMOJI -gt 0 ]; then $T $D/img/twemoji $TB/twemoji --shapes $B_LEN --first 100 --count $B_EMOJI; fi
if [ $B_PHOTO -gt 0 ]; then $T $D/img/coco_train $TB/coco --shapes $B_LEN --count $B_PHOTO; fi
B="$D/img/twemoji:$TB/twemoji:$B_EMOJI $D/img/coco_train:$TB/coco:$B_PHOTO"
pack2=$D/pack_b${B_EMOJI}_${B_PHOTO}x$B_LEN pack1=$pack2 A=
if [ $((A_EMOJI + A_PHOTO)) -gt 0 ]; then
  A="$D/img/twemoji:$TA$A_EMOJI_LEN/twemoji:$A_EMOJI $D/img/coco_train:$TA$A_PHOTO_LEN/coco:$A_PHOTO"
  pack1=$D/pack_a${A_EMOJI}x${A_EMOJI_LEN}_${A_PHOTO}x$A_PHOTO_LEN
  pack2=${pack2}_${pack1#$D/pack_}
  once $D ${pack1#$D/} python pack.py $pack1 $A
fi
once $D ${pack2#$D/} python pack.py $pack2 $B $A
[ -n "$ONLY_DATA" ] && exit 0

# supervised from scratch; RL on short episodes; supervised on both sets, adding glimpse windows;
# RL with a mixed reward on 256 elements, then 512; then through 2x magnified tiles; heads last
E="--eval_pack $D/pack_eval"
P1="--pack $pack1 $E" P2="--pack $pack2 $E"
S2="$P2 --bs_synth 4 --bs_teacher 12" M="$P2 --reward mse:0.5,lpips:0.5"
train sl1 $SL1 python train_sl.py --out $R/sl1 ${last:+--init $last} $P1 --lr 4e-4 --warmup 300 --early 0
train rl1 $RL1 python train_rl.py --out $R/rl1 --init $last $P1 --max_shapes $RL12_LEN --reward_res 128 --decay 0
train rl2 $RL2 python train_rl.py --out $R/rl2 --init $last $P1 --max_shapes $RL12_LEN --reward_res 128 --lr 4e-5
train sl2a $SL2A python train_sl.py --out $R/sl2a --init $last --glimpse 1 $S2 --lr 1.5e-4 --warmup 200
train sl2 $SL2 python train_sl.py --out $R/sl2 --init $last $S2 --lr 1e-4 --warmup 100
train rl3 $RL3 python train_rl.py --out $R/rl3 --init $last $M --max_shapes $RL34_LEN --decay 0
train rl4 $RL4 python train_rl.py --out $R/rl4 --init $last $M --max_shapes $RL34_LEN --lr 4e-5
train rl5 $RL5 python train_rl.py --out $R/rl5 --init $last $M --max_shapes $RL5_LEN
train zoom $ZOOM python train_rl.py --out $R/zoom --init $last $M --max_shapes $ZOOM_LEN --zoom 1
head_stage heads1 $HEADS1
head_stage heads2 $HEADS2

once $R eval python evaluate.py --ckpt $last ${heads:+--heads $heads} --pack $D/pack_eval --out $R/eval
if [ -n "$heads" ]; then
  once $R export python export_onnx.py --ckpt $last --heads $heads --eval_pack $D/pack_eval --out web
fi

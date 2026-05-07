export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=/path/to/image
SPLIT_ROOT=/path/to/split
EVAL_MODEL=/path/to/best_checkpoint.pth
# tokenizer
SENTENCEPIECE_MODEL=/path/to/beit3.spm
OUTPUT_DIR=/path/to/output

python eval_VistaRef.py \
  --num_workers 4 \
  --batch_size 128 \
  --imsize 384 --max_query_len 64 \
  --model beit3_large_patch16_384 \
  --task grounding \
  --dataset egopoint \
  --eval_set test \
  --data_root ${DATA_ROOT} \
  --split_root ${SPLIT_ROOT} \
  --sentencepiece_model ${SENTENCEPIECE_MODEL} \
  --eval_model ${EVAL_MODEL} \
  --output_dir ${OUTPUT_DIR};
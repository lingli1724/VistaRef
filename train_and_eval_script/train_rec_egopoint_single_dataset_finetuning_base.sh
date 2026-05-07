export CUDA_VISIBLE_DEVICES=0
echo "train_rec_egopoint_single_dataset_finetuning_base.sh"


# ====== TIME STAMP ======
TIME_TAG=$(date +"%Y%m%d_%H%M%S")

# ====== YOUR PATHS ======
DATA_ROOT=/path/to/image
SPLIT_ROOT=/path/to/split
OUTPUT_ROOT_BASE=/path/to/output_base
OUTPUT_ROOT=${OUTPUT_ROOT_BASE}/${TIME_TAG}

mkdir -p ${OUTPUT_ROOT}

echo "Output dir: ${OUTPUT_ROOT}"

# ====== OFFICIAL-STYLE BASE INIT (NO MRefM for single-dataset) ======
FINETUNE_INIT=/path/to/rec_mrefm_pretrain_base_patch16_384.pth

# tokenizer
SENTENCEPIECE_MODEL=/path/to/beit3.spm

#warmup 
python train_VistaRef.py \
  --epochs 10 \
  --batch_size 64 \
  --lr 0.00025 \
  --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 \
  --task grounding \
  --dataset egopoint \
  --use_regress_box \
  --loss_ema_momentum 0.99 \
  --loss_ema_eps 1e-6 \
  --loss_ema_warmup_iters 100 \
  --loss_orig_weight 0.7 \
  --loss_new_weight 0.3 \
  --loss_hand_ratio 0.2 \
  --loss_kp_ratio 0.4 \
  --loss_ray_ratio 0.3 \
  --loss_angle_ratio 0.1 \
  --frozen_backbone \
  --finetune ${FINETUNE_INIT} \
  --data_root ${DATA_ROOT} \
  --split_root ${SPLIT_ROOT} \
  --sentencepiece_model ${SENTENCEPIECE_MODEL} \
  --output_dir ${OUTPUT_ROOT}/v001/egopoint;


#finetuning training 
python train_VistaRef.py \
  --epochs 20 \
  --batch_size 8 \
  --lr 0.00003 \
  --lr_scheduler cosine \
  --aug_crop --aug_scale --aug_translate \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 \
  --task grounding \
  --dataset egopoint \
  --use_regress_box \
  --use_box_mask_constraints \
  --loss_ema_momentum 0.99 \
  --loss_ema_eps 1e-6 \
  --loss_ema_warmup_iters 100 \
  --loss_orig_weight 0.7 \
  --loss_new_weight 0.3 \
  --loss_hand_ratio 0.2 \
  --loss_kp_ratio 0.4 \
  --loss_ray_ratio 0.3 \
  --loss_angle_ratio 0.1 \
  --finetune ${OUTPUT_ROOT}/v001/egopoint/best_checkpoint.pth \
  --data_root ${DATA_ROOT} \
  --split_root ${SPLIT_ROOT} \
  --sentencepiece_model ${SENTENCEPIECE_MODEL} \
  --output_dir ${OUTPUT_ROOT}/v002/egopoint;


####### eval (val + test)  
python eval_VistaRef.py \
  --num_workers 4 \
  --batch_size 128 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 \
  --task grounding \
  --dataset egopoint \
  --eval_set val \
  --data_root ${DATA_ROOT} \
  --split_root ${SPLIT_ROOT} \
  --sentencepiece_model ${SENTENCEPIECE_MODEL} \
  --eval_model ${OUTPUT_ROOT}/v002/egopoint/best_checkpoint.pth \
  --output_dir ${OUTPUT_ROOT}/v002/egopoint;

python eval_VistaRef.py \
  --num_workers 4 \
  --batch_size 128 \
  --imsize 384 --max_query_len 64 \
  --model beit3_base_patch16_384 \
  --task grounding \
  --dataset egopoint \
  --eval_set test \
  --data_root ${DATA_ROOT} \
  --split_root ${SPLIT_ROOT} \
  --sentencepiece_model ${SENTENCEPIECE_MODEL} \
  --eval_model ${OUTPUT_ROOT}/v002/egopoint/best_checkpoint.pth \
  --output_dir ${OUTPUT_ROOT}/v002/egopoint;

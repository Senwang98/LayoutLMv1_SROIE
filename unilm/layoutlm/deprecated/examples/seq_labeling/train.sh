# CUDA_VISIBLE_DEVICES=0 \
# python3 run_seq_labeling.py \
python3 -m torch.distributed.launch --nproc_per_node=2 run_seq_labeling.py \
    --data_dir '../../../../../data/working/dataset' \
    --labels '../../../../../data/working/dataset/labels.txt' \
    --model_name_or_path "../../../../../SROIE2019/layoutlm-base-uncased/" \
    --model_type layoutlm \
    --max_seq_length 512 \
    --do_lower_case \
    --do_train \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --save_steps -1 \
    --output_dir output \
    --overwrite_output_dir \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 

# python -m torch.distributed.launch --nproc_per_node=4 run_seq_labeling.py \
#         --model_name_or_path microsoft/layoutlm-base-uncased \
#         --output_dir /tmp/test-ner \
#         --do_train \
#         --do_predict \
#         --max_steps 1000 \
#         --warmup_ratio 0.1 \
#         --fp16

# For evaluate
python3 -m torch.distributed.launch --nproc_per_node=2 run_seq_labeling.py \
        --data_dir '../../../../../data/working/dataset' \
        --labels '../../../../../data/working/dataset/labels.txt' \
        --model_name_or_path "../../../../../SROIE2019/layoutlm-base-uncased/" \
        --model_type layoutlm \
        --do_lower_case \
        --max_seq_length 512 \
        --do_predict \
        --logging_steps 10 \
        --save_steps -1 \
        --output_dir output \
        --per_gpu_eval_batch_size 8
export CUDA_VISIBLE_DEVICES=0

model=GPT4TS
percent=100
stride=2

for data in BTC_Daily
do
for seq_len in 200
do
for patch_size in 5
do 
for pred_len in 7
do


python3 main.py \
    --root_path ./datasets/BTC/ \
    --data_path $data'.csv' \
    --model_id $data'_'$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$patch_size \
    --data custom \
    --num_features 8 \
    --features M \
    --seq_len $seq_len \
    --target close \
    --label_len 7 \
    --pred_len $pred_len \
    --batch_size 3 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 8 \
    --d_ff 2048 \
    --gpt_layers 7 \
    --scale 1 \
    --dropout 0.05 \
    --pct 0 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --lradj type2 \
    --patch_size $patch_size \
    --stride $stride \
    --percent $percent \
    --itr 1 \
    --voc 15 \
    --loss_func mse \
    --model $model \
    --is_gpt 1 \
    --isllama 0 \
    --pca 0 \
 
done
done
done
done

export CUDA_VISIBLE_DEVICES=0

model=llamatest
percent=100
stride=2

for data in BTC_Daily
do
for seq_len in 100
do
for patch_size in 11
do 
for pred_len in 7
do

python3 main.py \
    --root_path ./datasets/BTC/ \
    --data_path $data'.csv' \
    --model_id $data'_'$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$patch_size \
    --data custom \
    --num_features 5 \
    --features M \
    --seq_len $seq_len \
    --target close \
    --label_len 7 \
    --pred_len $pred_len \
    --batch_size 3 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 5120 \
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
    --voc 200 \
    --loss_func mse \
    --model $model \
    --is_gpt 0 \
    --isllama 1 \
    --pca 0 \
 
done
done
done
done

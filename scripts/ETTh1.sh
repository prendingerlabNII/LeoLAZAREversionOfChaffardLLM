
export CUDA_VISIBLE_DEVICES=0

seq_len=96 
model=Llama_inverted_PCA

for percent in 100
do
for pred_len in 96
do
for lr in 0.0001
do

python main.py \
    --root_path ./datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 5 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 6 \
    --decay_fac 0.5 \
    --d_model 5120 \
    --n_heads 4 \
    --d_ff 5120 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 24 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --voc 100 \
    --isllama 0 \
    --pct 0 \
    --scale 1 \
    --pca 1 \

done
done
done
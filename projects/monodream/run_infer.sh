CHUNKS=16
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo $(( IDX % 8 ))
    CUDA_VISIBLE_DEVICES=$(( IDX % 8 )) python infer.py \
    --model-path  {YOUR/MODEL/PATH} \
    --conv-mode auto \
    --exp-config VLN_CE/vlnce_baselines/config/r2r_baselines/navid_r2r.yaml \
    --split-num $CHUNKS \
    --split-id $IDX \
    --visualize \
    --result-path {YOUR/RESULT/PATH} &
done
wait
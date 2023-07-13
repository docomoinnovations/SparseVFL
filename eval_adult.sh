export CUDA_VISIBLE_DEVICES=0

# Original
python sparsevfl.py --layer_no_bias                            --interface_dims 8,8,8 --reduction org --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv

# Dim
python sparsevfl.py --layer_no_bias                            --interface_dims 6,6,6 --reduction int --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction int --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias                            --interface_dims 3,3,3 --reduction int --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias                            --interface_dims 2,2,2 --reduction int --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv

# Quantization
python sparsevfl.py --layer_no_bias                            --interface_dims 8,8,8 --reduction q16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias                            --interface_dims 8,8,8 --reduction q8  --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv


# Top-W
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.0100
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.0100
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.0100

python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.0158
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.0158
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.0158

python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.0251
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.0251
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.0251

python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.0398
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.0398
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.0398

python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.0631
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.0631
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.0631

python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.1
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.1
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.1

python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 4096 --sparse_embed_lambda 0.1585
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 2048 --sparse_embed_lambda 0.1585
python sparsevfl.py --layer_no_bias  --interface_dims 8,8,8 --reduction topk16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv --top_k 1024 --sparse_embed_lambda 0.1585


# SparseVFL
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0100 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0158 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0251 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0398 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0631 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.1000 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.1585 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1.tsv

# PCA
python compressvfl.py --interface_dims 8,8,8 --reduction org --compress pca --model_header compressvfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1_compress.tsv

# AutoEncoder
python compressvfl.py --interface_dims 8,8,8 --reduction org --compress ae --model_header compressvfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name result_1_compress.tsv


exit 0

########################
# Computation time
########################
export CUDA_VISIBLE_DEVICES=0
python sparsevfl.py --layer_no_bias                            --interface_dims 8,8,8 --reduction org --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name computation_time.tsv --n_epochs 200

export CUDA_VISIBLE_DEVICES=0
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0251 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name computation_time.tsv --n_epochs 200

########################
# Ablation
########################
# sparse+relu+l1+run1+nobias
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.01 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name ablation_1.tsv --client_activation relu --norm 1 --run_axis 1

# sparse+selu+l1+run1+nobias
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.01 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name ablation_1.tsv --client_activation selu --norm 1 --run_axis 1
# sparse+elu+l1+run1+nobias
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.01 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name ablation_1.tsv --client_activation elu --norm 1 --run_axis 1

# sparse+relu+l2+run1+nobias
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.01 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name ablation_1.tsv --client_activation relu --norm 2 --run_axis 1
# sparse+relu+NO+run1+nobias
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0    --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name ablation_1.tsv --client_activation relu          --run_axis 1

# sparse+relu+l1+run0+nobias
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.01 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3 --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name ablation_1.tsv --client_activation relu --norm 1 --run_axis 0


########################
# Embedding sparsity
########################
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0100 --interface_dims 8,8,8 --reduction svfl16 --model_header sparsevfl_adult3_emb --data_dir /mnt/share/data/adult3 --lr 0.01 --tsv_name emb_1.tsv

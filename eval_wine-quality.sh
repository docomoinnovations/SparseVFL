export CUDA_VISIBLE_DEVICES=1

# Original
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction org --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv

# Dim
python sparsevfl.py --layer_no_bias                            --interface_dims 3,3,3 --reduction int --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias                            --interface_dims 2,2,2 --reduction int --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv

# Quantization
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction q16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction q8  --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv

# Top-W
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.0100
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.0100

python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.0158
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.0158

python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.0251
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.0251

python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.0398
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.0398

python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.0631
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.0631

python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.1
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.1

python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 2048 --sparse_embed_lambda 0.1585
python sparsevfl.py --layer_no_bias  --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv --top_k 1024 --sparse_embed_lambda 0.1585

# SparseVFL
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0100 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0158 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0251 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0398 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0631 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.1000 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.1585 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3.tsv


# PCA
python compressvfl.py --interface_dims 4,4,4 --reduction org --compress pca --model_header compressvfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3_compress.tsv

# AutoEncoder
python compressvfl.py --interface_dims 4,4,4 --reduction org --compress ae --model_header compressvfl_wine-quality3 --data_dir data/wine-quality3 --lr 0.01 --tsv_name result_3_compress.tsv

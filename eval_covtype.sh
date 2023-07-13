export CUDA_VISIBLE_DEVICES=3

# Original
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction org --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7

# Dim
python sparsevfl.py --layer_no_bias                            --interface_dims 3,3,3 --reduction int --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias                            --interface_dims 2,2,2 --reduction int --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7

# Quantization
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction q16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias                            --interface_dims 4,4,4 --reduction q8  --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7

# Top-W
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.0100
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.0100

python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.0158
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.0158

python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.0251
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.0251

python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.0398
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.0398

python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.0631
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.0631

python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.1
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.1


python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 2048 --loss ce --out_size 7 --sparse_embed_lambda 0.1585
python sparsevfl.py --layer_no_bias --interface_dims 4,4,4 --reduction topk16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --top_k 1024 --loss ce --out_size 7 --sparse_embed_lambda 0.1585


# SparseVFL
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0100 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0158 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0251 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0398 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.0631 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.1000 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7
python sparsevfl.py --layer_no_bias --sparse_embed_lambda 0.1585 --interface_dims 4,4,4 --reduction svfl16 --model_header sparsevfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7.tsv --loss ce --out_size 7

# PCA
python compressvfl.py --interface_dims 4,4,4 --reduction org --compress pca --model_header compressvfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7_compress.tsv --loss ce --out_size 7

# AutoEncoder
python compressvfl.py --interface_dims 4,4,4 --reduction org --compress ae --model_header compressvfl_covtype3 --data_dir data/covtype3 --lr 0.01 --tsv_name result_7_compress.tsv --loss ce --out_size 7

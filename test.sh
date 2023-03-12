CUDA_VISIBLE_DEVICES=2 python infer_single.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v0/swa_model.pth  --resultPath firstResult/EO_swa_v0.csv &
CUDA_VISIBLE_DEVICES=2 python infer_single.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v0/model_best.pth.tar  --resultPath firstResult/EO_best_v0.csv &
CUDA_VISIBLE_DEVICES=2 python infer_single.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v1/swa_model.pth  --resultPath firstResult/EO_swa_v1.csv &
CUDA_VISIBLE_DEVICES=2 python infer_single.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v1/model_best.pth.tar  --resultPath firstResult/EO_best_v1.csv &
CUDA_VISIBLE_DEVICES=1 python infer_single.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v3/swa_model.pth  --resultPath firstResult/EO_swa_v3.csv &
CUDA_VISIBLE_DEVICES=0 python infer_fusion.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v0_fusion/model_best.pth.tar  --resultPath firstResult/fusion_best_v0.csv &
CUDA_VISIBLE_DEVICES=3 python infer_fusion.py --num_classes 10 --checkpoint_path checkpoints/checkpoint-EO-v0_fusion/swa_model.pth  --resultPath firstResult/fusion_swa_v0.csv &
CUDA_VISIBLE_DEVICES=3 python infer_single4.py --num_classes 4 --checkpoint_path checkpoints/checkpoint-EO-v0_4class/model_best.pth.tar  --resultPath secondResult/class4_best_v0.csv & 

python postCluster.py

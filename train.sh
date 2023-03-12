python main.py --num_classes 10 --use_swa 1 --train_file data/train_task2_v0.txt --val_file data/val_task2_v0.txt --checkpoint_path checkpoints/checkpoint-EO-v0/
python main.py --num_classes 10 --use_swa 1 --train_file data/train_task2_v1.txt --val_file data/val_task2_v1.txt --checkpoint_path checkpoints/checkpoint-EO-v1/
python main.py --num_classes 10 --use_swa 1 --train_file data/train_task2_v2.txt --val_file data/val_task2_v2.txt --checkpoint_path checkpoints/checkpoint-EO-v2/
python main4.py --num_classes 4 --use_swa 1 --train_file data/train_task2_v0_4class.txt --val_file data/val_task2_v0_4class.txt --checkpoint_path checkpoints/checkpoint-EO-v0_4class/
python mainfusion.py --num_classes 10 --use_swa 1 --train_file data/train_task2_v0_fusion.txt --val_file data/val_task2_v0_fusion.txt --checkpoint_path checkpoints/checkpoint-EO-v0_fusion/

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25635 train2.py \
#     --cfg configs/Pure/vit_small_pre_coder_livec.yaml \
#     --tensorboard --tag 1.0_0.5_new \
#     --visual \
#     --alpha 1.0 --beta 0.5 \
#     --epoch 30 --seed 1024 --repeat --rnum 10

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25635 train2.py \
#     --cfg configs/Pure/vit_small_pre_coder_livec.yaml \
#     --tensorboard --tag 1.0_0.1_new \
#     --visual \
#     --alpha 1.0 --beta 0.1 \
#     --epoch 30 --seed 1024 --repeat --rnum 10

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25635 train2.py \
#     --cfg configs/Pure/vit_small_pre_coder_livec.yaml \
#     --tensorboard --tag onlysmooth \
#     --only_smooth \
#     --visual \
#     --alpha 1.0 --beta 0.1 \
#     --epoch 30 --seed 1024 --repeat --rnum 10
# 7
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25635 train2.py \
#     --cfg configs/Pure/vit_small_pre_coder_livec.yaml \
#     --tensorboard --tag 1.0_0.5_wovisual \
#     --alpha 1.0 --beta 0.5 \
#     --epoch 30 --seed 1024 --repeat --rnum 10

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train2.py \
#     --cfg configs/Pure/vit_small_pre_coder_livec.yaml  \
#     --tensorboard --tag nodistprompt \
#     --scene \
#     --visual \
#     --alpha 1.0 --beta 0.5 \
#     --epoch 30 --seed 1024 --repeat --rnum 10

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train2.py \
#     --cfg configs/Pure/vit_small_pre_coder_livec.yaml  \
#     --tensorboard --tag multi-label \
#     --dist \
#     --scene \
#     --visual \
#     --alpha 1.0 --beta 0.5 \
#     --epoch 30 --seed 1024 --repeat --rnum 10

#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train2.py \
#    --cfg configs/Pure/vit_small_pre_coder_livec.yaml  \
#    --tensorboard --tag final \
#    --dist --scene \
#    --visual \
#    --alpha 1.0 --beta 0.5 \
#    --epoch 30 --seed 1024 --repeat --rnum 10
#
#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train2.py \
#    --cfg configs/Pure/vit_small_pre_coder_live.yaml  \
#    --tensorboard --tag prompt_16 \
#    --dist --scene \
#    --visual \
#    --prompt 16 \
#    --alpha 1.0 --beta 0.1 \
#    --epoch 30 --seed 1024 --repeat --rnum 10
#
#CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train2.py \
#    --cfg configs/Pure/vit_small_pre_coder_live.yaml  \
#    --tensorboard --tag prompt_4 \
#    --dist --scene \
#    --visual \
#    --prompt 4 \
#    --alpha 1.0 --beta 0.1 \
#    --epoch 30 --seed 1024 --repeat --rnum 10

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train1.py \
    --cfg configs/Pure/vit_small_pre_coder_livec.yaml  \
    --tensorboard --tag baseline \
    --dist --scene \
    --visual \
    --print \
    --alpha 1.0 --beta 0.5 \
    --epoch 30 --seed 1024 --repeat --rnum 10
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train.py \
    --cfg configs/Pure/vit_small_pre_coder_livec.yaml  \
    --tensorboard --tag livec \
    --dist --scene \
    --visual \
    --alpha 1.0 --beta 0.5 \
    --epoch 30 --seed 1024 --repeat --rnum 10
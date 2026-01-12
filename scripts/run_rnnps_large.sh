torchrun --standalone --nproc_per_node=8 \
      MARS/train_rnnps.py \
      config/train_gpt2_large_rnnps.py \
      --batch_size=5 \
      --gradient_accumulation_steps=12 \
      --max_iters=40000 \
      --lr_decay_iters=40000

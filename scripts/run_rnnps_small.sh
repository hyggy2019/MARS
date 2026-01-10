torchrun --standalone --nproc_per_node=8 \
      MARS/train_rnnps.py \
      config/train_gpt2_small_rnnps.py \
      --batch_size=15 \
      --gradient_accumulation_steps=4

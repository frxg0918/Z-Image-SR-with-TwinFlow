export CUDA_VISIBLE_DEVICES=1
# *[Specify the WANDB API key]
# export WANDB_MODE=disabled
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY='e93e9915028934cd0ecb2f0863386b4eb7b6abb9'

accelerate launch --main_process_port 15123 \
  ./train.py \
  --dataset_base_path "" \
  --dataset_txt_paths "./datasets_utils/gt_path.txt" \
  --height 512 \
  --width 512 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Tongyi-MAI/Z-Image:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 5e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./runs" \
  --lora_base_model "dit" \
  --lora_target_modules "noise_refiner,layers" \
  --lora_rank 128 \
  --dataset_num_workers 8 \
  --task "twinflow" \
  --weight_decay 0.0 \
  --batch_size 2 \
  --save_steps 5000 \
  --max_grad_norm 0.0 \
  --use_gradient_checkpointing \
  --alpha 1.0 \
  --enable_2_temb \
  --enable_alpha1 \
  # --lora_checkpoint "/remote-home/share/xianggao/DiffSynth-Studio/runs/20260121-125648/models/step-45000.safetensors" \
  # --condition_timestep_zero \
  # --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3,t_embedder.mlp.*"
  # --lora_target_modules "to_q,to_k,to_v,to_out.0,w1,w2,w3,all_final_layer,adaLN" \
  # --output_path "./models/train/Z-Image-Turbo_lora" \

# TODO all_final_layers是对transformerblock的输出做投影，不涉及到噪声分支和条件分支交互，如果冻结噪声分支的话不用微调的
  # --model_id_with_origin_paths "Tongyi-MAI/Z-Image-Turbo:transformer/*.safetensors,Tongyi-MAI/Z-Image-Turbo:text_encoder/*.safetensors,Tongyi-MAI/Z-Image-Turbo:vae/diffusion_pytorch_model.safetensors" \

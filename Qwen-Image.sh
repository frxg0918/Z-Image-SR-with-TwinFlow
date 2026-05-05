qwen_path="/remote-home/share/xianggao/DiffSynth-Studio-20260119/DiffSynth-Studio/models"


CUDA_VISIBLE_DEVICES=1 accelerate launch train_qwen.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata.csv \
  --height 512 \
  --width 512 \
  --dataset_repeat 50 \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 128 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --model_id_with_origin_paths "Qwen/Qwen-Image-2512:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  # --model_paths "${qwen_path}/Qwen/Qwen-Image-2512:transformer/diffusion_pytorch_model*.safetensors,${qwen_path}/Qwen/Qwen-Image:text_encoder/model*.safetensors,${qwen_path}/Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \

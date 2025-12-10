## 📖 Inference

Here we give some demo inference codes in dir `infer/`. To inference, you need to first **clone the official repo** and put the inference code into the root directory of its repo.

* Inference with BAGEL

    ~~~
    python infer/infer_bagel.py \
        --image_paths demo/input_1.png demo/input_2.png \
        --prompt "The versatile and compact bike with different designs from image 1 is parked neatly against a wall illuminated by a sleek brushed nickel wall light from image 2. The soft glow highlights the bike’s varied features, creating a stylish juxtaposition in a contemporary urban setting." \
        --output_path output_bagel.png \
        --model_path /path/to/ft/ckpt \
        --seed 42 \
        --cuda_visible_devices 0
    ~~~

* Infence with BLIP3o-Next

    ~~~
    python infer/infer_blip3o.py \
        --model_path "/path/to/ft/ckpt" \
        --input_image demo/input_1.png demo/input_2.png \
        --instruction "The versatile and compact bike with different designs from image 1 is parked neatly against a wall illuminated by a sleek brushed nickel wall light from image 2. The soft glow highlights the bike’s varied features, creating a stylish juxtaposition in a contemporary urban setting." \
        --output output_blip3o.png \
        --use_und_image_vae \
        --use_und_image_vae_as_noise \
        --seed 42 \
        --device "cuda:0"
    ~~~

* Inference with Lumina-DiMOO

    ~~~
    python -u infer/infer_dimoo.py \
        --checkpoint /path/to/ft/ckpt \
        --vae_ckpt /path/to/vqvae \
        --prompt "The versatile and compact bike with different designs from image 1 is parked neatly against a wall illuminated by a sleek brushed nickel wall light from image 2. The soft glow highlights the bike’s varied features, creating a stylish juxtaposition in a contemporary urban setting." \
        --image_path "demo/input_1.png,demo/input_2.png" \
        --edit_type "multi_inference" \
        --output_dir "output_dimoo/"
    ~~~

* Inference with OmniGen2

    ~~~
    python infer_omnigen2.py \
    --model_path "/path/to/OmniGen2" \
    --num_inference_step 50 \
    --text_guidance_scale 4.0 \
    --input_image_path demo/input_1.png demo/input_2.png \
    --instruction "The versatile and compact bike with different designs from image 1 is parked neatly against a wall illuminated by a sleek brushed nickel wall light from image 2. The soft glow highlights the bike’s varied features, creating a stylish juxtaposition in a contemporary urban setting." \
    --seed 42 \
    --transformer_path "/path/to/ft/transformer" \
    --output_image_path output_omnigen.png
    ~~~


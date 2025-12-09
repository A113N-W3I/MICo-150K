# 🎯 Enabling Community Models with Multi-Image Composition Capability

Here we organized some training guidelines to equip community models with multi-image composition capability.

Just for your inference, you donnot have to follow this guideline to train these models.

## Train BAGEL with MICo

Follow the [official guideline](https://github.com/ByteDance-Seed/Bagel/blob/main/TRAIN.md) with the following modifications:

1. Add a new dataset for multi-image inputs in `bagel/data/interleave_datasets/edit_dataset.py`:

    ~~~py
    class MultiInputEditDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
        def parse_row(self, row):
            image_num = len(row["image_list"])

            data = self._init_data()
            for idx in range(image_num):
                if idx != image_num - 1:
                    data = self._add_image(
                        data, pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=False, need_vae=True, need_vit=True
                    )
                else:
                    data = self._add_text(data, row["instruction"], need_loss=False)
                    data = self._add_image(
                        data, pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
                        need_loss=True, need_vae=False, need_vit=False
                    )

            return data
    ~~~

2. Prepare the parquet files and annotations following the format in `bagel/data/bagel_example/editing`.

3. Register the dataset in `bagel/data/dataset_info.py`:

    ~~~py
    from .interleave_datasets import MultiInputEditDataset

    DATASET_REGISTRY = {
        # ...
        'mico-150k': MultiInputEditDataset,
    }

    DATASET_INFO = {
        # ...
        'mico-150k': {
            'mico-150k': {
                'data_dir': '/PATH/TO/MICO-150K',
                'num_files': 300,
                'num_total_samples': 150000,
                "parquet_info_path": '/PATH/TO/METADATA',
            }
        }
    }
    ~~~

4. Prepare your config with `dataset_name = mico-150k` and start your training with `--dataset_config_file ./data/configs/mico-150k.yaml`.


## Train BLIP3o-Next with MICo

Using the [official SFT script](https://github.com/JiuhaiChen/BLIP3o/blob/BLIP3o-NEXT/scripts/sft.sh) with the following modifications:

1. Add a new dataset for multi-image inputs in `blip3o-next-edit/blip3o/data/dataset.py`:

    ~~~py
    elif 'nano_banana' in dataset_path.lower():
        # ...
    elif 'mico' in dataset_path.lower():
        mico_path = "/PATH/TO/BLIP3o-MICo.json"
        train_dataset = load_dataset("json", data_files=mico_path, split="train", num_proc=1)
        train_dataset = train_dataset.rename_column("output_image", "image_path")
        train_dataset = train_dataset.rename_column("input_images", "input_images_paths")
        train_dataset = train_dataset.add_column('input_images', len(train_dataset) * [[]])
        train_dataset = train_dataset.rename_column("instruction", "txt")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['X2I_nano_banana'])
        train_dataset = train_dataset.add_column('image', len(train_dataset) * [None])
        repeat_time = 1
        train_dataset = concatenate_datasets([train_dataset] * dataset_weight)
        rank0_print("loaded MICo dataset with ", len(train_dataset), " samples", "from ", mico_path, "with weight ", dataset_weight)
    else:
        # ...
    ~~~

    with the format of `BLIP3o-MICo.json`:

    ~~~
    [
        {
            "input_images": [
                "/PATH/TO/INPUT-IMAGE-1",
                "/PATH/TO/INPUT-IMAGE-2"
            ],
            "output_image": "/PATH/TO/OUTPUT-IMAGE",
            "instruction": "..."
        },
        ...
    ]
    ~~~

2. Pass `DATASET_LIST="mico"`, `--model_max_length 5120` in `sft.sh` and start training.


## Train Lumina-DiMOO with MICo

Follow the [official guideline](https://github.com/Alpha-VLLM/Lumina-DiMOO#-quick-start) with the following modifications:

1. Implement the `multi_image`-to-`image` logic in `train/train.py`:

    ~~~py
    class ItemProcessor(ItemProcessorBase):
        def __init__(self, tokenizer, max_len, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def process_item(self, data_item: dict, training_mode=False) -> Tuple[List, List]:
            if isinstance(data_item["user_image"], list) and len(data_item["user_image"]) > 0 and data_item["answer_image"] != "":
                if np.random.rand() < 0.1:
                    instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user><uncondition></user>"
                    instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()
                    instruction_label = [-100] * len(instruction_token)
                else:
                    instruction = "<system>" + data_item["system_prompt"] + "</system>" + "<user>" + data_item["user_prompt"] + "</user>"
                    instruction_token = self.tokenizer(instruction, truncation=True, max_length=1024, padding=False, return_tensors="pt").input_ids[0].tolist()

                    all_image_tokens = []
                    for img_path in data_item["user_image"]:
                        with open(img_path, "rb") as f:
                            data_pkl = pickle.load(f)
                        image_tokens = data_pkl["input_ids"]
                        assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
                        image_height, image_width = data_pkl["width"]//16, data_pkl["height"]//16
                        image_tokens = add_break_line(image_tokens, image_height, image_width, new_number=126084)
                        all_image_tokens += [126349] + image_tokens + [126350]

                    instruction_token = instruction_token[:-1] + all_image_tokens + instruction_token[-1:]
                    instruction_label = [-100] * len(instruction_token)

                with open(data_item["answer_image"], "rb") as f:
                    data_pkl = pickle.load(f)
                image_tokens = data_pkl["input_ids"]
                assert data_pkl["height"] % 16 == 0 and data_pkl["width"] % 16 == 0
                image_height, image_width = data_pkl["width"]//16, data_pkl["height"]//16
                image_masked_codes, image_labels = mask_codes(image_tokens)
                image_tokens = add_break_line(image_masked_codes, image_height, image_width, new_number=126084)
                image_labels = add_break_line(image_labels, image_height, image_width, new_number=-100)

                all_token = instruction_token + [126354] + [126349] + image_tokens + [126350] + [126355]
                all_label = instruction_label + [-100] + [-100] + image_labels + [-100] + [-100]

            elif isinstance(data_item["user_image"], str) and data_item["user_image"] != "" and data_item["answer_image"] == "":
                # ...
    ~~~

2. Prepare `dimoo-mico.json` with the format:

    ~~~json
    [
        {
            "image_path": [
                "/PATH/TO/INPUT-IMAGE-1",
                "/PATH/TO/INPUT-IMAGE-2"
            ],
            "edit_path": "/PATH/TO/OUTPUT-IMAGE",
            "prompt": "..."
        },
        ...
    ]
    ~~~

3. Pre-tokenize the images for training by `bash pre_tokenizer/run_pre_token.sh` and start training.

## Train OmniGen2 with MICo

Since OmniGen2 naturally supports multi-image inputs for editing tasks, simply follow the [official guideline](https://github.com/VectorSpaceLab/OmniGen2/blob/main/docs/FINETUNE.md) can work perfectly on MICo. We prepared MICo metadata with the following `.jsonl` format:

~~~json
{"task_type": "edit", "instruction": "...", "input_images": ["/PATH/TO/INPUT-IMAGE-1","/PATH/TO/INPUT-IMAGE-2"],"output_image": "/PATH/TO/OUTPUT-IMAGE"}
{"task_type": "edit", "instruction": "...", "input_images": ["/PATH/TO/INPUT-IMAGE-1","/PATH/TO/INPUT-IMAGE-2"],"output_image": "/PATH/TO/OUTPUT-IMAGE"}
...
~~~
# <img src="assets/mico-logo.png" width="60px" align="center"> MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition

Official repository for the paper [MICo-150K: A Comprehensive Dataset for Multi-Image Composition](https://arxiv.org/abs/2512.07348).

## 📢 News

* **Dec 10, 2025:** 🚀 We released finetuned checkpoint [BAGEL-MICo](https://huggingface.co/kr-cen/BAGEL-MICo), [BLIP3o-Next-MICo](https://huggingface.co/kr-cen/BLIP3o-Next-MICo), [Lumina-DiMOO-MICo](https://huggingface.co/kr-cen/Lumina-DiMOO-MICo), and [OmniGen2-MICo](https://huggingface.co/kr-cen/OmniGen2-MICo), with impressive multi-image composition capability. <u>Our **MICo-150K dataset** coming soon, stay tuned! 👀</u>
* **Dec 10, 2025:** 📖 We released multi-image composition [training](https://github.com/A113N-W3I/MICo-150K/blob/main/TRAIN.md) & [inference](https://github.com/A113N-W3I/MICo-150K/blob/main/INFER.md) guideline for community models. ~~Our finetuned checkpoints coming soon, stay tuned! 👀~~
* **Dec 9, 2025:** 🔥 Our paper on [arXiv](https://arxiv.org/abs/2512.07348).
* **Dec 2, 2025:** 🎬 We released the official [project page](https://mico-150k.github.io/) for MICo-150K.

## Introduction

* **We present MICo-150K**, a large-scale, high-quality dataset for **Multi-Image Composition (MICo)** in controllable image generation. MICo focuses on synthesizing coherent and identity-consistent images from multiple reference inputs—a long-standing challenge due to the lack of suitable training data.
* MICo-150K covers **7 representative MICo tasks**, constructed from carefully curated source images and diverse composition prompts. The dataset is synthesized using strong proprietary models and refined via **human-in-the-loop filtering**, ensuring high fidelity and identity consistency. We further introduce a **Decomposition-and-Recomposition (De&Re)** subset, where real-world complex images are decomposed into components and recomposed, supporting both real and synthetic compositions.
* To enable systematic evaluation, we release **MICo-Bench**, consisting of **1000 curated test cases**, and propose **Weighted-Ref-VIEScore**, a new metric tailored specifically for MICo. We also provide strong baselines, including **Qwen-MICo**, which demonstrates competitive performance with proprietary models while supporting arbitrary multi-image inputs.

![mico-dataset](assets/dataset-case.jpg)

## 📑 Open-Source Plan

- [ ] MICo-150K dataset
- [X] Finetuned checkpoints
- [ ] MICo-Bench
- [X] Training and inference guidelines
- [X] [Technical Report](https://arxiv.org/abs/2512.07348)

## 🧱 Download Finetuned Models

| Models       | Download Link   |
|------------|-----------------|
| BAGEL-MICo      | 🤗 [Huggingface](https://huggingface.co/kr-cen/BAGEL-MICo)    |
| BLIP3o-Next-MICo | 🤗 [Huggingface](https://huggingface.co/kr-cen/BLIP3o-Next-MICo)    | 
| Lumina-DiMOO-MICo | 🤗 [Huggingface](https://huggingface.co/kr-cen/Lumina-DiMOO-MICo)     |
| OmniGen2-MICo     | 🤗 [Huggingface](https://huggingface.co/kr-cen/OmniGen2-MICo) |   
| Qwen-Image-MICo     | 🤗 Coming Soon |   


## Train

See [TRAIN.md](https://github.com/A113N-W3I/MICo-150K/blob/main/TRAIN.md) for details.

## Inference

See [INFER.md](https://github.com/A113N-W3I/MICo-150K/blob/main/INFER.md) for details.

## 🌟 Citation

~~~
@article{wei2025mico,
  title={MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition},
  author={Wei, Xinyu and Cen, Kangrui and Wei, Hongyang and Guo, Zhen and Li, Bairui and Wang, Zeqing and Zhang, Jinrui and Zhang, Lei},
  journal={arXiv preprint arXiv:2512.07348},
  year={2025}
}
~~~

## 🙋‍♂️ Questions?

If you have any questions or suggestions, feel free to open an [issue](https://github.com/A113N-W3I/MICo-150K/issues) or start a [discussion](https://github.com/A113N-W3I/MICo-150K/discussions).
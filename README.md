# <img src="assets/mico-logo.png" width="60px" align="center"> MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition

Official repository for the paper [MICo-150K: A Comprehensive Dataset for Multi-Image Composition](https://arxiv.org/abs/2512.07348).

## 📢 News

* **Dec 10, 2025:** 📖 We released multi-image composition [training](https://github.com/A113N-W3I/MICo-150K/blob/main/TRAIN.md) & [inference](https://github.com/A113N-W3I/MICo-150K/blob/main/INFER.md) guideline for community models. Our finetuned checkpoints coming soon, stay tuned! 👀
* **Dec 9, 2025:** 🔥 Our paper on [arXiv](https://arxiv.org/abs/2512.07348).
* **Dec 2, 2025:** We released the official [project page](https://mico-150k.github.io/) for MICo-150K.

## Introduction

* **We present MICo-150K**, a large-scale, high-quality dataset for **Multi-Image Composition (MICo)** in controllable image generation. MICo focuses on synthesizing coherent and identity-consistent images from multiple reference inputs—a long-standing challenge due to the lack of suitable training data.
* MICo-150K covers **7 representative MICo tasks**, constructed from carefully curated source images and diverse composition prompts. The dataset is synthesized using strong proprietary models and refined via **human-in-the-loop filtering**, ensuring high fidelity and identity consistency. We further introduce a **Decomposition-and-Recomposition (De&Re)** subset, where real-world complex images are decomposed into components and recomposed, supporting both real and synthetic compositions.
* To enable systematic evaluation, we release **MICo-Bench**, consisting of **1000 curated test cases**, and propose **Weighted-Ref-VIEScore**, a new metric tailored specifically for MICo. We also provide strong baselines, including **Qwen-MICo**, which demonstrates competitive performance with proprietary models while supporting arbitrary multi-image inputs.

![mico-dataset](assets/dataset-case.jpg)

## 📑 Open-Source Plan

- [ ] MICo-150K dataset pt.1
- [ ] MICo-150K dataset pt.2 (carefully post-refined subset)
- [ ] Finetuned checkpoints
- [ ] MICo-Bench
- [X] Training and inference guidelines
- [X] [Technical Report](https://arxiv.org/abs/2512.07348)

## Train

See [TRAIN.md](https://github.com/A113N-W3I/MICo-150K/blob/main/TRAIN.md) for details.

## Inference

See [INFER.md](https://github.com/A113N-W3I/MICo-150K/blob/main/INFER.md) for details.

## 🌟 Citation

~~~
@misc{wei2025mico150kcomprehensivedatasetadvancing,
      title={MICo-150K: A Comprehensive Dataset Advancing Multi-Image Composition}, 
      author={Xinyu Wei and Kangrui Cen and Hongyang Wei and Zhen Guo and Bairui Li and Zeqing Wang and Jinrui Zhang and Lei Zhang},
      year={2025},
      eprint={2512.07348},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.07348}, 
}
~~~

## 🙋‍♂️ Questions?

If you have any questions or suggestions, feel free to open an [issue](https://github.com/A113N-W3I/MICo-150K/issues) or start a [discussion](https://github.com/A113N-W3I/MICo-150K/discussions).
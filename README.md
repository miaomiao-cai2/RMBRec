<div align="center">

# RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors

[![arXiv](https://img.shields.io/badge/arXiv-2601.08705-b31b1b.svg)](https://arxiv.org/abs/2601.08705)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3774904.3792617-blue.svg)](https://doi.org/10.1145/3774904.3792617)
[![Conference](https://img.shields.io/badge/WWW-2026-4b8bbe.svg)](https://www.thewebconf.org/)
[![Python](https://img.shields.io/badge/python-3.9.23-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.3.1-ee4c2c.svg)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/miaomiao-cai2/RMBRec?style=social)](https://github.com/miaomiao-cai2/RMBRec)

**Official PyTorch implementation of our WWW 2026 paper**
*"RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors"*

[Paper](https://doi.org/10.1145/3774904.3792617) · [arXiv](https://arxiv.org/abs/2601.08705) · [Citation](#-citation)

</div>

## 📖 Overview

## 🔧 Requirements

```
python == 3.9.23
torch == 2.3.1+cu121
numba == 0.60.0
numpy == 1.26.3
pandas == 2.3.2
```

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/miaomiao-cai2/RMBRec.git
cd RMBRec

# 2. Create and activate a conda environment
conda create -n rmbrec python=3.9.23
conda activate rmbrec

# 3. Install dependencies matching the versions above
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install numba==0.60.0 numpy==1.26.3 pandas==2.3.2
```

> A GPU with CUDA 12.1 support is recommended for training at scale.

## 📁 Repository Structure

```
RMBRec/
├── main.py          # Entry point: training / evaluation loop and argument parsing
├── model.py          # RMBRec model definition (RRM + ORM on top of the backbone encoder)
├── lightGCN.py       # LightGCN backbone encoder
├── data_set.py       # Multi-behavior dataset loading and preprocessing
├── trainer.py        # Training loop, optimization, early stopping
├── metrics.py        # HR@K, NDCG@K implementations
├── tool.py           # Helper utilities
├── utils.py          # Additional helper functions
├── data/             # Multi-behavior interaction data (Taobao / Beibei / Tmall)
└── README.md
```

> Directory descriptions above reflect the current repository layout — feel free to adjust the wording if any file's actual contents differ.

## 📊 Datasets

Experiments are conducted on three public, widely-used multi-behavior recommendation benchmarks. In each dataset, **purchase** is the target behavior, and the remaining interaction types are treated as auxiliary behaviors.

| Dataset | #Users | #Items | #View | #Collect | #Add-to-cart | #Purchase |
|---|---:|---:|---:|---:|---:|---:|
| Taobao | 48,749 | 39,493 | 1,548,162 | — | 193,747 | 259,771 |
| Tmall | 41,738 | 11,953 | 1,813,498 | 221,514 | 1,996 | 287,158 |
| Beibei | 21,716 | 7,977 | 2,412,586 | — | 642,622 | 304,576 |

Sources: [Taobao](https://tianchi.aliyun.com/dataset/649) · [Tmall](https://tianchi.aliyun.com/dataset/140281) · Beibei (a maternal-and-infant e-commerce dataset commonly used in prior multi-behavior recommendation work).

Behavioral alignment varies sharply across datasets — Beibei's auxiliary behaviors are highly predictive of purchase, whereas Tmall's `add-to-cart` behavior is almost uncorrelated with it — which is precisely the kind of inconsistency RMBRec is designed to be robust to. See the paper's Behavioral Alignment Ratio (BAR) analysis for details.

## 🚀 Usage

RMBRec uses LightGCN as its backbone encoder. Example commands for the datasets currently configured in this repository:

### Taobao

```bash
python main.py --data_name='taobao' --irm_mode='rex' \
    --penalty_irm_coeff=1 --reg_coeff=0.001 --lambda_cl=1 --temperature=0.07
```

### Beibei

```bash
python main.py --data_name='beibei' --irm_mode='rex' \
    --penalty_irm_coeff=1 --reg_coeff=0.001 --lambda_cl=0.8 --temperature=0.07
```

### Tmall

```bash
python main.py --data_name='tmall' --irm_mode='rex' \
    --penalty_irm_coeff=1 --reg_coeff=0.001 --lambda_cl=1 --temperature=0.07
```

**Argument reference:**

| Argument | Meaning |
|---|---|
| `--data_name` | Dataset to use (`taobao`, `beibei`, or `tmall`) |
| `--irm_mode` | Invariance-regularization variant used by ORM (`rex`, `irmv1`, `irmv2`) |
| `--penalty_irm_coeff` | Weight of the Optimization Robustness Module (ORM) loss |
| `--lambda_cl` | Weight of the Representation Robustness Module (RRM) contrastive loss |
| `--reg_coeff` | L2 regularization weight |
| `--temperature` | Temperature used in the RRM contrastive loss |

## 🎛️ Hyperparameters

- **Optimizer:** Adam
- **Backbone:** LightGCN
- **Evaluation protocol:** leave-one-out splitting with full-ranking evaluation, reporting HR@K and NDCG@K for K = 10 and K = 20, averaged over 5 random seeds
- **ORM variant:** three invariance-regularization strategies are supported via `--irm_mode` (`irmv1`, `irmv2`, `rex`); the paper finds **REx** most stable and effective across all three datasets, so it is the recommended default

For the full hyperparameter search space and per-dataset optimal settings, please see Appendix B.2 of the [paper](https://arxiv.org/abs/2601.08705).

## 📝 Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@inproceedings{cai2026rmbrec,
  author    = {Cai, Miaomiao and Zhang, Zhijie and Fang, Junfeng and Cheng, Zhiyong and Wang, Xiang and Wang, Meng},
  title     = {RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  pages     = {},
  publisher = {ACM},
  address   = {Dubai, United Arab Emirates},
  doi       = {10.1145/3774904.3792617}
}
```

## 📄 License

No license file is currently included in this repository. If you intend to release this code under an open-source license (e.g., MIT or Apache-2.0), please add a `LICENSE` file at the repository root — this makes reuse terms explicit for others building on this work.

## 📬 Contact

For questions about the paper or code, please open a [GitHub issue](https://github.com/miaomiao-cai2/RMBRec/issues).

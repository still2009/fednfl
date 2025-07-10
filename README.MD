# 🛡️ Codebase for Privacy-Preserving Federated Learning (FedNFL)

This repository contains the code for our paper:

**“Theoretically Principled Federated Learning for Balancing Privacy and Utility”**  
[arXiv:2305.15148](https://arxiv.org/abs/2305.15148)

---

## 📁 Project Structure

```
├── README.md              # Project documentation
├── basic/                 # Common utility functions
├── data/                  # Dataset files
├── attack/                # Diverse DLG attack implementations
├── inversefed/            # Attack methods cloned from the InverseFed repo
├── fl/                    # FL simulation core
│   ├── client.py          # Federated client logic
│   └── server.py          # Federated server logic
├── demo.yaml              # Example config file with all configurable options
├── experiments.py         # Main entry for launching experiments
├── main.py                # Internal single-experiment runner (used by experiments.py)
└── result_ana.py          # Utilities for automatic result analysis
```

---

## 🚀 Getting Started

The main entry point is `experiments.py`. It supports:
- Handcrafted YAML config files (for flexible single/multi-experiment setup)
- Python-based config functions (for batch or automated tuning)

### 🔧 Example Usages

**Run a single experiment from YAML config:**
```bash
python experiments.py --config debug.yaml --gpu 0 --nproc 1
```

**Run multiple experiments from a group config file:**
```bash
python experiments.py --resubmit --config grp_x.yaml --gpu 0,1,2,3 --nproc 10
```

**Use Customized Python config logic (recommended for batch runs):**
```bash
# Debug mode, the config fuction is raw_dp() in experiments.py
python experiments.py --config raw_dp --gpu 0,1,2,3 --nproc 10 --debug

# Full run
python experiments.py --config raw_dp --gpu 0,1,2,3 --nproc 10
```

*notice: paths and directoies with the code should be manualy changed to ur own*
---

## 🔗 Citation

If you find this repo useful for your research or development, please cite our paper:

```
@article{zhang2023theoretically,
  title={Theoretically principled federated learning for balancing privacy and utility},
  author={Zhang, Xiaojin and Li, Wenjie and Chen, Kai and Xia, Shutao and Yang, Qiang},
  journal={arXiv preprint arXiv:2305.15148},
  year={2023}
}
```
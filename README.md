# CRTFS: Cross-Resolution Transformer Fusion Network for RGB-D Salient Object Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0%2B-red)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

The official implementation of "A Color Information Driven Collaborative Training of Dual Task Parallel Network for Visible and Thermal Infrared Image Fusion and Saliency Object Detection" (IJCV 2026)

## ✨ Key Features

- **Dual-Encoder Architecture**: Combines Vision Transformer (T2T-ViT) for global feature extraction and CNN for local feature extraction
- **Cross-Modal Fusion**: Effective fusion of RGB, Thermal Infrared, and color (YCbCr) information
- **Multi-Scale Processing**: Handles features at multiple resolutions (1/1, 1/4, 1/8, 1/16)
- **Comprehensive Evaluation**: Supports all major SOD evaluation metrics (S-measure, F-measure, E-measure, MAE, etc.)
- **Pretrained Models**: Includes pretrained weights for immediate inference

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Citation](#citation)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.9.0+

### Step-by-step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yukarizz/CRTFS.git
   cd CRTFS
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you have CUDA 12.1, you might need to install PyTorch separately:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation

The project expects data in the following structure:

```
data/
├── VT5000/                    
   +---testset
   |   +---contour
   |   +---GT
   |   +---RGB
   |   \---T_map_soft
   \---train
      +---contour
      +---GT
      +---RGB
      \---T_map_soft
```

**Supported Datasets**: VT5000, NJUD, NLPR, DUTLF-Depth, ReDWeb-S, STERE, SSD, SIP, RGBD135, LFSD

## 🏃 Quick Start

### Inference with Pretrained Model

1. **Download pretrained weights** (already included in `checkpoint/`):
   - `CRTFS.pth` (436MB): Main model weights
   - `80.7_T2T_ViT_t_14.pth.tar` (86MB): Pretrained T2T-ViT backbone

2. **Run inference on your data**:
   ```bash
   python train_test_eval.py --Testing True --data_root ./data/ --test_paths demoset
   ```

   Predictions will be saved to `preds/` directory.

## 🏋️ Training

### Training from Scratch

1. **Prepare your training dataset** in the `data/` directory (see [Dataset Preparation](#dataset-preparation))

2. **Start training**:
   ```bash
   python train_test_eval.py --Training True --data_root ./data/ --trainset VT5000 --epochs 200 --batch_size 8
   ```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--Training` | Enable training mode | `False` |
| `--data_root` | Path to dataset | `./data/` |
| `--trainset` | Training dataset name | `VT5000` |
| `--epochs` | Number of training epochs | `200` |
| `--batch_size` | Batch size | `2` |
| `--lr` | Learning rate | `1e-4` |
| `--img_size` | Input image size | `224` |
| `--save_model_dir` | Model save directory | `checkpoint/` |

### Resume Training
```bash
python train_test_eval.py --Training True --resume checkpoint/latest.pth
```

## 🔍 Testing

### Generate Predictions

```bash
python train_test_eval.py --Testing True --data_root ./data/ --test_paths demoset
```

**Output**: Saliency maps saved in `preds/` directory.

### Test Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--Testing` | Enable testing mode | `True` |
| `--test_model_name` | Model checkpoint path | `./checkpoint/CRTFS.pth` |
| `--save_test_path_root` | Output directory | `preds/` |
| `--test_paths` | Test dataset name(s) | `test` |

## 📈 Evaluation

### Comprehensive Evaluation

The project includes a complete evaluation toolkit with all standard SOD metrics:

```bash
python train_test_eval.py --Evaluation True --methods CRTFS --save_dir ./results/
```

### Evaluation Metrics

The evaluation includes:
- **S-measure (Sₘ)**: Structural similarity measure
- **F-measure (Fₘ)**: Weighted harmonic mean of precision and recall
  - max F-measure
  - mean F-measure
  - adaptive F-measure
- **E-measure (Eₘ)**: Enhanced alignment measure
  - max E-measure
  - mean E-measure
  - adaptive E-measure
- **MAE**: Mean Absolute Error
- **Fbw-measure**: Weighted F-measure

### Detailed Evaluation (Optional)

For more control over evaluation:
```bash
cd Evaluation/SOD_Evaluation_Metrics-main/
python main.py --pred_root_dir ../../preds/ --gt_root_dir ../../data/ --save_dir ../../score/
```

### Visualization
Generate PR curves and F-measure curves:
```bash
cd Evaluation/SOD_Evaluation_Metrics-main/
python draw_curve.py
```


## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex

```

## 🙏 Acknowledgments

- This project builds upon [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT)
- Evaluation metrics from [SOD_Evaluation_Metrics](https://github.com/lartpang/SOD_Evaluation_Metrics)
- Thanks to all open-source contributors in the computer vision community

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or issues, please:
1. Open an issue on GitHub
2. Provide detailed information about your problem
3. Include relevant code snippets and error messages

---

**Happy Coding!** 🚀
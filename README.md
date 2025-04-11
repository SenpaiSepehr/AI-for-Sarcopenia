# Automated Sarcopenia Measurement from CT Images

This repository contains the full pipeline developed by Team 10 (UBC Biomedical Engineering) for automating sarcopenia assessment from abdominal CT scans using deep learning. Our system performs:

1. **Third Lumbar Vertebra (L3) Localization**  
2. **Skeletal Muscle Segmentation on the L3 Slice**

This project significantly reduces clinical workload by automating a process that previously required manual review of CT volumes.

## 🔍 Project Overview

Sarcopenia is a musculoskeletal disorder defined by loss of muscle mass and function, and CT-based skeletal muscle analysis at the L3 vertebra is the gold standard for diagnosis. However, manual annotation is time-consuming.

### Pipeline Breakdown:

- **Input:** 3D CT volume
- **Step 1:** Frontal and sagittal MIP projections generated
- **Step 2:** YOLOv11 + Transformer models localize L3 slice from MIPs
- **Step 3:** Hybrid CNN-Transformer model segments skeletal muscle on the L3 slice
- **Output:** Muscle segmentation mask + sarcopenia indices (e.g., SMA, SMI)

## 📊 Results

| Module                     | Metric                        | Performance      |
|---------------------------|-------------------------------|------------------|
| L3 Localization (Frontal) | Absolute Error (mm)           | Mean: 2.11 ± 1.95 |
| L3 Localization (Sagittal)| Absolute Error (mm)           | Mean: 3.44 ± 4.36 |
| Muscle Segmentation       | Dice Similarity Coefficient   | Mean: 0.913 ± 0.03 |

Our models match or outperform radiologist-level annotation performance across the test set.

## 📁 Download Pretrained Weights

To run the full pipeline, please download the pretrained weights from the link below and place them into a `weights/` folder:

🔗 [Download Weights from Google Drive](https://drive.google.com/drive/folders/1GqGfKJEG5JqPYbE4j2RNQixb5FWINHJN?usp=drive_link)

Files to download:
- `best_frontal_l3localization.pt`
- `best_sagittal_l3localization.pt`
- `best_segmentation.pt`

## 🛠️ Usage

Clone the repository and ensure dependencies are installed:

```bash
git clone https://github.com/yourusername/sarcopenia-ct-pipeline.git
cd sarcopenia-ct-pipeline
pip install -r requirements.txt
```

## Run Streamlit UI
```bash
python -m streamlit run main.py --server.maxUploadSize=500
```



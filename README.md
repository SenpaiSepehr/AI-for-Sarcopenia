# Automated Sarcopenia Measurement from CT Images

This repository contains the full pipeline developed by Team 10 (UBC Biomedical Engineering) for automating sarcopenia assessment from abdominal CT scans using deep learning. Our system performs:

1. **Third Lumbar Vertebra (L3) Localization**  
2. **Skeletal Muscle Segmentation on the L3 Slice**

This project significantly reduces clinical workload by automating a process that previously required manual review of CT volumes.

## 🔍 Project Overview

Sarcopenia is a musculoskeletal disorder defined by loss of muscle mass and function, and CT-based skeletal muscle analysis at the L3 vertebra is the gold standard for diagnosis. However, manual annotation is time-consuming.

### Pipeline Breakdown:

- **Input:** 3D CT volume
- **Step 1:** Generation of frontal and sagittal maximum intensity projections (MIP)
- **Step 2:** YOLOv11 model localizes L3 slice from MIPs
- **Step 3:** Hybrid CNN-Transformer model segments skeletal muscle on the L3 slice
- **Output:** Skeletal muscle segmentation mask and area measurements (SMA)

## 📊 Results

| Module                     | Metric                        | Test Performance      |
|---------------------------|-------------------------------|------------------|
| L3 Localization (Frontal) | Absolute Error (mm)           | Mean: 2.11 ± 1.95 |
| L3 Localization (Sagittal)| Absolute Error (mm)           | Mean: 3.44 ± 4.36 |
| Muscle Segmentation       | Dice Similarity Coefficient   | Mean: 0.913 ± 0.03 |

Our models match or outperform radiologist-level annotation performance when assessed on an independent test set.

## 📁 Download Pretrained Weights

To run the full pipeline, please download the pretrained weights from the link below and place them into a `weights/` folder:

🔗 [Download Weights from Google Drive](https://drive.google.com/drive/folders/1GqGfKJEG5JqPYbE4j2RNQixb5FWINHJN?usp=drive_link)

Files to download:
- `best_frontal_l3localization.pt`
- `best_sagittal_l3localization.pt`
- `best_segmentation.pt`

## 📁 Download Example CT Volumes
🔗 [Download Example Data](https://drive.google.com/drive/folders/185lembeKTAYMUEB5z3hS212w7fDph88w?usp=sharing)

Five example CT volumes are provided for testing the UI and reproducing results in the Juptyer Notebook/vingette. The volumes are open-source and obtained from the Kaggle Liver and Liver Tumor Segmentation (LiTS) challenge. 

## 🛠️ Usage

Clone the repository and ensure dependencies are installed:

```bash
git clone https://github.com/SenpaiSepehr/AI-for-Sarcopenia.git
cd AI-for-Sarcopenia
pip install -r requirements.txt
```

## Run Streamlit UI
```bash
cd UI
python -m streamlit run main.py --server.maxUploadSize=500
```




# YOLOv8 Domain Adaptation for Challenging Object Detection

This project focuses on improving the YOLOv8 object detection model's performance on difficult domains such as occluded, rotated, and unusually angled objects.

## 🔍 Objective
To enhance YOLOv8's accuracy by applying a complete data-centric ML pipeline on a custom dataset collected from various sources.

## 📌 Highlights
- ✅ Manual collection of object images from unusual domains (e.g., occluded or rotated views)
- ✅ Multithreaded web scraping using Python
- ✅ Logged metadata (image validity, response code) to `download_metadata.csv`
- ✅ Removed corrupt/broken images
- ✅ Applied data augmentation to over 3000 images (`DATA ARGUMENTATION.ipynb`)
- ✅ Pseudolabel generation and noise filtering using KMeans clustering (8 clusters)
- ✅ Retained only high-confidence pseudolabels (confidence > 0.4)
- ✅ Retrained YOLOv8 with first 10 layers frozen (`yolov8newdomain (2).ipynb`)
- ✅ Achieved **precision = 0.67** and **recall = 0.814**
- ✅ Final model: `best_custom.pt`

## 📁 Repository Contents
| File                          | Description                              |
|-------------------------------|------------------------------------------|
| `DATA COLLECTION.ipynb`       | Scraping and downloading images          |
| `DATA ARGUMENTATION.ipynb`    | Image augmentation logic                 |
| `DATAPREPROCESSING.ipynb`     | Preprocessing images for training        |
| `pseudolabelandkmeans.ipynb`  | Label filtering and clustering           |
| `yolov8newdomain (2).ipynb`   | Final training script with freezing      |
| `realtime.ipynb`              | Real-time detection (optional)           |
| `download_metadata.csv`       | Metadata of scraped images               |
| `results(1).csv`              | Results                                  |
| `F1_curve.png`                | Precision-Recall / F1 Curve              |
| `best_custom.pt`              | Final trained YOLOv8 model               |

## 🚀 Results
- Precision: **0.67**
- Recall: **0.814**
- Training using frozen layers: First 10
- KMeans Clustering: **8 clusters**
- Filtered pseudolabels: Confidence threshold **0.4**

## 🛠️ Tech Stack
- Python, OpenCV, Ultralytics YOLOv8
- Jupyter Notebooks, CSV Logging, KMeans (Sklearn)
- Data Augmentation, Multithreading

## 📄 License
This project is licensed under the MIT License.

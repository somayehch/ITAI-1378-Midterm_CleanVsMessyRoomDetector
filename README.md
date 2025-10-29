[README-2.md](https://github.com/user-attachments/files/23221130/README-2.md)
# 🧠 Clean vs Messy Room Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Proposal%20%2F%20MVP-brightgreen)

> **Purpose:** A practical computer-vision project that classifies room photos as **Clean** or **Messy** to reduce manual inspection time for homes, hotels, and offices.

---

## ✨ Summary (Resume-Ready)
**Built an image classification system** using **ResNet50 + PyTorch** to predict room cleanliness with a target of **≥85% accuracy** and **<1s inference**. Implemented **transfer learning** and **data augmentation** to deliver results with a small, custom dataset. Designed for **real-world use** and future integration with **smart-home or facility management** workflows.

---

## 👤 Project & Course
- **Author:** *Somayeh Choboderazi*
- **Course:** ITAI 1378 – Computer Vision
- **Instructor:** Prof. Patricia McManus
- **Tier:** **Tier 1 – Practical Application** (uses pre-trained models; scoped for a single developer)

---

## 🚀 Problem → Solution
- **Problem:** Manual room checks are slow, inconsistent, and costly—especially in large facilities.
- **Who cares:** Homeowners, students, hotels, offices, and cleaning services.
- **Solution:** A lightweight AI model that classifies room images as *Clean* or *Messy*, enabling targeted cleaning and time savings.

---

## 🧱 Technical Architecture
**System Flow:**  
`[Input Image] → [Preprocessing] → [ResNet50 Feature Extraction] → [Classifier Layer] → [Output: Clean/Messy]`

**Stack**
- **Model:** ResNet50 (pre-trained; fine-tuned for binary classification)
- **Framework:** PyTorch (Google Colab GPU)
- **Data Handling:** TorchVision & OpenCV (resize, normalize, augment)
- **Training:** Transfer learning, Cross-Entropy Loss, Adam optimizer
- **Targets:** Accuracy ≥ 85%, Inference ≤ 1s/image

---

## 📚 Dataset Plan
- **Sources:** Google Images + self-captured photos
- **Classes:** `clean`, `messy`
- **Size:** ~200 images (balanced)
- **Split:** 70% train / 20% val / 10% test
- **Preprocessing:** Resize 224×224, normalize
- **Augmentation:** Flip, rotation, brightness/contrast
- **Reference:** Inspired by open datasets on Roboflow

> **Note:** If using public data, attribute sources in `data/README.md`.

---

## 📏 Evaluation
| Metric | Target | Why it matters |
|---|---|---|
| **Accuracy** | ≥ 85% | Core correctness for binary decision |
| **Latency** | ≤ 1s/image | Practical for real-time checks |
| **Precision / Recall** | Balanced | Avoids bias toward either class |
| **Confusion Matrix** | N/A | Reveals edge cases (e.g., “cluttered but clean”) |

---

## 📅 Milestones (Weeks 10–15)
- **W10:** Collect & label data → *Dataset ready*
- **W11:** Train & fine-tune model → *Baseline working*
- **W12:** Test & optimize → *≥85% accuracy*
- **W13:** Demo & visuals → *Usable prototype*
- **W14:** Docs & validation → *Submission-ready*
- **W15:** Present → *Deliver & reflect*

---

## ⚠️ Risks & Mitigation
| Risk | Probability | Mitigation |
|---|---|---|
| Low accuracy | Medium | Add data, tune LR, augment more |
| Not enough data | High | Pull additional sets from Roboflow |
| Lighting/angles vary | Medium | Normalize + augmentation |
| Overfitting | Medium | Dropout, early stopping, regularization |
| Colab GPU limits | High | Save checkpoints; smaller batch size |
| Time constraints | Medium | Follow weekly plan; lock scope |

---

## 🛠️ Setup & Usage

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Inference (example)
```python
from PIL import Image
import torch
from torchvision import transforms, models

# load model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # clean vs messy
model.load_state_dict(torch.load("weights/best.pt", map_location="cpu"))
model.eval()

# preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("sample.jpg").convert("RGB")
x = preprocess(img).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=1).item()

print("Prediction:", ["clean", "messy"][pred])
```

> Save your trained weights to `weights/best.pt` to use the above snippet.

---

## 📦 Repository Structure
```
Clean-vs-Messy-Room-Detector/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_exploration.ipynb
├── data/
│   └── README.md  # describe sources/attribution
├── docs/
│   └── proposal.pdf  # your slides export
└── src/
    ├── train.py
    ├── infer.py
    └── utils.py
```

---

## 📎 Requirements (suggested)
```
torch
torchvision
opencv-python
pillow
numpy
matplotlib
scikit-learn
tqdm
```

---

## 🧾 AI Usage Log

I used AI tools in a limited and responsible way during this project to improve clarity and organization.  
Below is a short record of how AI was used:

| Date | AI Tool | How I Used It | What I Did After |
|------|----------|---------------|------------------|
| Oct 2025 | ChatGPT (GPT-5) | Helped me organize my project plan and write the README layout | I edited all content and verified every section myself |
| Oct 2025 | ChatGPT (GPT-5) | Explained some technical terms (like transfer learning, metrics) in simple words | I rephrased them in my own style |
| Oct 2025 | ChatGPT (GPT-5) | Suggested example code structure for training and inference scripts | I customized and wrote the actual code myself |

> I used AI only for guidance and explanation. All writing, editing, and final decisions were done by **Somayeh Choboderazi**.


---

## ✅ Evaluation Criteria (Instructor Rubric Reference)

| Component | Description |
|------------|--------------|
| **Problem & Solution Clarity** | Defines a real problem with a clear, feasible solution |
| **Technical Approach** | Appropriate use of Computer Vision / ML methods and tools |
| **Data Plan & Feasibility** | Achievable with available datasets and resources |
| **Organization & Timeline** | Logical week-by-week plan with milestones |
| **Presentation Quality** | Professional slides, clear and concise communication |
| **GitHub Setup** | Proper repository structure, documentation, and licensing |
| **Creativity & Originality** | Demonstrates innovative or well-scoped application |

---

## 🔗 Citations & References
1. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition (ResNet)*. CVPR.  
2. PyTorch (2025). *Documentation*. https://pytorch.org  
3. Roboflow (2024). *Open Datasets*. https://roboflow.com/datasets  
4. scikit-learn (2025). *Metrics*. https://scikit-learn.org  
5. OpenCV (2025). *Library*. https://opencv.org  
6. Google Colab (2025). *Docs*. https://colab.research.google.com

---

## 🧾 License
MIT — feel free to use and adapt for learning or demos.

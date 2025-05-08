# YouTube Sentiment Analysis (Telugu/English/Transliterated) - Model Development Notebooks

This repository houses the Jupyter notebooks used during the development, fine-tuning, and testing phases of the sentiment analysis model for YouTube comments, supporting Telugu, English, and transliterated text.

## Repository Contents

- `finetune_model.ipynb`: Notebook detailing the process of fine-tuning the sentiment analysis model.
- `test.ipynb`: Notebook containing scripts and examples for testing the performance of the trained model.

## Running Notebooks Locally

To run these notebooks:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/gajula21/finetune_sentiment_telugu.git](https://github.com/gajula21/finetune_sentiment_telugu.git)
    cd finetune_sentiment_telugu
    ```

2.  **Set up a Python Environment:** Ensure you have a Python environment (e.g., using `venv` or `conda`). Activate it.
    ```bash
    # Example using venv
    python -m venv venv
    # On Windows: venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```

3.  **Install Dependencies Manually:** Based on the notebooks, you will need libraries like `transformers`, `torch` (or `tensorflow`), `pandas`, `jupyterlab` or `notebook`. Install them manually using pip:
    ```bash
    pip install transformers torch pandas jupyterlab # Add other libraries as needed
    ```

4.  **Launch JupyterLab or Jupyter Notebook:**
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
    Your browser will open, allowing you to access and run the `.ipynb` files.

## Model Information

### Model Overview

This model is a fine-tuned version of [AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual](https://huggingface.co/AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual). While the base model was trained on multilingual English-only YouTube comments, this version has been fine-tuned on a large dataset of Telugu comments, enabling it to classify Telugu (native script), transliterated Telugu, and English YouTube comments into sentiment categories: Negative, Neutral, and Positive.

### Model Details

- **Base model:** AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual
- **Fine-tuned for:** Telugu + English YouTube comment sentiment analysis
- **Languages Supported:** Telugu (native script), Transliterated Telugu, English
- **Labels:** 0: Negative, 1: Neutral, 2: Positive

### Dataset & Labeling

- **Source:** Comments extracted from YouTube using the YouTube Data API.
- **Comment Count:** Train set: 73,943 comments, Validation set: 8,216 comments
- **Labeling Method:** Comments were labeled using Gemini 1.5 Pro (Google’s LLM) via a sentiment classification prompt.

### How to Use (via Transformers Library)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "gajula21/youtube-sentiment-model-telugu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

comments = [
    "ఈ సినిమా చాలా బాగుంది!",
    "ఈ వీడియో చాలా బోరు పడింది",
    "ఇది మామూలు వీడియో",
]

inputs = tokenizer(comments, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=1)
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
sentiments = [label_mapping[p.item()] for p in predictions]
print(sentiments)
```
## Training Configuration
Framework: Hugging Face Transformers (PyTorch)
Tokenizer: AutoTokenizer from base model
Loss Function: CrossEntropyLoss with label_smoothing=0.1
Batch Size: 1176 (per device)
Gradient Accumulation Steps: 2
Learning Rate: 1e-5
Weight Decay: 0.05
Epochs: 3
Evaluation Strategy: Every 125 steps
Early Stopping: Patience of 5 evaluation steps
Mixed Precision: Enabled (fp16)

### Evaluation Results

| Step | Training Loss | Validation Loss | Accuracy |
|------|---------------|-----------------|----------|
| 125  | 0.7637        | 0.7355          | 72.97%   |
| 250  | 0.7289        | 0.7110          | 74.57%   |
| 375  | 0.7155        | 0.6982          | 75.72%   |
| 500  | 0.6912        | 0.7005          | 75.58%   |
| 625  | 0.6851        | 0.6821          | 76.79%   |
| 750  | 0.6606        | 0.6897          | 76.61%   |
| 875  | 0.6464        | 0.6838          | 76.68%   |
| 1000 | 0.6542        | 0.6676          | 77.45%   |
| 1125 | 0.6501        | 0.6602          | 78.04%   |
| 1250 | 0.6374        | 0.6730          | 77.81%   |
| 1375 | 0.6143        | 0.6682          | 77.99%   |
| 1500 | 0.6175        | 0.6665          | 78.10%   |
| 1625 | 0.6183        | 0.6646          | 78.16%   |

## Related Projects
- **Streamlit Application:** The web application client that uses the sentiment analysis API.(https://github.com/gajula21/youtube-sentiment-telugu-app)
- **Hosted API:** The Hugging Face Space hosting the FastAPI sentiment analysis API.(https://huggingface.co/spaces/gajula21/telugu-sentiment-api)
- **Hugging Face Model Card:** The direct link to the model on Hugging Face Hub.(https://huggingface.co/gajula21/youtube-sentiment-model-telugu)

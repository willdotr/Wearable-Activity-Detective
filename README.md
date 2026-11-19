# Wearable Activity Detective

**Can a neural network tell walking from climbing stairs using only a phone's accelerometer?**
Guess the activity from a raw sensor trace, then see what the model thought, and explore where it gets confused.

### 👉 [Open the live exhibit](https://willdotr.github.io/Wearable-Activity-Detective/)

---

## What's in the exhibit

| Section | What you do |
|---|---|
| **1. Guess the activity** | Look at a 2.56 s accelerometer trace, pick an activity, and compare your guess with the model's. |
| **2. Why did the model get this wrong?** | Browse real mistakes next to typical traces of the true and predicted activity. |
| **3. Test results** | See the model vs. baseline and the full confusion matrix. |
| **4. Ask Jose (bot)** | Chat with an in-browser language model (WebLLM, needs WebGPU) that answers from a short guide about the project. |

## Results

| Model | Test accuracy |
|---|---|
| Baseline (always predict the most common training activity) | 18.2% |
| **Neural network (561 → 32 → 6)** | **95.05%** |

The biggest mistakes were **Sitting → Standing** (73 windows) and **Walking Upstairs → Walking** (20 windows). Sitting and standing are both nearly motionless, and stair climbing looks a lot like walking in a single short window.

## How it works

- **Data:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (30 volunteers, six activities, pre-split by person).
- **Model:** one-hidden-layer MLP written in plain NumPy (Adam, dropout), using the dataset's 561 engineered features.
- **Evaluation:** hyperparameters tuned on 5 held-out *training* subjects; the test set was used once at the end.
- **Site:** static HTML/JS, with predictions and raw traces exported from `train.py`.
- **Chat:** [WebLLM](https://webllm.mlc.ai/) running Qwen2.5-1.5B in the browser. Jose is given a guide with the project's data, results and limitations, and is told to say when the guide doesn't cover a question.

## Run it yourself

```bash
# View the site locally (http://localhost:8000)
bash run_app.sh

# Retrain the model and regenerate docs/data/*
# (first download and unzip the UCI HAR dataset into a folder named "UCI HAR Dataset")
py -I train.py
```

Requires Python 3 with `numpy`. Chat needs a WebGPU browser (recent Chrome or Edge); the rest of the exhibit works without it.

## Limitations

Only 30 volunteers wore the same phone at the waist, and the model was tested only on this dataset's test subjects. We did **not** test it on other datasets, phones, body positions or people. It is not a medical or safety tool.

## Credits

Data: Anguita et al., 2013, UCI Machine Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

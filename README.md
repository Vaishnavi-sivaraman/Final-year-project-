# Final-year-project-
# ReadMe - Tamil Medical Assistant

## 1. URL / Source for Dataset
The dataset used for training the models is sourced from:
- **Custom Siddha Medical Dataset**: Created by extracting text from Tamil Siddha medicine books and synthesizing it into a question-answer format.
- **Audio Dataset**: [https://asr.iitm.ac.in/dataset](https://asr.iitm.ac.in/dataset)
---

## 2. Software and Hardware Requirements

### Software Requirements:
- Python 3.8+
- Required Python Libraries:
  - `torch`
  - `torchaudio`
  - `transformers`
  - `gradio`
  - `deep_translator`
  - `gtts`
- Pre-trained Models:
  - **Whisper ASR Model** (Fine-tuned on Tamil-English code-mixed speech)
  - **XLM-RoBERTa-based QA Model** (Fine-tuned on Tamil QA data)
- Operating System: Ubuntu 20.04+ / Windows 10+
- Jupyter Notebook / Google Colab (Optional for training/testing)

### Hardware Requirements:
- GPU (Recommended for faster inference)
- Minimum 8GB RAM (16GB+ preferred)
- 50GB+ Storage (for models and datasets)

---

## 3. Instructions to Execute the Source Code

### Step 1: Clone or Download the Repository
```bash
git clone <repository_link>
cd <repository_folder>
```

### Step 2: Install Dependencies
```bash
pip install torch torchaudio transformers gradio deep_translator gtts
```

### Step 3: Load Pretrained Models
Ensure the following models are available in the respective directories:
- Whisper ASR model at `/content/drive/MyDrive/IIT/Whisper_Tamil_model`
- XLM-RoBERTa-based QA model at `/content/drive/MyDrive/IIT/review_2_25/3ep_1lakh_distilled_xlm_roberta`

If not available, download them from Hugging Face and place them in the specified directories.

### Step 4: Run the Application
```bash
python tamil_medical_assistant.py
```

### Step 5: Access the Gradio Interface
Once the script is running, access the web UI using the displayed URL.

### Step 6: Usage Instructions
1. **Record an audio query** using the Gradio interface.
2. The application will:
   - Transcribe the Tamil-English speech.
   - Translate it into pure Tamil.
   - Extract the most relevant medical answer.
   - Convert the answer to speech.
3. The outputs (text and audio) will be displayed on the interface.

### Step 7: Stop the Application
Press `CTRL + C` in the terminal to stop the server.

---

## Additional Notes
- Ensure your microphone is enabled if using the recording feature.
- For GPU acceleration, install `torch` with CUDA support (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`).
- Modify the `context_list` in the script to add more medical contexts if needed.
- If deploying on a server, use `interface.launch(server_name='0.0.0.0', server_port=7860)` for public access.

---



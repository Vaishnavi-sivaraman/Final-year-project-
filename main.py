import torch
import torchaudio
import os
import base64
import numpy as np
import gradio as gr
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from deep_translator import GoogleTranslator  # ✅ Replaced googletrans with deep_translator
from gtts import gTTS

# Load models
whisper_model_directory = "/content/drive/MyDrive/IIT/Whisper_Tamil_model"
processor = WhisperProcessor.from_pretrained(whisper_model_directory)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_directory)

model_name = "/content/drive/MyDrive/IIT/review_2_25/3ep_1lakh_distilled_xlm_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=tokenizer)

translator = GoogleTranslator(source="auto", target="ta")  # ✅ Deep Translator initialized

# Medical contexts (rotate through them)
context_list = [
    """கோழி, வசம்பு, வெள்ளைப்பூண்டு, நிலவேர், முசுறுமுட்டை, கோழிமுட்டை (வகைக்கு 35 கி) ஆகியவற்றை நெய், ஆமணக் கெண்ணெய், வேப்பெண்ணெய் (வகைக்கு 1.3 லி) ஆகியவற்றுடன் சேர்த்து, எண்ணெய் கலவையில் காய்ச்சி, பின்னர் வடிகட்டவும். கோழியை சுத்தம் செய்து, கிழியாக கட்டி, அந்த வடிகட்டிய எண்ணெயில் காய்ச்சி, பத்திரமாகப் பாதுகாக்கவும். இதனை காலை மற்றும் மாலை 16 மில்லி அளவில் பயன்படுத்தவும்.""",

    """எள்ளெண்ணெய் 325 மில்லி, ஏலம் 35 கி, கார்போக அரிசி 35 கி ஆகியவற்றை எடுத்துக் கொண்டு, ஏலத்தையும் கார்போக அரிசியையும் பொடியாக்கி, அந்தப் பொடியை எள்ளெண்ணெயில் சேர்த்து நன்றாகக் கலக்கி, வெயிலில் வைக்கவும். பின்னர், அக்கி கண்ட இடமெல்லாம் கோழி இறகால் எடுத்து தடவவும்.""",

    """பூவரசம் பட்டை, அத்திப்பட்டை, வெள்ளருகுச் சாறு (35 மி.லி.), கோவைச் சாறு, சங்கங்குப்பி சாறு, சிற்றாமணக்கு எண்ணெய் (4 லி), அமுக்கிராக்கிழங்கு, பரங்கிச் சக்கை (வகைக்கு 105 கி), இலவங்கப் பட்டை, கார்போக அரிசி, கடுகுரோகணி, வாய் விளங்கம் (வகைக்கு 35 கி), திப்பிலி, மிளகு (வகைக்கு 17.5 கி), பூரம் (8.5 கி) ஆகியவற்றில் அமுக்கிராக்கிழங்கு முதல் பூரம் வரை உள்ள பொருட்களைத் தூளாக்கி, மேலே குறிப்பிட்ட சாறுகளை சிற்றாமணக்கு எண்ணெயில் ஊற்றி அடுப்பில் வைத்து எரித்து, கொதிக்கும் போது அந்தப் பொடிகளை சேர்த்து நன்றாகக் கலக்கி வடிக்கவும். இதனை 17.5 கி வீதம் மூன்று வேளை, ஏழு நாட்கள் விட்டு ஏழு நாட்கள் சாப்பிடவும்.""",

    """சின்னி இலைச்சாறு, அவுரி இலைச்சாறு, கொடிக் கழற்சி இலைச்சாறு, நாவி இலைச்சாறு, காட்டு ஆமணக்கு இலைச்சாறு, கோழி அவரைச்சாறு (வகைக்கு 650 மி.லி.), பூண்டு (20 கி), பெருங்காயம் (4 கி), கருஞ்சீரகம், கஸ்தூரி மஞ்சள், வசம்பு, கார்போக அரிசி, காட்டுச்சீரகம் (வகைக்கு 1.3 கி, 8 கி), மற்றும் சிற்றாமணக்கு ஆகியவற்றை மேலே குறிப்பிட்ட சாறு வகைகளுடன் சேர்த்து, கடை மருந்துகளையும் சேர்த்து, எண்ணெய் பதத்திற்கு காய்ச்சி வடித்து வைத்துக் கொள்ளவும். இதனை வேளைக்கு 25 மி.லி வீதம் ஆறு நாட்கள் பயன்படுத்தவும்.""",

    """கழுதைப்பாலும் நல்லெண்ணெய்யும் சம அளவில் எடுத்துக் கொண்டு, இரண்டையும் சேர்த்து எரித்து, பின்னர் வடித்துக் கொள்ளவும். 16 மி.லி. அளவில் உள்ளுக்குக் கொடுக்கவும். பத்தியமாக காரமும் புளிப்பும் தவிர்க்க வேண்டும்."""
]

# Function to rotate context list
def get_next_context():
    global context_list
    current_context = context_list.pop(0)
    context_list.append(current_context)
    return current_context

# Transcribe audio to text
def transcribe_audio(audio_file):
    audio_waveform, sample_rate = torchaudio.load(audio_file)

    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_waveform = transform(audio_waveform)

    audio_np = audio_waveform.squeeze().numpy()
    input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features

    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# ✅ Translate text to Tamil using Deep Translator
def convert_to_pure_tamil(text):
    return translator.translate(text)

# Extract answer from context
def get_answer(question):
    context = get_next_context()
    answer = qa_pipeline({'context': context, 'question': question})
    return context[:200],answer['answer']  # Returning a snippet of the context for reference

# Convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang="ta")
    tts.save("output_speech.mp3")
    return "output_speech.mp3"

# Full pipeline function
def process_audio(audio_file):
    transcription = transcribe_audio(audio_file)
    pure_tamil_output = convert_to_pure_tamil(transcription)
    answer, context_snippet = get_answer(pure_tamil_output)
    speech_output = text_to_speech(answer)
    return transcription, pure_tamil_output, answer, speech_output

# Fix for Gradio Audio Component
interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(sources=["microphone"], type="filepath"),  # Fixed the deprecated argument
    outputs=[
        gr.Textbox(label="Mixed Output (Tamil-English)"),
        gr.Textbox(label="Pure Tamil Output"),
        gr.Textbox(label="Extracted Answer"),
        gr.Audio(label="Answer Speech Output")
    ],
    title="Tamil Medical Assistant",
    description="Record your question, get the transcribed text, translated Tamil text, relevant medical answer, and hear the answer in speech."
)

# Launch the Gradio app
interface.launch(debug=True, share=True)

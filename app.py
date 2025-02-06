!pip install flask flask-ngrok transformers sentencepiece torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the translation model
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
def translate_english_to_spanish(text):
    tokenizer.src_lang = "en"  # Set source language to English
    encoded_text = tokenizer(text, return_tensors="pt")  # Tokenize input text
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id["es"])  # Translate to Spanish
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]  # Decode output
    return translated_text
import gradio as gr

# Create Gradio Interface
interface = gr.Interface(
    fn=translate_english_to_spanish,  # Function to run when input is given
    inputs=gr.Textbox(lines=2, placeholder="Enter text in English..."),  # Input field
    outputs="text",  # Output text
    title="English to Spanish Translator",
    description="Enter an English sentence, and the model will translate 
import gradio as gr

# Create Gradio Interface
interface = gr.Interface(
    fn=translate_english_to_spanish,  # Function to run when input is given
    inputs=gr.Textbox(lines=2, placeholder="Enter text in English..."),  # Input field
    outputs="text",  # Output text
    title="English to Spanish Translator",
    description="Enter an English sentence, and the model will translate it into Spanish."
)

# Launch the Gradio web app
interface.launch(share=True)


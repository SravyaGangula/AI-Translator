import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Translation function
def translate_text(text, target_language):
    tokenizer.src_lang = "en"
    encoded_text = tokenizer(text, return_tensors="pt")
    lang_codes = {"Spanish": "es", "Telugu": "te"}
    target_lang_code = lang_codes.get(target_language, "es")
    generated_tokens = model.generate(
        **encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang_code]
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Gradio app
interface = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter text in English...", label="Enter English Text"),
        gr.Dropdown(["Spanish", "Telugu"], value="Spanish", label="Select Target Language"),
    ],
    outputs="text",
    title="🌍 AI Translator",
    description="Enter an English sentence, and the model will translate it into Spanish or Telugu.",
)
interface.launch()model_name = "facebook/m2m100_418M"
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


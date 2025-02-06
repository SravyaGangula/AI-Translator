import gradio as gr
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model
model_name = "facebook/m2m100_418M"  # Fixed model name
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

interface.launch()

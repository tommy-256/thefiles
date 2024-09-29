from unsloth import FastLanguageModel
import gradio as gr

# Load the LoRA model directly using Unsloth
MODEL_NAME = "thatstommy/lora_model"  # Replace with your actual LoRA model name
max_seq_length = 2048  # Set maximum sequence length
load_in_4bit = True  # Enable 4-bit quantization for memory efficiency

# Initialize the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
)

# Initialize chatbot messages list
chatbot_messages = []
contextual_prompt = "You are a friendly chatbot."

def generate(user_input):
    # Add user input to messages history
    chatbot_messages.append(f"User: {user_input}")

    # Create input text for the model with contextual prompt and history
    input_text = f"{contextual_prompt}\n" + "\n".join(chatbot_messages[-3:])  # Limit to last 3 messages

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text from the model
    outputs = model.generate(**inputs)

    # Decode the output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add model response to messages history
    chatbot_messages.append(f"Bot: {response}")

    # Limit memory retention to last 3 messages
    while len(chatbot_messages) > 3:
        chatbot_messages.pop(0)  # Remove oldest message

    return response  # Return only the response for output textbox

# Create Gradio interface with Blocks
with gr.Blocks() as interface:
    gr.Markdown("<div style='background-color: #003366; padding: 20px; text-align: center;'><h1 style='color: white;'>School of Tomorrow</h1></div>")
    
    output_text = gr.Textbox(label="Output:", interactive=False)
    input_text = gr.Textbox(label="Enter your prompt:", placeholder="Type your prompt here...")
    
    btn = gr.Button("Generate")
    
    btn.click(fn=generate, inputs=input_text, outputs=output_text)

# Launch the interface
if __name__ == "__main__":
    interface.launch(share=True)

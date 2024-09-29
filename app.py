import gradio as gr  
from transformers import AutoModelForCausalLM, AutoTokenizer  

# Load the model and tokenizer  
MODEL_NAME = "thatstommy/lora_model"  
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  

def generate(user_input):  
    # Tokenize the input  
    inputs = tokenizer(user_input, return_tensors="pt")  
    
    # Generate text from the model  
    outputs = model.generate(**inputs)  
    
    # Decode the output  
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()  
    
    return response  # Return only the response  

# Create the Gradio interface  
interface = gr.Interface(fn=generate, inputs="text", outputs="text", title="Chatbot")  

# Launch the interface  
if __name__ == "__main__":  
    interface.launch(share=True)

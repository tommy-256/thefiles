import gradio as gr  
from transformers import AutoModelForCausalLM, AutoTokenizer  

# Load the model and tokenizer  
MODEL_NAME = "thatstommy/lora_model"  
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  

# Initialize the chatbot messages list  
chatbot_messages = []  
contextual_prompt = "You are a Eduacational chatbot and you should try to explain as simply as possible."  

def generate(user_input):  
    # Add user input
    chatbot_messages.append(f"User: {user_input}")  
    
    # Create the final input text
    input_text = f"{contextual_prompt}\n" + "\n".join(chatbot_messages[-3:])  # Limit to last 3 messages  
    
    # Tokenize the input  
    inputs = tokenizer(input_text, return_tensors="pt")  
    
    # Generate text from the model  
    outputs = model.generate(**inputs)  
    
    # Decode the output  
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()  
    
    # Add model response to the messages history  
    chatbot_messages.append(f"Bot: {response}")  

    # Limit the memory retention to the last 3 messages  
    while len(chatbot_messages) > 3:  
        chatbot_messages.pop(0) 

    return response  # Return only the response for the output textbox  

# Interface
with gr.Blocks() as interface:  
    # Dark header with brand name  
    gr.Markdown("<div style='background-color: #003366; padding: 20px; text-align: center;'><h1 style='color: white;'>School of Tomorrow</h1></div>")  
    
    # Output box
    output_text = gr.Textbox(label="Output:", interactive=False)  
    input_text = gr.Textbox(label="Enter your prompt:", placeholder="Type your prompt here...")  

    # generation function  
    btn = gr.Button("Generate")  

    # On button click, link the input to the model function and output  
    btn.click(fn=generate, inputs=input_text, outputs=output_text) 
    
    # Custom footer  
    gr.Markdown(  
        "<footer style='text-align: center; padding: 10px; background-color: #f1f1f1;'>"  
        "<p>© 2023 School of Tomorrow </p>"  
        "<p>Developed by Daniel Getachew.</p>"  
        "</footer>"  
    )  

if __name__ == "__main__":  
    interface.launch(share=True)

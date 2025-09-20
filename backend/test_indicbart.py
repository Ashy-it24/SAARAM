import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Testing IndicBART...")

try:
    # Load model
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indicbart')
    model = AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indicbart')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Device: {device}")
    print("Model loaded successfully!")
    
    # Test Tamil text
    test_text = "இன்று சென்னையில் மழை பெய்தது. மக்கள் மகிழ்ச்சியடைந்தனர். போக்குவரத்து சுமூகமாக நடைபெற்றது."
    print(f"Input: {test_text}")
    
    # Tokenize
    inputs = tokenizer(test_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    
    # Generate summary
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=50,
            min_length=10,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Summary: {summary}")
    print("IndicBART is working!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
import os
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

print("Starting IndicBART download...")
print(f"Cache directory: {os.environ['HF_HOME']}")

try:
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indicbart')
    print("[OK] Tokenizer downloaded successfully")
    
    print("Downloading model (this may take several minutes)...")
    model = AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indicbart')
    print("[OK] Model downloaded successfully")
    
    print("Testing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Test with simple Tamil text
    test_text = "இன்று வானிலை நல்லது. மக்கள் மகிழ்ச்சியாக உள்ளனர்."
    inputs = tokenizer(test_text, return_tensors='pt', max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,
            min_length=10,
            num_beams=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[OK] Test successful: {summary}")
    print("IndicBART is ready to use!")
    
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    import traceback
    traceback.print_exc()
"""Test embeddings computation step by step."""
import torch
from transformers import AutoModel, AutoTokenizer
from load_phase1_data import load_all_data

print("Loading data...")
all_data = load_all_data()

print(f"\nReal: {len(all_data['real'])} samples")
print(f"Qwen: {len(all_data['qwen'])} samples")
print(f"Llama: {len(all_data['llama'])} samples")

print("\nLoading BERT model...")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = model.to(device)
model.eval()

print("\nTesting with 2 samples...")
test_samples = all_data['real'][:2]

with torch.no_grad():
    texts = [s['text'] for s in test_samples]
    print(f"Text 1 length: {len(texts[0])} chars")
    print(f"Text 2 length: {len(texts[1])} chars")

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    print(f"Encoded shape: {encoded['input_ids'].shape}")

    outputs = model(**encoded)
    last_hidden = outputs.last_hidden_state
    print(f"Hidden states shape: {last_hidden.shape}")

    # CLS token
    cls_embeddings = last_hidden[:, 0, :].cpu().numpy()
    print(f"CLS embeddings shape: {cls_embeddings.shape}")

print("\nEmbedding test successful!")

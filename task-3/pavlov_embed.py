from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import torch

def generate_data(data_path: Path | str, model, tokenizer):
    data = open(data_path, 'r', encoding='utf-8').read().split("\n")[1:]

    better_data = []
    targets = []
    for item in data:
        # print(item[1:-3], item[-1:])
        text, response = item[:-2], item[-1:]
        
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)[0][0][0]

        response = 1 if response == '+' else 0
        
        better_data.append(outputs)
        targets.append(torch.tensor(response))
    
    better_data = torch.stack(better_data).squeeze()
    targets = torch.stack(targets)
    print(better_data.shape, targets.shape)
    torch.save(better_data, open("data/X.pt", 'wb'))
    torch.save(targets, open("data/y.pt", 'wb'))
    return

tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-sentence")

data = generate_data('data/task-3-dataset.csv', model, tokenizer)

# phrase = "Привет, Андрей!"
# inputs = tokenizer(phrase, return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)
# print(outputs[0][0][0].shape)
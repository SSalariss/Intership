from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("google/byt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

model.save_pretrained("./byt5_models/byt5_small")
tokenizer.save_pretrained("./byt5_models/byt5_small")

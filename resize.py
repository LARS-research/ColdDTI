from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("../Model/prot_bert")
model = AutoModel.from_pretrained("../Model/prot_bert")

new_tokens = ["[H]", "[E]", "[T]", "[B]", "[C]", "[secondary_start]", "[secondary_end]", "[tertiary_start]", "[tertiary_end]"]


num_added_tokens = tokenizer.add_tokens(new_tokens)
print(f"Added {num_added_tokens} tokens.")

model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained('../Model/prot_resize')
model.save_pretrained('../Model/prot_resize')
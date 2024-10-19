from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "bigscience/bloom"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Teste se o modelo está carregado
input_text = "Qual é a capital da França?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

# Treina com base no data.txt
tokenizer.train(files=["data.txt"], vocab_size=118, min_frequency=1, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>", "<|fim_da_conversa|>"
])

tokenizer.save_model(".", "meu-bpe")

print("Tokenizador salvo com sucesso!")

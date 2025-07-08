from tokenizers import ByteLevelBPETokenizer

try:
    tokenizer = ByteLevelBPETokenizer("meu-bpe-vocab.json", "meu-bpe-merges.txt")
except Exception as e:
    print("Erro ao carregar tokenizer:", e)
    exit()

vocab_size = tokenizer.get_vocab_size()

def encode(s: str) -> list[int]:
    return tokenizer.encode(s).ids

def decode(l: list[int]) -> str:
    return tokenizer.decode(l)

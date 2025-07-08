import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import NanoGPT, device, block_size
from tokenizer import encode, decode, vocab_size

# --- CONFIGURAÇÃO ---
model_path = "nanogpt_energy.pth"
temperature = 0.8
max_tokens_to_generate = 10  # Quantidade de horas que queremos prever

# --- PROMPT DE ENTRADA REAL (exemplo de consumo real da casa) ---
prompt = "6 6 5 5 6 5 6 6 6 7"

# Codifica o prompt para tokens
input_ids = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

# --- CARREGA O MODELO ---
model = NanoGPT(vocab_size=vocab_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --- GERAÇÃO DE NOVOS VALORES ---
for _ in range(max_tokens_to_generate):
    input_cond = input_ids[:, -block_size:]
    with torch.no_grad():
        logits, _ = model(input_cond)

    logits = logits[:, -1, :] / temperature
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    input_ids = torch.cat((input_ids, next_id), dim=1)

# Decodifica toda a sequência (prompt + previsão)
output_text = decode(input_ids[0].tolist())
print("\n--- Sequência completa (real + previsão) ---")
print(output_text)

# --- PREPARA OS DADOS PARA O GRÁFICO ---
all_values = list(map(int, output_text.strip().split()))
prompt_values = list(map(int, prompt.strip().split()))
predicted_values = all_values[len(prompt_values):]

# --- CRIA EIXO DE TEMPO EM HORAS ---
time_real = list(range(len(prompt_values)))  # Horas do consumo real
time_pred = list(range(len(prompt_values), len(prompt_values) + len(predicted_values)))  # Horas previstas

# --- PLOT ---
plt.figure(figsize=(12,6))
plt.plot(time_real, prompt_values, label="Consumo Real (últimas horas)", marker='o')
plt.plot(time_pred, predicted_values, label=f"Previsão próximas {max_tokens_to_generate} horas", marker='x')
plt.xlabel("Tempo (horas)")
plt.ylabel("Consumo de Energia (unidades do dataset)")
plt.title("Previsão de Consumo de Energia com NanoGPT")
plt.legend()
plt.grid(True)
plt.show()

# --- EXPLICAÇÃO ---
print(f"""
Estamos estimando o consumo (demanda) de energia para as próximas {max_tokens_to_generate} horas,
com base no histórico recente de consumo da casa (últimas {len(prompt_values)} horas).

No gráfico:
- Os pontos circulares mostram o consumo real nas horas anteriores.
- Os pontos com 'x' indicam a previsão do modelo para as próximas horas.

Essa previsão ajuda a planejar melhor o uso e distribuição de energia residencial,
identificando picos de consumo futuros e otimizando recursos.
""")

from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = Flask(__name__)

# Carregar o modelo treinado
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-final")  # Ajuste o caminho

# Dicionário de palavras-chave (mesmo do treinamento)
palavras_chave = {
    "excelente": 2, "péssimo": -2, "horrível": -2, "maravilhoso": 2,
    "ruim": -1, "bom": 1, "ótimo": 2, "terrível": -2,
    "gostei": 1, "odeio": -2, "recomendo": 1, "evitar": -2,
    "top": 1, "furada": -2, "show": 1, "lixo": -2,
    "incrível": 2, "decepção": -2, "fantástico": 2, "horroroso": -2,
    "adoro": 2, "detesto": -2, "perfeito": 2, "desastre": -2
}


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\bnão\s+\w+', 'não_', text)
    text = re.sub(r'\bnem\s+\w+', 'nem_', text)
    text = re.sub(r'[^\w\sáéíóúâêîôûãõàèìòùç]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def analisar_sentimento(texto):
    texto_pp = preprocess_text(texto)
    inputs = tokenizer(texto_pp, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
    prob_neg, prob_pos = probs[0] * 100, probs[1] * 100
    diferenca = abs(prob_pos - prob_neg)

    limiar_neutro = max(40, min(70, 60 - (diferenca / 10)))

    if max(prob_neg, prob_pos) < limiar_neutro:
        return "Neutro", max(prob_neg, prob_pos)
    else:
        return "Positivo" if prob_pos > prob_neg else "Negativo", max(prob_neg, prob_pos)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        texto = request.form['texto']
        resultado, confianca = analisar_sentimento(texto)

        # Detectar palavras-chave
        palavras = [p for p in palavras_chave if p in preprocess_text(texto)]

        return render_template('resultado.html',
                               texto=texto,
                               resultado=resultado,
                               confianca=f"{confianca:.1f}%",
                               palavras=", ".join(palavras) if palavras else "Nenhuma")

    return render_template('formulario.html')


if __name__ == '__main__':
    app.run(debug=True)
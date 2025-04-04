import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback
import re
import random
from collections import Counter

# ==============================================
# 1. PREPARAÇÃO DOS DADOS
# ==============================================

# Dados de treinamento ampliados e balanceados
base_data = {
    'tweet_text': [
        # Sentimentos positivos (40 exemplos)
        "Amei o novo filme! Incrível do começo ao fim!",
        "Que lugar maravilhoso! Com certeza vou voltar!",
        "O atendimento foi impecável, todos muito atenciosos!",
        "Me diverti muito no parque, foi um dia incrível!",
        "O produto superou minhas expectativas, recomendo muito!",
        "Esse restaurante tem uma comida maravilhosa, adorei!",
        "O serviço foi rápido e eficiente, fiquei muito satisfeito!",
        "A experiência foi maravilhosa, recomendo para todos!",
        "Adorei a minha experiência de compra, tudo perfeito!",
        "Fui muito bem atendido, o lugar é sensacional!",
        "Tudo estava perfeito, até o atendimento foi de primeira!",
        "Simplesmente incrível, adorei o produto!",
        "O hotel superou minhas expectativas, muito confortável!",
        "Fui muito bem atendido e o produto chegou no prazo!",
        "O ambiente do restaurante é super agradável, voltarei com certeza!",
        "Eu amei o atendimento, todos muito simpáticos!",
        "O evento foi incrível, adorei cada momento!",
        "Adorei o sabor da comida, super recomendo!",
        "Muito bom, voltarei com certeza!",
        "Foi a melhor experiência que já tive, tudo perfeito!",
        "Estou completamente satisfeito com o serviço!",
        "Excelente atendimento, recomendo a todos!",
        "O produto chegou em perfeito estado, muito feliz com a compra!",
        "Recomendo a todos, a experiência foi excelente!",
        "Tudo perfeito, desde o atendimento até a qualidade do produto!",
        "Top demais, nota mil!",
        "Gostei bastante do serviço prestado",
        "Bom atendimento e qualidade excelente",
        "bom demais, superou expectativas",
        "gostei do inicio ao fim, perfeito",
        "ótima empresa para se trabalhar",
        "salário excelente e benefícios bons",
        "ambiente de trabalho agradável",
        "líderes compreensivos e atenciosos",
        "oportunidades de crescimento reais",
        "recomendo a empresa sem dúvidas",
        "equipe unida e colaborativa",
        "gestão transparente e justa",
        "benefícios acima da média",
        "valorização dos funcionários clara",

        # Sentimentos negativos (40 exemplos)
        "O filme foi muito chato, não gostei nem um pouco.",
        "O atendimento foi horrível, nunca mais volto aqui!",
        "A comida estava fria e sem sabor, uma decepção.",
        "O lugar estava sujo e desorganizado, não gostei.",
        "O produto não atendeu às minhas expectativas, fiquei muito insatisfeito.",
        "O serviço foi muito demorado e não valeu a pena.",
        "O hotel estava mal cuidado e não me senti confortável.",
        "Fui mal atendido, o lugar é uma bagunça!",
        "Não gostei do atendimento, muito rude e impessoal.",
        "A comida estava sem gosto, não recomendo.",
        "A experiência foi frustrante, não voltarei mais.",
        "O serviço foi péssimo, demoraram demais para me atender.",
        "O ambiente estava sujo e desorganizado, não gostei.",
        "O filme foi uma grande decepção, não valeu o ingresso.",
        "Não gostei do produto, achei de má qualidade.",
        "A comida estava muito sem gosto, um desperdício de dinheiro.",
        "A experiência foi muito ruim, não recomendo.",
        "Foi a pior experiência de todas, muito insatisfeito.",
        "O lugar estava bagunçado e sem organização, não gostei.",
        "O atendimento foi péssimo, não voltei mais.",
        "O serviço demorou demais e estava mal feito.",
        "O produto que comprei veio com defeito, fiquei decepcionado.",
        "A comida estava ruim e o atendimento nem se fala, não volto.",
        "O hotel não estava nada bem cuidado, uma decepção total.",
        "A experiência foi abaixo das expectativas, muito insatisfeito.",
        "Empresa não dá benefícios bons, salário baixo.",
        "Não tem evolução profissional, estagnado.",
        "Funcionário não é bem tratado, ambiente tóxico.",
        "Líderes são ruins e mal educados, falta respeito.",
        "Não voltaria a trabalhar lá de jeito nenhum.",
        "Empresa não ajuda no crescimento dos funcionários.",
        "Péssimas condições de trabalho, evitável.",
        "Salário abaixo do mercado, exploração.",
        "Chefes abusivos e sem preparo técnico.",
        "Clima organizacional insuportável e pesado.",
        "Não valorizam os colaboradores, descartáveis.",
        "Processos burocráticos e ineficientes.",
        "Comunicação interna inexistente ou confusa.",
        "Falta de transparência na gestão, obscuro.",
        "Promessas não cumpridas, enganação.",
    ],
    'sentiment': [
        *["Positivo"] * 40,
        *["Negativo"] * 40
    ]
}

# Dados adicionais com exemplos mais complexos
additional_data = {
    'tweet_text': [
        # Exemplos positivos adicionais
        "Não é perfeito, mas atendeu minhas necessidades básicas",
        "Esperava mais, mas foi satisfatório no geral",
        "Cumpre o prometido, nada excepcional",
        "Bom considerando o preço pago",
        "Tem seus defeitos, mas gostei no geral",

        # Exemplos negativos adicionais
        "Até que não foi ruim, mas esperava muito mais",
        "Poderia ser pior, mas não recomendo",
        "Medíocre, nem bom nem horrível",
        "Aceitável, mas não vale o preço",
        "Não é terrível, mas decepcionante",

        # Expressões idiomáticas e gírias
        "Manda muito bem! Top demais!",
        "Péssimo, uma verdadeira furada",
        "Me deixou na mão, não curti",
        "Show de bola, adorei!",
        "Uma bosta, não vale nada",

        # Textos mais longos e complexos
        "Inicialmente fiquei desconfiado, mas no final superou todas as expectativas positivas que eu tinha",
        "Pensei que seria melhor, mas depois de usar por uma semana percebi que não atende ao que promete",
    ],
    'sentiment': [
        1, 1, 1, 1, 1,  # Positivos
        0, 0, 0, 0, 0,  # Negativos
        1, 0, 0, 1, 0,  # Gírias
        1, 0  # Longos
    ]
}

# Combinando os dados
data = pd.DataFrame(base_data)
additional_df = pd.DataFrame(additional_data)
data = pd.concat([data, additional_df], ignore_index=True)

# ==============================================
# 2. PRÉ-PROCESSAMENTO
# ==============================================

# Dicionário de palavras-chave com pesos
palavras_chave = {
    "excelente": 2, "péssimo": -2, "horrível": -2, "maravilhoso": 2,
    "ruim": -1, "bom": 1, "ótimo": 2, "terrível": -2,
    "gostei": 1, "odeio": -2, "recomendo": 1, "evitar": -2,
    "top": 1, "furada": -2, "show": 1, "lixo": -2,
    "incrível": 2, "decepção": -2, "fantástico": 2, "horroroso": -2,
    "adoro": 2, "detesto": -2, "perfeito": 2, "desastre": -2
}


# Função de pré-processamento avançada
def preprocess_text(text):
    text = str(text).lower()

    # Substitui expressões negativas
    text = re.sub(r'\bnão\s+\w+', 'não_', text)
    text = re.sub(r'\bnem\s+\w+', 'nem_', text)

    # Remove caracteres especiais mas mantém acentos
    text = re.sub(r'[^\w\sáéíóúâêîôûãõàèìòùç]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Aplicando pré-processamento
data["tweet_text"] = data["tweet_text"].apply(preprocess_text)
data["sentiment"] = data["sentiment"].map({"Positivo": 1, "Negativo": 0})

# Verificar e remover NaN
print("Valores NaN em sentiment:", data['sentiment'].isna().sum())
if data['sentiment'].isna().sum() > 0:
    data = data.dropna(subset=['sentiment'])
    print(f"Removidas {data['sentiment'].isna().sum()} linhas com NaN")


# ==============================================
# 3. AUMENTAÇÃO DE DADOS (OPCIONAL)
# ==============================================

def simple_augmentation(text, num_aug=2):
    """Função alternativa de aumento de dados sem dependências externas"""
    words = text.split()
    augmented = []

    for _ in range(num_aug):
        # Random swap (troca aleatória de palavras)
        if len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words_copy = words.copy()
            words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]
            augmented.append(' '.join(words_copy))

        # Random deletion (deleção aleatória)
        if len(words) > 2:
            words_copy = words.copy()
            del words_copy[random.randint(0, len(words_copy) - 1)]
            augmented.append(' '.join(words_copy))

    return list(set(augmented))  # Remove duplicados


# Divisão dos dados
X = data["tweet_text"].values
y = data["sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================
# 4. TOKENIZAÇÃO
# ==============================================

# Modelo principal
model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Função de tokenização corrigida
def tokenize_data(texts):
    """Função corrigida para lidar com diferentes tipos de entrada"""
    # Se for numpy array, converter para lista
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    # Garantir que é lista mesmo se vier como string única
    if isinstance(texts, str):
        texts = [texts]

    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
        add_special_tokens=True
    )


# Tokenização
try:
    train_encodings = tokenize_data(X_train)
    test_encodings = tokenize_data(X_test)
except Exception as e:
    print(f"Erro na tokenização: {str(e)}")
    print("Tipos encontrados:")
    print(f"X_train: {type(X_train)}, primeiro elemento: {type(X_train[0]) if len(X_train) > 0 else 'vazio'}")
    print(f"X_test: {type(X_test)}, primeiro elemento: {type(X_test[0]) if len(X_test) > 0 else 'vazio'}")
    raise


# ==============================================
# 5. DATASETS E MODELO
# ==============================================

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SentimentDataset(train_encodings, y_train)
test_dataset = SentimentDataset(test_encodings, y_test)

# Carregando o modelo
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    problem_type="single_label_classification",
)

# ==============================================
# 6. TREINAMENTO
# ==============================================

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  # Reduzido para demonstração
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    seed=42,
    fp16=torch.cuda.is_available(),
    report_to="none",
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

print("\nIniciando treinamento...")
trainer.train()

# ==============================================
# 7. AVALIAÇÃO
# ==============================================

print("\nAvaliando no conjunto de teste...")
results = trainer.evaluate(test_dataset)

# Métricas detalhadas
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, preds, target_names=["Negativo", "Positivo"]))

print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, preds))

print("\nResultados da avaliação:")
print(results)


model.save_pretrained("./results/checkpoint-final")
tokenizer.save_pretrained("./results/checkpoint-final")


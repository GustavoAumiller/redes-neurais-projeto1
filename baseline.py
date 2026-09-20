import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
DATASET = Path("dataset_projeto1.csv")

EPOCHS = 1000
BATCH_SIZE = 16

CONFIGURACOES = [
    {"nome": "MLP_4",   "hidden": [4],      "activation": "tanh",    "lr": 0.01},
    {"nome": "MLP_8",   "hidden": [8],      "activation": "tanh",    "lr": 0.01},
    {"nome": "MLP_16",  "hidden": [16],     "activation": "tanh",    "lr": 0.01},
    {"nome": "MLP_8_4", "hidden": [8, 4],   "activation": "tanh",    "lr": 0.01},
    {"nome": "MLP_16_8","hidden": [16, 8],  "activation": "tanh",    "lr": 0.01},
    {"nome": "MLP_8",   "hidden": [8],      "activation": "relu",    "lr": 0.01},
    {"nome": "MLP_16",  "hidden": [16],     "activation": "relu",    "lr": 0.01},
    {"nome": "MLP_8_4", "hidden": [8, 4],   "activation": "relu",    "lr": 0.01},
]


def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPRegressor(nn.Module):
    def __init__(self, hidden, activation="tanh"):
        super().__init__()

        layers = []
        input_size = 1

        activation_cls = {"tanh": nn.Tanh,"relu": nn.ReLU,}[activation]
        for neurons in hidden:
            layers.append(nn.Linear(input_size, neurons))
            layers.append(activation_cls())
            input_size = neurons
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def carregar_dados():
    df = pd.read_csv(DATASET)
    X = df[["x"]].values.astype(np.float32)
    y = df[["y"]].values.astype(np.float32)
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.90,random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp,test_size=8 / 9,random_state=SEED)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

def treinar_modelo(model, X_train, y_train, X_val, y_val, lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(X_train, y_train),batch_size=BATCH_SIZE,shuffle=True)
    historico_train = []
    historico_val = []
    melhor_val = float("inf")
    melhor_estado = None
    melhor_epoca = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        historico_train.append(train_loss)
        historico_val.append(val_loss)
        if val_loss < melhor_val:
            melhor_val = val_loss
            melhor_estado = copy.deepcopy(model.state_dict())
            melhor_epoca = epoch
    model.load_state_dict(melhor_estado)
    return {
        "train_loss": historico_train,
        "val_loss": historico_val,
        "best_val_loss": melhor_val,
        "best_epoch": melhor_epoca,
    }

def calcular_metricas(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).numpy().ravel()
    real = y.numpy().ravel()
    mae = mean_absolute_error(real, pred)
    mse = mean_squared_error(real, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(real, pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

def main():
    fix_seed()

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET.resolve()}")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = carregar_dados()
    resultados = []
    modelos = {}

    for config in CONFIGURACOES:
        fix_seed()
        model = MLPRegressor(hidden=config["hidden"],activation=config["activation"])
        historico = treinar_modelo(model,X_train,y_train,X_val,y_val,config["lr"])
        metricas_val = calcular_metricas(model, X_val, y_val)
        resultados.append({
            "Modelo": config["nome"],
            "Arquitetura": str(config["hidden"]),
            "Ativação": config["activation"],
            "Learning Rate": config["lr"],
            "Época escolhida": historico["best_epoch"],
            "Val_MAE": metricas_val["MAE"],
            "Val_MSE": metricas_val["MSE"],
            "Val_RMSE": metricas_val["RMSE"],
            "Val_R2": metricas_val["R2"],
        })
        modelos[config["nome"]] = {"model": model,"config": config,"historico": historico,}
    tabela = pd.DataFrame(resultados)
    tabela = tabela.sort_values("Val_MSE").reset_index(drop=True)
    melhor_nome = tabela.iloc[0]["Modelo"]
    baseline = modelos[melhor_nome]
    metricas_teste = calcular_metricas(baseline["model"],X_test,y_test)

    for nome, valor in metricas_teste.items():
        print(f"{nome:6s}: {valor:.6f}")
    hist = baseline["historico"]
    plt.figure(figsize=(9, 5))
    plt.plot(hist["train_loss"], label="Treino")
    plt.plot(hist["val_loss"], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title("Evolução do treinamento - Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("baseline_treinamento.png", dpi=150)
    plt.show()
    baseline["model"].eval()
    with torch.no_grad():
        pred = baseline["model"](X_test).numpy().ravel()
    real = y_test.numpy().ravel()
    plt.figure(figsize=(7, 7))
    plt.scatter(real, pred, alpha=0.7)
    minimo = min(real.min(), pred.min())
    maximo = max(real.max(), pred.max())
    plt.plot([minimo, maximo],[minimo, maximo],linestyle="--",label="Predição perfeita")
    plt.xlabel("Valor real")
    plt.ylabel("Valor previsto")
    plt.title("Baseline - Valores reais x previstos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("baseline_real_vs_predito.png", dpi=150)
    plt.show()
    tabela.to_csv("resultados_busca_baseline.csv",index=False,encoding="utf-8-sig")

if __name__ == "__main__":
    main()

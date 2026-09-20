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

MAX_EPOCHS = 5000
BATCH_SIZE = 16
LEARNING_RATE = 0.01
L1_LAMBDA = 0.001
L2_LAMBDA = 0.001
DROPOUT_P = 0.2
MOMENTUM = 0.9

def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MLPRegressor(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        layers = [nn.Linear(1, 8), nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers += [nn.Linear(8, 4), nn.ReLU()]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(4, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def carregar_dados():
    df = pd.read_csv(DATASET)
    X = df[["x"]].values.astype(np.float32)
    y = df[["y"]].values.astype(np.float32)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.90, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=8 / 9, random_state=SEED)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return tuple(torch.tensor(a, dtype=torch.float32) for a in
                 (X_train, y_train, X_val, y_val, X_test, y_test))

def penalizacao_l1(model):
    return sum(p.abs().sum() for p in model.parameters())

def penalizacao_l2(model):
    return sum((p ** 2).sum() for p in model.parameters())

def treinar(model, X_train, y_train, X_val, y_val, tipo):
    momentum = MOMENTUM if tipo == "momentum" else 0.0
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train),batch_size=BATCH_SIZE,shuffle=True)
    train_history = []
    val_history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_mse = []
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            mse = criterion(pred, yb)
            loss = mse
            if tipo == "l1":
                loss = loss + L1_LAMBDA * penalizacao_l1(model)
            elif tipo == "l2":
                loss = loss + L2_LAMBDA * penalizacao_l2(model)
            loss.backward()
            optimizer.step()
            batch_mse.append(mse.item())
        train_mse = float(np.mean(batch_mse))
        model.eval()
        with torch.no_grad():
            val_mse = criterion(model(X_val), y_val).item()
        train_history.append(train_mse)
        val_history.append(val_mse)
        if val_mse < best_val:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    model.load_state_dict(best_state)
    return {"train_loss": train_history,"val_loss": val_history,"best_val_loss": best_val,"best_epoch": best_epoch,}

def metricas(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).numpy().ravel()
    real = y.numpy().ravel()
    mse = mean_squared_error(real, pred)
    return {"MAE": mean_absolute_error(real, pred),"MSE": mse,"RMSE": np.sqrt(mse),"R2": r2_score(real, pred),}

def nome_arquivo(nome, sufixo):
    return f"ablation_{nome.lower().replace('+', 'mais').replace(' ', '_')}_{sufixo}.png"

def salvar_graficos(nome, hist, model, X_test, y_test):
    train = np.asarray(hist["train_loss"])
    val = np.asarray(hist["val_loss"])
    plt.figure(figsize=(10, 5))
    plt.plot(train, label="Treino")
    plt.plot(val, label="Validação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(f"Evolução do treinamento - {nome}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(nome_arquivo(nome, "treinamento"), dpi=150)
    plt.close()
    start = max(0, len(train) - 5000)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(start + 1, len(train) + 1), train[start:], label="Treino")
    plt.plot(np.arange(start + 1, len(val) + 1), val[start:], label="Validação")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(f"Últimas 5000 épocas - {nome}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(nome_arquivo(nome, "ultimas_5000"), dpi=150)
    plt.close()
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy().ravel()
    real = y_test.numpy().ravel()
    lo, hi = min(real.min(), pred.min()), max(real.max(), pred.max())
    plt.figure(figsize=(7, 7))
    plt.scatter(real, pred, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="Predição perfeita")
    plt.xlabel("Valor real")
    plt.ylabel("Valor previsto")
    plt.title(f"{nome} - Valores reais x previstos")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(nome_arquivo(nome, "real_vs_predito"), dpi=150)
    plt.close()

def main():
    fix_seed()
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATASET.resolve()}")
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_dados()
    experiments = [
        ("Baseline", "baseline", 0.0),
        ("L1", "l1", 0.0),
        ("L2", "l2", 0.0),
        ("Dropout", "dropout", DROPOUT_P),
        ("Momentum", "momentum", 0.0),
    ]
    results = []
    histories = {}

    for name, tipo, dropout_p in experiments:
        fix_seed()
        model = MLPRegressor(dropout_p=dropout_p)
        hist = treinar(model, X_train, y_train, X_val, y_val, tipo)
        val = metricas(model, X_val, y_val)
        test = metricas(model, X_test, y_test)
        results.append({
            "Modelo": name,
            "Melhor época": hist["best_epoch"],
            "Val_MAE": val["MAE"],
            "Val_MSE": val["MSE"],
            "Val_RMSE": val["RMSE"],
            "Val_R2": val["R2"],
            "Teste_MAE": test["MAE"],
            "Teste_MSE": test["MSE"],
            "Teste_RMSE": test["RMSE"],
            "Teste_R2": test["R2"],
        })
        histories[name] = hist
        salvar_graficos(name, hist, model, X_test, y_test)
    table = pd.DataFrame(results)
    table.to_csv("resultados_ablation.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(11, 6))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=name)
    plt.xlabel("Época")
    plt.ylabel("MSE de validação")
    plt.title("Comparação dos modelos - validação")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ablation_comparacao_validacao.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()

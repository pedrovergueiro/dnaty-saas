# Deploy dNATY SaaS no Railway

## Pré-requisitos
- Conta em [railway.app](https://railway.app)
- [Railway CLI](https://docs.railway.app/develop/cli): `npm install -g @railway/cli`
- Git instalado

---

## 1. Preparar o repositório

```powershell
# Na raiz do projeto dNATY (não dentro de dnaty_saas)
git init
git add .
git commit -m "chore: initial commit dNATY SaaS"
```

> O Railway detecta o `Dockerfile` automaticamente.
> O `railway.json` dentro de `dnaty_saas/` configura build e start command.

---

## 2. Login e criação do projeto

```powershell
railway login
railway init          # cria novo projeto
railway link          # ou: seleciona projeto existente
```

---

## 3. Configurar variáveis de ambiente

Via CLI:
```powershell
railway variables set API_KEY=sua_chave_secreta_aqui
railway variables set DEBUG=false
railway variables set LOG_LEVEL=INFO
railway variables set ALLOWED_ORIGINS=["*"]
```

Ou via dashboard: **Railway → seu projeto → Variables → New Variable**

> `PORT` é injetado automaticamente pelo Railway — não precisa setar.

---

## 4. Fazer o deploy

```powershell
railway up
```

O Railway vai:
1. Detectar o `Dockerfile`
2. Fazer build da imagem
3. Subir o container com `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Fazer health check em `GET /health`

---

## 5. Obter a URL pública

```powershell
railway open          # abre o dashboard com a URL
railway status        # mostra URL e status do deploy
```

Formato da URL: `https://<projeto>.up.railway.app`

---

## 6. Testar os endpoints

```powershell
# Health check básico
python health_check.py https://<seu-projeto>.up.railway.app

# Teste manual
curl https://<seu-projeto>.up.railway.app/health
curl https://<seu-projeto>.up.railway.app/docs
```

Teste de treino:
```powershell
$body = @{
    dataset="mnist"; n_pop=4; n_generations=5
    t_local=1; lr=0.001; batch_size=256
    init_hidden=@(64,32); device="cpu"
} | ConvertTo-Json

Invoke-RestMethod "https://<seu-projeto>.up.railway.app/api/v1/train" `
    -Method POST -Body $body -ContentType "application/json" `
    -Headers @{"X-API-Key"="sua_chave_secreta_aqui"}
```

---

## 7. Ver logs em tempo real

```powershell
railway logs          # stream de logs do container
```

---

## Variáveis de ambiente disponíveis

| Variável         | Descrição                              | Padrão   |
|------------------|----------------------------------------|----------|
| `API_KEY`        | Chave de autenticação (vazio = sem auth) | `""`   |
| `DEBUG`          | Ativa modo debug do FastAPI            | `false`  |
| `LOG_LEVEL`      | Nível de log (INFO/DEBUG/WARNING)      | `INFO`   |
| `ALLOWED_ORIGINS`| CORS origins (JSON array)              | `["*"]`  |
| `PORT`           | Porta (injetado pelo Railway)          | auto     |

---

## Estrutura esperada no repositório

```
dNATY/
├── dnaty/               ← pacote Python do modelo
├── dnaty_saas/
│   ├── Dockerfile
│   ├── railway.json
│   ├── .dockerignore
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── models/
│   └── routes/
└── ...
```

> O `Dockerfile` copia `../dnaty` para dentro do container.
> Por isso o `docker build` deve ser executado a partir da raiz do projeto:
> ```powershell
> docker build -f dnaty_saas/Dockerfile -t dnaty-saas .
> ```

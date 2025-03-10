# Predicting User Behavior & Battery Drain 📱🔋 100% - MLOps


<div align="center" style="max-width:68rem;">
<table>
  <tr>
    <td align="center">
        <a href="https://github.com/gianvr"><img src="https://avatars.githubusercontent.com/gianvr" alt="Giancarlo Ruggiero" width="100" style="border-radius: 50%;" /><br />
        <sub><b>Giancarlo Ruggiero</b></sub></a><br />
        Developer
    </td>
</table>
</div>

## Vídeo de Demonstração

[Link](https://www.youtube.com/watch?v=yr8ccf73KNE)

## 0. Setup
## 0.1 Dependências

```bash
pip install -r requirements.txt
```

### 0.2 .env

Na raiz do projeto existe um arquivo `.env.example` que deve ser copiado e renomeado para `.env` e preenchido com as credenciais da AWS.

### 0.3 AWS

Também é necessário configurar as credenciais da AWS. Para isso, execute:

```bash
aws configure
```

## 1. Input de Dados

Os dados são baixados e pré-processados automaticamente. Para isso, execute:
```bash
python src/process_data.py
```

## 2. Mlflow

Para inicializar o mlflow, execute:

```bash
mlflow ui
```

## 3. Treinamento

Para treinar o modelo, execute:

```bash
python src/train.py
```

O treinamento do modelo irá gerar um novo experimento no mlflow. Caso deseje registrar o modelo, altere a flag `register_model` para `True` no arquivo `src/train.py`.

Gera logs no diretório `logs/` e um gráfico da matriz de confusão no diretório `results/`.

## 4. Data Drift

O arquivo `src/data_drift.py` contém a análise de data drift, em que é possível verificar comparar a distribuição dos dados utilizados na produção com os dados novos.

Gera logs no diretório `logs/` e um gráfico da matriz de confusão no diretório `results/`.

Também são gerados artefatos no mlflow.

## 5. Deploy Local

Para realizar o deploy local, execute:

```bash
mlflow models serve -m runs:/<run_id>/user_behavior_model --no-conda -p 8080
```

### 5.1. Teste

Após o deploy, é possível realizar um teste com o arquivo `test/test_prediction.py`:

```bash
python test/test_prediction.py
```

## 6. Logging

Cada arquivo gera um log no diretório `logs/`.

## 7. DVC

É possível utilizar o DVC para versionar os dados e os modelos. Para isso, execute:

```bash
dvc init
dvc add data/user_behavior_dataset_processed.csv
```
### 7.1. S3


Para armazenar os dados no S3, primeiro crie um bucket no S3 (modifique o arquivo), executando:

```bash
python src/s3/create_bucket.py
```

Para deletar o bucket, execute:

```bash
python src/s3/delete_bucket.py
```
E então, execute:

```bash
dvc remote add -d myremote s3://<bucket-name>
dvc remote default myremote
dvc push
```

### 7.2. Pipeline

Para executar o pipeline de treinamento, execute:

```bash
dvc repro
```

## Referências

- Notebook: [User Behavior Notebook](https://www.kaggle.com/code/pavankumar4757/predicting-user-behavior-battery-drain-100#Model-For-Classification-of-User-Behavior)

- Dataset: [User Behavior Dataset](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset)
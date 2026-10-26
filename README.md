# Curriculum Learning

## Executar um experimento

Instale as dependências com `uv sync` e execute:

```bash
uv run curriculum-learning
```

O treinamento grava os modelos e os resultados em `outputs/`. Cada execução
inclui `training_config.json` e `training_summary.json`, com parâmetros,
seeds, revisões/fingerprints dos dados e do modelo e versões do ambiente.
O app Gradio e o notebook `notebooks/evaluate.ipynb` usam esse mesmo diretório.

## Reprodutibilidade

O treinamento fixa as seeds do Transformers para inicialização, treinamento e
embaralhamento, e ativa operações determinísticas por padrão. Isso pode reduzir
o desempenho ou não ser suportado por algumas operações/hardwares; nesses casos,
use `full_determinism=false` e mantenha a seed registrada.

Para repetir um experimento usando referências imutáveis do Hugging Face,
informe os commits do modelo e do dataset:

```bash
uv run curriculum-learning \
  seed=42 \
  model_revision=<commit-do-modelo> \
  dataset_revision=<commit-do-dataset>
```

As revisões são opcionais; quando omitidas, o Hugging Face usa a revisão padrão
do repositório, que pode mudar. O resumo registra a revisão resolvida do modelo
e o fingerprint de cada split carregado para ajudar a identificar os dados.

## Custo e armazenamento

Os experimentos continuam avaliando e salvando ao fim de cada época, sem parada
antecipada, para manter a mesma curva de avaliação entre métodos. O número de
checkpoints mantidos é limitado a dois; as métricas de todas as avaliações
permanecem nos logs do Trainer, e o melhor modelo é preservado para avaliação.

Em experimentos com currículo, os escores de dificuldade são armazenados em
`outputs/.cache/difficulty` e reutilizados entre execuções quando dataset,
heurística, métrica e parâmetros correspondem. Para a heurística aleatória, a
seed também faz parte da chave do cache.

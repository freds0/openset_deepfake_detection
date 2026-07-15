# configs/experiment — variações do PLAN_v0.2

Um experimento = um arquivo. Cada YAML é `# @package _global_` e contém apenas os
overrides sobre o baseline (config atual). Execução:

```bash
./scripts/train.sh +experiment=a1a_lora_r4              # FF++ (data default)
./scripts/train.sh +experiment=a1a_lora_r4 data=ntire  # variação × dataset
./scripts/train.sh +experiment=e1_loo_deepfakes        # protocolo open-set
```

Convenções:
- `logger.wandb.name` = id do experimento (rastreabilidade no W&B).
- `logger.wandb.tags` = `[v0.2, <grupo>, <id>]`.
- Grupos: A (arquitetura, só YAML) · C (pipeline, só YAML) · E (open-set, usa o
  flag `train_classes`/`test_classes`). Grupos B/D do plano exigem código e são
  condicionais — não incluídos aqui.

Protocolo de triagem/confirmação e critérios de promoção: ver `PLAN_v0.2.md` §0.2.

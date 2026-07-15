# AUDIT.md — Auditoria Técnica de Performance (OSDFD / SigLIP 2 + PEFT)

**Data:** 2026-07-07
**Ambiente de medição:** RTX 3060 (12 GB), batch 48, `bf16-mixed`, torch 2.2.2+cu121*,
transformers 4.53.2, dados em `/home` (HDD Seagate ST1000DX002, 99% de ocupação).
Treino de produção alvo: DGX B200 (`/raid`, NVMe).
**Escopo:** pipeline de dados, utilização de GPU, arquitetura (FLOPs/convergência),
escalabilidade. Todas as cifras abaixo foram **medidas**, não estimadas, salvo
indicação explícita.

\* Ambiente local; o DGX usa torch >= 2.7 (cu128), ver `requirements.txt`.

---

## 1. Sumário executivo

| # | Achado | Impacto | Ação |
|---|--------|---------|------|
| A1 | Treino **IO-bound nesta workstation**: loader entrega 88 img/s (HDD seek-bound) vs. teto de GPU de 146 img/s. | Alto (local) / nulo (DGX) | Diagnóstico documentado; mitigação = SSD ou shards sequenciais. Não afeta o DGX (NVMe). |
| A2 | Backward atravessa os 12 blocos do ViT congelado porque há LoRA no bloco 0. Truncar o PEFT nos últimos 6 blocos dá **+44% steps/s e −42% VRAM**. | Alto | Implementado como `peft.start_layer` (default 0 = paper-faithful; ablation opt-in). |
| A3 | SDPA (kernels fusionados de atenção) **já está ativo** por default no transformers 4.53; FlashAttention-2 não instalado e marginal em seq_len 197. | Nulo/baixo | Verificado no código-fonte instalado; exposto `backbone.attn_implementation` para experimentos no DGX. |
| A4 | Adam `fused=True`: **ganho zero medido** e **incompatível** com `gradient_clip_val` sob o plugin AMP do Lightning (erro em runtime). | Nulo | Flag `optimizer.fused` implementada, default `false`, motivos documentados no YAML. |
| A5 | FSDP/DeepSpeed **não se justificam**: 2.3M params treináveis (~28 MB de estado de otimizador), 3.65 GB de pico em batch 48. | — | DDP é a estratégia correta (`trainer.devices=N trainer.strategy=ddp`). |
| A6 | Estabilidade de gradientes OK por construção (backbone congelado + adapters zero-init + clip 1.0). Sem risco de vanishing/exploding. | — | Nenhuma ação necessária. |

---

## 2. Metodologia

1. **Perfil por estágio do pipeline de dados** — leitura fria vs. quente (page cache)
   vs. decode+transform, single-thread, 200 imagens aleatórias de
   `data/ffpp_frames/train/real`; depois throughput do `DataLoader` real
   (8 workers, 100 batches após warmup).
2. **Teto de GPU** — benchmark A/B com dados sintéticos (sem I/O): step completo
   de treino (forward + loss + backward + clip + optimizer step) sob
   `autocast(bf16)`, 50 steps + 10 de warmup, `cudnn.benchmark=True`, TF32 on.
   Script: efêmero (scratchpad), lógica idêntica ao `training_step` real.
3. **Validação controlada** — run real de 100 steps no FF++
   (`trainer.max_steps=100 trainer.val_check_interval=100`), exercitando
   train → val → checkpoint → test com os novos defaults.
4. **Verificação de atenção** — inspeção do código-fonte instalado
   (`transformers.models.siglip.modeling_siglip`): `SiglipAttention.forward`
   despacha via `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`, e
   `config._attn_implementation == "sdpa"` no checkpoint carregado.

---

## 3. Diagnóstico detalhado

### 3.1 Pipeline de dados (A1)

Medições no FF++ (crops 224×224, PNG ~59 KB):

| Estágio | Medido |
|---|---|
| Leitura fria (aleatória) + decode PNG | **18.16 ms/img** (55 img/s single-thread) |
| Leitura quente (page cache) + decode | 1.56 ms/img (643 img/s single-thread) |
| Decode + transform (resize/normalize) | 2.19 ms/img (457 img/s single-thread) |
| **DataLoader completo (8 workers, leitura fria)** | **1.82 batch/s = 88 img/s** |
| **Teto de GPU (modelo, dados sintéticos)** | **3.04 steps/s = 146 img/s** |

**Interpretação.** ~90% do custo por imagem é *seek* de disco (18.16 vs. 1.56 ms):
`/home` está num HDD rotacional com 99% de ocupação, e o shuffle do treino gera
leituras aleatórias espalhadas por 103k arquivos. Com loader (88 img/s) < GPU
(146 img/s), **o treino local é IO-bound** — otimizações de modelo não reduzem o
tempo de época nesta máquina. O decode em si é barato (1.56 ms) e o transform
adiciona apenas 0.6 ms: **não há gargalo de CPU**.

**No DGX** (`/raid`, NVMe): sem custo de seek, o pipeline vira decode-bound a
~640 img/s *por worker* → 8–16 workers saturam a GPU com folga. Para o NTIRE
(JPEGs full-res), o decode acelerado via `Image.draft` (`jpeg_draft_size: 448`)
e `num_workers: 16` já estão aplicados (PLAN v0.1, item 1.8).

**Mitigações locais** (apenas se for treinar de verdade nesta workstation):
mover `data/ffpp_frames` para o SSD (`sdb`, Kingston SA400) ou empacotar os
PNGs em shards de leitura sequencial (WebDataset/tar, LMDB). Não implementado —
fora do caminho de produção.

### 3.2 Utilização de recursos (A3, A4)

- **Mixed precision / TF32 / cuDNN benchmark**: já aplicados
  (`precision: bf16-mixed`, `set_float32_matmul_precision("high")`,
  `trainer.benchmark: true`, `deterministic: warn`) — PLAN v0.1, item 1.4.
- **Atenção (A3)**: SDPA já ativo por default (verificado no fonte instalado).
  FlashAttention-2 exigiria `pip install flash-attn` e daria ganho marginal em
  seq_len 197 (196 patches + MAP pooling). Exposto como
  `backbone.attn_implementation: null|eager|sdpa|flash_attention_2` para
  experimentação no DGX.
- **Adam fused (A4)**: medido 3.04 → 3.00 steps/s (paridade, dentro do ruído) —
  o update de 2.3M params é fração desprezível do step; o default `foreach` já
  agrupa os ~150 tensores pequenos. Além disso, o run controlado revelou que o
  plugin de mixed precision do Lightning **rejeita optimizers fused quando
  `gradient_clip_val` está setado** (`RuntimeError: ... does not allow for
  gradient clipping because it performs unscaling of gradients internally`).
  Default mantido `false`; a flag funciona com `trainer.gradient_clip_val=null`
  ou `precision=32-true`.

### 3.3 Arquitetura — FLOPs e convergência (A2, A6)

- **Convergência (A6)**: o desenho é estável por construção — backbone
  congelado (sem drift dos pesos pré-treinados), LoRA `up` e CDC `conv_up`
  inicializados em zero (o modelo começa numericamente idêntico ao
  pré-treinado), `gradient_clip_val=1.0` como cinto de segurança. Nenhuma
  camada com risco de vanishing/exploding identificada.
- **FLOPs (A2) — achado principal**: com LoRA injetado no bloco 0, o autograd
  precisa retropropagar pelos 12 blocos inteiros do ViT congelado para alcançar
  os primeiros parâmetros treináveis. Restringindo a injeção de PEFT aos
  últimos 6 blocos, o backward **trunca** no bloco 6 (os blocos 0–5 ficam fora
  do grafo):

  | Variante | steps/s | img/s | VRAM pico | Params treináveis |
  |---|---|---|---|---|
  | baseline (`start_layer=0`, paper) | 3.04 | 146 | 3.65 GB | 2.30M |
  | Adam fused | 3.00 | 144 | 3.67 GB | 2.30M |
  | `start_layer=6` (+fused) | **4.38 (+44%)** | **210** | **2.13 GB (−42%)** | 1.23M |

  ⚠️ **Isto altera o método** — o paper OSDFD adapta todos os blocos. O ganho
  de throughput só deve ser adotado se `val/auc` (e as submissões NTIRE) não
  regredirem. Recomenda-se adicioná-lo como run **R6** à matriz de experimentos
  do `PLAN_v0.1.md` (§2.7). Default `0` preserva o comportamento paper-faithful.

### 3.4 Escalabilidade — FSDP / DeepSpeed / FlashAttention (A5)

**FSDP e DeepSpeed não se justificam para este modelo.** Sharding resolve
estado de otimizador/gradientes que não cabem em uma GPU; aqui o estado do Adam
ocupa ~28 MB (2.3M params) e o pico total é 3.65 GB em batch 48 — numa B200
(192 GB) há espaço para ~20× o batch antes de sharding fazer sentido, e o mesmo
vale para o `siglip2_large`. **DDP é a estratégia correta e já é suportada**:

```bash
./scripts/train.sh trainer.devices=4 trainer.strategy=ddp optimizer.lr=1.2e-4
```

(escala de LR linear com o batch efetivo — ver comentário em
`configs/trainer/default.yaml`). FlashAttention-2 é a única sugestão dessa
família que vale um teste no DGX (`pip install flash-attn` +
`backbone.attn_implementation=flash_attention_2`), com expectativa de ganho
baixo (§3.2).

---

## 4. Refinamentos implementados

Todos config-gated e retrocompatíveis (defaults preservam o comportamento
anterior/paper-faithful):

| Mudança | Arquivos | Default |
|---|---|---|
| `peft.start_layer` — primeiro bloco a receber PEFT; trunca o backward | `src/models/peft_inject.py`, `src/models/osdfd.py`, `src/lightning/module.py`, `configs/peft/lora_cdc.yaml` | `0` (paper) |
| `backbone.attn_implementation` — seleção de kernel de atenção | `src/models/backbone.py`, `src/models/osdfd.py`, `src/lightning/module.py`, `configs/backbone/*.yaml` | `null` (= SDPA) |
| `optimizer.fused` — Adam de kernel único (guard de CUDA) | `src/lightning/module.py`, `configs/optimizer/adam.yaml` | `false` (ver A4) |
| `data.prefetch_factor` — fila por worker mais funda | `src/data/datamodule.py`, `configs/data/*.yaml` | `null` (=2); NTIRE: `4` |
| Teste `test_peft_start_layer_truncates_injection` | `tests/test_smoke.py` | — |

---

## 5. Validação controlada

1. **Benchmark GPU A/B** (sintético, 50 steps + 10 warmup): tabela em §3.3.
2. **Run real de 100 steps** (FF++, defaults pós-auditoria):
   ```bash
   python train.py trainer.max_steps=100 trainer.val_check_interval=100 \
       logger.wandb.enabled=false
   ```
   Completou train → validação no step 100 → checkpoint → test (exit 0).
   Este run capturou em runtime a incompatibilidade fused+clip (A4) que o
   benchmark sintético não detectou — motivo da reversão do default.
3. **Suíte de testes**: `pytest tests/ -q` → **22/22 passando**.

**Snippet de validação para o DGX** (dataset pré-processado em
`data/ffpp_frames`, derivado de `data/FaceForensics++_C23`):

```bash
# baseline pós-auditoria
./scripts/train.sh trainer.max_steps=100 trainer.val_check_interval=100 \
    data.root=data/ffpp_frames logger.wandb.enabled=false

# candidato a +44% de throughput (validar val/auc antes de adotar — run R6):
./scripts/train.sh trainer.max_steps=100 trainer.val_check_interval=100 \
    data.root=data/ffpp_frames peft.start_layer=6 logger.wandb.enabled=false
```

---

## 6. Impacto esperado (resumo)

| Proposta | Impacto | Confiança |
|---|---|---|
| `peft.start_layer=6` | +44% steps/s, −42% VRAM (**medido**) | Alta no throughput; **acurácia a validar** (R6) |
| Dados em SSD/NVMe (workstation local) | destrava o loader de 88 → ~600+ img/s | Alta (decode quente medido) |
| `prefetch_factor=4` (NTIRE/DGX) | suaviza jitter de decode; ganho pequeno | Média |
| FlashAttention-2 (DGX) | marginal em seq_len 197 | Baixa |
| Adam fused / FSDP / DeepSpeed | ~zero para este modelo | Alta (medido/analisado) |

## 7. Fora de escopo / follow-ups

- Empacotamento dos frames FF++ em shards sequenciais (WebDataset/LMDB) — só
  se a workstation local virar ambiente de treino recorrente.
- `torch.compile`: **conhecido e quebrado** com o FSM (controle de fluxo
  dependente de dado → `BackendCompilerFailed` no inductor); ver comentário em
  `train.py`. Um follow-up viável seria compilar apenas o backbone congelado.
- Armazenar os pesos congelados do backbone em bf16 (economia de ~170 MB de
  VRAM e banda) — benefício pequeno demais para justificar o risco de edge
  cases de dtype; não implementado.

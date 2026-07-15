# OPEN.md — Arquitetura do Modelo e Pipeline de Treinamento

Reimplementação do **OSDFD** — *"Open-Set Deepfake Detection: A Parameter-Efficient
Adaptation Method with Forgery Style Mixture"* (Kong et al., arXiv:2408.12791v3,
submetido ao IEEE TCSVT) — sobre um backbone **SigLIP 2** com PyTorch Lightning + Hydra.

- §1–2: arquitetura do modelo implementado
- §3: pipeline de treinamento
- §4: **comparação ponto a ponto com o artigo** (docs/)
- §5: melhorias propostas → ver `PLAN_v0.2.md`

---

## 1. Visão geral

```
pixel_values (B, 3, 224, 224)
    │
    ▼
SigLIP 2 vision tower (CONGELADO, ~92.9M params)
  ├── embeddings (patch 16×16 + pos. embedding, sem [CLS])
  └── 12 blocos transformer, cada um com:
        ├── Self-Attention: Q/K/V → LoRA (r=8, treinável)      ← Eq. (4)-(5)
        └── FFN:  MLP(h) + CDCAdapter(h)  (adapter treinável)  ← Eq. (1)-(3)
    │
    ▼  patch_tokens (B, 196, 768)  [pós post_layernorm]
Forgery Style Mixture (FSM)            ← Eq. (6)-(9); só em treino, só nos fakes
    │
    ▼
MAP pooling head (attention pooling do SigLIP 2, congelado)
    │
    ▼  pooled (B, 768)      [opcional: concat com média dos tokens → global_local]
ClassifierHead (MLP treinável)
  ├── fc1 768→256 + GELU  → scl_features (B, 256)  → L_SCL   ← Eq. (11)-(13)
  └── fc2 256→1           → logit (B,)             → L_BCE
                                                      │
                              L = L_BCE + λ·L_SCL  ←──┘        ← Eq. (10)
```

**Parâmetros treináveis (siglip2-base, default):**

| Componente | Cálculo | Params |
|---|---|---|
| LoRA (12 blocos × Q/K/V) | 12 × 3 × (768·8 + 8·768) | 442.368 |
| CDC adapter (12 blocos) | 12 × (conv_down 49.216 + CDC 3×3 36.864 + conv_up 49.920) | 1.632.000 |
| ClassifierHead | fc1 196.864 + fc2 257 | 197.121 |
| **Total treinável** | | **≈ 2,27M** (2,4% dos ~95M totais) |

Referências no código: `src/models/osdfd.py` (montagem), `backbone.py`, `lora.py`,
`cdc_adapter.py`, `fsm.py`, `head.py`, `peft_inject.py`.

## 2. Componentes

### 2.1 Backbone — SigLIP 2 (`src/models/backbone.py`)

`google/siglip2-base-patch16-224` (default) ou `siglip2-large-patch16-256`, carregado
via `transformers.AutoModel` e **totalmente congelado**. Diferente do CLIP/ViT clássico:

- **Sem token [CLS]** — todos os 196 tokens são patches espaciais, o que torna o
  reshape token→grid do CDC adapter exato (14×14).
- **MAP head** (Multihead Attention Pooling) para o embedding global, aplicado
  **depois** do FSM (fluxo da Fig. 4a do artigo: blocos → FSM → head).
- `attn_implementation` configurável (`sdpa` é o default do HF em torch ≥ 2.1.1).
- `pretrained=false` reconstrói a arquitetura sem download (usado por
  `load_for_inference` para restaurar checkpoints offline).

### 2.2 LoRA (`src/models/lora.py`)

`LoRALinear` envolve as projeções Q/K/V congeladas de cada bloco:
`out = base(x) + (α/r) · up(down(x))`, com `r=8`, `α=8` (escala 1.0 — equivalente à
Eq. 5 do artigo, que não usa fator de escala), init Kaiming/zeros (residual nulo no
início). Alvos configuráveis via `peft.lora.targets` (qualquer atributo de
`layer.self_attn`, ex.: `out_proj`).

### 2.3 CDC Adapter (`src/models/cdc_adapter.py`)

Paralelo ao FFN de cada bloco: `h_out = MLP(h) + Adapter(h)` (Eq. 1), com
`Adapter = Conv1×1_up(GELU(CDC(Conv1×1_down(h))))` (Eq. 2). Os tokens são
remodelados para um mapa 2-D 14×14 antes das convoluções. O operador CDC segue a
formulação generalizada do CDCN:

```
y = vanilla_conv(x) − θ · x_c · Σw        (θ=1.0 ⇔ Eq. 3 pura do artigo)
```

Default `θ=0.7` (valor consagrado do CDCN), `bottleneck=64`, kernel 3×3.
`conv_up` inicializado em zero → o adapter começa como identidade residual.

### 2.4 Forgery Style Mixture (`src/models/fsm.py`)

Aplicado nos patch tokens da última camada, **apenas em treino** e **apenas nos
fakes** do batch:

1. Com probabilidade `prob=0.5` o módulo dispara (senão, identidade).
2. Cada fake é pareado com um fake de **domínio de manipulação diferente**
   (amostragem uniforme vetorizada entre candidatos válidos).
3. Estatísticas AdaIN (μ, σ por canal, sobre a dimensão de tokens) são misturadas
   com peso `δ ~ Beta(0.1, 0.1)` (Eqs. 7–8) e reaplicadas ao conteúdo original (Eq. 9).

**Extensão além do artigo** — `single_domain_fallback`: quando o batch tem um único
domínio fake (ex.: NTIRE, sem rótulo por gerador), `"random"` pareia com outro fake
qualquer (estilo MixStyle); `"off"` reproduz o no-op original. O flag
`last_fired` é logado como `train/fsm_fired` para monitorar a taxa de disparo
(o FSM pode virar no-op silencioso — diagnóstico D1 do PLAN_v0.1).

### 2.5 Cabeça e perdas (`src/models/head.py`, `src/losses/`)

- `ClassifierHead`: `fc1(in→256) + GELU → fc2(256→1)`. A saída de `fc1` é a feature
  da "penúltima FC" usada pela SCL (Sec. III-D do artigo).
- `SingleCenterLoss` (Eqs. 11–13): centro `C` = média batch das features **reais**;
  compacta reais e afasta fakes com margem hinge. `margin=0.01` absoluto (artigo) ou
  `margin_scale=sqrt_dim` (formulação SCL original, margem relativa a √D).
- `OSDFDLoss`: `L = BCE + λ·SCL`, `λ=1.0`; `pos_weight` opcional para desbalanceio.

## 3. Pipeline de treinamento

### 3.1 Dados

| | FaceForensics++ c23 | NTIRE RobustAIGenDetection |
|---|---|---|
| Fonte | `data/ffpp_frames/<split>/<classe>/<vídeo>/*.png` (via `scripts/preprocess_ffpp.py`: MTCNN, margem 1.3, splits oficiais 720/140/140) | manifest CSV `path,label,domain,split` (via `scripts/preprocess_ntire.py`, split estratificado 90/5/5 do train) |
| Domínios FSM | 0=real, 1–4 = DF/F2F/FS/NT (da estrutura de pastas) | `domain=label` (1 domínio fake) → fallback `random`; ou pseudo-domínios k-means (`scripts/assign_pseudo_domains.py`) |
| Balanceamento | `real_oversample=4` (artigo, Sec. IV-B) | `real_oversample=2` (razão ~36/64); alternativa `balance_sampler` |
| Augmentation | **desligada** (fiel ao artigo) | **ligada** (hflip, jitter leve, RRC, blur, JPEG 30–95, downscale — robustez "in the wild" do challenge) |
| Decode | PNG | JPEG com `PIL.Image.draft(448)` (decode rápido), `prefetch_factor=4`, 16 workers |

Val/test usam sempre resize + normalização pura (`resize_mode: squash` default;
`crop` disponível como experimento).

### 3.2 Loop de treino (`src/lightning/module.py`, `configs/trainer/default.yaml`)

- **Otimização** (fiel ao artigo, Sec. IV-A): Adam `lr=3e-5`, β=(0.9, 0.999), sem
  weight decay, **sem** LR decay (`scheduler: none`), batch 48, **30k steps**,
  `gradient_clip_val=1.0`, validação a cada 1000 steps monitorando `val/auc`.
- **Precisão**: `bf16-mixed` (AMP), TF32 ativo, `benchmark=true`,
  `deterministic=warn` (repro bit-exata documentada no README).
- **Métricas**: torchmetrics (`BinaryAUROC/AP/Acc/F1`) — corretas sob DDP; EER e
  curvas via caminho numpy. O **threshold de decisão é calibrado no EER da
  validação** e persistido no checkpoint (`calibrated_threshold`), usado por
  `predict_step` e pelo `Predictor`.
- **Callbacks**: checkpoint top-1 por `val/auc` + last, LR monitor, EMA opcional
  (`callbacks.ema.enabled`), early stopping desligado (treino de duração fixa).
- **Multi-GPU**: DDP opt-in (`trainer.devices=N trainer.strategy=ddp`), com escala
  linear de LR documentada. FSDP/DeepSpeed avaliados e rejeitados (2,3M params
  treináveis; ver `AUDIT.md` A5).
- **Logging**: W&B + TensorBoard; `train/fsm_fired`, tempos de época, pico de VRAM.
- `torch.compile`: **quebrado** com o FSM (controle de fluxo dependente de dados;
  inductor `BackendCompilerFailed`) — default `false`, documentado.

### 3.3 Inferência

- `test.py`: passe único de predição, métricas + figuras nos loggers.
- `predict.py` / `src/inference/predictor.py`: `predict_folder` em batch
  (DataLoader próprio), threshold calibrado por default.
- `scripts/make_ntire_submission.py`: gera o CSV `image_name,pred` do challenge.

## 4. Comparação com o artigo (docs/)

### 4.1 Correspondências fiéis

| Elemento do artigo | Implementação | Status |
|---|---|---|
| LoRA em Q/K/V, r=8, backbone congelado (Eqs. 4–5) | `LoRALinear`, r=8, α/r=1 | ✅ fiel |
| Adapter no FFN: Conv1×1↓ → CDC → Conv1×1↑ (Eqs. 1–2) | `CDCAdapter`, bottleneck 64 | ✅ fiel |
| FSM: AdaIN stats, δ~Beta(0.1,0.1), prob 0.5, train-only, só fakes, pareamento entre domínios distintos (Eqs. 6–9, Fig. 7) | `ForgeryStyleMixture` | ✅ fiel |
| Posição do FSM: blocos → FSM → head (Fig. 4a) | tokens pós-encoder → FSM → MAP pool → MLP | ✅ fiel |
| L = BCE + λ·SCL, λ=1, margin 0.01, SCL na penúltima FC (Eqs. 10–13) | `OSDFDLoss` + `ClassifierHead` | ✅ fiel |
| Adam 3e-5, β=(0.9,0.999), sem LR decay, batch 48, 30k iters, AUC como métrica de val | `configs/optimizer/adam.yaml`, `trainer/default.yaml` | ✅ fiel |
| Reais aumentados 4× para balanceio (Sec. IV-B) | `real_oversample: 4` (FF++) | ✅ fiel |
| Margem da face 1.3×, entrada 224×224 | preprocess FF++ (`face_margin=1.3`) + resize 224 | ✅ fiel |
| Sem data augmentation | FF++: augmentation desligada | ✅ fiel |

### 4.2 Divergências deliberadas

| Aspecto | Artigo | Esta implementação | Racional |
|---|---|---|---|
| **Backbone** | ViT-B/16 ImageNet-21K (1,34M treináveis) e CLIP ViT-L/14 (2,89M) | **SigLIP 2** base/large | SigLIP 2 é um encoder VL mais recente com melhores features densas; decisão de projeto do v0.1. Consequência: números do artigo não são diretamente comparáveis. |
| **Pooling global** | MLP head sobre a saída dos blocos (ViT-B tem [CLS]) | **MAP attention-pooling** do SigLIP 2 (congelado; treinável via `train_pool_head`) | SigLIP 2 não tem [CLS]; o MAP head é o mecanismo nativo. O FSM mistura estatísticas de tokens *antes* do pooling — atenção do MAP re-pondera os tokens misturados, um caminho que não existe no artigo. |
| **CDC θ** | Eq. 3 pura (equivale a θ=1.0) | **θ=0.7** (default CDCN), configurável | 0.7 é o valor validado na literatura CDCN; θ=1.0 disponível para ablação. |
| **Detector de face** | dlib | MTCNN (facenet-pytorch) | Qualidade/velocidade; mesma margem 1.3. |
| **Params treináveis** | 1,34M (ViT-B) | ≈2,27M (siglip2-base) | Bottleneck 64 do CDC domina (1,63M). O artigo não detalha o bottleneck usado; reduzi-lo é ablação proposta no PLAN_v0.2. |
| **FSM em domínio único** | Pressupõe FF++ multi-domínio; sem fallback | `single_domain_fallback: random` + métrica `fsm_fired` | Sem isso o FSM é no-op silencioso no NTIRE (achado D1). |
| **Augmentation (NTIRE)** | Nenhuma | Pipeline "in the wild" (JPEG/blur/crop) | O challenge NTIRE avalia robustez a pós-processamento; divergência restrita ao config NTIRE. |
| **Threshold de decisão** | Não especificado (métricas AUC/EER) | Calibrado no EER da validação, persistido no ckpt | Necessário para `pred` binário reprodutível (submissão NTIRE). |
| **Precisão numérica** | Não especificado | bf16-mixed + TF32 | Custo/benefício em B200/RTX; caminho fp32 documentado. |

### 4.3 Capacidades extras (ausentes no artigo)

Config-gated, desligadas por default quando alteram o método: `feature_fusion:
global_local` (concat MAP + média dos tokens), `train_norm`, `train_pool_head`,
`peft.start_layer` (trunca backward, +44% steps/s medido — AUDIT A2), EMA de
parâmetros treináveis, `balance_sampler`, `scl_margin_scale: sqrt_dim`,
scheduler cosine, pseudo-domínios k-means para FSM no NTIRE.

### 4.4 Protocolo de avaliação — diferença importante

O artigo avalia em **cross-manipulation** (treina em 3 manipulações do FF++, testa
na 4ª) e **cross-dataset** (treina FF++ c23, testa CDF/WDF/DFDC/DFDC-P/DFR/FFIW).
O pipeline atual treina com as 4 manipulações juntas e avalia nos splits oficiais
do FF++ (in-domain) e no split interno do NTIRE. **Reproduzir o protocolo open-set
do artigo exige configs de treino leave-one-manipulation-out** — proposto no
`PLAN_v0.2.md` (grupo E).

## 5. Melhorias propostas

Ver **`PLAN_v0.2.md`**: variações de arquitetura e pipeline organizadas como
configs Hydra de experimento (`configs/experiment/`), com hipótese, critério de
decisão e ordem de execução para cada variação.

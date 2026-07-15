# PLAN_v0.2 — Variações de Arquitetura e Pipeline a Avaliar

**Escopo:** este plano NÃO altera defaults nem implementa mudanças. Ele define
(1) uma infraestrutura de *configs de experimento* Hydra e (2) um catálogo de
variações de arquitetura/pipeline, cada uma com hipótese, config proposto,
métrica e critério de decisão. A implementação se resume a criar arquivos YAML
(grupos A/C/E) ou pequenas extensões config-gated (grupos B/D, marcadas).

**Pré-requisitos:** Fases 0–2 do `PLAN_v0.1.md` concluídas (22/22 testes);
achados do `AUDIT.md` (A1–A6); descrição da arquitetura em `OPEN.md`.

---

## 0. Infraestrutura de experimentos

### 0.1 Grupo `configs/experiment/`

Criar o grupo Hydra padrão de experimentos. Cada variação é um arquivo
`configs/experiment/<id>_<slug>.yaml` com `# @package _global_`, contendo apenas
os overrides em relação ao baseline. Execução:

```bash
./scripts/train.sh +experiment=a1a_lora_r4          # uma variação
./scripts/train.sh +experiment=a1a_lora_r4 data=ntire  # variação × dataset
```

Regras:
- **Um experimento = um arquivo.** Nada de override manual na CLI em runs oficiais
  (garante rastreabilidade: o nome do experimento vai para o W&B run name).
- Todo experimento define `logger.wandb.tags` com `[v0.2, <grupo>, <id>]`.
- O baseline é a config atual sem overrides (registrar como `a0_baseline.yaml`
  vazio exceto tags, para ter um run nomeado de referência).

### 0.2 Protocolo de avaliação comum

| Item | Valor |
|---|---|
| Triagem (screening) | 5.000 steps, `val_check_interval=500`, seed 0 |
| Confirmação | 30.000 steps (budget completo), seeds {0, 1} para os vencedores |
| Métrica primária | `val/auc` (FF++ val ou NTIRE val interno) |
| Métricas secundárias | `val/eer`, `val/ap`, `train/fsm_fired`, steps/s, VRAM pico |
| Critério de promoção | melhora ≥ +0,3 p.p. AUC sobre o baseline no budget completo, 2 seeds, sem regressão de EER |
| Critério de descarte | −0,5 p.p. AUC na triagem → não avança para confirmação |

Variações de **eficiência** (A5, C-perf) têm critério invertido: promover se a
AUC ficar dentro de ±0,2 p.p. do baseline com ganho de throughput ≥ 20%.

---

## Grupo A — Arquitetura, config-only (só criar YAML)

### A0 — Baseline nomeado
`configs/experiment/a0_baseline.yaml`: apenas tags. Referência de todas as comparações.

### A1 — Rank do LoRA
**Hipótese:** r=8 pode estar sub/sobre-dimensionado para SigLIP 2 (o artigo calibrou
para ViT-B ImageNet-21K). Manter α=r (escala 1.0 constante) para isolar o efeito.

```yaml
# a1a_lora_r4.yaml            # a1b_lora_r16.yaml
# @package _global_
peft:
  lora: {r: 4, alpha: 4.0}    #   lora: {r: 16, alpha: 16.0}
```

### A2 — CDC: θ e bottleneck
**Hipótese θ:** a Eq. 3 do artigo é CDC pura (θ=1.0); nosso default 0.7 vem do CDCN.
Vale medir qual extrai melhor os artefatos de alta frequência.
**Hipótese bottleneck:** o CDC domina os params treináveis (1,63M de 2,27M);
bottleneck 32 aproxima a contagem do artigo (1,34M) e testa se há redundância.

```yaml
# a2a_cdc_theta10.yaml: peft.cdc.theta: 1.0
# a2b_cdc_bneck32.yaml: peft.cdc.bottleneck: 32
# a2c_cdc_bneck128.yaml: peft.cdc.bottleneck: 128
```

### A3 — Fusão global+local
**Hipótese:** o artigo enfatiza pistas locais (CDC); concatenar a média dos patch
tokens ao embedding MAP dá à cabeça acesso direto a elas.

```yaml
# a3_global_local.yaml
model: {feature_fusion: global_local}
```

### A4 — MAP head treinável
**Hipótese:** o MAP head do SigLIP 2 foi treinado para semântica VL, não para
forense; destravá-lo (~2,4M params extras) pode readaptar a atenção de pooling —
ou causar overfitting/esquecimento. Comparar também com A3 (caminhos alternativos
para o mesmo problema).

```yaml
# a4_train_pool_head.yaml
model: {train_pool_head: true}
```

### A5 — PEFT só nas camadas finais (R6 do AUDIT)
**Hipótese:** adaptar só os blocos 6–11 corta o backward pela metade
(+44% steps/s, −42% VRAM medidos) com perda de qualidade possivelmente nula —
as camadas iniciais de um encoder pré-treinado são genéricas. **Risco:** o CDC
nas camadas iniciais é justamente onde artefatos de alta frequência são mais
visíveis; este experimento decide o trade-off. Avaliar também a variante
intermediária `start_layer=3`.

```yaml
# a5a_start_layer6.yaml: peft.start_layer: 6
# a5b_start_layer3.yaml: peft.start_layer: 3
```

### A6 — Backbone large
**Hipótese:** análogo ao salto ViT-B → CLIP ViT-L do artigo (+6 p.p. AUC médio na
Tabela I). siglip2-large-patch16-256 (24 blocos, d=1024, 256 tokens).

```yaml
# a6_siglip2_large.yaml
backbone: siglip2_large        # troca de grupo, não de chave
data: {batch_size: 24}         # VRAM; compensar com accumulate_grad_batches: 2
trainer: {accumulate_grad_batches: 2}
```

### A7 — LoRA também no out_proj
**Hipótese:** o artigo adapta só Q/K/V; incluir a projeção de saída da atenção é
padrão em LoRA moderno e custa +147k params. Suportado hoje: `targets` aceita
qualquer atributo de `self_attn`.

```yaml
# a7_lora_oproj.yaml
peft:
  lora: {targets: [q_proj, k_proj, v_proj, out_proj]}
```

### A8 — Capacidade da cabeça
**Hipótese:** 256 de largura SCL pode limitar a separabilidade; dropout leve pode
regularizar com augmentation ligada (NTIRE).

```yaml
# a8a_head512.yaml: model: {head_hidden_dim: 512}
# a8b_head_dropout.yaml: model: {head_dropout: 0.1}
```

---

## Grupo B — Arquitetura, exige código (config-gated, default = comportamento atual)

> Cada item vira flag em config existente, com valor default reproduzindo o
> comportamento atual — mesmo padrão das Fases 1–2 do v0.1. Implementar somente
> se aprovado; cada um acompanha teste unitário próprio.

### B1 — FSM multi-profundidade
**Proposta:** aplicar FSM também na saída de blocos intermediários (ex.: 4 e 8),
não só na última camada. Estatísticas de estilo em camadas rasas capturam textura
de baixo nível — onde os artefatos de gerador vivem. MixStyle original aplica em
camadas rasas justamente por isso.
**Config:** `fsm.layers: [last]` (default) | `[4, 8, last]`.
**Custo:** hook nos blocos do encoder; FSM já é stateless, reutilizável.
**Risco:** interação com o CDC adapter (que também mira alta frequência).

### B2 — LoRA no FFN (fc1/fc2)
**Proposta:** `peft.lora.ffn_targets: []` (default) | `[fc1, fc2]`. Complementa A7;
a literatura (AdaLoRA, LoRA+ ablations) mostra que adaptar o FFN às vezes rende
mais que atenção. +1,2M params com r=8.

### B3 — Centro SCL com EMA
**Proposta:** o centro `C` é a média dos reais **do batch** (Eq. 13) — ruidoso com
poucos reais por batch (batch 48, ~50% reais com oversample). Manter um centro
EMA global (`scl_center: batch` (default) | `ema`, `scl_center_momentum: 0.9`)
estabiliza o alvo da compactação.
**Custo:** buffer registrado no loss; sincronização DDP via `all_reduce` na média
dos reais.

### B4 — Normalização L2 das features SCL
**Proposta:** `loss.scl_normalize: false` (default) | `true` — projeta as features
na hiperesfera antes de distâncias; torna a margem adimensional e remove a
interação margem×escala que motivou `sqrt_dim`. Combinar com `scl_margin≈0.1–0.3`.

### B5 — Warmup de LR
**Proposta:** `scheduler.warmup_steps: 0` (default) | `500`. Com adapter/LoRA
zero-init o modelo começa idêntico ao pré-treinado; warmup curto evita passos
grandes iniciais na cabeça. Necessário para A6 (large) e para lr maiores.

---

## Grupo C — Pipeline, config-only

### C1 — Scheduler cosine
**Hipótese:** o artigo não usa decay, mas treina um ViT-B ImageNet; com SigLIP 2 +
augmentation (NTIRE), cosine até `eta_min=1e-7` pode ganhar no fim do treino.

```yaml
# c1_cosine.yaml
scheduler: cosine              # t_max já é 30000
```

### C2 — WeightedRandomSampler vs. oversample por duplicação
**Hipótese:** amostragem ponderada dá balanceio exato por batch sem inflar a época.

```yaml
# c2_balance_sampler.yaml
data: {balance_sampler: true, real_oversample: 1}
```

### C3 — EMA dos parâmetros treináveis
**Hipótese:** EMA (decay 0.999) suaviza o fim de treino; barato (só 2,3M params).

```yaml
# c3_ema.yaml
callbacks: {ema: {enabled: true, decay: 0.999}}
```

### C4 — Resize preservando aspecto
**Hipótese:** `squash` distorce faces não-quadradas; `crop` (Resize+CenterCrop /
RRC no treino) preserva a geometria dos artefatos. Relevante sobretudo no NTIRE
(imagens heterogêneas).

```yaml
# c4_resize_crop.yaml
data: {resize_mode: crop}
```

### C5 — Augmentation leve no FF++
**Hipótese:** o artigo não usa augmentation no FF++, mas hflip+JPEG leve não muda
a natureza do artefato e pode melhorar generalização cross-manipulation (grupo E).

```yaml
# c5_ffpp_aug_light.yaml
data:
  augmentation:
    enabled: true
    hflip: 0.5
    jpeg: 0.3
    jpeg_quality: [60, 95]
    color_jitter: 0.0
    gaussian_blur: 0.0
    random_resized_crop: false
    downscale: 0.0
    random_erasing: 0.0
```

### C6 — Pseudo-domínios k-means para FSM (NTIRE)
**Hipótese:** o fallback `random` mistura fakes arbitrários; pseudo-domínios por
cluster de embedding aproximam a premissa do artigo (domínios = geradores).
Manifests gerados por `scripts/assign_pseudo_domains.py` com k ∈ {4, 8, 16}.

```yaml
# c6a_ntire_k8.yaml  (analogamente c6b k=4, c6c k=16)
data: {manifest: data/ntire_manifest_k8.csv}
fsm: {single_domain_fallback: off}   # exige domínios reais; monitora fsm_fired
```

**Atenção:** validar `train/fsm_fired ≈ prob` no início do run; se cair, os
clusters colapsaram e o experimento é inválido.

### C7 — Calibração da SCL
**Hipótese:** margem absoluta 0.01 é desprezível frente a `dist_r` típico (~5–15
em 256-d não normalizado) — a hinge quase nunca ativa; o termo efetivo vira só
compactação. `sqrt_dim` com margem 0.3 (≈4.8 absoluto) ativa a hinge de fato.

```yaml
# c7a_scl_sqrtdim.yaml: loss: {scl_margin: 0.3, scl_margin_scale: sqrt_dim}
# c7b_scl_w05.yaml:     loss: {scl_weight: 0.5}
# c7c_scl_w2.yaml:      loss: {scl_weight: 2.0}
```

### C8 — FSM: intensidade
```yaml
# c8a_fsm_prob08.yaml: fsm: {prob: 0.8}
# c8b_fsm_alpha03.yaml: fsm: {alpha: 0.3}   # Beta menos bimodal → misturas mais intermediárias
# c8c_fsm_off.yaml:     fsm: {prob: 0.0}    # ablação de controle (quanto o FSM contribui aqui?)
```

---

## Grupo D — Pipeline, exige código

### D1 — Cache de decode / formato binário (condicional)
Só se o treino no DGX se mostrar IO-bound (no local é — AUDIT A1; no DGX NVMe
provavelmente não). Proposta mínima: `data.cache: none|memmap` com tensores
uint8 pré-redimensionados. Não iniciar sem medição no DGX.

### D2 — Val de robustez (NTIRE)
Segunda passada de validação com corrupções fixas (JPEG q50, blur σ1.0, resize
0.5×) — mede exatamente o que o challenge pontua. `data.robust_val: false`
(default) | `true`; loga `val_robust/auc`. Hoje a augmentation é só de treino;
a métrica de robustez não existe.

---

## Grupo E — Protocolo open-set do artigo (config-only)

O pipeline atual treina com as 4 manipulações do FF++ juntas; o artigo avalia
**leave-one-manipulation-out** (Tabelas I–II). Sem isso, nenhuma comparação com o
artigo é válida. `domain_map` + a estrutura de pastas já permitem expressar isso
por config (filtrar classes de treino via `data.train_classes`, que **exige um
flag novo pequeno** — único item deste grupo com código, ~10 linhas no datamodule).

```yaml
# e1_loo_deepfakes.yaml — treina F2F/FS/NT + real, testa DF
data:
  train_classes: [real, Face2Face, FaceSwap, NeuralTextures]
  test_classes: [real, Deepfakes]
# e2_loo_face2face.yaml, e3_loo_faceswap.yaml, e4_loo_neuraltextures.yaml análogos
```

Meta de referência (artigo, OSDFD ViT-B, c23): AUC médio 0.843. Rodar os 4 splits
com o baseline A0 estabelece nossa âncora SigLIP 2 antes de qualquer variação.

---

## Ordem de execução e prioridades

**Fase E (âncora, primeiro):** E1–E4 com A0 → estabelece comparabilidade com o
artigo. Sem âncora, os grupos A/C não têm régua externa.

**Fase 1 (triagem 5k steps, FF++, maior impacto esperado):**

| Prioridade | Experimentos | Por quê |
|---|---|---|
| P0 | A5a/A5b (start_layer) | maior ganho de custo já medido; decide o default de eficiência |
| P0 | C7a (SCL sqrt_dim) | suspeita concreta de hinge inativa no baseline |
| P1 | A2a/A2b (CDC θ, bneck) | fidelidade ao artigo + params |
| P1 | A1a/A1b (LoRA r) | barato, mexe no núcleo do método |
| P1 | C8c (FSM off) | ablação de controle indispensável |
| P2 | A3, A4, A7, A8, C1, C2, C3, C4, C5, C8a/b | segunda leva |

**Fase 2 (NTIRE, após vencedores da Fase 1):** C6a–c (pseudo-domínios), C4, D2;
combinar os 2–3 vencedores da Fase 1 num run composto.

**Fase 3 (confirmação):** budget completo 30k, 2 seeds, só para candidatos a
promoção de default. A6 (large) entra direto aqui (caro demais para triagem).

**Grupo B:** implementar apenas os itens cujo experimento config-only vizinho
sinalizar potencial (ex.: B4 se C7a ganhar; B1 se C8 mostrar FSM sensível).

## Riscos e salvaguardas

- **Triagem de 5k steps pode inverter rankings** vs. 30k (LoRA converge tarde).
  Mitigação: nunca descartar por menos de −0,5 p.p.; confirmar top-3 sempre.
- **Interações entre variações** (ex.: A4 × A3 redundantes; C7 × B4): runs
  compostos só na Fase 2/3, um fator por vez antes disso.
- **Comparabilidade:** qualquer mudança promovida a default exige atualizar
  `OPEN.md` §4 e re-rodar A0.

## Entregáveis deste plano (sem tocar em código, exceto onde marcado)

1. `configs/experiment/` com os YAMLs dos grupos A, C e E (E exige o flag
   `train_classes`/`test_classes` — único código do escopo imediato, com teste).
2. Runs E1–E4 (âncora) + Fase 1 de triagem no DGX.
3. Tabela de resultados versionada (ex.: `docs/RESULTS_v0.2.md`) com AUC/EER/steps-s
   por experimento e decisão (promover / descartar / confirmar).

# PLAN v0.1 — Melhorias de Arquitetura e Pipeline de Treinamento

**Escopo:** OSDFD (SigLIP 2 + LoRA + CDC + FSM + SCL) treinado em FF++ e NTIRE 2026
(Robust AI-Generated Image Detection in the Wild).
**Estado analisado:** commit `cb221ef` + mudanças locais (`preprocess_ntire.py`,
`configs/data/ntire.yaml`, `download_ntire_val_dataset.py`).
**Regra geral deste plano:** cada item define (a) motivação, (b) arquivos e funções
exatas a alterar, (c) especificação de implementação, (d) critério de aceite.
Nenhum item altera o comportamento default paper-faithful do FF++, salvo indicação
explícita.

---

## 0. Diagnóstico (resumo da análise do código)

| # | Achado | Severidade | Onde |
|---|--------|-----------|------|
| D1 | **FSM é um no-op silencioso no NTIRE**: o manifesto usa `domain = label`, logo todo fake tem `domain=1`; `_domain_shuffle` exige ≥2 domínios distintos e devolve identidade → o módulo central do paper nunca atua nesse dataset. | **Alta** | `src/models/fsm.py:82`, `scripts/preprocess_ntire.py` |
| D2 | O challenge NTIRE é explicitamente sobre **robustez a distorções** (crop/resize/JPEG/blur — o CSV de val tem colunas `distortions`), mas `configs/data/ntire.yaml` treina com `augmentation.enabled=false`. | **Alta** | `configs/data/ntire.yaml` |
| D3 | `trainer.devices: auto` + `strategy: auto` → num DGX com 8×B200 o Lightning sobe **DDP em todas as GPUs silenciosamente**, mudando o batch efetivo (48→384) sem ajuste de LR. | **Alta** | `configs/trainer/default.yaml` |
| D4 | Não existe script de **submissão do challenge** (formato `image_name,pred` sobre `val_images/`), e `OSDFDPredictor.predict_folder` infere imagem a imagem (sem batch) — inviável para 10k+ imagens. | **Alta** | `src/inference/predictor.py`, `scripts/` |
| D5 | `test.py` percorre o test set **duas vezes** (`trainer.test` + `trainer.predict`). | Média | `test.py` |
| D6 | Margem do SCL (0.01) é comparada a distâncias L2 **não normalizadas** de features 256-d (escala ~10¹); a formulação original do SCL escala a margem por `sqrt(D)`. Hoje o termo hinge é dominado pelas distâncias e a margem é ruído. | Média | `src/losses/single_center_loss.py` |
| D7 | `load_from_checkpoint` reconstrói o modelo com `pretrained=True` → **baixa o SigLIP do HuggingFace** (rede/cache) só para sobrescrever com o ckpt. Falha em ambiente offline. | Média | `src/lightning/module.py:25` (`build_model`), `test.py`, `src/inference/predictor.py` |
| D8 | Sob DDP, o `all_gather` de métricas de val inclui os **samples duplicados de padding** do DistributedSampler (viés pequeno, mas o ranking de checkpoints por `val/auc` fica não determinístico). | Média | `src/lightning/module.py:153` (`_finalise_eval`) |
| D9 | `precision: 16-mixed` (fp16+GradScaler) em B200; `bf16-mixed` é numericamente mais estável e é o padrão da arquitetura. `deterministic: true` desliga kernels rápidos (cuDNN benchmark) sem necessidade em runs de produção. TF32 matmul não configurado (Lightning inclusive emite warning). | Média | `configs/trainer/default.yaml`, `config.yaml`, `train.py` |
| D10 | `_domain_shuffle` faz loop Python O(F) com `nonzero` por amostra a cada forward. | Baixa | `src/models/fsm.py:84` |
| D11 | Não há visibilidade de **quantas vezes o FSM realmente dispara** (pode ficar inoperante sem ninguém perceber — vide D1). | Baixa | `src/lightning/module.py` |
| D12 | `predict.py --image-size` default 224 fixo — dessincroniza com ckpt treinado em 256 (`siglip2_large`). | Baixa | `predict.py`, `src/inference/predictor.py` |
| D13 | Threshold de decisão fixo em 0.5 no `predict_step`/`predict.py`; não há calibração (ex.: threshold de EER da validação). | Baixa | `src/lightning/module.py:181`, `predict.py` |
| D14 | Decodificação de JPEGs grandes do NTIRE no CPU é o gargalo provável do dataloader (imagens "in the wild", full-res, reduzidas a 224 via PIL bicubic). | Baixa | `src/data/dataset.py:142` |
| D15 | `Resize((224,224))` **esmaga o aspect ratio**; para detecção de artefatos de geração, resample destrói traços de alta frequência — política de crop deveria ser configurável. | Baixa (ablation) | `src/data/transforms.py:120` |
| D16 | A MAP pooling head do SigLIP fica congelada sem opção de destravar (candidata natural a fine-tuning leve). `real_oversample` duplica registros (época inflada) em vez de usar sampler ponderado. | Baixa (ablation) | `src/models/backbone.py`, `src/data/datamodule.py` |

O plano abaixo é dividido em três fases. **Fase 0 destrava o treino correto no NTIRE**
(objetivo atual do projeto); Fase 1 corrige fidelidade/eficiência; Fase 2 são
ablations opcionais. Dentro de cada fase, os itens são independentes entre si e
podem ser implementados em qualquer ordem, exceto onde marcado.

---

## Fase 0 — Correções críticas para o treino NTIRE

### 0.1 FSM: fallback de pareamento aleatório quando há um único domínio de forgery (D1, D11)

**Motivação.** No NTIRE todos os fakes compartilham `domain=1` e o FSM vira
identidade — o mecanismo central de generalização do paper fica desligado sem
aviso. Os fakes do NTIRE vêm de **múltiplos geradores desconhecidos**; parear
fakes aleatoriamente entre si ainda mistura estatísticas de estilos de geradores
distintos com alta probabilidade (é exatamente o MixStyle clássico, citado como
[73] no paper). O pareamento informado por domínio real continua sendo o
comportamento quando há ≥2 domínios (FF++ intacto).

**Arquivos:** `src/models/fsm.py`, `src/models/osdfd.py`,
`src/lightning/module.py` (`build_model` e `training_step`),
`configs/fsm/default.yaml`.

**Especificação:**

1. `ForgeryStyleMixture.__init__` ganha o parâmetro
   `single_domain_fallback: str = "random"` (valores válidos: `"random"`,
   `"off"`; validar com `ValueError` no construtor) e o atributo de estado
   `self.last_fired: bool = False`.
2. `_domain_shuffle` deixa de ser `@staticmethod` e passa a:
   ```python
   def _domain_shuffle(self, domains: torch.Tensor) -> torch.Tensor:
       f = domains.numel()
       if torch.unique(domains).numel() < 2:
           if self.single_domain_fallback == "off":
               return torch.arange(f, device=domains.device)
           # MixStyle-style: permutação aleatória entre fakes. Pontos fixos
           # ocasionais são inofensivos (mistura consigo mesmo = identidade
           # naquela linha).
           return torch.randperm(f, device=domains.device)
       ...  # caminho multi-domínio (ver item 1.6 para a vetorização)
   ```
3. Em `forward`: setar `self.last_fired = False` na primeira linha; setar
   `self.last_fired = True` imediatamente antes do `return out` final (após a
   mistura efetiva). O early-return `torch.equal(perm, arange)` existente
   permanece — com o fallback `random` ele quase nunca dispara, e com `"off"`
   preserva o comportamento atual.
4. `OSDFDModel.__init__` ganha `fsm_single_domain_fallback: str = "random"` e
   repassa para `ForgeryStyleMixture`.
5. `build_model` (em `src/lightning/module.py`) lê
   `cfg.fsm.get("single_domain_fallback", "random")` e repassa.
6. `configs/fsm/default.yaml` adiciona:
   ```yaml
   single_domain_fallback: random   # "random" (MixStyle entre fakes) | "off"
   ```
7. Logging do fire-rate (resolve D11): em `training_step`, após o forward:
   ```python
   self.log("train/fsm_fired", float(self.model.fsm.last_fired), batch_size=bs)
   ```
   A média dessa métrica no W&B/TB deve gravitar em torno de `fsm.prob` (0.5).
   Se ficar em 0.0, o FSM está inoperante — exatamente o alarme que faltou.

**Aceite:**
- Teste novo em `tests/test_smoke.py::test_fsm_single_domain_fallback`: batch
  sintético com 4 fakes todos `domain=1`, `prob=1.0`, `training=True`,
  `torch.manual_seed(0)` → com `fallback="random"` a saída difere da entrada
  nas linhas fake e é idêntica nas linhas real; com `fallback="off"` a saída é
  idêntica à entrada. `last_fired` reflete cada caso.
- Teste existente de FF++ multi-domínio continua passando sem alteração.
- Num treino NTIRE de 50 steps, `train/fsm_fired` média ≈ 0.5.

### 0.2 Ligar a augmentation de robustez por default no NTIRE (D2)

**Motivação.** O challenge avalia sob distorções (JPEG, resize, blur, crop). O
pipeline de augmentation já existe e já implementa exatamente essas corrupções
(`RandomJPEG`, `RandomDownscale`, `GaussianBlur`, `RandomResizedCrop`) — está só
desligado. O default paper-faithful continua valendo **apenas para FF++**
(`configs/data/faceforensics.yaml` não muda).

**Arquivo:** `configs/data/ntire.yaml` (somente).

**Especificação:** alterar o bloco `augmentation` para:
```yaml
augmentation:
  enabled: true
  hflip: 0.5
  color_jitter: 0.2          # leve; distorção fotométrica do "in the wild"
  hue: 0.05
  random_resized_crop: true  # robustez a crop/reframe
  rrc_scale: [0.5, 1.0]      # crops mais agressivos que o default (challenge usa crop)
  gaussian_blur: 0.3
  blur_sigma: [0.1, 2.0]
  jpeg: 0.5                  # recompressão é a distorção mais comum na prática
  jpeg_quality: [30, 95]
  downscale: 0.3
  downscale_range: [0.25, 0.9]
  random_erasing: 0.0        # cutout pode apagar o artefato-alvo; manter off
```
Sem mudança de código. Val/test continuam com resize+normalize puro
(comportamento já garantido por `build_transform(train=False)`).

**Aceite:** `python train.py data=ntire --cfg job` imprime o bloco acima;
smoke train de 20 steps roda sem erro; `configs/data/faceforensics.yaml`
permanece com `enabled: false` (diff vazio).

### 0.3 Tornar o uso de GPUs explícito (D3)

**Motivação.** `devices: auto` num nó com 8 GPUs sobe DDP com batch efetivo 8×
sem o usuário pedir, invalidando a comparação com o paper e com runs anteriores.
Regra: paralelismo é decisão explícita.

**Arquivo:** `configs/trainer/default.yaml`.

**Especificação:**
```yaml
accelerator: auto
devices: 1                   # explícito; multi-GPU é opt-in (ver comentário)
strategy: auto               # com devices>1 use: trainer.strategy=ddp
# Multi-GPU: batch efetivo = devices * data.batch_size. Ao escalar devices,
# escale o LR linearmente (ex.: 4 GPUs -> optimizer.lr=1.2e-4) ou reduza
# data.batch_size para manter o batch efetivo de referência (48).
```
Comando documentado no README (seção Training):
```bash
./scripts/train.sh data=ntire trainer.devices=4 trainer.strategy=ddp optimizer.lr=1.2e-4
```

**Aceite:** `python train.py --cfg job | grep devices` mostra `1`; um treino
default no DGX usa exatamente 1 GPU (verificar `nvidia-smi`).

### 0.4 Predictor em batch + script de submissão NTIRE (D4)

**Motivação.** A submissão do challenge exige `submission.csv` com colunas
`image_name` (stem, sem extensão) e `pred` (P(fake) ∈ [0,1]) sobre as 10k
imagens de `val_images/` (e `val_images_hard/`). Inferência imagem a imagem com
overhead Python por chamada não escala.

**Arquivos:** `src/inference/predictor.py`, novo
`scripts/make_ntire_submission.py`.

**Especificação — predictor em batch:**

1. Em `predictor.py`, adicionar dataset interno mínimo:
   ```python
   class _ImageFolderDataset(torch.utils.data.Dataset):
       def __init__(self, paths, transform, cropper=None):
           self.paths, self.transform, self.cropper = paths, transform, cropper
       def __len__(self): return len(self.paths)
       def __getitem__(self, i):
           img = Image.open(self.paths[i]).convert("RGB")
           if self.cropper is not None:
               img = self.cropper(img)
           return self.transform(img), i
   ```
2. Reescrever `predict_folder` com a assinatura
   `predict_folder(self, folder: str, batch_size: int = 64, num_workers: int = 8) -> list[Prediction]`:
   glob recursivo idêntico ao atual → `DataLoader(_ImageFolderDataset(...),
   batch_size=batch_size, num_workers=num_workers, pin_memory=True)` → loop
   `@torch.no_grad()` movendo o batch para `self.device`, um único forward por
   batch (`self.module.model(x, apply_fsm=False)`), montando `Prediction` por
   índice. A ordem de saída segue `sorted(files)` como hoje.
3. `predict_image` permanece como está (caminho single-image).

**Especificação — script de submissão** (`scripts/make_ntire_submission.py`):
```
Args (argparse):
  --ckpt        (obrigatório) caminho do .ckpt
  --images-dir  default: data/NTIRE-RobustAIGenDetection-val/val_images
  --out         default: submission.csv
  --batch-size  default: 128
  --num-workers default: 8
  --device      default: None (auto)
```
Corpo: instancia `OSDFDPredictor(ckpt_path=..., device=...)` (image_size vem do
ckpt — item 1.7), roda `predict_folder(images_dir, batch_size, num_workers)`,
valida que todos os arquivos são `.jpg`, e escreve:
```python
df = pd.DataFrame({
    "image_name": [Path(r.path).stem for r in results],
    "pred": [r.probability for r in results],
})
df.to_csv(args.out)   # index=True, igual ao snippet oficial dos organizadores
print(f"{len(df)} predictions -> {args.out}")
```
Para o conjunto hard: `--images-dir data/NTIRE-RobustAIGenDetection-val/val_images_hard --out submission_hard.csv`.

**Aceite:**
- Teste novo `tests/test_smoke.py::test_predict_folder_batched`: com o modelo
  tiny offline (`pretrained=False`) e 5 imagens sintéticas em tmpdir, as
  probabilidades de `predict_folder(batch_size=2)` igualam as de
  `predict_image` uma a uma (`atol=1e-5`).
- `python scripts/make_ntire_submission.py --ckpt <ckpt> --images-dir <dir com 5 jpgs>`
  gera CSV com 5 linhas, colunas `image_name` (sem extensão) e `pred` ∈ [0,1].

### 0.5 `test.py` em passada única (D5)

**Motivação.** Hoje o test set é percorrido duas vezes (uma por `trainer.test`,
outra por `trainer.predict`) — em NTIRE são ~14k imagens × 2.

**Arquivo:** `test.py`.

**Especificação:** remover a chamada `trainer.test(...)`. Manter apenas
`trainer.predict(...)` e, a partir do dataframe montado (que já tem `prob` e
`label`), computar `compute_metrics(y_true, probs)` (já feito) e adicionar a
geração de figuras que era responsabilidade do `test_step`:
```python
from src.training.metrics import log_figures
for logger in trainer.loggers:
    log_figures(logger, y_true, probs, step=0, prefix="test")
```
Também logar as métricas nos loggers (`logger.log_metrics({f"test/{k}": v ...}, step=0)`)
para manter paridade com o comportamento anterior no W&B/TB. Nada muda em
`src/lightning/module.py` (o `test_step` continua existindo para `trainer.fit`
→ `trainer.test(ckpt_path="best")` do `train.py`).

**Aceite:** `python test.py ckpt_path=<ckpt> data=ntire` produz o mesmo
`predictions.csv` e as mesmas métricas de antes (tolerância numérica), em ~metade
do tempo de eval; figuras `test/confusion_matrix|roc_curve|pr_curve` aparecem no
logger.

---

## Fase 1 — Fidelidade, métricas e eficiência

### 1.1 SCL: margem escalável por `sqrt(D)` (D6)

**Motivação.** Com features 256-d não normalizadas, `dist_r`/`dist_f` têm escala
≫ 0.01; a margem atual é numericamente irrelevante no hinge. A formulação
original do Single-Center Loss define a margem como `m·sqrt(D)` justamente para
ser invariante à dimensionalidade. Manter o default paper-faithful (OSDFD diz
0.01 absoluto) e expor a variante correta como opção de experimento.

**Arquivos:** `src/losses/single_center_loss.py`, `src/losses/combined.py`,
`configs/loss/bce_scl.yaml`.

**Especificação:**
1. `SingleCenterLoss.__init__(margin=0.01, margin_scale: str = "none")` com
   valores `"none" | "sqrt_dim"` (validar). Em `forward`:
   ```python
   margin = self.margin
   if self.margin_scale == "sqrt_dim":
       margin = self.margin * math.sqrt(features.size(1))
   hinge = torch.clamp(dist_r - dist_f + margin, min=0.0)
   ```
2. `OSDFDLoss.__init__` ganha `scl_margin_scale: str = "none"` e repassa.
3. `build_model`/`OSDFDLightningModule.__init__`: repassar
   `cfg.loss.get("scl_margin_scale", "none")`.
4. `configs/loss/bce_scl.yaml`:
   ```yaml
   scl_margin_scale: none    # "none" (paper OSDFD) | "sqrt_dim" (SCL original; use scl_margin~0.3)
   ```
5. Experimento recomendado (não default):
   `loss.scl_margin_scale=sqrt_dim loss.scl_margin=0.3`.

**Aceite:** teste unitário com features controladas: para `margin_scale="sqrt_dim"`,
`hinge` usa `0.3*16=4.8` quando `D=256`... (usar D=16 no teste: margem efetiva
`0.3*4=1.2`); para `"none"`, valor atual inalterado (regressão bit-a-bit nos
testes existentes).

### 1.2 Métricas de validação DDP-corretas via torchmetrics (D8)

**Motivação.** O `all_gather` atual inclui amostras de padding duplicadas pelo
DistributedSampler. `torchmetrics` (já em `requirements.txt`) sincroniza estados
corretamente entre ranks.

**Arquivo:** `src/lightning/module.py`.

**Especificação:**
1. No `__init__`:
   ```python
   from torchmetrics.classification import (
       BinaryAUROC, BinaryAveragePrecision, BinaryAccuracy, BinaryF1Score)
   self.val_auc = BinaryAUROC()
   self.val_ap = BinaryAveragePrecision()
   self.val_acc = BinaryAccuracy()   # threshold 0.5
   self.val_f1 = BinaryF1Score()
   ```
2. `validation_step`: além do fluxo atual, `probs = torch.sigmoid(out.logits)`;
   `self.val_auc.update(probs, batch["label"])` (idem para as demais).
3. `on_validation_epoch_end`: logar
   `val/auc`, `val/ap`, `val/acc`, `val/f1` a partir dos objetos torchmetrics
   (`self.log("val/auc", self.val_auc, prog_bar=True)` — Lightning faz
   compute+reset). **Remover** essas quatro chaves do dicionário vindo de
   `compute_metrics` no caminho `val` para não haver duplicidade; o caminho
   numpy continua responsável apenas por `val/eer`, `val/fpr`, `val/fnr` e pelo
   threshold de calibração (item 1.3), mantendo o `all_gather` atual para isso
   (viés de padding nessas métricas secundárias é aceitável e documentado em
   comentário).
4. O caminho `test` (single-GPU no `test.py`) permanece 100% numpy/sklearn como
   hoje.

**Aceite:** run single-GPU: `val/auc` via torchmetrics difere do valor sklearn
anterior por < 1e-6 (verificar num smoke run comparando com
`compute_metrics`); o checkpoint continua sendo selecionado por `val/auc`.

### 1.3 Calibração de threshold por EER da validação (D13)

**Motivação.** 0.5 raramente é o ponto de operação ótimo; o threshold de EER da
validação é o ponto padrão em face forensics e viaja junto com o checkpoint.

**Arquivos:** `src/training/metrics.py`, `src/lightning/module.py`,
`src/inference/predictor.py`, `predict.py`.

**Especificação:**
1. `compute_metrics` passa a incluir `metrics["eer_threshold"]` (já calculado por
   `equal_error_rate`, hoje descartado).
2. `OSDFDLightningModule.__init__`: `self.calibrated_threshold: float = 0.5`.
   Em `_finalise_eval`, quando `prefix == "val"` e `"eer_threshold" in metrics`
   e for finito: `self.calibrated_threshold = float(metrics["eer_threshold"])`.
3. Persistência:
   ```python
   def on_save_checkpoint(self, checkpoint: dict) -> None:
       checkpoint["calibrated_threshold"] = self.calibrated_threshold
   def on_load_checkpoint(self, checkpoint: dict) -> None:
       self.calibrated_threshold = checkpoint.get("calibrated_threshold", 0.5)
   ```
4. `predict_step`: `pred = (probs >= self.calibrated_threshold).long()...`.
5. `OSDFDPredictor`: ler `self.threshold = getattr(self.module, "calibrated_threshold", 0.5)`
   e usar em `label = "fake" if prob >= self.threshold else "real"`. `predict.py`
   ganha `--threshold` (float, default `None` = usar o calibrado do ckpt).
6. A probabilidade contínua (`prob`) não muda em nenhum fluxo — a submissão
   NTIRE (item 0.4) usa `prob` e é insensível a threshold.

**Aceite:** treinar 30 steps com val; inspecionar
`torch.load(ckpt)["calibrated_threshold"]` ≠ 0.5 (dado val não-trivial);
`predict.py` sem `--threshold` usa o valor do ckpt (imprimir no stdout:
`threshold=0.xxxx (calibrated)`).

### 1.4 Precisão numérica e velocidade em B200 (D9)

**Arquivos:** `configs/trainer/default.yaml`, `configs/config.yaml`,
`src/utils/lightning_setup.py`, `train.py`, `test.py`.

**Especificação:**
1. `configs/trainer/default.yaml`: `precision: bf16-mixed` e novo flag
   `benchmark: true`; `build_trainer` repassa `benchmark=tc.benchmark`.
2. `configs/config.yaml`: `deterministic: warn` (Lightning aceita a string;
   mantém reprodutibilidade onde não custa performance e só avisa onde não há
   kernel determinístico). `build_trainer` já repassa o valor como está.
3. `train.py` e `test.py`, primeira linha de `main()` após o parse do config:
   ```python
   torch.set_float32_matmul_precision("high")   # TF32 em matmuls fp32
   ```
   (import `torch` no topo).
4. Documentar no README que para reproduzir bit-a-bit usa-se
   `trainer.precision=32-true deterministic=true trainer.benchmark=false`.

**Aceite:** smoke train de 20 steps sem warnings de matmul precision; velocidade
steps/s ≥ à do fp16 anterior (comparar no log `train/epoch_time_s` de um run
curto); nenhum GradScaler no log (bf16 não usa).

### 1.5 Carregar checkpoint sem depender do HuggingFace Hub (D7)

**Arquivos:** `src/lightning/module.py` (`build_model`),
`configs/backbone/siglip2_base.yaml`, `configs/backbone/siglip2_large.yaml`,
`test.py`, `src/inference/predictor.py`.

**Especificação:**
1. `build_model`: passar `pretrained=cfg.backbone.get("pretrained", True)` ao
   `OSDFDModel` (o parâmetro já existe e já é plumbing completo até o
   `Siglip2Backbone`). Adicionar `pretrained: true` aos dois YAMLs de backbone.
   **Atenção:** com `pretrained=False` o `Siglip2Backbone` atual constrói de
   `config_overrides`; para manter a arquitetura real é preciso construir da
   config do checkpoint HF **sem baixar pesos**. Ajustar o ramo em
   `Siglip2Backbone.__init__`:
   ```python
   else:
       from transformers import AutoConfig, SiglipVisionModel
       if config_overrides:
           cfg = SiglipVisionConfig(**config_overrides)
       else:
           cfg = AutoConfig.from_pretrained(model_name).vision_config
       self.vision_model = SiglipVisionModel(cfg).vision_model
   ```
   (`AutoConfig.from_pretrained` usa o cache local de config — poucos KB, já
   presente após o primeiro treino; não baixa pesos.)
2. Helper novo em `src/lightning/module.py`:
   ```python
   @classmethod
   def load_for_inference(cls, ckpt_path: str, map_location=None) -> "OSDFDLightningModule":
       ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
       cfg = OmegaConf.create(ckpt["hyper_parameters"])
       with open_dict(cfg):
           cfg.backbone.pretrained = False   # pesos vêm do state_dict do ckpt
       return cls.load_from_checkpoint(ckpt_path, map_location=map_location, cfg=cfg)
   ```
3. `test.py` e `OSDFDPredictor.__init__` trocam
   `OSDFDLightningModule.load_from_checkpoint(...)` por
   `OSDFDLightningModule.load_for_inference(...)`.

**Aceite:** com `HF_HUB_OFFLINE=1` e cache de pesos renomeado (somente configs
presentes), `predict.py --ckpt <ckpt> --input <img>` funciona; as probabilidades
são idênticas às do caminho antigo (mesmo ckpt, mesma imagem, `atol=1e-6`).

### 1.6 FSM: `_domain_shuffle` vetorizado (D10)

**Arquivo:** `src/models/fsm.py`.

**Especificação:** substituir o loop do caminho multi-domínio por:
```python
diff = domains.unsqueeze(0) != domains.unsqueeze(1)   # (F, F); [i, j] = domínios diferem
no_cand = ~diff.any(dim=1)                            # linhas sem par válido
weights = diff.float()
idx = torch.arange(f, device=domains.device)
weights[no_cand, idx[no_cand]] = 1.0                  # degenera para identidade nessas linhas
perm = torch.multinomial(weights, 1).squeeze(1)
```
Semântica idêntica à atual (amostragem uniforme entre candidatos de outro
domínio; identidade quando não há candidato). O guard de `<2` domínios (com o
fallback do item 0.1) fica antes deste bloco, inalterado.

**Aceite:** teste de propriedade: para 200 sorteios com
`domains=[1,1,2,2,3]`, todo `perm[i]` satisfaz `domains[perm[i]] != domains[i]`;
distribuição aproximadamente uniforme entre candidatos (qui-quadrado informal
não requerido — basta a garantia de domínio distinto e ausência do loop).

### 1.7 `image_size` e transform derivados do checkpoint (D12)

**Arquivos:** `src/inference/predictor.py`, `predict.py`.

**Especificação:** em `OSDFDPredictor.__init__`, `image_size: int | None = None`;
quando `None`, resolver do ckpt: `int(self.module.cfg.backbone.image_size)`
(fallback `224` se ausente, com `print` de aviso). `predict.py --image-size`
passa a default `None`. O `scripts/make_ntire_submission.py` (item 0.4) não
expõe `--image-size` — usa sempre o do ckpt.

**Aceite:** ckpt tiny com `backbone.image_size=256` nos hparams → predictor
constrói transform 256 sem flag; com flag explícita, a flag vence.

### 1.8 Dataloader NTIRE: decode JPEG rápido e tuning (D14)

**Arquivos:** `src/data/dataset.py`, `src/data/datamodule.py`,
`configs/data/ntire.yaml`.

**Especificação:**
1. `ForgeryFrameDataset.__init__(..., jpeg_draft_size: int | None = None)`.
   Em `__getitem__`:
   ```python
   image = Image.open(rec.path)
   if self.jpeg_draft_size and image.format == "JPEG":
       # Decodifica no maior fator DCT que ainda cobre o alvo (2-8x mais rápido).
       image.draft("RGB", (self.jpeg_draft_size, self.jpeg_draft_size))
   image = image.convert("RGB")
   ```
2. `ForgeryDataModule.__init__(..., jpeg_draft_size: int | None = None)`;
   repassar na construção dos três datasets.
3. `configs/data/ntire.yaml`: `jpeg_draft_size: 448` (2× o input 224 — o
   `draft` só reduz em fatores 1/2,1/4,1/8 e nunca abaixo do pedido, então o
   resize bicubic final sempre parte de ≥448px; perda de qualidade irrelevante
   para um alvo 224). `configs/data/faceforensics.yaml`: adicionar
   `jpeg_draft_size: null` (PNGs; no-op explícito).
4. `configs/data/ntire.yaml`: subir `num_workers: 16` (o DGX tem cores de
   sobra; 8 é conservador para decode JPEG full-res).

**Aceite:** benchmark manual de 200 batches com e sem draft
(`python -m timeit`-style script descartável, não commitado): ganho esperado
≥1.5× no throughput do loader com imagens NTIRE; probabilidades de um modelo
treinado variam de forma desprezível (draft muda pixels sub-visualmente em 448→224).

---

## Fase 2 — Ablations e extensões (opt-in, ordem sugerida)

Cada item desta fase é um experimento com config própria; nenhum vira default
sem um run comprovando ganho de `val/auc` (in-domain) **e** nas submissões
val/val_hard do challenge.

### 2.1 Pseudo-domínios de gerador para o FSM no NTIRE (extensão do 0.1)

**Motivação.** O fallback aleatório (0.1) mistura estilos às cegas. Se
estimarmos o gerador por clustering, o FSM volta a fazer pareamento
*informado* entre domínios distintos, como no FF++.

**Arquivo novo:** `scripts/assign_pseudo_domains.py`.

**Especificação:**
```
Args: --manifest data/ntire_manifest.csv
      --out data/ntire_manifest_k8.csv
      --k 8
      --backbone google/siglip2-base-patch16-224
      --batch-size 256 --num-workers 8 --device cuda --seed 0
```
1. Carrega o manifesto; seleciona linhas `label==1` (todos os splits — domínio
   é usado só em treino, mas manter consistência).
2. Extrai o embedding MAP-pooled do backbone **congelado e sem PEFT**
   (`Siglip2Backbone(model_name, freeze=True)`, `pool(forward(x))`), em
   `torch.no_grad()`, bf16, transform de val (`build_transform(train=False)`).
3. `sklearn.cluster.MiniBatchKMeans(n_clusters=k, random_state=seed)` sobre os
   embeddings L2-normalizados.
4. Escreve o manifesto de saída com `domain = 1 + cluster_id` para fakes
   (1..k) e `domain = 0` para reais. Imprime o tamanho de cada cluster.
5. Treino: `./scripts/train.sh data=ntire data.manifest=data/ntire_manifest_k8.csv
   fsm.single_domain_fallback=off` (com k domínios o fallback não é acionado;
   `off` garante que qualquer regressão de domínio único falhe visível via
   `train/fsm_fired`=0 em vez de mascarar).

**Aceite:** manifesto de saída tem `domain` ∈ {0..k}; nenhum real com domínio
≠0; run de treino mostra `train/fsm_fired` ≈ `fsm.prob`. Comparar `val/auc` e
submissão vs. run 0.1.

### 2.2 Política de resize configurável (D15)

**Arquivos:** `src/data/transforms.py`, `src/data/datamodule.py`, configs de data.

**Especificação:** `build_transform(..., resize_mode: str = "squash")`, valores:
- `"squash"` (default, comportamento atual): `Resize((s, s))`.
- `"crop"`: train → `RandomResizedCrop(s, scale=rrc_scale)` mesmo sem
  augmentation habilitada; val/test → `Resize(s)` (lado menor) + `CenterCrop(s)`.
Plumbing: `data.resize_mode` nos YAMLs (faceforensics: `squash`; ntire:
experimento com `crop`). Quando `augmentation.random_resized_crop=true` e
`resize_mode="crop"`, não duplicar o RRC (o RRC da augmentation vence no train).

**Aceite:** teste de shape para os dois modos (entrada 640×480 → tensor
3×224×224); run comparativo NTIRE `squash` vs `crop`.

### 2.3 `WeightedRandomSampler` em vez de duplicação de registros (D16)

**Arquivos:** `src/data/datamodule.py`, configs de data.

**Especificação:** novo hparam `balance_sampler: bool = false`. Quando `true`:
- `setup`: **não** aplicar `oversample_real`; erro claro se
  `real_oversample > 1` e `balance_sampler=true` simultaneamente.
- `train_dataloader`: `WeightedRandomSampler(weights=1/contagem_da_classe[label_i],
  num_samples=len(dataset), replacement=True)`; `shuffle` omitido (mutuamente
  exclusivo com sampler); manter `drop_last=True`.
**Aceite:** teste: proporção real/fake por batch ≈ 50/50 (±5 p.p.) em 50
batches sintéticos; época tem `len(dataset)` amostras (não inflada).

### 2.4 MAP pooling head treinável (D16)

**Arquivos:** `src/models/peft_inject.py` (`mark_trainable`),
`configs/model/default.yaml`, `src/models/osdfd.py`, `src/lightning/module.py`.

**Especificação:** flag `model.train_pool_head: false`. Quando `true`,
`mark_trainable` ganha parâmetro `train_pool_head: bool` e reativa
`requires_grad` em `backbone.vision_model.head.parameters()`. Plumbing igual ao
`train_norm`. Reportar o novo total de parâmetros treináveis no log existente
de `on_fit_start`.
**Aceite:** com a flag, `num_trainable_parameters()` cresce exatamente pelo
tamanho da MAP head; sem a flag, inalterado.

### 2.5 EMA dos parâmetros treináveis

**Arquivo novo:** `src/utils/ema.py` (callback Lightning).

**Especificação:** callback `EMACallback(decay: float = 0.999)`:
- `on_train_batch_end`: `shadow = decay*shadow + (1-decay)*param` apenas para
  parâmetros com `requires_grad=True` (≈2.3M — custo desprezível).
- `on_validation_start`/`on_validation_end` (e test/predict): swap in/out dos
  pesos EMA (guardar originais em buffer CPU).
- `state_dict`/`load_state_dict` para resume.
Config: `callbacks.ema.enabled: false`, `callbacks.ema.decay: 0.999`;
`build_callbacks` instancia quando enabled.
**Aceite:** teste unitário do decay em 3 steps sintéticos; run comparativo com
`enabled=true` valida via `val/auc`.

### 2.6 `torch.compile` opcional

**Arquivos:** `train.py`, `configs/trainer/default.yaml`.

**Especificação:** `trainer.compile: false`; em `train.py`, após construir o
módulo: `if cfg.trainer.compile: model.model = torch.compile(model.model)`.
Nota em comentário: FSM introduz shapes/branches dinâmicos → esperar
recompilações; medir antes de adotar. Não compilar em `test.py`/predictor.
**Aceite:** smoke run com flag ligada completa 20 steps; se steps/s não
melhorar ≥15% num run de 500 steps, manter default off e registrar no README.

### 2.7 Matriz de experimentos de referência (NTIRE)

Ordem de execução e comparação (todas com seed 0, 30k steps, 1×B200, batch 48):

| Run | Config | Compara com |
|-----|--------|-------------|
| R0 | baseline atual (pós-Fase 0.2/0.3, FSM fallback **off**, sem aug) | — (controle histórico) |
| R1 | + augmentation robusta (0.2) | R0 |
| R2 | + FSM fallback random (0.1) | R1 |
| R3 | + SCL sqrt_dim, margin 0.3 (1.1) | R2 |
| R4 | + pseudo-domínios k=8 (2.1) | R2 |
| R5 | + resize_mode crop (2.2) | melhor de R2–R4 |

Métricas de decisão, nesta ordem: (1) `val/auc` interno; (2) score da submissão
em `val_images`; (3) score em `val_images_hard`. Registrar tudo no W&B com o
nome do run = `ntire-R<k>`.

---

## Testes e verificação global

Novos testes em `tests/test_smoke.py` (todos offline, modelo tiny
`pretrained=False`, CPU):
1. `test_fsm_single_domain_fallback` (item 0.1).
2. `test_fsm_multidomain_vectorized_pairs` (item 1.6).
3. `test_scl_margin_scale` (item 1.1).
4. `test_predict_folder_batched` (item 0.4).
5. `test_manifest_ignores_extra_columns`: manifesto com colunas extras
   (`distortions`, ...) carrega sem erro (guarda contra CSVs futuros do NTIRE).
6. `test_calibrated_threshold_roundtrip` (item 1.3): salvar/carregar ckpt
   preserva o threshold.

Gate de cada fase antes de seguir para a próxima:
```bash
pytest tests/ -q                                            # tudo verde
./scripts/train.sh data=ntire trainer.max_steps=20 \
    trainer.val_check_interval=10 logger.wandb.enabled=false  # smoke NTIRE
./scripts/train.sh trainer.max_steps=20 \
    trainer.val_check_interval=10 logger.wandb.enabled=false  # smoke FF++ (sem regressão)
```

## Fora de escopo desta versão (registrado para v0.2)

- Distillação/ensemble entre backbones (siglip2_base + large).
- Test-time augmentation na submissão (média de flips/crops) — ganho provável,
  mas medir custo primeiro.
- Avaliação de robustez local sintética (aplicar o pipeline de distorções na
  val interna para prever o score do `val_images_hard`).
- Treino conjunto FF++ + NTIRE (multi-dataset, domínios unificados 0..k).

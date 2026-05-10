# Few-Shot Gaze - Docker Setup Guide

Este guia explica como rodar a demo de eye tracking em Docker localmente, com suporte para GPU (NVIDIA) e CPU fallback.

## Pré-requisitos

- Docker instalado
- NVIDIA Docker Runtime (para suporte a GPU)
- Câmera USB conectada
- X Server rodando (para visualização)
- Arquivo `demo_weights.zip` baixado (veja seção de Setup)

## Setup Inicial

### 1. Download dos pesos pré-treinados

```bash
cd demo
wget https://files.ait.ethz.ch/projects/faze/demo_weights.zip
unzip demo_weights.zip
cd ..
```

### 2. Build da imagem Docker

```bash
docker build -t few-shot-gaze-demo .
```

## Rodando a Demo

### Opção 1: Com GPU NVIDIA (Recomendado)

Se sua GPU for compatível com a versão do CUDA compilada:

```bash
sudo docker run --gpus all -it --rm \
    --device /dev/video1 \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/work \
    -w /work/demo \
    few-shot-gaze-demo \
    python run_demo.py
```

### Opção 2: Com CPU (CPU Fallback - recomendado para GPUs Blackwell como RTX5070)

Se encontrar erro CUDA `no kernel image is available for execution on the device`:

```bash
sudo docker run --gpus all -it --rm \
    --device /dev/video1 \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e FORCE_CPU=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/work \
    -w /work/demo \
    few-shot-gaze-demo \
    python run_demo.py
```

**Nota:** CPU é ~10-20x mais lento que GPU, mas funciona em qualquer sistema.

## Variáveis de Ambiente

### FORCE_CPU

Força execução em CPU em vez de GPU:
```bash
-e FORCE_CPU=1
```

### CAMERA_INDEX

Define qual câmera usar (padrão: 1):
```bash
-e CAMERA_INDEX=1
```

## Flags do Docker Explicadas

| Flag | Descrição |
|------|-----------|
| `--gpus all` | Ativa acesso a todas as GPUs NVIDIA |
| `--device /dev/video1` | Monta o device da câmera no container |
| `-e DISPLAY=$DISPLAY` | Passa a variável X11 DISPLAY |
| `-e QT_X11_NO_MITSHM=1` | Evita problemas de memória compartilhada com X11 |
| `-v /tmp/.X11-unix:/tmp/.X11-unix` | Monta socket X11 para visualização |
| `-v "$PWD":/work` | Monta diretório atual como /work no container |
| `-w /work/demo` | Define diretório de trabalho para /work/demo |

## Workflow da Demo

### 1. Calibração da Câmera (primeira vez apenas)

```bash
sudo docker run --gpus all -it --rm \
    --device /dev/video1 \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e FORCE_CPU=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$PWD":/work \
    -w /work/demo \
    few-shot-gaze-demo \
    python calibrate_camera.py
```

Isso gera `calib_cam1.pkl` (guardado no seu diretório local).

### 2. Rodar a Demo

A demo coleta dados de calibração pessoal e fine-tuna a rede no seu gaze:

1. Execute o comando acima (Opção 1 ou 2)
2. Aparecerá uma janela pedindo seu nome (ex: "diego")
3. Uma tela branca mostrará 13 pontos de calibração (3x3 grid + 4 aleatórios)
4. Olhe para cada alvo e pressione a seta correspondente (↑↓←→)
5. A rede será fine-tuned em ~1 minuto
6. A bolinha vermelha seguirá seu olhar na tela

### 3. Parar a Demo

Pressione `q` na janela de visualização para sair.

## Troubleshooting

### Erro: "Failed to initialize module context"

X11 não está configurado. Verifique:
```bash
echo $DISPLAY
# Deve retorgar algo como :0 ou :1
```

Se vazio, inicie o X Server.

### Erro: "CUDA error: no kernel image is available"

Sua GPU requer compilação diferente. Use:
```bash
-e FORCE_CPU=1
```

### Câmera não detectada

Verifique qual câmera está disponível:
```bash
ls -la /dev/video*
```

Ajuste `--device /dev/video1` para a câmera correta.

### Visualização preta ou sem saída

Verifique se X11 está forwarding corretamente:
```bash
xhost +local:
```

Antes de rodar o container.

## Performance

| Modo | Velocidade | Compatibilidade |
|------|-----------|-----------------|
| GPU (NVIDIA) | 30-60 FPS | Apenas arquiteturas pré-compiladas |
| CPU | 3-10 FPS | Universal, compatível com RTX5070 |

## Melhorando a Precisão

1. **Calibração da câmera:** Execute `calibrate_camera.py` com um padrão de xadrez impresso
2. **Posicionamento:** Coloque a câmera no centro horizontal do monitor, alinhada com os olhos
3. **Óculos:** Use óculos o tempo todo (não mude entre calibração e uso)
4. **Iluminação:** Ambiente bem iluminado reduz erros
5. **Mais pontos de calibração:** Já coleta 13 pontos (bom padrão)

## Salvando Resultados

Os arquivos gerados ficam no seu diretório local:
- `diego_calib.avi` - vídeo de calibração
- `diego_calib_target.pkl` - alvo de calibração
- `diego_gaze_network.pth.tar` - modelo fine-tuned

## Referências

- [Documentação Original](./demo/README.md)
- [Paper FAZE](https://files.ait.ethz.ch/projects/faze/)
- [GitHub Few-Shot Gaze](https://github.com/NVlabs/few_shot_gaze)

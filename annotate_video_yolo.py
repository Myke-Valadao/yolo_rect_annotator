#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Rect Annotator
-------------------
- Desenho de múltiplos BBOX (retângulos) com o mouse.
- Após soltar o mouse, você digita no terminal o ID numérico da classe daquele BBOX.
- Mostra o ID da classe no topo central do bbox (com fundo preto).
- Salva frames/imagens (.jpg) e labels no formato YOLO (.txt) com coordenadas NORMALIZADAS.
- A janela exibe a imagem no tamanho original, reduzida apenas se for maior que a tela
  (os arquivos salvos sempre mantêm a resolução original).
- Funciona com um VÍDEO ou com IMAGENS (uma pasta, um padrão glob ou um único arquivo).

Teclas:
    q = sair
    n = próximo frame/imagem (descarta BBOX não salvos)
    s = salvar BBOX do frame/imagem atual (gera .jpg e .txt)
    u = desfazer último BBOX do frame/imagem
    r = remover TODOS os BBOX do frame/imagem atual

Uso (vídeo):
    python annotate_video_yolo.py \
        --video /caminho/para/video.mp4 \
        --outdir dataset \
        --img-prefix myke \
        --skip 0 \
        --start 0

Uso (pasta de imagens):
    python annotate_video_yolo.py \
        --images /caminho/para/pasta_de_imagens \
        --outdir dataset \
        --img-prefix myke

Uso (padrão glob / imagem única):
    python annotate_video_yolo.py --images "/caminho/para/fotos/*.jpg" --outdir dataset
    python annotate_video_yolo.py --images /caminho/para/foto.png --outdir dataset
"""

import argparse
import glob
import itertools
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import cv2
import numpy as np

# ==============================
# Globais controladas pela UI
# ==============================
ref_point: List[Tuple[int, int]] = []
drawing: bool = False
image = None
image_copy = None

# Cada item: {"pt1": (x1,y1), "pt2": (x2,y2), "cls": int}
bboxes: List[Dict] = []

# Sinaliza que um bbox acabou de ser desenhado e precisamos pedir a classe no loop principal
awaiting_class_input: bool = False
pending_bbox: Dict = {}

# Fator de escala entre a imagem original (usada para salvar) e o que é exibido na janela
# (imagens maiores que a tela são reduzidas apenas para exibição; coordenadas de mouse são
# convertidas de volta para a resolução original antes de guardar o bbox).
display_scale: float = 1.0

WINDOW_NAME = "Frame"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clip_rect(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(x1, w - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    return x1, y1, x2, y2


def to_yolo(x1, y1, x2, y2, img_w, img_h):
    # garante ordem
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    # converte p/ centro-largura-altura (normalizado)
    xc = ((x1 + x2) / 2.0) / img_w
    yc = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def put_label(canvas, text, x1, y1, x2, y2):
    """Escreve o rótulo (texto) no topo central do bbox, com fundo para contraste."""
    tx = int((x1 + x2) / 2)
    ty = max(0, y1 - 8)
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # caixa de fundo (preta)
    cv2.rectangle(canvas,
                  (tx - tw // 2 - 4, ty - th - 4),
                  (tx + tw // 2 + 4, ty + 2),
                  (0, 0, 0), -1)
    # texto (amarelo)
    cv2.putText(canvas, text, (tx - tw // 2, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)


def draw_all_boxes(canvas, boxes, color=(0, 255, 0)):
    """Desenha todos os bboxes com suas classes no topo central."""
    for item in boxes:
        (x1, y1) = item["pt1"]
        (x2, y2) = item["pt2"]
        cls = item["cls"]
        # retângulo
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        # rótulo com a classe
        put_label(canvas, f"{cls}", x1, y1, x2, y2)


def draw_pending_box(canvas, pending):
    """Desenha o bbox recém-desenhado (ainda sem classe), em cor distinta."""
    if not pending:
        return
    (x1, y1) = pending["pt1"]
    (x2, y2) = pending["pt2"]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
    put_label(canvas, "id?", x1, y1, x2, y2)


def get_screen_size(default: Tuple[int, int] = (1600, 900)) -> Tuple[int, int]:
    """Tenta detectar a resolução da tela; usa um valor padrão se não conseguir (ex.: ambiente sem GUI)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return default


def compute_display_scale(w: int, h: int, max_w: int, max_h: int) -> float:
    """Fator (<=1.0) para reduzir a imagem à tela, sem nunca ampliar."""
    return min(1.0, max_w / w, max_h / h)


def show_frame(canvas):
    """Exibe o canvas (em resolução original) já reduzido pelo display_scale, se necessário."""
    if display_scale < 1.0:
        disp_w = max(1, int(round(canvas.shape[1] * display_scale)))
        disp_h = max(1, int(round(canvas.shape[0] * display_scale)))
        canvas = cv2.resize(canvas, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, canvas)


def to_original_coords(x: int, y: int) -> Tuple[int, int]:
    """Converte coordenadas recebidas da janela (possivelmente reduzida) para a imagem original."""
    if display_scale < 1.0:
        return int(round(x / display_scale)), int(round(y / display_scale))
    return x, y


def bring_window_to_front():
    """Traz a janela para frente/foco (útil após ler input() no terminal, que rouba o foco)."""
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 0)
        cv2.waitKey(1)
    except Exception:
        pass


def mouse_cb(event, x, y, flags, param):
    global ref_point, drawing, image, image_copy, awaiting_class_input, pending_bbox

    x, y = to_original_coords(x, y)

    if event == cv2.EVENT_LBUTTONDOWN and not awaiting_class_input:
        ref_point = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing and not awaiting_class_input:
        image_copy[:] = image  # restaura base
        # desenha os já existentes
        draw_all_boxes(image_copy, bboxes)
        cv2.rectangle(image_copy, ref_point[0], (x, y), (0, 255, 0), 2)
        show_frame(image_copy)

    elif event == cv2.EVENT_LBUTTONUP and drawing and not awaiting_class_input:
        ref_point.append((x, y))
        drawing = False

        # armazena bbox pendente; classe será pedida no loop principal
        (x1, y1) = ref_point[0]
        (x2, y2) = ref_point[1]
        pending_bbox = {"pt1": (x1, y1), "pt2": (x2, y2)}
        awaiting_class_input = True

        # mostra visualmente o bbox pendente (sem classe ainda), em cor distinta
        image_copy[:] = image
        draw_all_boxes(image_copy, bboxes)
        draw_pending_box(image_copy, pending_bbox)
        show_frame(image_copy)


def collect_image_paths(images_arg: str) -> List[Path]:
    """Resolve --images para uma lista ordenada de arquivos de imagem.

    Aceita: uma pasta, um padrão glob (ex.: 'fotos/*.jpg') ou um único arquivo.
    """
    p = Path(images_arg)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
    if p.is_file():
        return [p]
    # trata como padrão glob
    return sorted(
        f for f in (Path(m) for m in glob.glob(images_arg))
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )


def iter_video_source(video_path: str, start: int, skip: int) -> Iterator[Tuple[np.ndarray, str]]:
    """Gera (frame, nome_base) a partir de um vídeo."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Erro ao abrir o vídeo: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    print(f"[INFO] Vídeo aberto. Frames: {total if total > 0 else 'desconhecido'}")

    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_idx = start
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fim do vídeo ou erro ao ler frame.")
                break
            if skip > 0 and (frame_idx - start) % (skip + 1) != 0:
                frame_idx += 1
                continue
            yield frame, str(frame_idx)
            frame_idx += 1
    finally:
        cap.release()


def iter_image_source(image_paths: List[Path], start: int, skip: int) -> Iterator[Tuple[np.ndarray, str]]:
    """Gera (imagem, nome_base) a partir de uma lista de arquivos de imagem."""
    for i, path in enumerate(image_paths[start:]):
        if skip > 0 and i % (skip + 1) != 0:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print(f"[AVISO] Não foi possível ler a imagem: {path}")
            continue
        yield img, path.stem


def main():
    global image, image_copy, bboxes, awaiting_class_input, pending_bbox, display_scale

    ap = argparse.ArgumentParser(description="Ferramenta simples para anotar BBOX retangulares no formato YOLO.")
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--video", help="Caminho do vídeo de entrada.")
    src_group.add_argument(
        "--images",
        help="Pasta de imagens, padrão glob (ex.: 'fotos/*.jpg') ou um único arquivo de imagem.",
    )
    ap.add_argument("--outdir", default="dataset", help="Diretório raiz de saída (terá subpastas images/ e labels/).")
    ap.add_argument("--img-prefix", default="frame", help="Prefixo do nome de arquivo das imagens/labels.")
    ap.add_argument("--skip", type=int, default=0, help="Pular N frames/imagens entre anotações (0 = anotar todos).")
    ap.add_argument("--start", type=int, default=0, help="Índice inicial (0-based) do frame/imagem.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    img_dir = outdir / "images"
    lbl_dir = outdir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        try:
            source = iter_video_source(args.video, args.start, args.skip)
        except RuntimeError as e:
            print(f"[ERRO] {e}")
            return
    else:
        image_paths = collect_image_paths(args.images)
        if not image_paths:
            print("[ERRO] Nenhuma imagem encontrada em --images.")
            return
        print(f"[INFO] {len(image_paths)} imagem(ns) encontrada(s).")
        source = iter_image_source(image_paths, args.start, args.skip)

    # lê o primeiro item para obter dimensões e configurar a janela
    try:
        first_image, first_stem = next(source)
    except StopIteration:
        print("[ERRO] Nenhum frame/imagem disponível para anotar (verifique --start/--skip).")
        return

    # tamanho máximo de exibição (a imagem original nunca é alterada, só a exibição na tela)
    screen_w, screen_h = get_screen_size()
    max_display_w = max(400, int(screen_w * 0.9))
    max_display_h = max(300, int(screen_h * 0.85))

    # WINDOW_AUTOSIZE: a janela sempre tem o mesmo tamanho da imagem exibida, garantindo que as
    # coordenadas do mouse informadas pelo OpenCV correspondam exatamente ao que foi desenhado.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_cb)
    print("[INFO] Se as teclas (s/n/u/r/q) não responderem, clique uma vez na janela da imagem "
          "para dar foco a ela antes de pressionar a tecla.")

    saved_count = 0
    full_source = itertools.chain([(first_image, first_stem)], source)

    for src_image, stem in full_source:
        image = src_image.copy()
        image_copy = src_image.copy()
        h, w = image.shape[:2]
        display_scale = compute_display_scale(w, h, max_display_w, max_display_h)
        bboxes = []
        awaiting_class_input = False
        pending_bbox = {}

        # traz a janela para frente ao trocar de frame/imagem
        bring_window_to_front()

        while True:
            # exibe a imagem com caixas já salvas + o bbox pendente (se houver), evitando que
            # ele "desapareça" enquanto aguardamos o ID da classe no terminal
            disp = image.copy()
            draw_all_boxes(disp, bboxes)
            draw_pending_box(disp, pending_bbox)
            show_frame(disp)

            # se acabamos de desenhar um bbox, pedir a classe no terminal
            if awaiting_class_input:
                (x1, y1) = pending_bbox["pt1"]
                (x2, y2) = pending_bbox["pt2"]
                x1, y1, x2, y2 = clip_rect(x1, y1, x2, y2, w, h)
                if x1 == x2 or y1 == y2:
                    print("[AVISO] BBOX com área zero ignorado.")
                    awaiting_class_input = False
                    pending_bbox = {}
                else:
                    try:
                        cls_str = input("Digite o ID numérico da classe para este BBOX (ex.: 0, 1, 2...): ").strip()
                        cls_id = int(cls_str)
                        bboxes.append({"pt1": (x1, y1), "pt2": (x2, y2), "cls": cls_id})
                        print(f"[OK] BBOX adicionado com classe {cls_id}. Total no frame/imagem: {len(bboxes)}")
                    except ValueError:
                        print("[ERRO] Valor inválido. BBOX descartado.")
                    awaiting_class_input = False
                    pending_bbox = {}

                    # refresh visual imediato + devolve o foco à janela (input() no terminal o rouba)
                    image_copy[:] = image
                    draw_all_boxes(image_copy, bboxes)
                    show_frame(image_copy)
                    bring_window_to_front()

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Saindo.")
                cv2.destroyAllWindows()
                return

            elif key == ord('u'):
                if bboxes:
                    dropped = bboxes.pop()
                    print(f"[INFO] Removido último BBOX (classe {dropped['cls']}). Restantes: {len(bboxes)}")

            elif key == ord('r'):
                bboxes = []
                print("[INFO] BBOX do frame/imagem atual limpos.")

            elif key == ord('n'):
                # descarta sem salvar
                print("[INFO] Indo para o próximo frame/imagem (sem salvar).")
                break

            elif key == ord('s'):
                # salva imagem e labels
                img_name = f"{args.img_prefix}-{stem}.jpg"
                lbl_name = f"{args.img_prefix}-{stem}.txt"
                img_path = img_dir / img_name
                lbl_path = lbl_dir / lbl_name

                cv2.imwrite(str(img_path), image)

                with open(lbl_path, "w") as f:
                    for item in bboxes:
                        (bx1, by1) = item["pt1"]
                        (bx2, by2) = item["pt2"]
                        bx1, by1, bx2, by2 = clip_rect(bx1, by1, bx2, by2, w, h)
                        xc, yc, ww, hh = to_yolo(bx1, by1, bx2, by2, w, h)
                        f.write(f"{item['cls']} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

                saved_count += 1
                print(f"[SALVO] {img_path.name} + {lbl_path.name}  (bboxes={len(bboxes)})")
                break

    cv2.destroyAllWindows()
    print(f"[DONE] Frames/imagens salvos: {saved_count}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO Rect Annotator
-------------------
- Desenho de múltiplos BBOX (retângulos) com o mouse.
- Após soltar o mouse, você digita no terminal o ID numérico da classe daquele BBOX.
- Mostra o ID da classe no topo central do bbox (com fundo preto).
- Salva frames (.jpg) e labels no formato YOLO (.txt) com coordenadas NORMALIZADAS.
- A janela é criada no TAMANHO ORIGINAL DO VÍDEO.

Teclas:
    q = sair
    n = próximo frame (descarta BBOX não salvos)
    s = salvar BBOX do frame atual (gera .jpg e .txt)
    u = desfazer último BBOX do frame
    r = remover TODOS os BBOX do frame atual

Uso:
    python annotate_video_yolo.py \
        --video /caminho/para/video.mp4 \
        --outdir dataset \
        --img-prefix myke \
        --skip 0 \
        --start 0
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

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


def mouse_cb(event, x, y, flags, param):
    global ref_point, drawing, image, image_copy, awaiting_class_input, pending_bbox

    if event == cv2.EVENT_LBUTTONDOWN and not awaiting_class_input:
        ref_point = [(x, y)]
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing and not awaiting_class_input:
        image_copy[:] = image  # restaura base
        # desenha os já existentes
        draw_all_boxes(image_copy, bboxes)
        cv2.rectangle(image_copy, ref_point[0], (x, y), (0, 255, 0), 2)
        cv2.imshow('Frame', image_copy)

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
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 200, 255), 2)
        put_label(image_copy, "id?", x1, y1, x2, y2)  # dica visual
        cv2.imshow('Frame', image_copy)


def main():
    global image, image_copy, bboxes, awaiting_class_input, pending_bbox

    ap = argparse.ArgumentParser(description="Ferramenta simples para anotar BBOX retangulares no formato YOLO.")
    ap.add_argument("--video", required=True, help="Caminho do vídeo de entrada.")
    ap.add_argument("--outdir", default="dataset", help="Diretório raiz de saída (terá subpastas images/ e labels/).")
    ap.add_argument("--img-prefix", default="frame", help="Prefixo do nome de arquivo das imagens/labels.")
    ap.add_argument("--skip", type=int, default=0, help="Pular N frames entre anotações (0 = anotar todo frame).")
    ap.add_argument("--start", type=int, default=0, help="Frame inicial (0-based).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    img_dir = outdir / "images"
    lbl_dir = outdir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    print(f"[INFO] Vídeo aberto. Frames: {total if total > 0 else 'desconhecido'}")

    # posiciona no frame inicial
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # lê um frame para obter dimensões e configurar a janela no tamanho ORIGINAL do vídeo
    ret, first_frame = cap.read()
    if not ret:
        print("Erro ao ler o primeiro frame.")
        return
    h0, w0 = first_frame.shape[:2]

    # cria janela e ajusta ao tamanho original do vídeo
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', w0, h0)
    cv2.setMouseCallback('Frame', mouse_cb)

    # reposiciona para o frame inicial novamente
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    frame_idx = args.start
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fim do vídeo ou erro ao ler frame.")
            break

        # pula frames se --skip > 0
        if args.skip > 0 and (frame_idx - args.start) % (args.skip + 1) != 0:
            frame_idx += 1
            continue

        image = frame.copy()
        image_copy = frame.copy()
        h, w = image.shape[:2]
        bboxes = []
        awaiting_class_input = False
        pending_bbox = {}

        while True:
            # exibe a imagem com caixas atuais
            disp = image.copy()
            draw_all_boxes(disp, bboxes)
            cv2.imshow('Frame', disp)

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
                        print(f"[OK] BBOX adicionado com classe {cls_id}. Total no frame: {len(bboxes)}")

                        # refresh visual imediato
                        image_copy[:] = image
                        draw_all_boxes(image_copy, bboxes)
                        cv2.imshow('Frame', image_copy)
                        cv2.waitKey(1)
                    except ValueError:
                        print("[ERRO] Valor inválido. BBOX descartado.")
                    awaiting_class_input = False
                    pending_bbox = {}

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("[INFO] Saindo.")
                cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord('u'):
                if bboxes:
                    dropped = bboxes.pop()
                    print(f"[INFO] Removido último BBOX (classe {dropped['cls']}). Restantes: {len(bboxes)}")

            elif key == ord('r'):
                bboxes = []
                print("[INFO] BBOX do frame atual limpos.")

            elif key == ord('n'):
                # descarta sem salvar
                print("[INFO] Indo para o próximo frame (sem salvar).")
                break

            elif key == ord('s'):
                # salva imagem e labels
                img_name = f"{args.img_prefix}-{frame_idx}.jpg"
                lbl_name = f"{args.img_prefix}-{frame_idx}.txt"
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

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Frames salvos: {saved_count}")


if __name__ == "__main__":
    main()


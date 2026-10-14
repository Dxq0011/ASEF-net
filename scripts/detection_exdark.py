"""Zero-shot ExDark detection with YOLOv11n over enhanced images (Sec. 4.8).

Protocol: the COCO-pretrained detector (yolo11n.pt) evaluates the original
low-light images and each method's enhanced images with identical weights,
input resolution and confidence/IoU thresholds. A fixed ExDark->COCO class
correspondence is applied identically to every method: ExDark annotations are
converted once to YOLO labels carrying COCO category ids, so the detector's
native COCO head is evaluated without any fine-tuning.

ExDark layout (official): <root>/<class>/<image>.jpg with sibling .txt files,
each line `class_index x y w h` (1-indexed classes, top-left pixel origin).
The conversion runs once into --workdir; afterwards only the image folder is
swapped between the original and each enhanced variant.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

EXDARK_TO_COCO = {  # ExDark class index (1-12) -> COCO category id
    1: 2, 2: 9, 3: 40, 4: 6, 5: 3, 6: 16,
    7: 57, 8: 42, 9: 17, 10: 4, 11: 1, 12: 61,
}
COCO_NAMES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,'
              'traffic light,fire hydrant,stop sign,parking meter,bench,bird,'
              'cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,'
              'umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,'
              'kite,baseball bat,baseball glove,skateboard,surfboard,tennis rack,'
              'bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,'
              'orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,'
              'potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,'
              'keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,'
              'clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')


def convert_annotations(exdark_root: Path, workdir: Path, image_exts) -> list[str]:
    images_dir = workdir / 'images'
    labels_dir = workdir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    stems = []
    for ann in sorted(exdark_root.rglob('*.txt')):
        stem = ann.stem
        img = next((p for p in (ann.parent / f'{stem}{e}' for e in image_exts)
                    if p.exists()), None)
        if img is None:
            continue
        target = images_dir / img.name
        if not target.exists():
            shutil.copy2(img, target)
        width, height = Image.open(target).size
        lines = []
        for line in ann.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = int(parts[0]), *map(float, parts[1:])
            if cls not in EXDARK_TO_COCO:
                continue
            xc, yc = (x + w / 2) / width, (y + h / 2) / height
            lines.append(f'{EXDARK_TO_COCO[cls]} {xc:.6f} {yc:.6f} '
                         f'{w / width:.6f} {h / height:.6f}')
        (labels_dir / f'{stem}.txt').write_text('\n'.join(lines), encoding='utf8')
        stems.append(img.name)
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exdark-root', required=True)
    ap.add_argument('--images', help='enhanced images to evaluate; omit for the original low-light set')
    ap.add_argument('--workdir', default='datasets/exdark_yolo')
    ap.add_argument('--weights', default='yolo11n.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--label', default=None)
    ap.add_argument('--out', default='results/exdark_detection.json')
    args = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install the detection extra: pip install ultralytics') from exc

    workdir = Path(args.workdir)
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    convert_annotations(Path(args.exdark_root), workdir, exts)
    if args.images:
        images_dir = workdir / 'images'
        for p in images_dir.iterdir():
            p.unlink()
        for src in sorted(Path(args.images).iterdir()):
            if src.suffix.lower() in exts:
                shutil.copy2(src, images_dir / src.name)
    data_yaml = workdir / 'data.yaml'
    data_yaml.write_text(
        f'path: {workdir.resolve()}\ntrain: images\nval: images\n'
        f'names: {{{", ".join(f"{i}: {n}" for i, n in enumerate(COCO_NAMES))}}}\n',
        encoding='utf8')

    model = YOLO(args.weights)
    res = model.val(data=str(data_yaml), split='val', imgsz=args.imgsz,
                    conf=args.conf, iou=args.iou, verbose=False)
    report = {'label': args.label or (args.images or 'original low-light'),
              'weights': args.weights, 'imgsz': args.imgsz,
              'conf': args.conf, 'iou': args.iou,
              'mAP50': float(res.box.map50) * 100,
              'mAP50-95': float(res.box.map) * 100,
              'precision': float(res.box.mp) * 100,
              'recall': float(res.box.mr) * 100}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

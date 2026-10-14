"""CUED construction-scene detection (Sec. 4.8): train YOLOv11n on enhanced
CUED training images and evaluate on the manually annotated test subset.

Protocol: for every enhancement method, an independent YOLOv11n detector is
trained for 100 epochs on that method's enhanced version of the 1,200 training
images, keeping the architecture, optimizer, batch size, schedule and split
identical; detectors are then evaluated on the 300 annotated test images.

Expected inputs (YOLO format, prepared from the manual annotations):
  --labels-root containing train/ and test/ label .txt files (YOLO txt,
     normalized xywh) and optionally a names list; enhanced image folders are
     linked in by this script, so the same label set is reused for each method.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def link_images(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in dst_dir.iterdir():
        p.unlink()
    n = 0
    for src in sorted(src_dir.iterdir()):
        if src.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(src, dst_dir / src.name)
            n += 1
    if not n:
        raise ValueError(f'no images in {src_dir}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-root', required=True,
                    help='YOLO-format annotation root with train/ and test/ label folders')
    ap.add_argument('--train-images', required=True, help='enhanced CUED training images')
    ap.add_argument('--test-images', required=True, help='enhanced CUED test images')
    ap.add_argument('--workdir', default='datasets/cued_yolo')
    ap.add_argument('--names', default='worker,vehicle,crane,material',
                    help='comma-separated class names matching the label ids')
    ap.add_argument('--model', default='yolo11n.yaml',
                    help='yolo11n.yaml trains from scratch; yolo11n.pt uses COCO pretraining')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--label', default=None)
    ap.add_argument('--out', default='results/cued_detection.json')
    args = ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Install the detection extra: pip install ultralytics') from exc

    workdir = Path(args.workdir)
    labels_root = Path(args.labels_root)
    link_images(Path(args.train_images), workdir / 'images' / 'train')
    link_images(Path(args.test_images), workdir / 'images' / 'test')
    for split in ('train', 'test'):
        src = labels_root / split
        if not src.exists():
            raise FileNotFoundError(f'missing label folder: {src}')
        dst = workdir / 'labels' / split
        dst.mkdir(parents=True, exist_ok=True)
        for ann in src.glob('*.txt'):
            shutil.copy2(ann, dst / ann.name)
    names = ', '.join(f'{i}: {n}' for i, n in enumerate(args.names.split(',')))
    data_yaml = workdir / 'data.yaml'
    data_yaml.write_text(
        f'path: {workdir.resolve()}\ntrain: images/train\nval: images/test\n'
        f'names: {{{names}}}\n', encoding='utf8')

    model = YOLO(args.model)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, seed=args.seed, verbose=False,
                project=str(Path('runs') / 'cued_detection'),
                name=args.label or Path(args.train_images).name)
    res = model.val(data=str(data_yaml), split='test', imgsz=args.imgsz,
                    verbose=False)
    report = {'label': args.label or Path(args.train_images).name,
              'model': args.model, 'epochs': args.epochs, 'seed': args.seed,
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

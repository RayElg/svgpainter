"""512x512 references: photos from a COCO zip (centre square, short side >= 400 px), or flat
graphics from a directory of SVGs (rendered at 352-448 px onto white or a light colour).

python prep_images.py coco val2017.zip data/img/coco_val [limit]
python prep_images.py svg twemoji-15.1.0/assets/svg data/img/twemoji
"""
import io
import os
import random
import sys
import zipfile
from multiprocessing import Pool
from PIL import Image

S = 512


def coco(args):
    path, names, out = args
    z, n = zipfile.ZipFile(path), 0
    for name in names:
        dst = os.path.join(out, os.path.basename(name))
        if os.path.exists(dst):
            continue
        try:
            im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
        except Exception:
            continue
        s = min(im.size)
        if s < 400:
            continue
        l, t = (im.width - s) // 2, (im.height - s) // 2
        im.crop((l, t, l + s, t + s)).resize((S, S), Image.LANCZOS).save(dst, quality=93)
        n += 1
    return n


def svg(args):
    import resvg_py
    paths, out = args
    n = 0
    for p in paths:
        dst = os.path.join(out, os.path.basename(p)[:-4] + ".png")
        if os.path.exists(dst):
            continue
        rnd = random.Random(p)
        size = rnd.choice([352, 400, 448])
        try:
            fg = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_path=p, width=size, height=size)))).convert("RGBA")
        except Exception:
            continue
        bg = (255, 255, 255) if rnd.random() < 0.6 else tuple(rnd.randint(150, 255) for _ in range(3))
        im = Image.new("RGB", (S, S), bg)
        im.paste(fg, ((S - fg.width) // 2, (S - fg.height) // 2), fg)
        im.save(dst)
        n += 1
    return n


def main():
    mode, src, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    k = os.cpu_count() or 4
    if mode == "coco":
        names = sorted(n for n in zipfile.ZipFile(src).namelist() if n.endswith(".jpg"))
        names = names[:int(sys.argv[4])] if len(sys.argv) > 4 else names
        jobs, fn = [(src, names[i::k], out) for i in range(k)], coco
    else:
        paths = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".svg"))
        jobs, fn = [(paths[i::k], out) for i in range(k)], svg
    with Pool(k) as p:
        print(out, sum(p.map(fn, jobs)), "images")


if __name__ == "__main__":
    main()

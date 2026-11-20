from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "data" / "image_exemple.jpg"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ---------- Chargement et inspection (Activité 1) ----------
bgr = cv.imread(str(IMAGE_PATH))
if bgr is None:
    raise FileNotFoundError(f"Image introuvable ou illisible : {IMAGE_PATH}")

h, w, c = bgr.shape
print("ACTIVITE 1 - INSPECTION")
print("Forme :", bgr.shape)
print("Type :", bgr.dtype)
print("Nombre de valeurs :", bgr.size)
print("Mémoire (octets) :", bgr.nbytes)
print("Nombre de pixels H*W :", h * w)
print("Nombre de valeurs H*W*C :", h * w * c)
print("Pixel central BGR :", bgr[h // 2, w // 2])

# ---------- Couleurs et ROI (Activité 2) ----------
rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
gris = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
roi = rgb[h // 4:3 * h // 4, w // 4:3 * w // 4].copy()
petite = cv.resize(rgb, (max(1, w // 4), max(1, h // 4)),
                   interpolation=cv.INTER_AREA)

print("\nACTIVITE 2 - FORMES")
print("RGB :", rgb.shape, "| Gris :", gris.shape, "| HSV :", hsv.shape)
print("ROI :", roi.shape, "| Image réduite :", petite.shape)
print("Rapport pixels image initiale / réduite :", (h*w) / (petite.shape[0]*petite.shape[1]))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(rgb); axes[0].set_title("Image RGB")
axes[1].imshow(gris, cmap="gray"); axes[1].set_title("Niveaux de gris")
axes[2].imshow(roi); axes[2].set_title("ROI centrale")
for ax in axes: ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "comparaison_image_gris_roi.png", dpi=150)
plt.close(fig)

# Affichage séparé des canaux RGB en niveaux de gris
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for i, (ax, nom) in enumerate(zip(axes, ["Rouge", "Vert", "Bleu"])):
    ax.imshow(rgb[:, :, i], cmap="gray"); ax.set_title(nom); ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "canaux_rgb.png", dpi=150)
plt.close(fig)

# ---------- Intensités et histogrammes (Activité 3) ----------
gris_float = gris.astype(np.float32)
clair = np.clip(gris_float + 40, 0, 255).astype(np.uint8)
contraste = np.clip(0.4 * (gris_float - 128) + 128, 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 5))
for image, label in [(gris, "Original"), (clair, "Éclaircie +40"),
                     (contraste, "Contraste réduit")]:
    plt.hist(image.ravel(), bins=32, range=(0, 256), alpha=0.45, label=label)
plt.xlabel("Intensité"); plt.ylabel("Effectif"); plt.title("Histogrammes")
plt.legend(); plt.tight_layout()
plt.savefig(OUT / "histogrammes.png", dpi=150)
plt.close()

# ---------- Quantification 4 niveaux (Activité 4) ----------
L = 4
q = np.rint(gris.astype(np.float32) * (L - 1) / 255)
q = np.rint(q * 255 / (L - 1)).astype(np.uint8)
print("\nACTIVITE 4 - QUANTIFICATION")
print("Niveaux présents :", np.unique(q))
print("Mémoire gris / q :", gris.nbytes, "/", q.nbytes, "octets")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(gris, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original")
axes[1].imshow(q, cmap="gray", vmin=0, vmax=255); axes[1].set_title("4 niveaux")
for ax in axes: ax.axis("off")
fig.tight_layout(); fig.savefig(OUT / "quantification_4_niveaux.png", dpi=150)
plt.close(fig)

# ---------- Masque rouge (Activité 5) ----------
m1 = cv.inRange(hsv, np.array([0, 80, 50], np.uint8),
                np.array([10, 255, 255], np.uint8))
m2 = cv.inRange(hsv, np.array([170, 80, 50], np.uint8),
                np.array([179, 255, 255], np.uint8))
masque = cv.bitwise_or(m1, m2)
selection = cv.bitwise_and(rgb, rgb, mask=masque)
taux = 100 * np.count_nonzero(masque) / masque.size
print("\nACTIVITE 5 - MASQUE ROUGE")
print(f"Pourcentage de pixels sélectionnés : {taux:.2f} %")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(rgb); axes[0].set_title("Image")
axes[1].imshow(masque, cmap="gray"); axes[1].set_title("Masque rouge")
axes[2].imshow(selection); axes[2].set_title("Pixels sélectionnés")
for ax in axes: ax.axis("off")
fig.tight_layout(); fig.savefig(OUT / "masque_rouge.png", dpi=150)
plt.close(fig)

# ---------- Stockage et fichiers (Activité 6) ----------
png_path = OUT / "copie.png"
jpg90_path = OUT / "qualite90.jpg"
jpg30_path = OUT / "qualite30.jpg"
cv.imwrite(str(png_path), bgr)
cv.imwrite(str(jpg90_path), bgr, [cv.IMWRITE_JPEG_QUALITY, 90])
cv.imwrite(str(jpg30_path), bgr, [cv.IMWRITE_JPEG_QUALITY, 30])

print("\nACTIVITE 6 - STOCKAGE")
for path in [png_path, jpg90_path, jpg30_path]:
    relu = cv.imread(str(path))
    print(path.name, "| taille fichier :", path.stat().st_size,
          "octets | tableau décodé :", relu.shape,
          "| mémoire :", relu.nbytes, "octets",
          "| identique au BGR initial :", np.array_equal(bgr, relu))

print("\nTerminé. Les figures et fichiers sont dans le dossier outputs/.")

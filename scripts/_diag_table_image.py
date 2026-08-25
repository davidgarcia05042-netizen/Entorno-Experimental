import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [
    ("MediaPipe Heavy", "7.85°", "27.98°", "Moderado"),
    ("MediaPipe Lite", "9.43°", "34.72°", "Moderado"),
    ("MediaPipe Full", "10.94°", "28.86°", "No aceptable"),
    ("YOLOv8-medium", "11.84°", "32.43°", "No aceptable"),
    ("YOLOv8-small", "13.87°", "30.48°", "No aceptable"),
    ("YOLOv8-nano", "15.31°", "36.65°", "No aceptable"),
]
columns = ["Variante", "Error medio", "Error máximo", "Clasificación"]

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.9)

class_colors = {
    "Moderado": "#fef3c7",
    "No aceptable": "#fee2e2",
}
for i, row in enumerate(rows, start=1):
    classification = row[3]
    color = class_colors.get(classification, "#ffffff")
    for j in range(len(columns)):
        table[i, j].set_facecolor(color)

for j in range(len(columns)):
    table[0, j].set_facecolor("#1f2937")
    table[0, j].set_text_props(color="white", fontweight="bold")

ax.set_title("Error de ángulo de rodilla vs. gold standard (rodilla izquierda, Marcha Katherine)", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(
    "data/gold_standard/marcha_katherine_2026-06-19/error_table.png",
    dpi=150,
    bbox_inches="tight",
)
print("guardado")

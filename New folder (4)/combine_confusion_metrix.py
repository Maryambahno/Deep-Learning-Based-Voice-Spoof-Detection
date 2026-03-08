from PIL import Image
import matplotlib.pyplot as plt


# Image paths
img1 = Image.open("New folder (4)/gru_confusion_matrix.png")
img2 = Image.open("New folder (4)/LSTM_confusion_matrix.png")
img3 = Image.open("New folder (4)/BiGRU_confusion_matrix.png")
img4 = Image.open("New folder (4)/VGGNET_confusion_matrix.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)

images = [img1, img2, img3, img4]
titles = ["(a) GRU", "(b) LSTM", "(c) BiGRU", "(d) VGGNET"]

for ax, img, title in zip(axes.ravel(), images, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=16, pad=10)

    # ticks hide kar do
    ax.set_xticks([])
    ax.set_yticks([])

    # border box show karo
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

plt.subplots_adjust(hspace=0.25, wspace=0.18)
plt.savefig("combined_confusion_matrices_boxed.png", dpi=300, bbox_inches="tight")
plt.show()



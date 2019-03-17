import matplotlib.pyplot as plt

plt.style.use('ggplot')


def plot_image_from_tensor(tensor, title="Image"):
    plt.imshow(tensor, cmap='gray')
    plt.grid(False)
    plt.title(title)
    plt.show()


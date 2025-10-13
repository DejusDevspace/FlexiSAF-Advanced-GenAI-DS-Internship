import matplotlib.pyplot as plt
from .helpers import generate_caption, calculate_bleu

def visualize_prediction(image_path, model, vocab, device):
    """
    Generate and display caption for an image.
    """
    caption, image = generate_caption(image_path, model, vocab, device)
    bleu_score = calculate_bleu(image_path, caption, captions_file='../dataset/captions.txt')
    caption += f"\nBLEU-4: {bleu_score:.2f}"

    # Display image with caption
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated Caption:\n{caption}", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('../img/prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Generated Caption: {caption}")
    return caption


def create_demo_grid(image_paths, model, vocab, device):
    """
    Create a grid showing multiple images with their captions.
    Useful for presentation/demo.
    """
    n_images = len(image_paths)
    cols = 2
    rows = (n_images + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx, img_path in enumerate(image_paths):
        caption, image = generate_caption(img_path, model, vocab, device)
        bleu_score = calculate_bleu(img_path, caption, captions_file='../dataset/captions.txt')
        caption += f"\nBLEU-4: {bleu_score:.2f}"

        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(caption, fontsize=12, wrap=True)

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('../img/demo_grid2.png', dpi=150, bbox_inches='tight')
    plt.show()

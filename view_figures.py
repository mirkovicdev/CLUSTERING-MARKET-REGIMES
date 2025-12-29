"""
Quick viewer to display all generated figures in a grid.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from glob import glob

# Dark mode
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#0d1117'

def view_figures(figures_dir="figures", cols=3):
    """Display all PNG figures in a grid."""

    # Get all PNG files
    png_files = sorted(glob(os.path.join(figures_dir, "*.png")))

    if not png_files:
        print(f"No PNG files found in {figures_dir}/")
        return

    n_images = len(png_files)
    rows = (n_images + cols - 1) // cols

    print(f"Found {n_images} images, displaying in {rows}x{cols} grid...")

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    fig.patch.set_facecolor('#0d1117')

    # Flatten axes for easy iteration
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Display each image
    for i, (ax, img_path) in enumerate(zip(axes, png_files)):
        img = mpimg.imread(img_path)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path), fontsize=9, color='white')
        ax.axis('off')

    # Hide empty subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def view_figures_pages(figures_dir="figures", per_page=9):
    """Display figures in pages of 9 (3x3 grid)."""

    png_files = sorted(glob(os.path.join(figures_dir, "*.png")))

    if not png_files:
        print(f"No PNG files found in {figures_dir}/")
        return

    n_images = len(png_files)
    n_pages = (n_images + per_page - 1) // per_page

    print(f"Found {n_images} images across {n_pages} page(s)")
    print("Close each window to see the next page...")

    for page in range(n_pages):
        start = page * per_page
        end = min(start + per_page, n_images)
        page_files = png_files[start:end]

        cols = 3
        rows = 3

        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.patch.set_facecolor('#0d1117')
        fig.suptitle(f"Page {page + 1} of {n_pages}", fontsize=14, color='white', y=0.98)

        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < len(page_files):
                img = mpimg.imread(page_files[i])
                ax.imshow(img)
                ax.set_title(os.path.basename(page_files[i]), fontsize=8, color='white')
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--pages":
        # Show in pages of 9
        view_figures_pages()
    else:
        # Show all at once
        view_figures()

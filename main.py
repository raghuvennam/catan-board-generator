import random
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import numpy as np

# Resource selection logic


def selectionLarge():
    # 6 of each resource, 2 deserts
    tiles = ["brick"] * 6 + ["wood"] * 6 + ["stone"] * 6 + ["sheep"] * 6 + ["wheat"] * 6
    tiles = tiles[:28]  # In case of any overfill (shouldn't happen)
    tiles += ["desert"] * 2
    random.shuffle(tiles)
    return tiles


def selectionSmall():
    # 4 of each resource, 1 desert
    tiles = ["brick"] * 4 + ["wood"] * 4 + ["stone"] * 4 + ["sheep"] * 4 + ["wheat"] * 4
    tiles = tiles[:18]  # In case of any overfill (shouldn't happen)
    tiles += ["desert"]
    random.shuffle(tiles)
    return tiles


def ensure_resized_image(board_type="Small"):
    import os
    from PIL import Image
    import matplotlib.image as mpimg
    import numpy as np
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    if os.path.exists("catan_board.png"):
        img = Image.open("catan_board.png")
        img = img.convert("RGB")
        img = img.resize((1200, 1200), Image.LANCZOS)
        return np.asarray(img)
    elif os.path.exists("catan_board.svg"):
        img = mpimg.imread("catan_board.svg")
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img = img.resize((1200, 1200), Image.LANCZOS)
        return np.asarray(img)
    else:
        # Generate a simple Catan-style hex board as an image
        def hex_corner(center, size, i):
            angle_deg = 60 * i - 30
            angle_rad = np.pi / 180 * angle_deg
            return (
                center[0] + size * np.cos(angle_rad),
                center[1] + size * np.sin(angle_rad),
            )

        # Use board_type to determine layout and tiles
        if board_type == "Large":
            tiles = selectionLarge()
            # Official Catan 5-6 player (30 hex) layout: only include these (q, r) pairs
            hex_centers = [
                (-3, 2),
                (-3, 3),
                (-2, 1),
                (-2, 2),
                (-2, 3),
                (-1, 0),
                (-1, 1),
                (-1, 2),
                (-1, 3),
                (0, -1),
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (1, -2),
                (1, -1),
                (1, 0),
                (1, 1),
                (1, 2),
                (1, 3),
                (2, -3),
                (2, -2),
                (2, -1),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, -3),
                (3, -2),
                (3, -1),
                (3, 0),
            ]

            # Correct 60-degree hex grid rotations
            def rotate_hex(q, r, times):
                for _ in range(times % 6):
                    q, r = -r, q + r
                return (q, r)

            rotation_times = random.randint(0, 5)
            hex_centers = [rotate_hex(q, r, rotation_times) for (q, r) in hex_centers]
            # Randomly reflect (flip q and r axes)
            if random.choice([True, False]):
                hex_centers = [(r, q) for (q, r) in hex_centers]
            # Shuffle both hex_centers and tiles before pairing
            random.shuffle(hex_centers)
            random.shuffle(tiles)
            paired = list(zip(hex_centers, tiles))
            hex_centers, tiles = zip(*paired)
            hex_centers = list(hex_centers)
            tiles = list(tiles)
        else:
            tiles = selectionSmall()
            # Small board: radius 2
            hex_centers = []
            for q in range(-2, 3):
                r1 = max(-2, -q - 2)
                r2 = min(2, -q + 2)
                for r in range(r1, r2 + 1):
                    hex_centers.append((q, r))
            random.shuffle(hex_centers)
            random.shuffle(tiles)
            paired = list(zip(hex_centers, tiles))
            hex_centers, tiles = zip(*paired)
            hex_centers = list(hex_centers)
            tiles = list(tiles)

        # Hex to pixel
        def hex_to_pixel(q, r, size, origin):
            x = size * 3 / 2 * q + origin[0]
            y = size * np.sqrt(3) * (r + q / 2) + origin[1]
            return (x, y)

        size = 40  # original hex size
        # Center the board visually
        if board_type == "Large":
            origin = (300, 250)  # Adjusted for large board centering
        else:
            origin = (300, 300)  # original center for small board
        colors = ["#967353", "#228B22", "#C0C0C0", "#fff44f", "#e0cda9", "#e2e2e2"]
        # Remove duplicate tile/hex assignment and shuffling here
        # Map resource names to color indices
        resource_to_idx = {
            "brick": 0,
            "wood": 1,
            "stone": 2,
            "sheep": 3,
            "wheat": 4,
            "desert": 5,
        }
        tile_types = [resource_to_idx[t] for t in tiles]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 600)
        ax.axis("off")
        size = 40
        origin = (300, 250)  # Adjusted for large board centering
        colors = ["#967353", "#228B22", "#C0C0C0", "#fff44f", "#e0cda9", "#e2e2e2"]
        for idx, (q, r) in enumerate(hex_centers):
            center = hex_to_pixel(q, r, size, origin)
            hexagon = mpatches.RegularPolygon(
                center,
                numVertices=6,
                radius=size,
                orientation=np.radians(30),
                facecolor=colors[tile_types[idx]],
                edgecolor="k",
                lw=2,
            )
            ax.add_patch(hexagon)
            hex_type = ["brick", "wood", "stone", "sheep", "wheat", "desert"][
                tile_types[idx]
            ]
            ax.text(
                center[0],
                center[1] - size * 0.25,
                str(idx + 1),
                color="red",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
            ax.text(
                center[0],
                center[1] + size * 0.15,
                hex_type,
                color=("#C2B280" if tile_types[idx] == 5 else "black"),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return img


# Map image selection


def get_map_image(cmd):
    # Use the resized image for both map sizes
    return ensure_resized_image(cmd)


# Matplotlib UI


def update_display(label):
    global ax_img, ax_text, fig
    ax_img.clear()
    ax_text.clear()
    if label == "Small (1-4 players)":
        cmd = "Small"
        tiles = selectionSmall()
    else:
        cmd = "Large"
        tiles = selectionLarge()
    # Show map image
    img = get_map_image(cmd)
    ax_img.imshow(img)
    ax_img.axis("off")
    # Show tile list
    tile_str = "\n".join([f"{i+1} {t}" for i, t in enumerate(tiles)])
    ax_text.text(0, 1, tile_str, va="top", ha="left", fontsize=12, family="monospace")
    ax_text.axis("off")
    fig.canvas.draw_idle()


def on_generate(event):
    update_display(radio.value_selected)


def regenerate_board(event=None):
    update_display(radio.value_selected)


# Ask user for board size at startup
import sys

board_type = "Small"  # Default to Small
if len(sys.argv) > 1 and sys.argv[1] in ["5", "6"]:
    board_type = "Large"
print(f"Generating {board_type} board...")

# Set up matplotlib figure
fig = plt.figure(figsize=(24, 12))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.5, right=0.55)
ax_img = fig.add_subplot(gs[2])
ax_text = fig.add_subplot(gs[0:2])
ax_text.axis("off")

# Only show one board at startup
if board_type == "Large":
    update_display("Large (5-6 players)")
else:
    update_display("Small (1-4 players)")

# Commenting out plt.show() to prevent the matplotlib window from popping up
# plt.show()


def save_board_image(output_path="catan_board_output", board_type="Small"):
    """
    Save the generated board image to disk in both SVG and PNG formats without displaying the matplotlib window.

    Parameters:
        output_path (str): The base file path to save the images (without extension).
        board_type (str): The type of board to generate ("Small" or "Large").
    """
    if board_type == "Large":
        tiles = selectionLarge()
    else:
        tiles = selectionSmall()

    # Generate the board image
    img = get_map_image(board_type)

    # Save as PNG
    png_path = f"{output_path}.png"
    plt.imsave(png_path, img)
    print(f"Board image saved as PNG to {png_path}")

    # Save as SVG
    svg_path = f"{output_path}.svg"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Board image saved as SVG to {svg_path}")


# Dynamically save the board image based on the command-line argument
save_board_image("output", board_type)

import random
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import numpy as np

# Resource selection logic


def selectionLarge():
    # 6 of each resource, 2 deserts (for 30 hexes)
    tiles = ["brick"] * 6 + ["wood"] * 6 + ["stone"] * 6 + ["sheep"] * 6 + ["wheat"] * 6
    tiles = tiles[:28]  # In case of any overfill (shouldn't happen)
    tiles += ["desert"] * 2
    random.shuffle(tiles)
    return tiles


def selectionSmall():
    # 3, 4, 5, 4, 3 = 19 hexes (classic small Catan board)
    # Standard: 3 of each resource, 1 desert (for 19 hexes)
    tiles = ["brick"] * 3 + ["wood"] * 4 + ["stone"] * 3 + ["sheep"] * 4 + ["wheat"] * 4
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

        # Use board_type to determine layout and tiles
        if board_type == "Large":
            tiles = selectionLarge()

            # Official Catan 5-6 player (30 hex) layout: 3, 4, 5, 6, 5, 4, 3 per row, horizontally centered
            # Build neighbor map for large board
            def get_neighbors(q, r):
                # Hex axial directions
                directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
                return [(q + dq, r + dr) for dq, dr in directions]

            # Generate hex positions
            hex_centers = []
            row_lengths = [3, 4, 5, 6, 5, 4, 3]
            q_vals = list(range(-3, 4))
            shift_map = {-3: 3, -2: 2, -1: 2, 0: 1, 1: 1}
            for q, row_length in zip(q_vals, row_lengths):
                r_start = -((row_length - 1) // 2) + shift_map.get(q, 0)
                for i in range(row_length):
                    r = r_start + i
                    hex_centers.append((q, r))
            # Build neighbor map
            pos_to_idx = {pos: i for i, pos in enumerate(hex_centers)}
            neighbors = [[] for _ in range(len(hex_centers))]
            for idx, (q, r) in enumerate(hex_centers):
                for nq, nr in get_neighbors(q, r):
                    if (nq, nr) in pos_to_idx:
                        neighbors[idx].append(pos_to_idx[(nq, nr)])
            # Greedy placement with retries
            import collections

            def try_place(tiles):
                assignment = [None] * len(hex_centers)
                for idx in range(len(hex_centers)):
                    random.shuffle(tiles)
                    for t in tiles:
                        if all(assignment[n] != t for n in neighbors[idx]):
                            assignment[idx] = t
                            tiles.remove(t)
                            break
                    else:
                        return None  # Failed
                return assignment

            # Try multiple times to find a valid placement
            max_tries = 1000
            for _ in range(max_tries):
                tiles = selectionLarge()
                candidate = try_place(tiles.copy())
                if candidate:
                    tiles = candidate
                    break
            else:
                # Fallback to random if not possible
                tiles = selectionLarge()
                random.shuffle(hex_centers)
                random.shuffle(tiles)
            hex_centers = list(hex_centers)
            tiles = list(tiles)
        else:
            tiles = selectionSmall()
            # Small board: 5 rows: 3, 4, 5, 4, 3 per row, horizontally centered
            hex_centers = []
            row_lengths = [3, 4, 5, 4, 3]
            q_vals = list(range(-2, 3))  # q from -2 to 2
            for q_idx, (q, row_length) in enumerate(zip(q_vals, row_lengths)):
                if q == -2:
                    r_start = -((row_length - 1) // 2) + 2
                elif q == -1:
                    r_start = -((row_length - 1) // 2) + 1
                elif q == 0:
                    r_start = -((row_length - 1) // 2) + 1
                else:
                    r_start = -((row_length - 1) // 2)
                for i in range(row_length):
                    r = r_start + i
                    hex_centers.append((q, r))

            # Build neighbor map for small board
            def get_neighbors(q, r):
                directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
                return [(q + dq, r + dr) for dq, dr in directions]

            pos_to_idx = {pos: i for i, pos in enumerate(hex_centers)}
            neighbors = [[] for _ in range(len(hex_centers))]
            for idx, (q, r) in enumerate(hex_centers):
                for nq, nr in get_neighbors(q, r):
                    if (nq, nr) in pos_to_idx:
                        neighbors[idx].append(pos_to_idx[(nq, nr)])

            # Greedy placement with retries
            def try_place(tiles):
                assignment = [None] * len(hex_centers)
                for idx in range(len(hex_centers)):
                    random.shuffle(tiles)
                    for t in tiles:
                        if all(assignment[n] != t for n in neighbors[idx]):
                            assignment[idx] = t
                            tiles.remove(t)
                            break
                    else:
                        return None  # Failed
                return assignment

            max_tries = 1000
            for _ in range(max_tries):
                tiles = selectionSmall()
                candidate = try_place(tiles.copy())
                if candidate:
                    tiles = candidate
                    break
            else:
                # Fallback to random if not possible
                tiles = selectionSmall()
                random.shuffle(hex_centers)
                random.shuffle(tiles)
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
        # Define contrasting text colors for each resource
        resource_text_colors = {
            "brick": "white",
            "wood": "white",
            "stone": "black",
            "sheep": "black",
            "wheat": "black",
            "desert": "#967353",  # brownish for desert
        }
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
            # Draw index vertically, centered, and moved 15 px right
            ax.text(
                center[0] + 15,  # move 15 px right
                center[1],  # center vertically
                str(idx + 1),
                color=resource_text_colors[hex_type],
                ha="center",
                va="center",
                fontsize=6,
                fontweight="bold",
                rotation=90,  # vertical
                rotation_mode="anchor",
            )
            # Draw resource name vertically
            ax.text(
                center[0],
                center[1],
                hex_type,
                color=resource_text_colors[hex_type],
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                rotation=90,  # vertical
                rotation_mode="anchor",
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

    # Save as SVG first
    svg_path = f"{output_path}.svg"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    ax.axis("off")
    # Increase DPI for better clarity
    fig.savefig(svg_path, format="svg", bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Convert SVG to PNG using cairosvg
    try:
        import cairosvg
        import time
        import os

        png_path = f"{output_path}.png"
        # Determine SVG size for scaling
        from xml.etree import ElementTree as ET

        tree = ET.parse(svg_path)
        root = tree.getroot()
        width = root.get("width")
        height = root.get("height")

        # Helper to parse SVG length with units
        def parse_svg_length(val):
            if val is None:
                return None
            import re

            m = re.match(r"([0-9.]+)([a-z%]*)", val.strip())
            if not m:
                return None
            num, unit = m.groups()
            num = float(num)
            if unit in ("", "px"):  # pixels
                return num
            elif unit == "pt":  # points (1pt = 1.3333px)
                return num * 96 / 72
            elif unit == "in":  # inches
                return num * 96
            elif unit == "cm":
                return num * 96 / 2.54
            elif unit == "mm":
                return num * 96 / 25.4
            else:
                return num  # fallback, treat as px

        width = parse_svg_length(width) if width else None
        height = parse_svg_length(height) if height else None
        # Fallback if width/height are not set as attributes
        if width is None or height is None:
            # Try viewBox
            viewbox = root.get("viewBox")
            if viewbox:
                _, _, w, h = viewbox.split()
                width = float(w)
                height = float(h)
        width = int(round(width)) if width else 600
        height = int(round(height)) if height else 600
        # Double the size for PNG
        out_w, out_h = width * 2, height * 2
        cairosvg.svg2png(
            url=svg_path, write_to=png_path, output_width=out_w, output_height=out_h
        )
        print(f"Board image saved as PNG to {png_path} (double size: {out_w}x{out_h})")
        # Ensure file is written and nonzero size
        for _ in range(10):
            if os.path.exists(png_path) and os.path.getsize(png_path) > 0:
                break
            time.sleep(0.1)
        else:
            print(f"Warning: PNG file {png_path} not found or empty after save.")
        # Crop to 2 inch margin around the actual board (not just canvas)
        from PIL import Image
        import numpy as np

        img = Image.open(png_path).convert("RGB")
        # Rotate by 270 degrees before cropping
        img = img.rotate(270, expand=True)
        arr = np.array(img)
        # Find non-white (background) pixels
        mask = np.any(arr < 250, axis=2)  # True for any non-white pixel
        coords = np.argwhere(mask)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            # Add 1-inch margin (96 dpi)
            dpi = img.info.get("dpi", (96, 96))[0]
            margin = int(1 * dpi)
            x0 = max(x0 - margin, 0)
            y0 = max(y0 - margin, 0)
            x1 = min(x1 + margin, img.width - 1)
            y1 = min(y1 + margin, img.height - 1)
            img_cropped = img.crop((x0, y0, x1, y1))
            img_cropped.save(png_path)
            print(
                f"Cropped PNG to board + 1-inch margin: {png_path} ({img_cropped.width}x{img_cropped.height})"
            )
        else:
            print("Warning: Could not find board content for cropping, skipping crop.")
        # Open the PNG image using the default viewer (macOS)
        import subprocess
        import webbrowser

        try:
            proc = subprocess.Popen(["open", png_path])
            print(f"Launched 'open' for PNG: {png_path} (PID: {proc.pid})")
        except Exception as e:
            print(f"Exception during 'open': {e}. Trying webbrowser fallback...")
            webbrowser.open(f"file://{png_path}")
    except ImportError:
        print("cairosvg is not installed. PNG will not be generated from SVG.")


# Dynamically save the board image based on the command-line argument
save_board_image("output", board_type)

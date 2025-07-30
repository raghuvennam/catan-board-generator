# catan-board-generator

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/catan-board-generator.git
   cd catan-board-generator
   ```
2. (Recommended) Create and activate a Python virtual environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To generate a Catan board image:

- For the classic (small) board (3-4 players):
  ```sh
  python main.py
  ```
- For the large board (5-6 players):
  ```sh
  python main.py 5
  # or
  python main.py 6
  ```

This will generate `output.svg` and `output.png` in the current directory. The PNG will be opened automatically if you are on macOS.

## Notes
- Requires Python 3.7 or newer.
- The PNG output is rotated for print/play convenience.
- You can further tweak the board by editing `main.py`.
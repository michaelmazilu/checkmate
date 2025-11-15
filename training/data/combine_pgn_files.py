"""
Combine all downloaded Stockfish PGN.gz files into a single PGN file.
"""

import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
STOCKFISH_DIR = Path(__file__).parent / "stockfish_pgns"
SNAPSHOT_DIR = (
    STOCKFISH_DIR
    / "datasets--official-stockfish--fishtest_pgns"
    / "snapshots"
)
OUTPUT_FILE = Path(__file__).parent / "data" / "combined_stockfish.pgn"

def combine_pgn_files():
    """Extract and combine all .pgn.gz files into one large PGN file."""
    
    print("=" * 80)
    print("STOCKFISH PGN COMBINER")
    print("=" * 80)
    print(f"Cache root: {STOCKFISH_DIR}")
    if SNAPSHOT_DIR.exists():
        print(f"Snapshot directory: {SNAPSHOT_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)
    
    # Find all .pgn.gz files
    print("\n[1/3] Finding .pgn.gz files...")
    search_root = SNAPSHOT_DIR if SNAPSHOT_DIR.exists() else STOCKFISH_DIR
    if not search_root.exists():
        print(f"❌ Directory not found: {search_root}")
        print("   Please run download_stockfish_data.py first")
        return

    if search_root == SNAPSHOT_DIR:
        print("   Looking inside Hugging Face snapshot cache for PGN archives...")
    else:
        print("   Snapshot cache missing, scanning entire Stockfish directory...")

    pgn_gz_files = list(search_root.rglob("*.pgn.gz"))
    
    if len(pgn_gz_files) == 0:
        print("❌ No .pgn.gz files found!")
        print(f"   Please run download_stockfish_data.py first")
        return
    
    print(f"✓ Found {len(pgn_gz_files)} .pgn.gz files")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in pgn_gz_files)
    total_size_gb = total_size / (1024 ** 3)
    print(f"  Total compressed size: {total_size_gb:.2f} GB")
    
    # Sort by date (newest first) - files are in YY-MM-DD directories
    pgn_gz_files.sort(key=lambda x: x.parent.parent.name, reverse=True)
    
    print("\nFirst 5 files:")
    for i, f in enumerate(pgn_gz_files[:5]):
        size_mb = f.stat().st_size / (1024 ** 2)
        print(f"  {i+1}. {f.parent.parent.name}/{f.parent.name}/{f.name} ({size_mb:.1f} MB)")
    if len(pgn_gz_files) > 5:
        print(f"  ... and {len(pgn_gz_files) - 5} more")
    
    # Ask for confirmation
    response = input(f"\nCombine {len(pgn_gz_files)} files into one? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Operation cancelled.")
        return
    
    # Create output directory
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Combine files
    print(f"\n[2/3] Extracting and combining files...")
    games_written = 0
    bytes_written = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for pgn_gz_file in tqdm(pgn_gz_files, desc="Processing"):
            try:
                with gzip.open(pgn_gz_file, 'rt', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    
                    # Add separator between files if not already present
                    if not content.endswith('\n\n'):
                        outfile.write('\n\n')
                    
                    bytes_written += len(content)
                    # Rough estimate: each game is ~200 chars
                    games_written += content.count('[Event ')
                    
            except Exception as e:
                print(f"\n⚠ Error processing {pgn_gz_file.name}: {e}")
                continue
    
    output_size_gb = bytes_written / (1024 ** 3)
    
    print(f"\n[3/3] Verifying output...")
    if OUTPUT_FILE.exists():
        actual_size = OUTPUT_FILE.stat().st_size / (1024 ** 3)
        print(f"✓ Output file created: {OUTPUT_FILE}")
        print(f"  Size: {actual_size:.2f} GB")
        print(f"  Estimated games: ~{games_written:,}")
    
    # Summary
    print("\n" + "=" * 80)
    print("COMBINATION COMPLETE")
    print("=" * 80)
    print(f"✓ Combined {len(pgn_gz_files)} files")
    print(f"✓ Output: {OUTPUT_FILE}")
    print(f"✓ Size: {output_size_gb:.2f} GB")
    print(f"✓ Estimated games: ~{games_written:,}")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Convert PGN games to training format (FEN + move + value)")
    print("   Run: python training/convert_pgn_to_training_data.py")
    print("=" * 80)


if __name__ == "__main__":
    combine_pgn_files()

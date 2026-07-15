#!/usr/bin/env python3
import glob
import os
import sys

from robodojo.core.storage import assets_root


def print_color(message, color_code):
    NC = "\033[0m"
    sys.stdout.write(f"{color_code}{message}{NC}\n")


BLUE = "\033[0;34m"
YELLOW = "\033[0;33m"
GREEN = "\033[0;32m"


def main():
    assets_path = assets_root()
    print_color(f"Assets root: {assets_path}", BLUE)
    if not (assets_path / "Robots").is_dir():
        print_color(f"Error: {assets_path / 'Robots'} is unavailable", YELLOW)
        sys.exit(1)

    # Counters
    count_total = count_updated = count_error = 0

    # Find *_tmp.yml files
    print_color("Searching for configuration template files...", BLUE)
    pattern = os.path.join(assets_path, "Robots", "**", "*_tmp.yml")
    config_files = glob.glob(pattern, recursive=True)

    if not config_files:
        print_color("No *_tmp.yml files found", YELLOW)
        sys.exit(1)

    print_color("Starting to process configuration files...", BLUE)
    for tmp_file in config_files:
        count_total += 1
        target_file = tmp_file.replace("_tmp.yml", ".yml")
        sys.stdout.write(f"Processing [{count_total}]: {tmp_file} -> {target_file}\n")

        try:
            with open(tmp_file) as f:
                content = f.read()

            new_content = content.replace("${ASSETS_PATH}/Assets", str(assets_path))
            new_content = new_content.replace("$ASSETS_PATH/Assets", str(assets_path))

            with open(target_file, "w") as f:
                f.write(new_content)

            print_color(f"  ✓ Successfully replaced ${{ASSETS_PATH}} -> {assets_path}", GREEN)
            count_updated += 1

            if "${ASSETS_PATH}" in content and str(assets_path) in new_content:
                print_color("  ✓ Confirmed path was correctly replaced", GREEN)
            elif "${ASSETS_PATH}" in content:
                print_color("  ! Warning: Could not confirm if path was correctly replaced", YELLOW)

        except Exception as e:
            print_color(f"  ✗ Replacement failed: {e}", YELLOW)
            count_error += 1

    # Summary
    sys.stdout.write("\n")
    print_color("Processing complete!", BLUE)
    sys.stdout.write(f"Total processed: {count_total} files\n")
    print_color(f"Successfully updated: {count_updated} files", GREEN)
    if count_error > 0:
        print_color(f"Failed to process: {count_error} files", YELLOW)

    sys.stdout.write("\n")
    print_color("All template files have been processed!", GREEN)
    sys.stdout.write("To use in a new environment, run this script again\n")


if __name__ == "__main__":
    main()

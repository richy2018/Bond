#!/usr/bin/env python3
"""
Bloomberg Screenshot Table Extractor

Extracts tabular bond data from Bloomberg terminal screenshots using OCR.
Handles Bloomberg's characteristic orange text on black background.

Usage:
    python extract_bloomberg.py screenshot.png                  # single file
    python extract_bloomberg.py ./screenshots/                  # folder
    python extract_bloomberg.py ./screenshots/ -o output.csv    # custom output
    python extract_bloomberg.py screenshot.png --debug          # save preprocessed images
"""

import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Known Bloomberg column headers (in expected left-to-right order)
EXPECTED_HEADERS = [
    "Issuer Name",
    "G-Spread",
    "BVAL Bid Yld",
    "Cpn",
    "Maturity",
    "BBG Composite",
    "Currency",
    "ISIN",
    "Issuer 1-yr Default Prob",
    "EBITDA to Interest",
    "Net Debt to EBITDA",
    "Payment Rank",
    "OAS Sprd (Mid)",
    "OAS Eff Dur (Mid)",
    "Mid Yield to Convention",
    "Bid-Ask Spread",
    "Issuer Industry",
]

# Shorter aliases found in screenshot headers (map to canonical names)
HEADER_ALIASES = {
    "issuer name": "Issuer Name",
    "issuer": "Issuer Name",
    "g-spread": "G-Spread",
    "g spread": "G-Spread",
    "g-sprea": "G-Spread",
    "bval bid yld": "BVAL Bid Yld",
    "bval bid": "BVAL Bid Yld",
    "bval": "BVAL Bid Yld",
    "cpn": "Cpn",
    "maturity": "Maturity",
    "maturity bbg com": "Maturity",
    "bbg composite": "BBG Composite",
    "bbg com": "BBG Composite",
    "bbg compo": "BBG Composite",
    "curr": "Currency",
    "currency": "Currency",
    "isin": "ISIN",
    "issuer 1-yr d": "Issuer 1-yr Default Prob",
    "issuer 1-yr default": "Issuer 1-yr Default Prob",
    "issuer 1-yr": "Issuer 1-yr Default Prob",
    "1-yr default": "Issuer 1-yr Default Prob",
    "ebitda to inter": "EBITDA to Interest",
    "ebitda to int": "EBITDA to Interest",
    "ebitda to interest": "EBITDA to Interest",
    "ebitda": "EBITDA to Interest",
    "net debt to ebit": "Net Debt to EBITDA",
    "net debt to ebitda": "Net Debt to EBITDA",
    "net debt": "Net Debt to EBITDA",
    "payment rank": "Payment Rank",
    "payment": "Payment Rank",
    "oas sprd (mid)": "OAS Sprd (Mid)",
    "oas sprd": "OAS Sprd (Mid)",
    "oas eff dur": "OAS Eff Dur (Mid)",
    "oas eff dur (mid)": "OAS Eff Dur (Mid)",
    "oas eff": "OAS Eff Dur (Mid)",
    "mid yield to con": "Mid Yield to Convention",
    "mid yield to convention": "Mid Yield to Convention",
    "mid yield": "Mid Yield to Convention",
    "bid-ask spread": "Bid-Ask Spread",
    "bid-ask": "Bid-Ask Spread",
    "bid ask spread": "Bid-Ask Spread",
    "issuer industry": "Issuer Industry",
    "industry": "Issuer Industry",
}

# Numeric columns for validation
NUMERIC_COLUMNS = {
    "G-Spread", "BVAL Bid Yld", "Cpn", "Issuer 1-yr Default Prob",
    "EBITDA to Interest", "Net Debt to EBITDA", "OAS Sprd (Mid)",
    "OAS Eff Dur (Mid)", "Mid Yield to Convention", "Bid-Ask Spread",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# =============================================================================
# Image Preprocessing
# =============================================================================

def preprocess_bloomberg(image_path: str, debug: bool = False) -> np.ndarray:
    """
    Preprocess a Bloomberg screenshot for OCR.
    Isolates orange/amber text from black background and converts to
    high-contrast black-on-white image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Bloomberg orange/amber text HSV range
    # Hue: 5-35 (orange range), Sat: 80-255, Val: 130-255
    lower_orange = np.array([5, 80, 130])
    upper_orange = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Check if mask captured enough pixels (at least 1% of image)
    mask_ratio = np.count_nonzero(mask) / mask.size
    if mask_ratio < 0.005:
        # Fallback: broader color range or grayscale approach
        print(f"  Warning: Orange mask captured only {mask_ratio:.3%} of pixels, trying broader range...")
        lower_orange = np.array([0, 40, 100])
        upper_orange = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_ratio = np.count_nonzero(mask) / mask.size

        if mask_ratio < 0.005:
            # Final fallback: grayscale inversion
            print("  Warning: Color mask failed, falling back to grayscale threshold...")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Light dilation to reconnect broken characters
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Invert: OCR expects black text on white background
    result = cv2.bitwise_not(mask)

    # Scale up small images for better OCR
    h, w = result.shape[:2]
    if w < 2000:
        scale = 2.0
        result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if debug:
        debug_dir = Path(image_path).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        stem = Path(image_path).stem
        cv2.imwrite(str(debug_dir / f"{stem}_preprocessed.png"), result)
        cv2.imwrite(str(debug_dir / f"{stem}_mask.png"), mask)
        print(f"  Debug images saved to {debug_dir}/")

    return result


# =============================================================================
# OCR and Column Detection
# =============================================================================

def run_ocr(image: np.ndarray, reader) -> list:
    """Run EasyOCR on preprocessed image. Returns list of (bbox, text, confidence)."""
    results = reader.readtext(image, paragraph=False, width_ths=0.5)
    return results


def bbox_x_center(bbox) -> float:
    """Get x-center of a bounding box. bbox is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]."""
    xs = [pt[0] for pt in bbox]
    return (min(xs) + max(xs)) / 2


def bbox_x_left(bbox) -> float:
    """Get left x of a bounding box."""
    return min(pt[0] for pt in bbox)


def bbox_y_center(bbox) -> float:
    """Get y-center of a bounding box."""
    ys = [pt[1] for pt in bbox]
    return (min(ys) + max(ys)) / 2


def bbox_height(bbox) -> float:
    """Get height of a bounding box."""
    ys = [pt[1] for pt in bbox]
    return max(ys) - min(ys)


def match_header(text: str) -> str | None:
    """Try to match OCR text to a known Bloomberg header."""
    normalized = text.strip().lower()
    # Direct alias match
    if normalized in HEADER_ALIASES:
        return HEADER_ALIASES[normalized]
    # Fuzzy: check if any alias is contained in the text or vice versa
    for alias, canonical in HEADER_ALIASES.items():
        if alias in normalized or normalized in alias:
            return canonical
    return None


def detect_columns(detections: list, image_width: int) -> tuple[list[dict], list]:
    """
    Detect header row and compute column boundaries.

    Returns:
        (columns, data_detections) where columns is a list of
        {"name": str, "x_center": float, "x_left": float, "x_right": float}
        and data_detections are all non-header detections.
    """
    if not detections:
        return [], []

    # Sort by y position to find header row (topmost cluster)
    sorted_by_y = sorted(detections, key=lambda d: bbox_y_center(d[0]))

    # Find the header row: top detections that match known headers
    # Estimate row height from first few detections
    if len(sorted_by_y) >= 2:
        typical_height = np.median([bbox_height(d[0]) for d in sorted_by_y[:20]])
    else:
        typical_height = 20

    # Header candidates: detections in the top portion that match header names
    top_y = bbox_y_center(sorted_by_y[0][0])
    header_threshold = top_y + typical_height * 2.5  # Allow for multi-line headers

    header_detections = []
    data_detections = []
    matched_headers = {}  # canonical_name -> detection

    for det in sorted_by_y:
        bbox, text, conf = det
        y = bbox_y_center(bbox)
        if y <= header_threshold:
            header_name = match_header(text)
            if header_name and header_name not in matched_headers:
                matched_headers[header_name] = det
                header_detections.append(det)
            # Even unmatched top-row text might be part of headers (multi-word)
        else:
            data_detections.append(det)

    # Also check detections just below threshold that didn't match
    # (some header text might be on a second line)
    if len(matched_headers) < 5:
        # Try extending the search range
        extended_threshold = top_y + typical_height * 4
        for det in sorted_by_y:
            bbox, text, conf = det
            y = bbox_y_center(bbox)
            if header_threshold < y <= extended_threshold:
                header_name = match_header(text)
                if header_name and header_name not in matched_headers:
                    matched_headers[header_name] = det
                    header_detections.append(det)

    # Rebuild data_detections excluding all header detections
    header_set = set(id(d) for d in header_detections)
    data_detections = [d for d in sorted_by_y if id(d) not in header_set]

    if not matched_headers:
        print("  Warning: No headers detected. Using fallback column assignment.")
        return [], data_detections

    # Build column definitions sorted by x position
    columns = []
    for name, det in matched_headers.items():
        bbox = det[0]
        columns.append({
            "name": name,
            "x_center": bbox_x_center(bbox),
            "x_left": bbox_x_left(bbox),
        })
    columns.sort(key=lambda c: c["x_center"])

    # Compute column boundaries (midpoints between adjacent columns)
    for i, col in enumerate(columns):
        if i == 0:
            col["x_bound_left"] = 0
        else:
            col["x_bound_left"] = (columns[i - 1]["x_center"] + col["x_center"]) / 2
        if i == len(columns) - 1:
            col["x_bound_right"] = image_width
        else:
            col["x_bound_right"] = (col["x_center"] + columns[i + 1]["x_center"]) / 2

    print(f"  Detected {len(columns)} columns: {[c['name'] for c in columns]}")
    return columns, data_detections


def assign_to_column(x_pos: float, columns: list[dict]) -> str | None:
    """Assign an x-position to the closest column."""
    for col in columns:
        if col["x_bound_left"] <= x_pos <= col["x_bound_right"]:
            return col["name"]
    # Fallback: nearest column
    if columns:
        nearest = min(columns, key=lambda c: abs(c["x_center"] - x_pos))
        return nearest["name"]
    return None


def cluster_into_rows(detections: list, row_threshold: float = None) -> list[list]:
    """
    Group detections into rows based on y-position proximity.
    Returns list of rows, each row is a list of (bbox, text, confidence).
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: bbox_y_center(d[0]))

    # Estimate row height
    if row_threshold is None:
        heights = [bbox_height(d[0]) for d in sorted_dets[:30]]
        row_threshold = np.median(heights) * 0.8 if heights else 15

    rows = []
    current_row = [sorted_dets[0]]
    current_y = bbox_y_center(sorted_dets[0][0])

    for det in sorted_dets[1:]:
        y = bbox_y_center(det[0])
        if abs(y - current_y) <= row_threshold:
            current_row.append(det)
        else:
            rows.append(current_row)
            current_row = [det]
            current_y = y

    if current_row:
        rows.append(current_row)

    return rows


def build_row_dicts(rows: list[list], columns: list[dict]) -> list[dict]:
    """Convert clustered rows into list of dicts with column names as keys."""
    result = []

    for row_dets in rows:
        row = {col["name"]: None for col in columns}
        row_confidences = {}

        # Sort detections in this row by x position
        row_dets_sorted = sorted(row_dets, key=lambda d: bbox_x_center(d[0]))

        for bbox, text, conf in row_dets_sorted:
            x = bbox_x_center(bbox)
            col_name = assign_to_column(x, columns)
            if col_name:
                # If column already has a value, append (handles wrapped text)
                if row[col_name] is not None:
                    row[col_name] = row[col_name] + " " + text.strip()
                    row_confidences[col_name] = min(row_confidences.get(col_name, 1.0), conf)
                else:
                    row[col_name] = text.strip()
                    row_confidences[col_name] = conf

        # Store confidence info
        row["_confidences"] = row_confidences

        # Skip rows that are mostly empty (likely artifacts)
        non_empty = sum(1 for k, v in row.items() if k != "_confidences" and v is not None)
        if non_empty >= 2:
            result.append(row)

    return result


# =============================================================================
# Validation and Confidence
# =============================================================================

def fix_isin_ocr(text: str) -> str:
    """Fix common OCR errors in ISIN codes."""
    if not text or len(text) < 10:
        return text
    # First two chars should be letters
    fixed = list(text.upper().replace(" ", ""))
    for i in range(min(2, len(fixed))):
        if fixed[i] == '0':
            fixed[i] = 'O'
        elif fixed[i] == '1':
            fixed[i] = 'I'
    return "".join(fixed)


def validate_row(row: dict) -> tuple[dict, list[str]]:
    """
    Validate and clean a row. Returns (cleaned_row, list_of_flags).
    """
    flags = []
    cleaned = dict(row)
    confidences = cleaned.pop("_confidences", {})

    for col, val in cleaned.items():
        if col.startswith("_"):
            continue
        if val is None or val.strip() == "--" or val.strip() == "—" or val.strip() == "":
            cleaned[col] = ""
            continue

    # ISIN validation
    isin_val = cleaned.get("ISIN", "")
    if isin_val:
        isin_val = fix_isin_ocr(isin_val)
        cleaned["ISIN"] = isin_val
        if not re.match(r'^[A-Z]{2}[A-Z0-9]{9,11}$', isin_val):
            flags.append(f"invalid_isin:{isin_val}")

    # Numeric columns
    for col in NUMERIC_COLUMNS:
        val = cleaned.get(col, "")
        if val and val.strip():
            # Clean common OCR artifacts in numbers
            clean_val = val.strip()
            clean_val = clean_val.replace(",", "").replace(" ", "")
            clean_val = clean_val.replace("O", "0").replace("o", "0")
            clean_val = clean_val.replace("l", "1").replace("I", "1")
            clean_val = clean_val.replace("S", "5").replace("B", "8")
            # Keep only valid numeric characters
            clean_val = re.sub(r'[^0-9.\-]', '', clean_val)
            try:
                float(clean_val)
                cleaned[col] = clean_val
            except ValueError:
                flags.append(f"parse_error:{col}={val}")

    # Date validation for Maturity
    maturity = cleaned.get("Maturity", "")
    if maturity and maturity.strip():
        # Bloomberg typically shows MM/DD/YYYY
        date_match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', maturity.strip())
        if not date_match:
            # Try other date formats
            date_match2 = re.match(r'(\d{4})-(\d{2})-(\d{2})', maturity.strip())
            if not date_match2:
                flags.append(f"parse_error:Maturity={maturity}")

    # Currency validation
    currency = cleaned.get("Currency", "")
    if currency and currency.strip():
        valid_currencies = {"USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", "SEK", "NOK", "DKK", "HKD", "SGD", "BRL", "MXN"}
        if currency.strip().upper() not in valid_currencies:
            flags.append(f"unknown_currency:{currency}")

    # Min OCR confidence
    min_conf = min(confidences.values()) if confidences else 0.0
    if min_conf < 0.5:
        flags.append(f"low_ocr_confidence:{min_conf:.2f}")

    cleaned["_min_ocr_confidence"] = round(min_conf, 3) if confidences else 0.0
    cleaned["_flags"] = "; ".join(flags) if flags else ""

    return cleaned, flags


# =============================================================================
# Main Processing
# =============================================================================

def process_image(image_path: str, reader, debug: bool = False) -> list[dict]:
    """Process a single Bloomberg screenshot. Returns list of row dicts."""
    print(f"\nProcessing: {image_path}")

    # Preprocess
    processed = preprocess_bloomberg(image_path, debug=debug)

    # OCR
    print("  Running OCR...")
    detections = run_ocr(processed, reader)
    print(f"  Found {len(detections)} text detections")

    if not detections:
        print("  Warning: No text detected in image")
        return []

    # Get image dimensions (of preprocessed image)
    h, w = processed.shape[:2]

    # Detect columns from headers
    columns, data_detections = detect_columns(detections, w)

    if not columns:
        print("  Warning: Could not detect columns, skipping image")
        return []

    # Cluster into rows
    rows = cluster_into_rows(data_detections)
    print(f"  Detected {len(rows)} data rows")

    # Build row dicts
    row_dicts = build_row_dicts(rows, columns)

    # Validate each row
    validated = []
    flagged_count = 0
    for row in row_dicts:
        cleaned, flags = validate_row(row)
        cleaned["_source_file"] = Path(image_path).name
        validated.append(cleaned)
        if flags:
            flagged_count += 1

    print(f"  Extracted {len(validated)} rows ({flagged_count} flagged for review)")
    return validated


def collect_images(input_path: str) -> list[str]:
    """Collect image file paths from a file or directory."""
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            return [str(p)]
        else:
            print(f"Error: {input_path} is not a supported image format")
            return []
    elif p.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(p.glob(f"*{ext}"))
            images.extend(p.glob(f"*{ext.upper()}"))
        images = sorted(set(str(i) for i in images))
        if not images:
            print(f"Error: No image files found in {input_path}")
        return images
    else:
        print(f"Error: {input_path} does not exist")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract tabular bond data from Bloomberg terminal screenshots"
    )
    parser.add_argument(
        "input",
        help="Path to a screenshot image or a folder containing screenshots"
    )
    parser.add_argument(
        "-o", "--output",
        default="bloomberg_extract.csv",
        help="Output CSV file path (default: bloomberg_extract.csv)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save preprocessed images for debugging"
    )
    args = parser.parse_args()

    # Collect images
    images = collect_images(args.input)
    if not images:
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process")

    # Initialize EasyOCR (once for all images)
    import easyocr
    print("Initializing OCR engine...")
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    # Process all images
    all_rows = []
    for img_path in images:
        rows = process_image(img_path, reader, debug=args.debug)
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo data extracted from any images.")
        sys.exit(1)

    # Build DataFrame with consistent column order
    output_columns = ["_source_file"] + EXPECTED_HEADERS + ["_min_ocr_confidence", "_flags"]
    df = pd.DataFrame(all_rows)

    # Ensure all expected columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = ""

    # Reorder columns (only include columns that exist)
    final_columns = [c for c in output_columns if c in df.columns]
    df = df[final_columns]

    # Write CSV
    df.to_csv(args.output, index=False)

    # Summary
    total = len(df)
    flagged = len(df[df["_flags"] != ""])
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Total rows: {total}")
    print(f"  Flagged for review: {flagged}")
    print(f"  Output: {args.output}")
    if flagged:
        print(f"\n  Rows with flags (check _flags column):")
        for _, row in df[df["_flags"] != ""].iterrows():
            issuer = row.get("Issuer Name", "Unknown")
            print(f"    - {issuer}: {row['_flags']}")


if __name__ == "__main__":
    main()

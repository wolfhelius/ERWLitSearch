#!/usr/bin/env python3
"""
Keyword Analysis for ERW Literature

This script analyzes PDF files in the ./doc folder to count occurrences of specific keywords
related to enhanced rock weathering (ERW) research, particularly focusing on toxicity and
geological material keywords.

Keywords analyzed:
- Toxicity: toxic*, hazard*, harm*, nickel, chromium
- Geological: dunite, olivine, ultramafic*, serpentine, mafic*, basalt, wollastonite

The script uses the first 4 digits of PDF filenames as unique identifiers (UIDs) to match
with records in the merged_outer_openalex.csv dataset.
"""

import os
import re
import pandas as pd
import PyPDF2
import fitz  # PyMuPDF - better text extraction than PyPDF2
from pathlib import Path
import argparse
import sys
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeywordAnalyzer:
    """Analyzes PDF files for keyword occurrences in ERW literature."""

    def __init__(self, data_path='./dat/merged_outer_openalex.csv', doc_path='./doc'):
        """Initialize the analyzer with data and document paths."""
        self.data_path = Path(data_path)
        self.doc_path = Path(doc_path)

        # Define keywords to search for
        self.keywords = {
            # Toxicity-related keywords
            'toxic': r'\btoxic\w*\b',
            'hazard': r'\bhazard\w*\b',
            'harm': r'\bharm\w*\b',
            'nickel': r'\bnickel\b',
            'chromium': r'\bchromium\b',

            # Geological material keywords
            'dunite': r'\bdunite\b',
            'olivine': r'\bolivine\b',
            'ultramafic': r'\bultramafic\w*\b',
            'serpentine': r'\bserpentine\b',
            'mafic': r'\bmafic\w*\b',
            'basalt': r'\bbasalt\b',
            'wollastonite': r'\bwollastonite\b'
        }

        # Case-sensitive chemical symbols (searched separately)
        self.case_sensitive_keywords = {
            'Ni': r'\bNi\b',
            'Cr': r'\bCr\b'
        }

        self.df = None
        self.pdf_files = []

    def load_data(self) -> None:
        """Load the merged dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with {len(self.df)} records")

            # Ensure number column exists and is properly formatted
            if 'number' not in self.df.columns:
                raise ValueError("Dataset must contain 'number' column")

            # Convert number to int for matching, handle NaN values
            self.df['number'] = pd.to_numeric(self.df['number'], errors='coerce')
            # Fill NaN values with -1 to avoid issues
            self.df['number'] = self.df['number'].fillna(-1).astype(int)

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def find_pdf_files(self) -> None:
        """Find all PDF files in the document directory."""
        if not self.doc_path.exists():
            raise FileNotFoundError(f"Document directory not found: {self.doc_path}")

        self.pdf_files = sorted(list(self.doc_path.glob("*.pdf")))
        logger.info(f"Found {len(self.pdf_files)} PDF files")

        if not self.pdf_files:
            logger.warning("No PDF files found in document directory")

    def extract_uid_from_filename(self, filename: str) -> int:
        """Extract the 4-digit UID from PDF filename."""
        try:
            logger.debug(f"Processing filename: {repr(filename)} (type: {type(filename)})")

            # Ensure filename is a string
            if not isinstance(filename, str):
                logger.error(f"Filename is not a string: {type(filename)}")
                return None

            # Extract first 4 digits before underscore
            match = re.match(r'^(\d{4})_', filename)
            if match:
                return int(match.group(1))
            else:
                # Try just the first 4 characters if they're digits
                if len(filename) >= 4 and filename[:4].isdigit():
                    return int(filename[:4])
            return None
        except Exception as e:
            logger.error(f"Error extracting UID from filename {repr(filename)}: {e}")
            return None

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        text = ""
        try:
            # Try PyMuPDF first (usually better)
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            doc.close()

        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path.name}, trying PyPDF2: {e}")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            except Exception as e2:
                logger.error(f"Both PDF readers failed for {pdf_path.name}: {e2}")
                return ""

        return text

    def count_keywords(self, text: str) -> Dict[str, int]:
        """Count occurrences of all keywords in text (case-insensitive and case-sensitive)."""
        text_lower = text.lower()
        counts = {}

        # Case-insensitive keywords
        for keyword, pattern in self.keywords.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            counts[keyword] = len(matches)

        # Case-sensitive keywords
        for keyword, pattern in self.case_sensitive_keywords.items():
            matches = re.findall(pattern, text)  # No case conversion
            counts[keyword] = len(matches)

        return counts

    def analyze_single_pdf(self, pdf_path: Path) -> Tuple[int, Dict[str, int]]:
        """Analyze a single PDF file and return UID and keyword counts."""
        try:
            filename = pdf_path.name
            logger.debug(f"Extracting UID from filename: {filename}")
            uid = self.extract_uid_from_filename(filename)
        except Exception as e:
            logger.error(f"Error in analyze_single_pdf with {pdf_path}: {e}")
            raise

        if uid is None:
            logger.warning(f"Could not extract UID from filename: {filename}")
            return None, {}

        logger.info(f"Analyzing {filename} (UID: {uid})")

        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)

        if not text.strip():
            logger.warning(f"No text extracted from {filename}")
            return uid, {keyword: 0 for keyword in self.keywords.keys()}

        # Count keywords
        keyword_counts = self.count_keywords(text)

        # Log interesting findings
        total_keywords = sum(keyword_counts.values())
        if total_keywords > 0:
            found_keywords = [k for k, v in keyword_counts.items() if v > 0]
            logger.info(f"  Found keywords: {found_keywords}")

        return uid, keyword_counts

    def initialize_keyword_columns(self) -> None:
        """Add keyword columns to the dataframe."""
        # Initialize keyword columns
        for keyword in self.keywords.keys():
            self.df[keyword] = 0

        # Initialize case-sensitive keyword columns
        for keyword in self.case_sensitive_keywords.keys():
            self.df[keyword] = 0

        # Initialize analysis tracking column
        self.df['keyword_analyzed'] = 0

        # Initialize cdrlit_meth column - 1 if methmax has a value, 0 otherwise
        if 'methmax' in self.df.columns:
            self.df['cdrlit_meth'] = self.df['methmax'].notna().astype(int)
            logger.info(f"Created cdrlit_meth column based on methmax availability")
        else:
            self.df['cdrlit_meth'] = 0
            logger.warning("No 'methmax' column found in dataset")

    def analyze_all_pdfs(self) -> None:
        """Analyze all PDF files and update the dataframe."""
        logger.info("Starting keyword analysis of all PDFs...")

        # Initialize keyword columns
        self.initialize_keyword_columns()

        processed = 0
        matched = 0
        total_pdfs = len(self.pdf_files)

        for i, pdf_path in enumerate(self.pdf_files, 1):
            try:
                logger.info(f"Processing file {i}/{total_pdfs}: {pdf_path.name}")
                uid, keyword_counts = self.analyze_single_pdf(pdf_path)
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue

            if uid is not None:
                processed += 1

                # Find matching row in dataframe
                mask = self.df['number'] == uid
                matching_rows = self.df[mask]

                if len(matching_rows) == 1:
                    matched += 1
                    # Update keyword counts for this record
                    for keyword, count in keyword_counts.items():
                        self.df.loc[mask, keyword] = count

                    # Mark as analyzed
                    self.df.loc[mask, 'keyword_analyzed'] = 1

                elif len(matching_rows) > 1:
                    logger.warning(f"Multiple records found for UID {uid}")
                else:
                    logger.warning(f"No record found for UID {uid}")

        logger.info(f"Analysis complete: {processed} PDFs processed, {matched} matched to records")

    def save_results(self, output_path='./dat/erwlit_keywords.csv') -> None:
        """Save the updated dataframe with keyword counts."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    def generate_summary_report(self) -> None:
        """Generate a summary report of keyword findings."""
        logger.info("\n" + "="*50)
        logger.info("KEYWORD ANALYSIS SUMMARY REPORT")
        logger.info("="*50)

        # Overall statistics
        total_records = len(self.df)
        records_with_keywords = len(self.df[self.df[list(self.keywords.keys())].sum(axis=1) > 0])

        logger.info(f"Total records: {total_records}")
        logger.info(f"Records with keywords: {records_with_keywords}")
        logger.info(f"Coverage: {records_with_keywords/total_records*100:.1f}%")

        logger.info("\nKeyword frequencies:")
        all_keywords = list(self.keywords.keys()) + list(self.case_sensitive_keywords.keys())
        for keyword in all_keywords:
            total_count = self.df[keyword].sum()
            records_with_keyword = len(self.df[self.df[keyword] > 0])
            logger.info(f"  {keyword}: {total_count} total occurrences in {records_with_keyword} records")

        # Analysis statistics
        analyzed_records = self.df['keyword_analyzed'].sum()
        logger.info(f"\nAnalysis coverage: {analyzed_records} records analyzed")

        # CDR method statistics
        cdr_meth_records = self.df['cdrlit_meth'].sum()
        logger.info(f"Records with CDR methods: {cdr_meth_records}")

        # Top records by keyword density
        logger.info("\nTop 5 records by total keyword count:")
        all_keywords = list(self.keywords.keys()) + list(self.case_sensitive_keywords.keys())
        self.df['total_keywords'] = self.df[all_keywords].sum(axis=1)

        # Handle missing columns gracefully
        display_columns = ['number', 'total_keywords']
        if 'author_short' in self.df.columns:
            display_columns.append('author_short')
        if 'title' in self.df.columns:
            display_columns.append('title')

        top_records = self.df.nlargest(5, 'total_keywords')[display_columns]

        for _, record in top_records.iterrows():
            author_info = record.get('author_short', 'Unknown')
            title_info = record.get('title', 'No title')
            # Ensure all values are strings to avoid concatenation errors
            author_str = str(author_info) if pd.notna(author_info) else 'Unknown'
            title_str = str(title_info) if pd.notna(title_info) else 'No title'
            logger.info(f"  UID {int(record['number'])}: {author_str} - {int(record['total_keywords'])} keywords")

    def run_analysis(self, output_path='./dat/erwlit_keywords.csv') -> None:
        """Run the complete keyword analysis pipeline."""
        try:
            logger.info("Starting ERW Literature Keyword Analysis")

            # Load data and find PDFs
            self.load_data()
            self.find_pdf_files()

            if not self.pdf_files:
                logger.error("No PDF files found. Exiting.")
                return

            # Run analysis
            self.analyze_all_pdfs()

            # Save results and generate report
            self.save_results(output_path)
            self.generate_summary_report()

            logger.info("Analysis completed successfully!")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='Analyze ERW literature PDFs for keyword occurrences')
    parser.add_argument('--data', default='./dat/merged_outer_openalex.csv',
                        help='Path to merged dataset CSV file')
    parser.add_argument('--docs', default='./doc',
                        help='Path to directory containing PDF files')
    parser.add_argument('--output', default='./dat/erwlit_keywords.csv',
                        help='Path for output CSV file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        analyzer = KeywordAnalyzer(args.data, args.docs)
        analyzer.run_analysis(args.output)

    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

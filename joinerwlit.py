#!/usr/bin/env python3
"""
Literature Database Merger

This script merges two literature datasets:
1. CDR Literature from https://climateliterature.org/#/project/cdrmap
2. ERW Literature from Google Sheets

Merge Convention:
- ERW Literature serves as the LEFT dataset (primary/base)
- CDR Literature serves as the RIGHT dataset (matched against ERW)
- Result: ERW records with matched CDR data merged in, plus unmatched CDR records

Column Mapping:
- ERW: 'number', 'author_short', 'year', 'DOI/ISSN/ISBN', 'title'
- CDR: 'idx', 'authors', 'publication_year', 'doi', 'title', 'openalex_id'

The script performs fuzzy matching on DOI, title, and author fields,
and optionally backfills missing DOIs from OpenAlex API.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz, process


def normalize_doi(doi_series):
    """Normalize DOI strings by removing URLs and converting to lowercase."""
    return (doi_series
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace('https://doi.org/', '', regex=False))


def is_doi_missing(doi_series):
    """Check if DOI is missing or empty."""
    return doi_series.isna() | (doi_series.astype(str).str.strip() == '')


class LiteratureMerger:
    """Main class for merging literature datasets."""

    def __init__(self, cdr_path='./dat/cdrlit_lueck.csv', erw_path='./dat/cdrlit_suhrhoff.csv'):
        """Initialize with input file paths."""
        self.cdr_path = Path(cdr_path)
        self.erw_path = Path(erw_path)
        self.cdrlit = None
        self.erw_lit = None
        self.merged_df = None

    def load_data(self):
        """Load the input CSV files."""
        print("📚 Loading datasets...")

        if not self.cdr_path.exists():
            raise FileNotFoundError(f"CDR literature file not found: {self.cdr_path}")
        if not self.erw_path.exists():
            raise FileNotFoundError(f"ERW literature file not found: {self.erw_path}")

        self.cdrlit = pd.read_csv(self.cdr_path)
        self.erw_lit = pd.read_csv(self.erw_path)

        print(f"  ✓ CDR Literature: {len(self.cdrlit)} records")
        print(f"  ✓ ERW Literature: {len(self.erw_lit)} records")

    def normalize_dois(self):
        """Normalize DOI strings for comparison."""
        print("🔧 Normalizing DOI fields...")

        self.cdrlit['doi_norm'] = normalize_doi(self.cdrlit['doi'])
        self.erw_lit['doi_norm'] = normalize_doi(self.erw_lit['DOI/ISSN/ISBN'])

        # Track which cdrlit rows were matched
        self.cdrlit['matched'] = False

    def find_best_match(self, row, cdrlit_df, title_thresh=90, author_thresh=85):
        """
        Find the best match for a row using DOI, title, and author matching.

        Tries matching in order of preference:
        1. Exact DOI match
        2. Fuzzy title match (token set ratio)
        3. Fuzzy author match (partial ratio)
        """
        # 1. Exact DOI match (only if DOI is actually present and not empty)
        if (pd.notnull(row['doi_norm']) and
            isinstance(row['doi_norm'], str) and
            row['doi_norm'].strip() != '' and
            row['doi_norm'].strip().lower() != 'nan'):

            exact_match = cdrlit_df[
                (cdrlit_df['doi_norm'] == row['doi_norm']) &
                (cdrlit_df['doi_norm'].notna()) &
                (cdrlit_df['doi_norm'].astype(str).str.strip() != '') &
                (cdrlit_df['doi_norm'].astype(str).str.strip().str.lower() != 'nan')
            ]
            if not exact_match.empty:
                return exact_match.iloc[0], 'doi'

        # 2. Fuzzy title match
        if pd.notnull(row['title']):
            best_title = process.extractOne(
                row['title'],
                cdrlit_df['title'],
                scorer=fuzz.token_set_ratio
            )
            if best_title and best_title[1] >= title_thresh:
                match_row = cdrlit_df[cdrlit_df['title'] == best_title[0]]
                if not match_row.empty:
                    return match_row.iloc[0], 'title'

        # 3. Fuzzy author match
        if pd.notnull(row['author_short']):
            best_author = process.extractOne(
                row['author_short'],
                cdrlit_df['authors'].astype(str),
                scorer=fuzz.partial_ratio
            )
            if best_author and best_author[1] >= author_thresh:
                match_row = cdrlit_df[cdrlit_df['authors'] == best_author[0]]
                if not match_row.empty:
                    return match_row.iloc[0], 'author'

        return None, None

    def merge_datasets(self):
        """
        Perform the main merge operation.

        Convention: ERW literature (left) is matched against CDR literature (right)
        - ERW records become the base, with CDR data merged in when matches are found
        - Unmatched CDR records are added as separate rows
        """
        print("🔗 Merging datasets...")

        merged_rows = []
        match_counts = {'doi': 0, 'title': 0, 'author': 0, 'none': 0}

        # For each ERW record (left), try to find a matching CDR record (right)
        for i, (_, row) in enumerate(self.erw_lit.iterrows()):
            if i % 100 == 0:
                print(f"  Processing ERW record {i+1}/{len(self.erw_lit)}")

            match, match_type = self.find_best_match(row, self.cdrlit[~self.cdrlit['matched']])
            left_part = row.to_dict()

            if match is not None:
                right_part = match.drop(['doi_norm', 'matched']).to_dict()
                self.cdrlit.loc[match.name, 'matched'] = True
                left_part['match_type'] = match_type
                match_counts[match_type] += 1
            else:
                right_part = {}
                left_part['match_type'] = None
                match_counts['none'] += 1

            merged_rows.append({**left_part, **right_part})

        # Handle unmatched cdrlit rows
        unmatched_cdrlit = self.cdrlit[~self.cdrlit['matched']].drop(columns=['doi_norm', 'matched'])

        for _, row in unmatched_cdrlit.iterrows():
            combined_row = {**{col: None for col in self.erw_lit.columns}, **row.to_dict()}
            combined_row['match_type'] = None
            merged_rows.append(combined_row)

        self.merged_df = pd.DataFrame(merged_rows)

        print(f"  ✓ Matches by DOI: {match_counts['doi']}")
        print(f"  ✓ Matches by title: {match_counts['title']}")
        print(f"  ✓ Matches by author: {match_counts['author']}")
        print(f"  ✓ No matches: {match_counts['none']}")
        print(f"  ✓ Unmatched CDR records: {len(unmatched_cdrlit)}")
        print(f"  ✓ Total merged records: {len(self.merged_df)}")

    def assign_synthetic_numbers(self):
        """
        Assign synthetic number IDs where missing.

        Logic:
        1. ERW records keep their original 'number' column values
        2. Unmatched CDR records (no 'number' but have 'idx') get: 1e5 + idx
        3. Fallback for any remaining records: 1e6 + row_index
        """
        print("🔢 Assigning synthetic numbers...")

        # For unmatched CDR records: where 'number' is missing but 'idx' is present
        # These are CDR records that didn't match any ERW record
        mask = self.merged_df['number'].isna() & self.merged_df['idx'].notna()
        self.merged_df.loc[mask, 'number'] = (1e5 + self.merged_df.loc[mask, 'idx']).astype(int)

        # Fallback for any remaining records without numbers (shouldn't happen but safety net)
        fallback_mask = self.merged_df['number'].isna()
        self.merged_df.loc[fallback_mask, 'number'] = (1e6 + self.merged_df[fallback_mask].index).astype(int)

    def format_authors(self, auth_string):
        """Format author string for display."""
        if pd.isna(auth_string) or not isinstance(auth_string, str):
            return None

        authors = [a.strip() for a in auth_string.split(';') if a.strip()]
        if len(authors) == 0:
            return None

        formatted = []
        for a in authors[:2]:  # Only format first 2 authors
            parts = a.strip().split(',')
            if len(parts) == 2:
                # "Last, First" format
                last = parts[0].strip()
                first = parts[1].strip()
            else:
                # "First Last" format
                tokens = a.strip().split()
                if len(tokens) < 2:
                    continue
                last = tokens[-1]
                first = tokens[0]

            # Create initials
            initials = ''.join([f"{x[0]}." for x in first.strip().split() if x])
            formatted.append(f"{last}, {initials}")

        if len(authors) <= 2:
            return ' & '.join(formatted)
        else:
            return f"{formatted[0]} et al."

    def fill_missing_fields(self):
        """Fill missing fields from available data."""
        print("📝 Filling missing fields...")

        # Fill author_short from authors where missing
        missing_author = self.merged_df['author_short'].isna()
        self.merged_df.loc[missing_author, 'author_short'] = (
            self.merged_df.loc[missing_author, 'authors'].apply(self.format_authors)
        )

        # Fill year from publication_year where missing
        missing_year = self.merged_df['year'].isna()
        self.merged_df.loc[missing_year, 'year'] = self.merged_df.loc[missing_year, 'publication_year']

        # Handle DOI fields - ensure we preserve DOI info from either source
        # Fill DOI/ISSN/ISBN from doi where missing (for CDR-only records)
        missing_doi_field = self.merged_df['DOI/ISSN/ISBN'].isna()
        self.merged_df.loc[missing_doi_field, 'DOI/ISSN/ISBN'] = self.merged_df.loc[missing_doi_field, 'doi']

        # Create normalized DOI field from the best available source
        self.merged_df['doi_norm'] = normalize_doi(self.merged_df['DOI/ISSN/ISBN'])

        # Create hyperlinks for all records with valid DOI data
        self.merged_df['Hyperlink'] = 'https://doi.org/' + self.merged_df['doi_norm']

        # Clear DOI fields only where there's no valid DOI data from either source
        no_valid_doi = is_doi_missing(self.merged_df['DOI/ISSN/ISBN'])
        self.merged_df.loc[no_valid_doi, ['DOI/ISSN/ISBN', 'doi_norm', 'Hyperlink']] = ''

    def clean_abstract(self, abstract_text):
        """Clean HTML tags and redundant text from abstract field."""
        if pd.isna(abstract_text) or not isinstance(abstract_text, str):
            return abstract_text

        import re

        # Remove HTML tags like <strong class="journal-contentHeaderColor">Abstract.</strong>
        cleaned = re.sub(r'<[^>]+>', '', abstract_text)

        # Remove "Abstract" prefix variations at the beginning
        cleaned = re.sub(r'^(Abstract\.?\s*|Abstract\s+)', '', cleaned.strip(), flags=re.IGNORECASE)

        return cleaned.strip()

    def final_cleanup(self):
        """Perform final data cleanup and column management."""
        print("🧹 Performing final cleanup...")

        # Clean abstract field if it exists
        if 'abstract' in self.merged_df.columns:
            self.merged_df['abstract'] = self.merged_df['abstract'].apply(self.clean_abstract)

        # Rename doi_norm to doi before dropping other columns
        if 'doi_norm' in self.merged_df.columns:
            self.merged_df = self.merged_df.rename(columns={'doi_norm': 'doi'})

        # Drop redundant DOI columns (keeping the renamed doi_norm as doi)
        columns_to_drop = []
        if 'DOI/ISSN/ISBN' in self.merged_df.columns:
            columns_to_drop.append('DOI/ISSN/ISBN')
        if 'publication_year' in self.merged_df.columns:
            columns_to_drop.append('publication_year')

        # Also drop the original 'doi' column from CDR data (we're keeping the normalized version)
        # But first check if we already renamed doi_norm to doi
        if 'doi_norm' not in self.merged_df.columns and 'doi' in self.merged_df.columns:
            # doi_norm was already renamed, don't drop doi
            pass
        elif 'doi' in self.merged_df.columns:
            # We still have original doi column, drop it
            columns_to_drop.append('doi')

        if columns_to_drop:
            self.merged_df = self.merged_df.drop(columns=columns_to_drop)

        # Reorder columns to match specified order
        self.reorder_columns()

    def reorder_columns(self):
        """Reorder columns to match the specified final column order."""
        print("📋 Reordering columns...")

        # Define the desired column order
        desired_order = [
            'number', 'idx', 'author_short', 'year', 'journal_short', 'doi',
            'Hyperlink', 'openalex_id', 'authors', 'title', 'institutions',
            'abstract', 'match_type', 'techmax', 'methmax', 'contmax',
            'tech|00', 'tech|01', 'tech|02', 'tech|03', 'tech|04', 'tech|05',
            'tech|06', 'tech|07', 'tech|08', 'tech|09', 'tech|10', 'tech|11',
            'tech|12', 'tech|13', 'tech|14', 'tech|15', 'tech|16',
            'meth|00', 'meth|01', 'meth|02', 'meth|03', 'meth|04', 'meth|05',
            'meth|06', 'meth|07', 'meth|08', 'meth|09',
            'cont|0', 'cont|1', 'cont|2', 'cont|3', 'cont|4', 'cont|5'
        ]

        # Get existing columns that are in our desired order
        existing_columns = [col for col in desired_order if col in self.merged_df.columns]

        # Add any remaining columns that weren't in our desired order (shouldn't happen but safety net)
        remaining_columns = [col for col in self.merged_df.columns if col not in existing_columns]

        # Final column order
        final_order = existing_columns + remaining_columns

        # Reorder the DataFrame
        self.merged_df = self.merged_df[final_order]

        print(f"  ✓ Reordered to {len(final_order)} columns")

    def fetch_doi_from_openalex(self, openalex_id):
        """Fetch DOI from OpenAlex API."""
        if pd.isna(openalex_id) or not isinstance(openalex_id, str):
            return None

        openalex_id_short = openalex_id.strip().split('/')[-1]  # Extract W...
        url = f'https://api.openalex.org/works/{openalex_id_short}'

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('doi')
        except Exception as e:
            print(f"  ⚠️  Error fetching {openalex_id_short}: {e}")

        return None

    def backfill_dois_from_openalex(self):
        """Backfill missing DOIs from OpenAlex API."""
        doi_missing = is_doi_missing(self.merged_df['doi'])
        has_openalex = self.merged_df['openalex_id'].notna()
        to_update = self.merged_df[doi_missing & has_openalex].index

        if len(to_update) == 0:
            print("ℹ️  No records need DOI backfill from OpenAlex.")
            return

        print(f"🔎 Backfilling {len(to_update)} DOIs from OpenAlex...")

        success_count = 0
        for i, idx in enumerate(to_update):
            if i % 10 == 0:
                print(f"  Progress: {i+1}/{len(to_update)} "
                      f"({(i + 1)/len(to_update)*100:.1f}%) - "
                      f"Found {success_count} DOIs so far")

            openalex_id = self.merged_df.at[idx, 'openalex_id']
            doi = self.fetch_doi_from_openalex(openalex_id)

            if doi:
                success_count += 1
                self.merged_df.at[idx, 'doi'] = doi
                self.merged_df.at[idx, 'DOI/ISSN/ISBN'] = doi
                self.merged_df.at[idx, 'doi_norm'] = doi.lower().replace('https://doi.org/', '').strip()
                self.merged_df.at[idx, 'Hyperlink'] = f"https://doi.org/{self.merged_df.at[idx, 'doi_norm']}"

            time.sleep(0.2)  # Respectful rate limiting

        print(f"  ✓ Successfully backfilled {success_count}/{len(to_update)} DOIs")

    def save_output(self, output_path='./dat/merged_outer_clean.csv'):
        """Save the merged dataset to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.merged_df.to_csv(output_path, index=False)
        print(f"💾 Saved merged dataset to: {output_path}")
        print(f"  Total records: {len(self.merged_df)}")

    def run(self, openalex_backfill=False):
        """Run the complete merge process."""
        print("🚀 Starting literature merge process...\n")

        self.load_data()
        self.normalize_dois()
        self.merge_datasets()
        self.assign_synthetic_numbers()
        self.fill_missing_fields()

        if openalex_backfill:
            self.backfill_dois_from_openalex()
        else:
            print("ℹ️  Skipping OpenAlex DOI backfill (use --openalex-backfill to enable)")

        self.final_cleanup()
        self.save_output()
        print("\n✅ Literature merge completed successfully!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Merge CDR and ERW literature datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Basic merge without OpenAlex backfill
  %(prog)s --openalex-backfill       # Include OpenAlex DOI backfill
  %(prog)s --cdr ./data/cdr.csv      # Use custom CDR file path
        """
    )

    parser.add_argument(
        '--cdr-path',
        default='./dat/cdrlit_lueck.csv',
        help='Path to CDR literature CSV file (default: ./dat/cdrlit_lueck.csv)'
    )

    parser.add_argument(
        '--erw-path',
        default='./dat/cdrlit_suhrhoff.csv',
        help='Path to ERW literature CSV file (default: ./dat/cdrlit_suhrhoff.csv)'
    )

    parser.add_argument(
        '--output-path',
        default='./dat/cdrlit_merged.csv',
        help='Path for output CSV file (default: ./dat/cdrlit_merged.csv)'
    )

    parser.add_argument(
        '--openalex-backfill',
        action='store_true',
        help='Enable DOI backfill from OpenAlex API (slow but comprehensive)'
    )

    args = parser.parse_args()

    try:
        merger = LiteratureMerger(args.cdr_path, args.erw_path)
        merger.run(openalex_backfill=args.openalex_backfill)

        # Print summary statistics
        match_stats = merger.merged_df['match_type'].value_counts()
        print("\n📊 Matching Statistics:")
        for match_type, count in match_stats.items():
            print(f"  {match_type or 'No match'}: {count:,}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

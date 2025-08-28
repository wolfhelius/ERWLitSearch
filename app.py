import streamlit as st
import pandas as pd
import urllib.parse
import os
import requests

# Cache OpenAlex API calls to avoid repeated requests
@st.cache_data
def fetch_openalex_landing_page(openalex_id):
    """Fetch landing page URL from OpenAlex API"""
    if pd.isna(openalex_id) or not isinstance(openalex_id, str):
        return None

    # Extract W... identifier from full URL if needed
    openalex_id_short = openalex_id.strip().split('/')[-1]
    url = f'https://api.openalex.org/works/{openalex_id_short}'

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            locations = data.get('locations', [])
            if locations and len(locations) > 0:
                # Get the primary location's landing page URL
                return locations[0].get('landing_page_url')
    except Exception as e:
        st.error(f"Error fetching OpenAlex data for {openalex_id_short}: {e}")

    return None

# Load and enhance DataFrame with OpenAlex data
@st.cache_data
def load_and_enhance_data():
    """Load data and add OpenAlex landing pages, with persistent caching"""

    # Try to load enhanced version first
    enhanced_file = "./dat/cdrlit_merged.csv"
    base_file = "./dat/cdrlit_merged.csv"

    if os.path.exists(enhanced_file):
        st.info("Loading cached OpenAlex data...")
        df = pd.read_csv(enhanced_file)
    else:
        st.info("Loading base data and fetching OpenAlex links (this may take a while)...")
        df = pd.read_csv(base_file)

        # Add OpenAlex landing page column
        df['openalex_landing_page'] = None

        # Get records with OpenAlex IDs
        openalex_records = df[df['openalex_id'].notna()].copy()

        if len(openalex_records) > 0:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, idx in enumerate(openalex_records.index):
                progress = (i + 1) / len(openalex_records)
                progress_bar.progress(progress)
                status_text.text(f"Fetching OpenAlex data: {i+1}/{len(openalex_records)}")

                openalex_id = openalex_records.at[idx, 'openalex_id']
                landing_page = fetch_openalex_landing_page(openalex_id)
                df.at[idx, 'openalex_landing_page'] = landing_page

            # Save enhanced version for future use
            df.to_csv(enhanced_file, index=False)
            st.success(f"Saved enhanced data to {enhanced_file}")

            progress_bar.empty()
            status_text.empty()

    # Clean up duplicate doi columns if they exist
    if 'doi' in df.columns:
        df['doi'] = df['doi'].astype(str).str.strip()
        # Remove any duplicate doi columns
        doi_cols = [col for col in df.columns if col == 'doi']
        if len(doi_cols) > 1:
            cols_to_keep = []
            doi_kept = False
            for col in df.columns:
                if col == 'doi' and not doi_kept:
                    cols_to_keep.append(col)
                    doi_kept = True
                elif col != 'doi':
                    cols_to_keep.append(col)
            df = df[cols_to_keep]

    return df

# Streamlit app configuration
st.set_page_config(page_title="Literature PDF Fetcher", layout="wide")

# Load data
df = load_and_enhance_data()

# App title and description
st.title("Literature PDF Fetcher")
st.markdown("Find PDFs via Sci-Hub, Google Scholar, and OpenAlex")

# Sci-Hub mirror selector
scihub_mirrors = [
    "https://sci-hub.se/",
    "https://sci-hub.ru/",
    "https://sci-hub.st/",
    "https://sci-hub.wf/",
    "https://sci-hub.hkvisa.net/",
    "https://scihub2024.vercel.app/"
]

# Mirror testing function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def test_mirror(url):
    try:
        r = requests.get(url, timeout=5)
        return r.status_code == 200
    except:
        return False

with st.spinner("Testing Sci-Hub mirrors..."):
    available_mirrors = [m for m in scihub_mirrors if test_mirror(m)]

if not available_mirrors:
    st.error("No Sci-Hub mirrors are currently available!")
    available_mirrors = scihub_mirrors  # Fallback to all mirrors

selected_mirror = st.selectbox("Choose Sci-Hub mirror:", available_mirrors)

# Download status filter
st.sidebar.markdown("### Filter by Download Status")
show_downloaded = st.sidebar.checkbox("✅ Downloaded", value=True)
show_failed = st.sidebar.checkbox("❌ Failed to download", value=True)
show_not_tried = st.sidebar.checkbox("❓ Not yet tried", value=True)

# Determine download status
doc_dir = "./doc"
if os.path.exists(doc_dir):
    # Extract number prefix before underscore for each PDF file
    existing_files = {}
    max_checked = -1
    for f in os.listdir(doc_dir):
        if f.endswith(".pdf"):
            # Extract digits before first underscore
            prefix = f.split('_')[0]
            if prefix.isdigit():
                number = int(prefix)
                existing_files[number] = prefix
                max_checked = max(max_checked, number)
    if max_checked == -1:
        max_checked = -1
else:
    existing_files = {}
    max_checked = -1

# Build table first to determine what to show
links = []

for _, row in df.iterrows():
    number = int(row['number']) if pd.notna(row['number']) else None
    if number is None:
        continue

    prefix = str(number)
    author_short = row.get('author_short', 'Unknown')
    title = row.get('title', 'No title')
    if len(str(title)) > 50:
        title = str(title)[:50] + "..."

    # Status determination
    if number in existing_files:
        status = "✅"
        prefix = existing_files[number]  # Use actual file prefix
    elif number <= max_checked:
        status = "❌"
    else:
        status = "❓"

    # DOI links
    doi = row.get('doi', '')
    sci_hub_link = None
    scholar_link = None

    if pd.notna(doi) and str(doi).strip() and str(doi).strip().lower() != 'nan':
        encoded_doi = urllib.parse.quote(str(doi))
        sci_hub_link = selected_mirror + encoded_doi
        scholar_link = f"https://scholar.google.com/scholar?q={encoded_doi}&btnG="

    # OpenAlex link (from cached data)
    openalex_link = row.get('openalex_landing_page', None)

    links.append({
        "ID": prefix,
        "Status": status,
        "Author": author_short,
        "Title": title,
        "Sci-Hub": sci_hub_link,
        "Scholar": scholar_link,
        "OpenAlex": openalex_link
    })

links_df = pd.DataFrame(links)

# Apply download status filter
filtered_links = []
for _, row in links_df.iterrows():
    status = row['Status']
    if ((status == "✅" and show_downloaded) or
        (status == "❌" and show_failed) or
        (status == "❓" and show_not_tried)):
        filtered_links.append(row)

filtered_df = pd.DataFrame(filtered_links) if filtered_links else pd.DataFrame(columns=links_df.columns)

# Dataset summary
st.markdown(f"### Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(links_df))
with col2:
    st.metric("Showing", len(filtered_df))
with col3:
    st.metric("Records with DOI", len(df[df['doi'].notna() & (df['doi'].astype(str).str.strip() != '') & (df['doi'].astype(str).str.strip().str.lower() != 'nan')]))
with col4:
    st.metric("Downloaded PDFs", len(existing_files))



# Display table with clickable links
def make_clickable(link, text="link"):
    if pd.isna(link) or not link:
        return "—"
    return f'<a href="{link}" target="_blank">{text}</a>'

if not filtered_df.empty:
    # Apply clickable formatting to filtered data
    display_df = filtered_df.copy()
    display_df['Sci-Hub'] = display_df['Sci-Hub'].apply(lambda x: make_clickable(x, "sci-hub"))
    display_df['Scholar'] = display_df['Scholar'].apply(lambda x: make_clickable(x, "scholar"))
    display_df['OpenAlex'] = display_df['OpenAlex'].apply(lambda x: make_clickable(x, "source"))

    st.markdown("### Literature Download Links")
    st.markdown("""
    **Legend:**
    - ✅ PDF already downloaded
    - ❌ Previously checked, no PDF found
    - ❓ Not yet checked
    """)

    # Display the table
    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.warning("No records match the selected filters.")

# Download statistics
st.markdown("### Download Statistics")
all_status_counts = links_df['Status'].value_counts()
filtered_status_counts = filtered_df['Status'].value_counts() if not filtered_df.empty else {}

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("✅ Downloaded",
              filtered_status_counts.get('✅', 0),
              delta=f"of {all_status_counts.get('✅', 0)} total")
with col2:
    st.metric("❌ Not Found",
              filtered_status_counts.get('❌', 0),
              delta=f"of {all_status_counts.get('❌', 0)} total")
with col3:
    st.metric("❓ Unchecked",
              filtered_status_counts.get('❓', 0),
              delta=f"of {all_status_counts.get('❓', 0)} total")

# Cache management
st.markdown("### Cache Management")
if os.path.exists("./dat/merged_outer_openalex.csv"):
    if st.button("🔄 Refresh OpenAlex Data"):
        os.remove("./dat/merged_outer_openalex.csv")
        st.experimental_rerun()
    st.success("Using cached OpenAlex data. Click refresh to update.")
else:
    st.info("OpenAlex data will be cached after first load.")

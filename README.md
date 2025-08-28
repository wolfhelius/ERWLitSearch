```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


# Analyze ERW Literature

This project analyzes ERW literature PDFs to categorize and extract key information
for enhanced rock weathering (ERW) literature review.

Analysis framework:
- Categorize papers by type: model, bench, field, metaanalysis
- Count keyword occurrences: toxic*, hazard*, harm*, nickel, chromium
- For empirical studies: determine if Ni was actually measured (vs mentioned)
- For Ni measurements: assess if levels were within hazardous ranges
- For hazardous ranges: check if controls showed same effects
- For biotic indicators: identify if harmful effects were observed

Manuscript structure outline:
Introduction and background:
* Prominent papers with easily identified flaws (eg Levy)
* Citation trail that includes no harms (eg Beerling)
* Deep dive on Ni
* Review EPA 503, EU Nickel risk, US Superfund Nickel risk
* Clear references to the role of the material in driving toxicity
* Analysis of how much was added relative to background
* Jenny and Anderson papers
* => leads to gap, we need to analyze
* Analysis of Cr(VI) work

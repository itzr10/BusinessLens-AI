# Third-party notices

BusinessLens AI does not copy or bundle any third-party code into this repository.
Users install the packages below with `pip install -r requirements.txt`; each stays
under its own license. License names are taken from the installed package metadata
(versions pinned in `requirements.txt`) and should be re-checked before any
redistribution of a bundled build, container image or hosted product.

| Package | Version | License (per package metadata) |
|---|---|---|
| streamlit | 1.66.0 | Apache-2.0 |
| pandas | 3.0.6 | BSD-3-Clause |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| plotly | 7.1.0 | MIT |
| google-genai | 2.31.0 | Apache-2.0 |
| openai | 3.35.1 | Apache-2.0 |
| httpx2 | 2.14.0 | BSD-3-Clause |
| pytest (development only) | 9.1.1 | MIT |

Transitive dependencies are not listed here; they carry their own licenses.

If you ever ship these packages together with the app (installer, Docker image,
hosted service with distributed binaries), you must include their copyright and
license texts, and for Apache-2.0 packages their NOTICE files. Merely publishing
this repository and a `requirements.txt` does not distribute them.

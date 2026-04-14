# Flask Hydro Data Intelligence

This is a Flask conversion of the uploaded Streamlit hydro dashboard.

## Features
- Predict Missing Day
- Validate Entered Values
- Streamlit-like layout rebuilt in Flask
- Separate HTML template and CSS
- Chart.js charts for the sidebar snapshot and trend panel

## Project structure

```text
flask_hydro_streamlit_like/
├── app.py
├── model.pkl                # place your trained bundle here
├── requirements.txt
├── templates/
│   ├── base.html
│   └── index.html
└── static/
    └── css/
        └── style.css
```

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes
- Put `model.pkl` beside `app.py`.
- If you also want `images.png` from the Streamlit app, add it under `static/` and reference it in the template.

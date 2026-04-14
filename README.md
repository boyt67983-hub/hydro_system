# Streamlit Hydro Data Intelligence



## Features
- Predict Missing Day
- Validate Entered Values
- Streamlit-like layout rebuilt in Flask
- Separate HTML template and CSS
- Chart.js charts for the sidebar snapshot and trend panel

## Project structure

```text
 Streamlit_Hydro_System/
├── app.py
├── model.pkl                # place your trained bundle here
├── requirements.txt
   
```

## Run

```bash
pip install -r requirements.txt
Streamlit run app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes
- Put `model.pkl` beside `app.py`.
- If you also want `images.png` from the Streamlit app, add it under `static/` and reference it in the template.

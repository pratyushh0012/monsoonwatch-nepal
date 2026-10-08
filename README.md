# MonsoonWatch Nepal

A 4-week early warning for dengue surges in Nepal, built from the weekly hospital reports published by
Nepal's Epidemiology and Disease Control Division (2016–2025).

Live demo: https://monsoonwatch-nepal.streamlit.app

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

This repository holds only what the demo needs: the app, the saved model, and the weekly table and
back-test results it displays. The model is a simple statistical one (logistic regression for the
warning, negative binomial regression for the case count), tested season by season on 2021–2025.

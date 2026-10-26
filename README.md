# Cyberbullying Detection — Render deployment

This is a small Flask web app for the college project. It trains a TF-IDF + Logistic Regression text classifier from `cyberbullying_dataset.csv` when the app starts.

## Files
- `app.py` — Flask app and model training
- `templates/index.html` — browser interface
- `cyberbullying_dataset.csv` — dataset
- `requirements.txt` — Python dependencies
- `render.yaml` — Render service configuration

## Deploy on Render
1. Download and unzip this project.
2. Create a GitHub repository and upload all files, keeping `templates/index.html` inside the `templates` folder.
3. Sign in to Render and choose **New + → Web Service**.
4. Connect the GitHub repository. Render should detect `render.yaml`; if it asks for settings, choose:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
5. Click **Deploy Web Service** and wait for the build to finish.
6. Open the public `onrender.com` URL to use the app.

## Run locally (optional)
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:10000`.

## Dataset notes
The CSV must contain columns named exactly `Text` and `Label`. The optional `ID` column is ignored. Keep the CSV in the same folder as `app.py`.

The uploaded dataset has only 40 rows. This is enough to demonstrate the deployment flow, but it is too small to establish reliable real-world accuracy. Add more diverse, correctly labelled examples and evaluate the model on a separate test set before presenting any accuracy claims. This is a traditional machine-learning demo, not an LLM.

# Cardiovascular Disease Prediction — React + FastAPI + scikit-learn

A full-stack academic/demo project for the cardiovascular dataset with columns:

`id;age;gender;height;weight;ap_hi;ap_lo;cholesterol;gluc;smoke;alco;active;cardio`

## Features
- CSV/semicolon dataset upload
- Data preprocessing: validation, missing values, duplicate removal, BMI, train/test split, scaling
- Exploratory analysis and plots
- Correlation matrix
- Model comparison: Logistic Regression, SVM, KNN, Decision Tree, Random Forest
- Metrics: accuracy, precision, recall, F1, ROC-AUC
- Selects the best model by validation accuracy
- Individual cardiovascular-risk prediction
- OpenAI API explanation/summary endpoint (API key stays on the backend)

## Dataset
Place your file at `data/cardio_train.csv`. The backend also accepts an uploaded `.csv` file.

Expected delimiter is `;`.

`cardio` is the target: 0 = no cardiovascular disease, 1 = cardiovascular disease.

## Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
python -m uvicorn main:app --reload --port 8000 or 
python -m uvicorn main:app --port 8000 --log-level debug
```

## Frontend
```bash
cd frontend
npm install
npm run dev
```
Open the Vite URL shown in the terminal.

## OpenAI
Put your key in `backend/.env`:
`OPENAI_API_KEY=...`

The browser never receives the key. The backend uses the OpenAI Python SDK and the Responses API for the natural-language explanation. The ML prediction itself is deterministic scikit-learn code; OpenAI is not used as the classifier.

## API
- `POST /api/analyze` — upload/train/analyze dataset
- `POST /api/predict` — predict one patient
- `POST /api/ai-explain` — explain returned metrics/prediction
- `GET /api/health`

---

## Designed By

<p align="center">
	<svg width="700" height="120" viewBox="0 0 700 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">
		<defs>
			<linearGradient id="g" x1="0" x2="1">
				<stop offset="0" stop-color="#ff7eb3"/>
				<stop offset="0.5" stop-color="#7a9bff"/>
				<stop offset="1" stop-color="#7ef2d6"/>
			</linearGradient>
		</defs>
		<rect rx="16" width="700" height="120" fill="url(#g)" />
		<text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" font-size="28" font-family="Segoe UI, Roboto, Arial, sans-serif" fill="#0b1020">Designed by <tspan font-weight="700">Sushma Kanna</tspan></text>
	</svg>
</p>

<p align="center"><em>Crafted with care — thank you for checking out this project.</em></p>


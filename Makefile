.PHONY: setup train validate predict export migrate-legacy prepare-data backend frontend test clean

# ─── Environment Setup ─────────────────────────────────────────────
setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -e ".[dev,web]"
	cd frontend && npm install

# ─── Training Pipeline ─────────────────────────────────────────────
migrate-legacy:
	python training/data/migrate_legacy.py

prepare-data:
	python training/data/prepare_dataset.py

train:
	python training/train.py

validate:
	python training/validate.py

predict:
	python training/predict.py --source datasets/test/

export:
	python training/export.py --format onnx
	python training/export.py --format tflite

augment:
	python training/data/augment_copypaste.py --input datasets/images/train/ --count 500

# ─── Backend ───────────────────────────────────────────────────────
backend:
	cd backend && python manage.py runserver 0.0.0.0:8000

backend-migrate:
	cd backend && python manage.py makemigrations && python manage.py migrate

backend-check:
	cd backend && python manage.py check

# ─── Frontend ──────────────────────────────────────────────────────
frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# ─── Testing ───────────────────────────────────────────────────────
test:
	cd backend && python -m pytest

# ─── Cleanup ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

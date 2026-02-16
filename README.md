# Clovr Fndr - 4-Leaf Clover Detection App

A mobile-first web application that uses computer vision (YOLO) to detect 4-leaf clovers in real-time video or photos.

## What is Clovr Fndr?

Point your phone camera at a patch of clovers, and Clovr Fndr will automatically detect and highlight any 4-leaf clovers with bounding boxes. No more eye strain!

## Features

- 📱 **Mobile Camera**: Real-time video detection
- 📸 **Photo Upload**: Analyze existing photos
- 🎯 **High Accuracy**: Custom-trained YOLO model
- 🎮 **Gamification**: Track discoveries, achievements
- 🌐 **PWA**: Works offline, install on home screen
- 🔒 **Privacy**: Images processed locally (optional cloud)

## How It Works

1. **Open app** on your phone
2. **Point camera** at clover patch
3. **Wait for alert** when 4-leaf clover detected
4. **Save discovery** to your gallery

## Technology

- **Model**: YOLOv8 (custom-trained on clover dataset)
- **Backend**: Django + ONNX Runtime
- **Frontend**: React PWA with WebRTC camera
- **Inference**: Browser (ONNX.js) or backend (GPU)

## Detection Classes

- 🍀 **3-leaf clover** (common, background)
- 🍀🍀 **4-leaf clover** (rare, 1 in 5,000)
- 🍀🍀🍀 **5-leaf clover** (very rare)
- 🍀🍀🍀🍀 **6+ leaf clover** (ultra-rare!)

## Development Status

- [x] Project initialization and documentation
- [ ] **PENDING**: Legacy code transfer to `legacy-code/` folder
- [ ] Model training (1000+ annotated clover images)
- [ ] Backend: Django API with YOLO inference
- [ ] Frontend: React camera component
- [ ] Real-time bounding box overlay
- [ ] Discovery gallery
- [ ] PWA offline mode

## Quick Start

```bash
# Clone repository
git clone https://github.com/x78163/clovr-fndr.git
cd clovr-fndr

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver

# Frontend setup
cd frontend
npm install
npm run dev
```

## Training Your Own Model

```bash
# Install Ultralytics YOLO
pip install ultralytics

# Collect and annotate clover images (use Roboflow)
# Export dataset in YOLO format

# Train YOLOv8
yolo train \
  model=yolov8n.pt \
  data=clovers.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16

# Export to ONNX for web
yolo export model=best.pt format=onnx
```

## Legacy Code

This project has existing legacy code that will be transferred to `legacy-code/` folder for analysis and potential reuse.

**Once legacy code is added**:
- Analyze existing model and dataset
- Determine if retraining needed
- Port useful components to new architecture

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Comprehensive AI continuity document (full system context)
- **[README.md](README.md)** - This file
- **[MODEL_TRAINING.md](MODEL_TRAINING.md)** - YOLO training guide (TBD)

## Use Cases

- **Casual**: Fun app for parks and backyards
- **Collectors**: Efficiently find rare clovers
- **Botanists**: Study clover mutations
- **Education**: Learn about genetics and probability

## Privacy

- **Local Processing**: Images can be processed entirely on-device (browser mode)
- **No Tracking**: No analytics or user tracking
- **Optional Cloud**: Opt-in to backend processing for better performance
- **Data Deletion**: Saved discoveries can be deleted anytime

## Fun Facts

- 4-leaf clovers occur in about 1 in 5,000 clovers
- 5-leaf clovers are about 1 in 20,000
- The record is a 56-leaf clover (2009, Japan)!
- Each leaf represents: Hope, Faith, Love, Luck

## Contributing

This is a personal project. Contributions welcome after initial development.

## License

TBD

## Contact

For questions or feedback, open an issue on GitHub.

---

**Happy clover hunting! 🍀**

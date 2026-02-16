# Clovr Fndr - AI Continuity Document

## Project Overview

**Clovr Fndr** (Clover Finder) is a computer vision web application that uses a custom-trained YOLO object detection model to identify 4-leaf clovers in real-time video streams or uploaded photos. The app uses the device camera to scan clover patches and highlights detected 4-leaf clovers with bounding boxes.

**Current Status**: Project initialization - continuity documents created, awaiting legacy code transfer

---

## Concept

### The Problem
Finding 4-leaf clovers is tedious and time-consuming. The average frequency is 1 in 5,000 to 1 in 10,000 3-leaf clovers. Human eyes get fatigued quickly when scanning dense clover patches.

### The Solution
A mobile-first web app that:
1. Accesses device camera (phone/tablet)
2. Processes video frames in real-time
3. Detects 4-leaf clovers using custom YOLO model
4. Draws bounding boxes around detected clovers
5. Alerts user when 4-leaf clover found
6. Saves photos of discoveries

### Use Cases
- **Casual Users**: Fun app for parks, backyards
- **Clover Collectors**: Efficiently find rare clovers
- **Botanists/Researchers**: Study clover mutations
- **Gamification**: "Pokémon GO for clovers"

---

## Technology Stack

### Computer Vision
- **Model**: YOLOv8 or YOLOv11 (Ultralytics)
- **Training**: Custom dataset of 4-leaf clover images
- **Inference**:
  - **Option 1**: ONNX Runtime (runs in browser via WebGL)
  - **Option 2**: Backend inference (Python + Flask/Django)
  - **Option 3**: TensorFlow.js (browser-based)
- **Performance Target**: 10-30 FPS on mobile devices

### Backend
- **Framework**: Django or Flask (lightweight)
- **API**: REST endpoints for image upload/inference
- **Model Serving**: ONNX Runtime or TorchServe
- **Database**: PostgreSQL (user stats, discoveries)
- **Storage**: S3 for saved clover photos

### Frontend
- **Framework**: React or Vue (mobile-first PWA)
- **Camera**: WebRTC MediaStream API
- **Video Processing**: Canvas API for frame extraction
- **Inference**:
  - **Browser**: ONNX.js or TensorFlow.js
  - **Backend**: POST frames to API for inference
- **UI**: Real-time bounding box overlay on video feed

### Mobile
- **Progressive Web App (PWA)**: Install on phone home screen
- **Responsive Design**: Optimized for phone cameras
- **Offline Capable**: Model cached locally (if using browser inference)

---

## Core Features

### 1. Real-Time Camera Detection
**User Flow**:
1. Open app on phone
2. Grant camera permission
3. Point camera at clover patch
4. App processes video frames (10-30 FPS)
5. Bounding boxes appear around detected 4-leaf clovers
6. Alert sound/vibration when clover found

**Technical Implementation**:
- WebRTC `getUserMedia()` captures video stream
- Extract frames to Canvas at 10-30 FPS
- Inference on each frame (YOLO model)
- Draw bounding boxes on Canvas overlay
- Highlight high-confidence detections (>80% confidence)

**Performance**:
- Lightweight YOLO model (YOLOv8n or YOLOv8s)
- Image resolution: 640x640 (YOLO standard)
- Frame skip if inference too slow (prioritize UX)

### 2. Photo Upload Detection
**User Flow**:
1. Upload photo of clover patch
2. Backend processes image
3. Results displayed with bounding boxes
4. Download annotated image

**Technical Implementation**:
- `POST /api/detect/` with image file
- Backend runs YOLO inference
- Return bounding boxes with confidence scores
- Frontend draws boxes on uploaded image

### 3. Discovery Gallery
**User Flow**:
1. User taps "Save" when 4-leaf clover found
2. Photo saved to gallery
3. Gallery shows all saved discoveries
4. Stats: total found, rarest (5-leaf, 6-leaf, etc.)

**Technical Implementation**:
- Save photo to S3 with timestamp, GPS location
- Database: `Discovery` model with metadata
- Gallery page: Grid of thumbnails
- Click for full-size annotated image

### 4. Gamification (Optional)
- **Streak**: Days in a row finding clovers
- **Leaderboard**: Most clovers found (global/local)
- **Achievements**: "Found 10 clovers", "Found 5-leaf clover"
- **Share**: Post discoveries to social media

---

## YOLO Model Training

### Dataset Requirements

**Minimum**: 500-1000 labeled images
- **3-leaf clovers** (background/negative examples): ~60%
- **4-leaf clovers**: ~35%
- **5-leaf clovers** (rare): ~4%
- **6+ leaf clovers** (very rare): ~1%

**Image Diversity**:
- Different lighting conditions (sunny, cloudy, shade)
- Different angles (top-down, oblique)
- Different clover sizes (close-up, wide shot)
- Different backgrounds (grass, soil, mulch)
- Different clover species (white clover, red clover)

**Annotation Format**: YOLO format (bounding boxes)
```
class_id center_x center_y width height
```
- Class 0: 3-leaf (background, not used for detection alert)
- Class 1: 4-leaf (target)
- Class 2: 5-leaf (bonus)
- Class 3: 6+ leaf (ultra-rare)

### Training Process

**Step 1: Data Collection**
- Take photos of clover patches (phone camera)
- 1000+ images in various conditions
- Focus on 4-leaf clovers (hardest to find)

**Step 2: Annotation**
- Use LabelImg, Roboflow, or CVAT
- Draw bounding boxes around each clover
- Label as 3-leaf, 4-leaf, 5-leaf, 6+ leaf

**Step 3: Training**
```bash
# Install Ultralytics YOLO
pip install ultralytics

# Train YOLOv8 model
yolo train \
  model=yolov8n.pt \
  data=clovers.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0  # GPU
```

**Step 4: Optimization**
- Export to ONNX for browser inference
- Quantization for mobile (FP16 or INT8)
- Test inference speed on target devices

**Step 5: Validation**
- Test in real-world conditions
- Measure false positive rate (misidentifying 3-leaf as 4-leaf)
- Measure false negative rate (missing 4-leaf clovers)
- Iterate training with hard negative examples

---

## Architecture

### Option A: Browser-Based Inference (Preferred)

**Pros**:
- Low latency (no network round-trip)
- Works offline (PWA)
- Scalable (no backend compute cost)
- Privacy-friendly (images never leave device)

**Cons**:
- Model size limited (~10MB max for web)
- Slower on old phones
- Battery drain

**Implementation**:
```
┌─────────────────────────┐
│   React PWA             │
│   - Camera (WebRTC)     │
│   - ONNX.js inference   │
│   - Canvas overlay      │
└─────────────────────────┘
         │ (Optional)
         ↓
┌─────────────────────────┐
│   Backend (Django)      │
│   - Save discoveries    │
│   - Gallery storage     │
│   - Leaderboard         │
└─────────────────────────┘
```

### Option B: Backend Inference

**Pros**:
- Larger model (better accuracy)
- Works on all devices
- Consistent performance

**Cons**:
- Network latency (~100-500ms)
- Backend compute cost (GPU inference)
- Requires internet connection
- Privacy concerns (images sent to server)

**Implementation**:
```
┌─────────────────────────┐
│   React PWA             │
│   - Camera (WebRTC)     │
│   - Send frames to API  │
│   - Display results     │
└──────────┬──────────────┘
           │ POST /api/detect
           ↓
┌─────────────────────────┐
│   Backend (Django)      │
│   - YOLOv8 inference    │
│   - Return bboxes       │
│   - GPU acceleration    │
└─────────────────────────┘
```

**Recommendation**: Start with **Option B** (backend inference) for MVP. Migrate to **Option A** (browser inference) once model is optimized.

---

## Database Schema

### Models

#### User (Optional)
- `id` (UUID)
- `username`
- `email`
- `created_at`

#### Discovery
- `id` (UUID)
- `user_id` (FK, nullable for anonymous)
- `image_path` (S3 URL)
- `clover_type` (3-leaf, 4-leaf, 5-leaf, 6+ leaf)
- `confidence` (float, 0-1)
- `latitude` (float, nullable)
- `longitude` (float, nullable)
- `location_name` (text, e.g. "Central Park")
- `timestamp`
- `likes` (int, if social features)

#### Achievement (Optional)
- `id`
- `user_id` (FK)
- `achievement_type` (enum: "first_clover", "10_clovers", "5_leaf", etc.)
- `unlocked_at`

---

## API Endpoints

### Detection
- `POST /api/detect/` - Upload image, get bounding boxes
  - Input: Image file (JPEG/PNG)
  - Output: `{"detections": [{"class": "4-leaf", "confidence": 0.95, "bbox": [x, y, w, h]}]}`

- `POST /api/detect/stream/` - Real-time video frame inference (if backend)
  - Input: Video frame (base64 or binary)
  - Output: Same as above

### Discoveries
- `GET /api/discoveries/` - List user's discoveries
- `POST /api/discoveries/` - Save discovery
- `GET /api/discoveries/{id}/` - Get discovery detail
- `DELETE /api/discoveries/{id}/` - Delete discovery

### Stats (Optional)
- `GET /api/stats/user/` - User stats (total found, streak, etc.)
- `GET /api/leaderboard/` - Global leaderboard

---

## Development Plan

### Phase 1: Model Training (2-3 weeks)
- [ ] Collect 1000+ clover images
- [ ] Annotate with bounding boxes
- [ ] Train YOLOv8 model
- [ ] Validate accuracy (>90% for 4-leaf detection)
- [ ] Export to ONNX format

### Phase 2: Backend API (1-2 weeks)
- [ ] Django project setup
- [ ] Models: User, Discovery
- [ ] `POST /api/detect/` endpoint
- [ ] YOLO inference integration
- [ ] S3 storage for discoveries
- [ ] Unit tests

### Phase 3: Frontend (2-3 weeks)
- [ ] React PWA setup
- [ ] Camera component (WebRTC)
- [ ] Video frame extraction
- [ ] API integration for inference
- [ ] Bounding box overlay (Canvas)
- [ ] Discovery gallery
- [ ] Responsive design (mobile-first)

### Phase 4: Real-Time Optimization (1-2 weeks)
- [ ] Frame rate optimization
- [ ] Model quantization (FP16)
- [ ] Browser-based inference (ONNX.js)
- [ ] Offline mode (PWA caching)
- [ ] Battery optimization

### Phase 5: Gamification (1-2 weeks)
- [ ] Achievement system
- [ ] Leaderboard
- [ ] Streak tracking
- [ ] Social sharing

### Phase 6: Testing & Launch (1 week)
- [ ] Field testing (real clover patches)
- [ ] False positive reduction
- [ ] Performance tuning
- [ ] App store submission (iOS/Android PWA)

---

## Legacy Code Integration

User mentioned existing legacy code in a different repository. Once transferred to `legacy-code/` folder:

### Review Checklist
- [ ] What YOLO version was used? (YOLOv3, v5, v7?)
- [ ] Training dataset size and quality?
- [ ] Model weights file location?
- [ ] Inference code (Python, JavaScript, other?)
- [ ] Any labeled training data?
- [ ] Performance metrics (accuracy, speed)?

### Migration Strategy
1. **Preserve legacy code**: Keep in `legacy-code/` folder, don't modify
2. **Extract useful components**:
   - Model weights (if compatible)
   - Training dataset (if available)
   - Inference logic (port to new framework)
   - UI patterns (if applicable)
3. **Document differences**: Create `LEGACY_NOTES.md` comparing old vs new approach

---

## Technical Challenges & Solutions

### Challenge 1: Similar Appearance
**Problem**: 3-leaf and 4-leaf clovers look very similar
**Solution**:
- High-quality training data (clear 4-leaf examples)
- Augmentation (rotation, flip, brightness) to increase dataset diversity
- Hard negative mining (train on 3-leaf clovers that look like 4-leaf)

### Challenge 2: Occlusion
**Problem**: Leaves overlap, making leaf count ambiguous
**Solution**:
- Annotate only clear, unoccluded clovers
- Conservative detection (only flag high confidence >80%)
- User verification: "Is this a 4-leaf clover?" feedback loop

### Challenge 3: Real-Time Performance
**Problem**: YOLO inference slow on mobile (~100ms per frame)
**Solution**:
- Use YOLOv8n (nano) for speed
- Reduce input resolution (640x640 → 320x320)
- Frame skipping (process every 3rd frame)
- Hardware acceleration (WebGL, GPU.js)

### Challenge 4: False Positives
**Problem**: Model detects 3-leaf as 4-leaf
**Solution**:
- Confidence threshold (>80% for alert)
- Post-processing: Leaf counting algorithm
- User feedback: "Report incorrect detection"
- Continual learning: Retrain with false positives

---

## Model Optimization

### For Browser (ONNX.js)
```bash
# Export to ONNX
yolo export model=best.pt format=onnx imgsz=640

# Quantize to FP16
python -m onnxruntime.quantization.quantize_dynamic \
  --model_input best.onnx \
  --model_output best_fp16.onnx \
  --per_channel \
  --reduce_range
```

### For Mobile (TFLite)
```bash
# Export to TensorFlow Lite
yolo export model=best.pt format=tflite imgsz=640

# Quantize to INT8
python convert_tflite.py --quantize int8
```

---

## Privacy & Ethics

### Privacy
- **Browser Inference**: Images never leave device (best)
- **Backend Inference**: Images deleted after processing
- **Saved Discoveries**: User controls what to save
- **Location Data**: Optional, with explicit permission

### Ethics
- **Environmental Impact**: Encourage "look but don't pick" (clover conservation)
- **Gamification Balance**: Avoid incentivizing clover destruction
- **Educational**: Include info about clover mutations, genetics

---

## Monetization (Optional)

### Free Tier
- Basic detection (10 detections/day)
- Gallery (save 10 discoveries)
- Ads (non-intrusive)

### Premium ($2.99/month or $19.99/year)
- Unlimited detections
- No ads
- Advanced stats (heatmaps, best locations)
- Export discoveries (CSV, map)

### Alternative: One-Time Purchase ($9.99)
- All features unlocked
- No subscription

---

## Marketing & Launch

### Target Audience
- **Kids**: Fun app for outdoor exploration
- **Families**: Activity for parks, picnics
- **Gardeners**: Clover enthusiasts
- **Botanists**: Research tool

### Launch Strategy
1. **Beta**: TestFlight (iOS) / Google Play (Android)
2. **Social Media**: TikTok, Instagram (demo videos)
3. **Press**: Tech blogs, gardening sites
4. **Reddit**: r/gardening, r/oddlysatisfying
5. **App Store Optimization**: Keywords "clover", "lucky", "4-leaf"

---

## Success Metrics

### Technical
- Model accuracy: >90% for 4-leaf detection
- Inference speed: <100ms per frame (mobile)
- False positive rate: <5%
- App load time: <2 seconds

### User Engagement
- DAU (Daily Active Users)
- Average session duration
- Discoveries saved per user
- Retention (7-day, 30-day)

---

## Future Enhancements

### Phase 2 Features
- **AR Mode**: Augmented reality overlay (ARKit/ARCore)
- **Plant Identification**: Identify other plants (not just clovers)
- **Community Map**: Global heatmap of 4-leaf clover locations
- **AI Insights**: "Best times to find clovers" (weather, season)

### Advanced Features
- **Live Multiplayer**: Race friends to find clovers
- **NFT Integration**: Mint rare clovers as NFTs (blockchain)
- **Scientific Contribution**: Crowdsource data for clover mutation research

---

## Related Resources

### YOLO Training
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- Roboflow (annotation): https://roboflow.com/
- Custom dataset guide: https://blog.roboflow.com/train-yolov8/

### WebRTC
- Camera access: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

### ONNX.js
- Browser inference: https://onnxruntime.ai/docs/tutorials/web/

---

## Questions for Development

### Q1: Legacy Code Status?
- What YOLO version?
- Model weights available?
- Training dataset available?
**Action**: User will transfer legacy code to `legacy-code/` folder for analysis

### Q2: Browser vs Backend Inference?
**Recommendation**: Start backend, migrate to browser once optimized

### Q3: Monetization Strategy?
**Options**: Free with ads, Freemium, One-time purchase
**Recommendation**: Free with optional $2.99 "Pro" upgrade (no ads, unlimited)

### Q4: Social Features?
**Scope**: Leaderboard? Community map? Social sharing?
**Recommendation**: MVP = no social, add later based on user demand

---

## Notes for AI Assistants

- **Custom YOLO Model**: Requires training before development can proceed
- **Legacy Code**: User will provide existing code for analysis
- **Mobile-First**: Camera access is core feature (desktop secondary)
- **Real-Time**: Video processing, not just static images
- **Gamification**: Balance fun with environmental responsibility

---

## Current Session Context

**Date**: 2026-02-16
**Status**: Project initialization, awaiting legacy code transfer
**Next Steps**:
1. User transfers legacy code to `legacy-code/` folder
2. Analyze legacy code (model, dataset, inference)
3. Decide: Retrain from scratch or fine-tune legacy model?
4. Set up Django backend project
5. Implement YOLO inference endpoint
6. Build React camera frontend

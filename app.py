from flask import Flask, render_template_string, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import base64
import threading
import time

# ============= CONFIGURATION =============
app = Flask(__name__)
CORS(app)

# Local storage for analysis data only (NO IMAGES)
DATA_DIR = 'analysis_data'
SESSIONS_FILE = os.path.join(DATA_DIR, 'sessions.json')
STATS_FILE = os.path.join(DATA_DIR, 'statistics.json')

os.makedirs(DATA_DIR, exist_ok=True)

# Live stream storage
latest_frames = {}  # Dictionary to store latest frame from each camera
frame_lock = threading.Lock()

print("🔄 Loading AI models...")
try:
    object_detector = YOLO('yolov8n.pt')
    pose_detector = YOLO('yolov8n-pose.pt')
    print("✅ AI models loaded successfully!\n")
except Exception as e:
    print(f"⚠️ Could not load YOLO models: {e}")
    print("Running in basic detection mode...\n")
    object_detector = None
    pose_detector = None

# ============= DATA STORAGE FUNCTIONS =============
def load_sessions():
    """Load session data from local JSON file"""
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_session(session_data):
    """Save session analysis (NO IMAGE DATA)"""
    sessions = load_sessions()
    sessions.insert(0, session_data)
    sessions = sessions[:1000]  # Keep last 1000 sessions
    
    with open(SESSIONS_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)
    
    update_statistics(session_data)

def update_statistics(session_data):
    """Update running statistics"""
    stats = load_statistics()
    
    stats['total_sessions'] += 1
    stats['activities'][session_data['activity']] = stats['activities'].get(session_data['activity'], 0) + 1
    
    if session_data['is_unusual']:
        stats['unusual_count'] += 1
    
    stats['last_updated'] = datetime.now().isoformat()
    
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

def load_statistics():
    """Load statistics from local storage"""
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {
        'total_sessions': 0,
        'unusual_count': 0,
        'activities': {},
        'last_updated': None
    }

# ============= AI DETECTION FUNCTIONS =============
def analyze_image_advanced(image, student_id):
    """Advanced AI analysis using YOLO models"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect objects
    objects_found = []
    if object_detector:
        results = object_detector(image, verbose=False)
        for detection in results[0].boxes.data:
            class_id = int(detection[5])
            confidence = float(detection[4])
            if confidence > 0.5:
                object_name = object_detector.model.names[class_id]
                objects_found.append(object_name)
    
    # Detect pose
    head_down = False
    hand_raised = False
    
    if pose_detector:
        pose_results = pose_detector(image, verbose=False)
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            keypoints = pose_results[0].keypoints
            if keypoints.xy is not None and len(keypoints.xy) > 0:
                points = keypoints.xy[0].cpu().numpy()
                
                if len(points) >= 11:
                    nose_y = points[0, 1]
                    left_shoulder_y = points[5, 1]
                    right_shoulder_y = points[6, 1]
                    left_wrist_y = points[9, 1]
                    right_wrist_y = points[10, 1]
                    
                    # Check head down
                    if nose_y > 0 and left_shoulder_y > 0:
                        avg_shoulder = (left_shoulder_y + right_shoulder_y) / 2
                        if nose_y > avg_shoulder + 30:
                            head_down = True
                    
                    # Check hand raised
                    if (left_wrist_y < left_shoulder_y - 50) or (right_wrist_y < right_shoulder_y - 50):
                        hand_raised = True
    
    # Determine activity
    activity = "Unknown"
    is_unusual = False
    confidence_score = 0.0
    
    if 'cell phone' in objects_found:
        activity = "Using Phone"
        is_unusual = True
        confidence_score = 0.95
    elif 'person' not in objects_found:
        activity = "Student Absent"
        is_unusual = True
        confidence_score = 0.90
    elif any(food in objects_found for food in ['fork', 'spoon', 'bowl', 'cup', 'bottle']):
        activity = "Eating/Drinking"
        is_unusual = True
        confidence_score = 0.85
    elif head_down and 'book' not in objects_found and 'laptop' not in objects_found:
        activity = "Sleeping/Distracted"
        is_unusual = True
        confidence_score = 0.75
    elif hand_raised:
        activity = "Hand Raised (Participating)"
        confidence_score = 0.90
    elif 'laptop' in objects_found:
        activity = "Working on Laptop"
        confidence_score = 0.85
    elif 'book' in objects_found:
        activity = "Reading/Studying"
        confidence_score = 0.80
    elif head_down:
        activity = "Writing/Focused"
        confidence_score = 0.70
    else:
        activity = "Attentive/Listening"
        confidence_score = 0.65
    
    brightness = np.mean(gray)
    
    return {
        'student_id': student_id,
        'activity': activity,
        'objects_detected': objects_found,
        'is_unusual': is_unusual,
        'confidence_score': round(confidence_score, 2),
        'brightness_level': round(float(brightness), 2),
        'head_down': head_down,
        'hand_raised': hand_raised,
        'timestamp': datetime.now().isoformat(),
        'image_dimensions': f"{width}x{height}"
    }

def analyze_image_basic(image, student_id):
    """Basic analysis when YOLO is not available"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Motion detection
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (width * height) * 100
    
    brightness = np.mean(gray)
    
    # Activity estimation
    if len(faces) == 0:
        activity = "Student Absent"
        is_unusual = True
    elif edge_density > 15:
        activity = "High Activity Detected"
        is_unusual = False
    elif brightness < 50:
        activity = "Low Visibility"
        is_unusual = True
    else:
        activity = "Attentive"
        is_unusual = False
    
    return {
        'student_id': student_id,
        'activity': activity,
        'objects_detected': [],
        'is_unusual': is_unusual,
        'confidence_score': 0.6,
        'brightness_level': round(float(brightness), 2),
        'face_count': len(faces),
        'edge_density': round(float(edge_density), 2),
        'timestamp': datetime.now().isoformat(),
        'image_dimensions': f"{width}x{height}"
    }

# ============= LIVE STREAMING FUNCTIONS =============
def generate_frames(student_id):
    """Generator function for video streaming"""
    while True:
        with frame_lock:
            if student_id in latest_frames:
                frame = latest_frames[student_id].copy()
            else:
                # Create a placeholder frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Waiting for {student_id}...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS

# ============= API ROUTES =============
@app.route('/')
def dashboard():
    """Main dashboard with modern UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze image from ESP32-CAM (IMAGE NOT SAVED, but stored for live stream)"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        student_id = request.form.get('student_id', 'unknown')
        image_file = request.files['image']
        
        # Read image
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Store frame for live streaming (replaces previous frame)
        with frame_lock:
            latest_frames[student_id] = image.copy()
        
        # Analyze image (choose method based on model availability)
        if object_detector and pose_detector:
            analysis = analyze_image_advanced(image, student_id)
        else:
            analysis = analyze_image_basic(image, student_id)
        
        # Save ONLY the analysis data (NOT the image to disk)
        save_session(analysis)
        
        print(f"✅ Analyzed: {student_id} - {analysis['activity']} | Active streams: {len(latest_frames)}")
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stream/<student_id>')
def video_feed(student_id):
    """Video streaming route for live view"""
    print(f"📹 Stream requested for: {student_id}")
    return Response(generate_frames(student_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/active_streams')
def active_streams():
    """Get list of active camera streams"""
    with frame_lock:
        streams = list(latest_frames.keys())
    print(f"📊 Active streams requested: {streams}")
    return jsonify({'streams': streams})

@app.route('/api/sessions')
def get_sessions():
    """Get recent sessions"""
    limit = request.args.get('limit', 50, type=int)
    sessions = load_sessions()[:limit]
    return jsonify({'sessions': sessions, 'count': len(sessions)})

@app.route('/api/stats')
def get_stats():
    """Get statistics"""
    stats = load_statistics()
    sessions = load_sessions()
    
    # Calculate additional stats
    if sessions:
        recent = sessions[:20]
        avg_confidence = sum(s.get('confidence_score', 0) for s in recent) / len(recent)
        
        # Activity distribution
        activity_dist = defaultdict(int)
        for s in sessions[:100]:
            activity_dist[s['activity']] += 1
        
        stats['average_confidence'] = round(avg_confidence, 2)
        stats['activity_distribution'] = dict(activity_dist)
        stats['recent_sessions'] = recent[:10]
    
    return jsonify(stats)

@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear all stored data"""
    if os.path.exists(SESSIONS_FILE):
        os.remove(SESSIONS_FILE)
    if os.path.exists(STATS_FILE):
        os.remove(STATS_FILE)
    return jsonify({'success': True, 'message': 'All data cleared'})

@app.route('/api/debug')
def debug_info():
    """Debug endpoint to check system status"""
    with frame_lock:
        active_streams = list(latest_frames.keys())
    
    sessions = load_sessions()
    stats = load_statistics()
    
    return jsonify({
        'status': 'running',
        'active_streams': active_streams,
        'total_sessions': len(sessions),
        'recent_sessions': sessions[:5] if sessions else [],
        'statistics': stats,
        'models_loaded': {
            'object_detector': object_detector is not None,
            'pose_detector': pose_detector is not None
        }
    })

# ============= HTML TEMPLATE =============
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Student Activity Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --dark: #0f172a;
            --gray-50: #f8fafc;
            --gray-100: #f1f5f9;
            --gray-800: #1e293b;
            --gray-900: #0f172a;
        }

        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
            animation: fadeIn 0.6s ease-out;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 0 2px 20px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .tab-container {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            justify-content: center;
        }

        .tab-btn {
            padding: 15px 30px;
            background: white;
            border: none;
            border-radius: 15px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }

        .tab-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }

        .tab-btn.active {
            background: var(--primary);
            color: white;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease-out;
        }

        .tab-content.active {
            display: block;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: slideUp 0.6s ease-out;
        }

        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        }

        .stat-icon {
            width: 60px;
            height: 60px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 30px;
            margin-bottom: 15px;
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--dark);
            margin-bottom: 5px;
        }

        .stat-label {
            color: #64748b;
            font-size: 0.95rem;
            font-weight: 500;
        }

        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }

        .panel {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            animation: slideUp 0.8s ease-out;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
        }

        .panel-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--dark);
        }

        .activity-list {
            max-height: 500px;
            overflow-y: auto;
        }

        .activity-item {
            padding: 20px;
            border-radius: 12px;
            background: var(--gray-50);
            margin-bottom: 15px;
            border-left: 4px solid var(--primary);
            transition: all 0.3s ease;
        }

        .activity-item:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .activity-item.unusual {
            border-left-color: var(--danger);
            background: #fef2f2;
        }

        .activity-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .activity-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--dark);
        }

        .activity-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .badge-normal {
            background: #dcfce7;
            color: #166534;
        }

        .badge-unusual {
            background: #fee2e2;
            color: #991b1b;
        }

        .activity-details {
            font-size: 0.9rem;
            color: #64748b;
        }

        .activity-time {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 8px;
        }

        .video-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }

        .video-container {
            background: white;
            border-radius: 20px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }

        .video-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .video-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: var(--dark);
        }

        .live-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 5px 12px;
            background: #dcfce7;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            color: #166534;
        }

        .live-dot {
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }

        .video-frame {
            width: 100%;
            height: auto;
            border-radius: 15px;
            background: #000;
            min-height: 300px;
            object-fit: contain;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.95rem;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
        }

        .btn-danger {
            background: var(--danger);
            color: white;
        }

        .btn-danger:hover {
            background: #dc2626;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(239, 68, 68, 0.4);
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #94a3b8;
        }

        .empty-state-icon {
            font-size: 4rem;
            margin-bottom: 20px;
            opacity: 0.5;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #64748b;
        }

        .spinner {
            border: 4px solid #f3f4f6;
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s ease-in-out infinite;
        }

        .status-active {
            background: var(--success);
        }

        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2rem;
            }

            .video-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 Smart Student Activity Monitor</h1>
            <p>AI-Powered Real-Time Analysis System with Live Streaming</p>
        </div>

        <div class="tab-container">
            <button class="tab-btn active" onclick="switchTab('dashboard')">📊 Dashboard</button>
            <button class="tab-btn" onclick="switchTab('livestream')">📹 Live Stream</button>
        </div>

        <!-- Dashboard Tab -->
        <div id="dashboard" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon" style="background: #dbeafe; color: #1e40af;">📊</div>
                    <div class="stat-value" id="totalSessions">0</div>
                    <div class="stat-label">Total Sessions</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon" style="background: #dcfce7; color: #166534;">✅</div>
                    <div class="stat-value" id="normalCount">0</div>
                    <div class="stat-label">Normal Activities</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon" style="background: #fee2e2; color: #991b1b;">⚠️</div>
                    <div class="stat-value" id="unusualCount">0</div>
                    <div class="stat-label">Unusual Activities</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon" style="background: #fef3c7; color: #92400e;">🎯</div>
                    <div class="stat-value" id="avgConfidence">0%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>

            <div class="main-content">
                <div class="panel">
                    <div class="panel-header">
                        <div>
                            <div class="panel-title">
                                <span class="status-indicator status-active"></span>
                                Recent Activities
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="refreshData()">🔄 Refresh</button>
                    </div>
                    <div class="activity-list" id="activityList">
                        <div class="loading">
                            <div class="spinner"></div>
                            <p>Loading activities...</p>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">
                        <div class="panel-title">Activity Distribution</div>
                    </div>
                    <div id="distributionChart"></div>
                    <div style="margin-top: 30px;">
                        <button class="btn btn-danger" onclick="clearAllData()" style="width: 100%;">
                            🗑️ Clear All Data
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Stream Tab -->
        <div id="livestream" class="tab-content">
            <div class="video-grid" id="videoGrid">
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Loading camera streams...</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let refreshInterval;
        let streamInterval;
        let currentTab = 'dashboard';

        function switchTab(tab) {
            currentTab = tab;
            
            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(tab).classList.add('active');
            
            // Load appropriate data
            if (tab === 'dashboard') {
                refreshData();
                if (streamInterval) {
                    clearInterval(streamInterval);
                    streamInterval = null;
                }
            } else if (tab === 'livestream') {
                loadStreams();
                streamInterval = setInterval(loadStreams, 5000);
            }
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                if (!response.ok) {
                    console.error('Stats API error:', response.status);
                    return;
                }
                const data = await response.json();
                
                document.getElementById('totalSessions').textContent = data.total_sessions || 0;
                document.getElementById('unusualCount').textContent = data.unusual_count || 0;
                document.getElementById('normalCount').textContent = 
                    (data.total_sessions || 0) - (data.unusual_count || 0);
                document.getElementById('avgConfidence').textContent = 
                    (data.average_confidence || 0) ? (data.average_confidence * 100).toFixed(0) + '%' : '0%';
                
                renderDistribution(data.activity_distribution || {});
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        async function loadActivities() {
            try {
                const response = await fetch('/api/sessions?limit=50');
                if (!response.ok) {
                    console.error('Sessions API error:', response.status);
                    return;
                }
                const data = await response.json();
                
                console.log('Loaded sessions:', data.count);
                
                const listEl = document.getElementById('activityList');
                
                if (data.sessions.length === 0) {
                    listEl.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">📭</div>
                            <h3>No Activities Yet</h3>
                            <p>Waiting for ESP32-CAM to send images...</p>
                            <p style="margin-top: 10px; font-size: 0.9rem;">Make sure ESP32 is connected and sending data</p>
                        </div>
                    `;
                    return;
                }
                
                listEl.innerHTML = data.sessions.map(session => {
                    const time = new Date(session.timestamp).toLocaleString();
                    const badgeClass = session.is_unusual ? 'badge-unusual' : 'badge-normal';
                    const itemClass = session.is_unusual ? 'unusual' : '';
                    
                    return `
                        <div class="activity-item ${itemClass}">
                            <div class="activity-header">
                                <div class="activity-name">${session.activity}</div>
                                <span class="activity-badge ${badgeClass}">
                                    ${session.is_unusual ? 'Unusual' : 'Normal'}
                                </span>
                            </div>
                            <div class="activity-details">
                                <strong>Student:</strong> ${session.student_id}<br>
                                <strong>Confidence:</strong> ${(session.confidence_score * 100).toFixed(0)}%<br>
                                ${session.objects_detected && session.objects_detected.length > 0 ? 
                                    `<strong>Objects:</strong> ${session.objects_detected.join(', ')}` : ''}
                            </div>
                            <div class="activity-time">🕐 ${time}</div>
                        </div>
                    `;
                }).join('');
            } catch (error) {
                console.error('Error loading activities:', error);
            }
        }

        function renderDistribution(distribution) {
            const chartEl = document.getElementById('distributionChart');
            
            if (Object.keys(distribution).length === 0) {
                chartEl.innerHTML = '<div class="empty-state"><p>No data to display</p></div>';
                return;
            }
            
            const total = Object.values(distribution).reduce((a, b) => a + b, 0);
            const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981'];
            
            chartEl.innerHTML = Object.entries(distribution).map(([activity, count], index) => {
                const percentage = (count / total * 100).toFixed(1);
                const color = colors[index % colors.length];
                
                return `
                    <div style="margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: 600; color: #334155;">${activity}</span>
                            <span style="color: #64748b;">${count} (${percentage}%)</span>
                        </div>
                        <div style="background: #e2e8f0; border-radius: 10px; height: 8px; overflow: hidden;">
                            <div style="background: ${color}; height: 100%; width: ${percentage}%; transition: width 0.5s ease;"></div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        async function loadStreams() {
            try {
                const response = await fetch('/api/active_streams');
                if (!response.ok) {
                    console.error('Streams API error:', response.status);
                    return;
                }
                const data = await response.json();
                
                console.log('Active streams:', data.streams);
                
                const gridEl = document.getElementById('videoGrid');
                
                if (data.streams.length === 0) {
                    gridEl.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">📹</div>
                            <h3>No Active Streams</h3>
                            <p>Waiting for cameras to connect...</p>
                            <p style="margin-top: 10px; font-size: 0.9rem;">
                                Make sure:<br>
                                1. ESP32-CAM is powered on<br>
                                2. Connected to WiFi<br>
                                3. Sending images to server<br>
                                4. Check Serial Monitor for errors
                            </p>
                        </div>
                    `;
                    return;
                }
                
                gridEl.innerHTML = data.streams.map(studentId => `
                    <div class="video-container">
                        <div class="video-header">
                            <div class="video-title">📷 ${studentId}</div>
                            <div class="live-indicator">
                                <div class="live-dot"></div>
                                LIVE
                            </div>
                        </div>
                        <img src="/api/stream/${studentId}?t=${Date.now()}" 
                             class="video-frame" 
                             alt="Live stream from ${studentId}"
                             onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'640\' height=\'480\'%3E%3Crect fill=\'%23333\' width=\'640\' height=\'480\'/%3E%3Ctext fill=\'%23fff\' font-size=\'24\' x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\'%3EStream Error%3C/text%3E%3C/svg%3E'">
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading streams:', error);
            }
        }

        async function refreshData() {
            await Promise.all([loadStats(), loadActivities()]);
        }

        async function clearAllData() {
            if (confirm('Are you sure you want to clear all data? This cannot be undone.')) {
                try {
                    await fetch('/api/clear', { method: 'POST' });
                    alert('All data cleared successfully!');
                    refreshData();
                } catch (error) {
                    alert('Error clearing data: ' + error.message);
                }
            }
        }

        // Initial load
        refreshData();

        // Auto-refresh every 5 seconds for dashboard
        refreshInterval = setInterval(() => {
            if (currentTab === 'dashboard') {
                refreshData();
            }
        }, 5000);
    </script>
</body>
</html>
'''

# ============= START SERVER =============
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   🎓 SMART STUDENT ACTIVITY DETECTOR")
    print("=" * 60)
    print(f"✅ Local storage: {DATA_DIR}/")
    print("✅ Live streaming enabled")
    print("✅ Server starting...")
    print("\n📱 Dashboard: http://localhost:5000")
    print("📡 ESP32-CAM Endpoint: POST http://localhost:5000/api/analyze")
    print("📹 Live Stream: http://localhost:5000 (Live Stream Tab)")
    print("\n⚠️  NOTE: Images analyzed in real-time, frames kept in memory")
    print("   Only analysis data is stored to disk")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    
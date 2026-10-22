from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
import json
import base64
import os

app = FastAPI(title="OMR ZipGrade Precision API (40 ข้อ)")

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def read_index():
    if os.path.exists("static/exam.html"):
        with open("static/exam.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>OMR API Running. Please place exam.html in static/ folder.</h1>"

# =========================================================
# 1. ค้นหามาร์เกอร์ 4 มุม
# =========================================================
def find_four_corners(gray_image):
    h_orig, w_orig = gray_image.shape
    
    target_w = 1000
    scale = w_orig / target_w
    target_h = int(h_orig / scale)
    
    resized = cv2.resize(gray_image, (target_w, target_h))
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    min_area = 100
    max_area = 8000
    
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            if h_b == 0: continue
            aspect_ratio = float(w_b) / h_b
            
            if 0.70 <= aspect_ratio <= 1.30:
                solidity = area / float(w_b * h_b)
                if solidity > 0.60:
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        candidates.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    if len(candidates) < 4:
        _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area <= max_area:
                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                if h_b == 0: continue
                aspect_ratio = float(w_b) / h_b
                if 0.70 <= aspect_ratio <= 1.30:
                    solidity = area / float(w_b * h_b)
                    if solidity > 0.60:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            candidates.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    if len(candidates) < 4:
        return None

    tl = min(candidates, key=lambda p: p[0]**2 + p[1]**2)
    tr = min(candidates, key=lambda p: (target_w - p[0])**2 + p[1]**2)
    br = min(candidates, key=lambda p: (target_w - p[0])**2 + (target_h - p[1])**2)
    bl = min(candidates, key=lambda p: p[0]**2 + (target_h - p[1])**2)

    selected = [tl, tr, br, bl]
    if len(set(selected)) < 4:
        return None

    orig_corners = np.array([[p[0] * scale, p[1] * scale] for p in selected], dtype="float32")
    return orig_corners

# =========================================================
# 2. อ่านพิกัด Y จากขีดมาร์กท้ายข้อ (Dynamic Timing Bar Y)
# =========================================================
def detect_row_y_positions(warped_gray, roi_x1, roi_x2, default_y1=354, default_y2=850):
    roi = warped_gray[310:880, roi_x1:roi_x2]
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []
    for c in contours:
        area = cv2.contourArea(c)
        if 25 <= area <= 600:
            x, y, w, h = cv2.boundingRect(c)
            aspect = float(w) / max(1, h)
            if 1.1 <= aspect <= 5.0:
                cy = 310 + y + h // 2
                bars.append(cy)
                
    bars = sorted(bars)
    unique_bars = []
    for b in bars:
        if not unique_bars or (b - unique_bars[-1]) > 12:
            unique_bars.append(b)
            
    if len(unique_bars) == 20:
        return unique_bars
        
    if len(unique_bars) >= 2:
        start_y = unique_bars[0]
        end_y = unique_bars[-1]
        span = end_y - start_y
        if 440 <= span <= 550:
            return [int(start_y + (i / 19.0) * span) for i in range(20)]
            
    return [int(default_y1 + (i / 19.0) * (default_y2 - default_y1)) for i in range(20)]

# =========================================================
# 3. อ่านพิกัด X จากขีดมาร์กหัวคอลัมน์
# =========================================================
def detect_col_x_positions(warped_gray, roi_x1, roi_x2, default_x_list):
    roi = warped_gray[305:340, roi_x1:roi_x2]
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    xs = []
    for c in contours:
        area = cv2.contourArea(c)
        if 20 <= area <= 500:
            x, y, w, h = cv2.boundingRect(c)
            if w > 3 and h > 2:
                cx = roi_x1 + x + w // 2
                xs.append(cx)
    
    xs = sorted(xs)
    unique_xs = []
    for x_val in xs:
        if not unique_xs or (x_val - unique_xs[-1]) > 15:
            unique_xs.append(x_val)
            
    if len(unique_xs) == 4:
        return unique_xs
        
    return default_x_list

# =========================================================
# 4. ถอดรหัส QR Code (พื้นที่ตรงกลางใต้ชุดกา)
# =========================================================
def decode_student_qrcode(warped_color):
    try:
        qr_detector = cv2.QRCodeDetector()
        
        # 1. สแกนทั้งภาพ
        val, _, _ = qr_detector.detectAndDecode(warped_color)
        if val:
            return val

        # 2. สแกนบนภาพ Grayscale
        gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        val_gray, _, _ = qr_detector.detectAndDecode(gray)
        if val_gray:
            return val_gray

        # 3. เจาะสแกนพื้นที่ตรงกลางใต้ชุดกา (y: 840-1040, x: 200-600)
        h, w = warped_color.shape[:2]
        qr_roi = warped_color[int(h*0.75):int(h*0.96), int(w*0.20):int(w*0.80)]
        val_roi, _, _ = qr_detector.detectAndDecode(qr_roi)
        if val_roi:
            return val_roi

        # 4. ภาพ ขาว-ดำ สแกนซ้ำ
        _, thresh_roi = cv2.threshold(cv2.cvtColor(qr_roi, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        val_thresh, _, _ = qr_detector.detectAndDecode(thresh_roi)
        if val_thresh:
            return val_thresh

    except Exception as e:
        print("QR Code decode error:", e)
    return ""

# =========================================================
# 5. API ตรวจข้อสอบ OMR 40 ข้อ
# =========================================================
@app.post("/api/scan")
async def scan_omr(
    file: UploadFile = File(...), 
    answer_key_json: str = Form(...),
    total_questions: int = Form(40),
    max_score: float = Form(15.0),
    has_subjective: bool = Form(False),
    subjective_score: float = Form(0.0)
):
    answer_key = json.loads(answer_key_json)
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. ดักจับ 4 มุม
    corners = find_four_corners(gray)
    if corners is None:
        return {"success": False, "message": "ไม่สามารถตรวจจับสี่เหลี่ยมดำ 4 มุมได้ กรุณาถ่ายภาพให้เห็นมุมกระดาษครบถ้วน"}

    # 2. ทำ Perspective Warp (ปรับจุดศูนย์กลางมาร์กเกอร์ระยะ 45px)
    dst = np.array([[45, 45], [755, 45], [755, 1055], [45, 1055]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners, dst)
    warped_gray = cv2.warpPerspective(gray, M, (800, 1100))
    warped_color = cv2.warpPerspective(image, M, (800, 1100))

    # 3. ถอดรหัส QR Code ดึงเลขที่นักเรียน
    qr_val = decode_student_qrcode(warped_color)
    student_roll = ""
    if qr_val:
        clean_val = qr_val.replace("STU:", "").replace("ROLL:", "").strip()
        if "-" in clean_val:
            student_roll = clean_val.split("-")[-1].strip()
        elif clean_val.isdigit():
            student_roll = clean_val.strip()

    # 4. อ่านพิกัด Y และ X ข้อสอบ 1-40
    col1_y_positions = detect_row_y_positions(warped_gray, 280, 325, 354, 850)
    col2_y_positions = detect_row_y_positions(warped_gray, 650, 695, 354, 850)

    default_col1_x = [167, 202, 237, 272]
    default_col2_x = [533, 568, 603, 638]

    col1_x = detect_col_x_positions(warped_gray, 150, 285, default_col1_x)
    col2_x = detect_col_x_positions(warped_gray, 515, 650, default_col2_x)

    THAI_CHOICES = ['ก', 'ข', 'ค', 'ง']
    ENG_TO_THAI = {'A': 'ก', 'B': 'ข', 'C': 'ค', 'D': 'ง'}

    detected_answers = {}
    raw_score = 0
    active_questions = min(max(1, total_questions), 40)
    radius = 10

    # 5. สแกนและวาดจุดแสดงผล
    for q in range(1, 40 + 1):
        is_col1 = q <= 20
        row_idx = (q - 1) if is_col1 else (q - 21)
        
        u_y = col1_y_positions[row_idx] if is_col1 else col2_y_positions[row_idx]
        x_coords = col1_x if is_col1 else col2_x

        base_x = 110 if is_col1 else 475
        baseline_samples = []
        for dy in range(-3, 4, 2):
            for dx in range(-3, 4, 2):
                py, px = u_y + dy, base_x + dx
                if 0 <= py < 1100 and 0 <= px < 800:
                    baseline_samples.append(warped_gray[py, px])
        
        baseline_brightness = np.mean(baseline_samples) if baseline_samples else 210.0

        darkest_choice = None
        min_avg_circle = 255.0

        for opt_idx, ch in enumerate(THAI_CHOICES):
            u_x = x_coords[opt_idx]

            mask = np.zeros(warped_gray.shape, dtype="uint8")
            cv2.circle(mask, (u_x, u_y), radius, 255, -1)
            circle_pixels = warped_gray[mask == 255]
            avg_circle = np.mean(circle_pixels) if len(circle_pixels) > 0 else 255.0

            is_filled = (avg_circle < baseline_brightness * 0.80) and (avg_circle < 170)

            if is_filled and avg_circle < min_avg_circle:
                min_avg_circle = avg_circle
                darkest_choice = ch

        detected_answers[str(q)] = darkest_choice

        if q <= active_questions:
            raw_ans = answer_key.get(str(q)) or answer_key.get(q, 'ก')
            correct_ans = ENG_TO_THAI.get(raw_ans, raw_ans)
            
            is_correct = (darkest_choice == correct_ans)
            if is_correct:
                raw_score += 1

            for opt_idx, ch in enumerate(THAI_CHOICES):
                u_x = x_coords[opt_idx]
                is_this_correct_key = (ch == correct_ans)
                is_this_student_pick = (ch == darkest_choice)

                if is_this_student_pick and is_this_correct_key:
                    # 🟢 ตอบถูก: วงกลมสีเขียวทึบ
                    cv2.circle(warped_color, (u_x, u_y), radius+2, (0, 200, 0), -1)
                    cv2.circle(warped_color, (u_x, u_y), radius+2, (0, 150, 0), 2)
                elif is_this_student_pick and not is_this_correct_key:
                    # 🔴 ตอบผิด: แสดงวงกลมสีแดงบนข้อผิด
                    cv2.circle(warped_color, (u_x, u_y), radius+2, (0, 0, 255), 3)
                elif is_this_correct_key and not is_this_student_pick:
                    # 🟢 เฉลย: แสดงวงกลมสีเขียวโปร่งตรงข้อที่ถูก
                    cv2.circle(warped_color, (u_x, u_y), radius+2, (0, 220, 0), 2)
                else:
                    cv2.circle(warped_color, (u_x, u_y), radius, (248, 189, 56), 1)

    # 6. คำนวณสรุปคะแนน
    subj_max = subjective_score if has_subjective else 0.0
    obj_max = max(0.0, max_score - subj_max)
    objective_scaled = round((raw_score / active_questions) * obj_max, 2)
    final_scaled = round(objective_scaled + subj_max, 2)

    _, buffer = cv2.imencode('.png', warped_color)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "success": True,
        "student_roll": student_roll,
        "raw_score": raw_score,
        "total_questions": active_questions,
        "objective_scaled": objective_scaled,
        "objective_max": obj_max,
        "final_scaled": final_scaled,
        "max_score": max_score,
        "scanned_answers": detected_answers,
        "annotated_image": f"data:image/png;base64,{img_base64}"
    }
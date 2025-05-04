BASE_FOLDER = 'static/images'  # Base folder path for images
CASE_FOLDERS = ['case1', 'case2', 'case3', 'case4']
def calculate_rust_percentage(base_img, test_img):
    # Convert to HSV
    hsv_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    hsv_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2HSV)

    # Define a range for rust color in HSV (you can adjust as needed)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # Create masks for rust areas
    base_mask = cv2.inRange(hsv_base, lower_red, upper_red)
    test_mask = cv2.inRange(hsv_test, lower_red, upper_red)

    # Find overlapping rust regions (base vs test)
    overlap = cv2.bitwise_and(base_mask, test_mask)
    match_pixels = np.count_nonzero(overlap)
    total_pixels = np.count_nonzero(base_mask)

    if total_pixels == 0:
        return 0.0  # Avoid division by zero if no rust in base

    match_percent = (match_pixels / total_pixels) * 100
    return match_percent
def detect_rust_and_damage_percentage(image):
    """Detect rust (red color) and damage (bare iron) percentage in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red ranges in HSV for rust detection
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define low saturation and high value for detecting bare iron (damaged surface)
    lower_bare = np.array([0, 0, 100])  # Low saturation, high value (bare metal)
    upper_bare = np.array([180, 50, 255])

    # Mask for rust (red) and bare metal (damage)
    mask_rust1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_rust2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_bare = cv2.inRange(hsv, lower_bare, upper_bare)

    # Combine the masks for rust and bare metal (damage)
    rust_pixels = cv2.countNonZero(mask_rust1) + cv2.countNonZero(mask_rust2)
    damage_pixels = cv2.countNonZero(mask_bare)
    total_pixels = image.shape[0] * image.shape[1]

    rust_percentage = (rust_pixels / total_pixels) * 100
    damage_percentage = (damage_pixels / total_pixels) * 100

    return rust_percentage, damage_percentage

def get_max_values_for_case(case_folder, base_image):
    """Get the max rust and damage percentage for a given case folder."""
    max_rust = 0
    max_damage = 0
    
    # Go through each side image in the case folder (side1.jpg, side2.jpg, side3.jpg, side4.jpg)
    for side in range(1, 5):
        side_image_path = os.path.join(BASE_FOLDER, case_folder, f"side{side}.jpg")
        
        # Check if the side image exists
        if not os.path.exists(side_image_path):
            continue
        
        # Read the side image
        side_img = cv2.imread(side_image_path)
        
        # Calculate rust and damage percentages
        rust_percent, damage_percent = detect_rust_and_damage_percentage(side_img)
        
        # Update the max values if current values are higher
        max_rust = max(max_rust, rust_percent)
        max_damage = max(max_damage, damage_percent)
    
    return max_rust, max_damage
    
@app.route('/analyze_corrosion', methods=['POST'])
def analyze_corrosion():
    uploaded_images = []
    rust_scores_by_case = {case: [] for case in CASE_FOLDERS}

    # Step 1: Read uploaded images (side1 to side4)
    for i in range(1, 5):
        file = request.files.get(f'side{i}')
        if not file:
            print(f"❌ Missing uploaded image: side{i}")
            return jsonify({"error": f"Missing image: side{i}"}), 400

        # print(f"✅ Received: {file.filename}")
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": f"Cannot decode image: side{i}"}), 400
        img = cv2.resize(img, (300, 300))
        uploaded_images.append(img)

    # Step 2: Compare each uploaded image to corresponding base image in each case
    for case in CASE_FOLDERS:
        # print(f"📁 Analyzing against: {case}")
        case_scores = []
        for i in range(1, 5):  # side1 to side4
            base_img_path = os.path.join(BASE_FOLDER, case, f'side{i}.jpg')
            if not os.path.exists(base_img_path):
                # print(f"⚠️ Missing base image: {base_img_path}")
                case_scores.append(0)
                continue

            base_img = cv2.imread(base_img_path)
            base_img = cv2.resize(base_img, (300, 300))

            # Calculate rust/damage % between base and uploaded
            rust_percent = calculate_rust_percentage(base_img, uploaded_images[i - 1])
            # print(f"🧪 Rust match {case} side{i}: {rust_percent:.2f}%")
            case_scores.append(round(rust_percent))

        rust_scores_by_case[case] = case_scores

    # Step 3: Return average rust match per case + individual side details
    rust_summary = [round(np.mean(rust_scores_by_case[case])) for case in CASE_FOLDERS]
    # print("📊 Final rust match summary:", rust_summary)

    return jsonify({
        "damage_per_image": rust_summary,
        "details": {
            case: {
                f"side{i+1}": rust_scores_by_case[case][i]
                for i in range(4)
            } for case in CASE_FOLDERS
        }
    })

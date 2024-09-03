import os
import cv2
from ultralytics import YOLO

# Define paths
input_path = os.path.join('.', 'input')  # Directory for video or image files
output_path = os.path.join('.', 'output')  # Directory for output

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Define input file (update this to your file name)
input_file = 'input_file.mp4'  # Change this to your input file name

# Set paths
input_path = os.path.join(input_path, input_file)
output_file = '{}_out.mp4'.format(os.path.splitext(input_file)[0]) if input_file.lower().endswith(('.mp4', '.avi', '.mov')) else '{}_out.jpg'.format(os.path.splitext(input_file)[0])
output_path = os.path.join(output_path, output_file)

# Load YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)
threshold = 0.5

def process_image(image):
    results = model(image)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            # Draw the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            
            # Prepare the label with confidence level in percentage
            label = f'{results.names[int(class_id)].upper()} {score*100:.2f}%'
            
            # Draw the label above the bounding box
            cv2.putText(image, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    return image

try:
    # Process based on input file type
    if input_file.lower().endswith(('.mp4', '.avi', '.mov')):
        # Video file processing
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError("Error: Unable to open video file.")
        
        ret, frame = cap.read()
        if ret:
            H, W, _ = frame.shape
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
            
            while ret:
                processed_frame = process_image(frame)
                out.write(processed_frame)
                ret, frame = cap.read()
            
            cap.release()
            out.release()
        else:
            raise IOError("Error: Unable to read video file.")
    else:
        # Image file processing
        image = cv2.imread(input_path)
        if image is not None:
            processed_image = process_image(image)
            cv2.imwrite(output_path, processed_image)
        else:
            raise IOError("Error: Unable to read image file.")
    
    print(f"Processing complete. Output saved to {output_path}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    cv2.destroyAllWindows()
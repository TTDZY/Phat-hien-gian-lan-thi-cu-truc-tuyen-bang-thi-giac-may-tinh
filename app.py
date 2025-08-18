import cv2
import typer
from ultralytics import YOLO
from supervision.annotators.core import BoundingBoxAnnotator, LabelAnnotator
import supervision as sv

#Khởi tạo ứng dụng và model
app = typer.Typer()
model = YOLO("best.pt")

#Hàm xử lý video/webcam
def process_webcam(output_file="output.mp4"):
    cap = cv2.VideoCapture("vid.mp4")  # Webcam, hoặc "vid.mp4" cho file video

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

#Tạo annotator
    bounding_box_annotator = BoundingBoxAnnotator()
    label_annotator = LabelAnnotator()
#Vòng lặp xử lý từng frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        out.write(annotated_frame)
        cv2.imshow("Webcam", annotated_frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
#Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()
#Lệnh CLI
@app.command()
def webcam(output_file: str = "output.mp4"):
    typer.echo("Starting webcam processing...")
    process_webcam(output_file)

if __name__ == "__main__":
    app()

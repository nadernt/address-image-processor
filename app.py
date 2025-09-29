import argparse
import boto3
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO

def main(sub, project_id, image_name, bucket_name="dev-projects-g2y"):
    s3 = boto3.client("s3", region_name="us-east-1")

    # Download file from S3
    s3_key = f"{sub}/{project_id}/{image_name}"
    tmp_in = tempfile.NamedTemporaryFile(delete=False)
    tmp_in.close()  # Close it so Windows allows boto3 to rename/overwrite
    s3.download_file(bucket_name, s3_key, tmp_in.name)
    img = cv2.imread(tmp_in.name)

    # Load models from local folder
    face_model = YOLO(os.path.join("models", "model.pt"))
    lp_model = YOLO(os.path.join("models", "best.pt"))

    # Blur faces
    faces = face_model(img)[0].boxes.xyxy
    for x1, y1, x2, y2 in faces:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    # Blur license plates
    plates = lp_model(img)[0].boxes.xyxy
    for x1, y1, x2, y2 in plates:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    # Save result to temporary file
    out_path = os.path.join(tempfile.gettempdir(), "out.jpg")
    cv2.imwrite(out_path, img)

    # Upload back to S3
    out_key = f"{sub}/{project_id}/out.jpg"
    s3.upload_file(out_path, bucket_name, out_key)

    print(f"✅ Anonymized image saved to s3://{bucket_name}/{out_key}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize faces and plates in images from S3")
    parser.add_argument("--clientSub", required=True, help="Project folder of client in S3 bucket")
    parser.add_argument("--projectId", required=True, help="Project folder in S3 bucket")
    parser.add_argument("--imageName", required=True, help="Image filename in S3 bucket")
    args = parser.parse_args()

    main(args.clientSub, args.projectId, args.imageName)

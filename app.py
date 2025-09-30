import os
import tempfile
import cv2
import boto3
from ultralytics import YOLO
import runpod


# Load models once when container starts (warm start optimization)
face_model = YOLO(os.path.join("models", "model.pt"))
lp_model = YOLO(os.path.join("models", "best.pt"))


def handler(event):
    """
    RunPod handler function
    event['input'] must contain:
        - clientSub
        - projectId
        - imageName
    """
    input_data = event["input"]

    client_sub = input_data.get("clientSub")
    project_id = input_data.get("projectId")
    image_name = input_data.get("imageName")
    bucket_name = os.environ.get("S3_BUCKET", "dev-projects-g2y")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    if not (client_sub and project_id and image_name):
        return {"error": "Missing one of: clientSub, projectId, imageName"}

    s3 = boto3.client("s3", region_name=region)
    s3_key = f"{client_sub}/{project_id}/{image_name}"

    # Download input file
    tmp_in = tempfile.NamedTemporaryFile(delete=False)
    tmp_in.close()
    try:
        s3.download_file(bucket_name, s3_key, tmp_in.name)
    except Exception as e:
        return {"error": f"Failed to download {s3_key}: {str(e)}"}

    img = cv2.imread(tmp_in.name)
    if img is None:
        return {"error": f"cv2.imread failed for {s3_key}"}

    # Blur faces
    faces = face_model(img)[0].boxes.xyxy
    for x1, y1, x2, y2 in faces:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    # Blur plates
    plates = lp_model(img)[0].boxes.xyxy
    for x1, y1, x2, y2 in plates:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)

    # Save result and upload
    out_path = os.path.join(tempfile.gettempdir(), "out.jpg")
    cv2.imwrite(out_path, img)

    out_key = f"{client_sub}/{project_id}/tmp/{image_name}"
    try:
        s3.upload_file(out_path, bucket_name, out_key)
    except Exception as e:
        return {"error": f"Failed to upload {out_key}: {str(e)}"}

    return {
        "status": "success",
        "output_s3_uri": f"s3://{bucket_name}/{out_key}"
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

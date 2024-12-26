import cv2
import os
import csv
import sys
from openpose import pyopenpose as op

# OpenPose 設定
params = dict()
params["model_folder"] = "./models/"  # OpenPose 模型目錄

# 定義輸入影片的目錄和輸出 CSV 的目錄
input_video_dir = "./videos"  # 輸入影片目錄
output_csv_dir = "./csv_outputs"  # 輸出 CSV 目錄

# 確保輸出目錄存在
os.makedirs(output_csv_dir, exist_ok=True)

# 遍歷所有 .MP4 格式的影片
video_files = [f for f in os.listdir(input_video_dir) if f.endswith(".MP4")]

if not video_files:
    print("No .MP4 files found in the directory.")
    sys.exit(1)

# 初始化 OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# 處理每個影片
for video_file in video_files:
    video_path = os.path.join(input_video_dir, video_file)
    output_csv = os.path.join(output_csv_dir, f"{os.path.splitext(video_file)[0]}.csv")
    
    # 載入影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_file}. Skipping...")
        continue

    # 開啟對應的 CSV 檔案寫入
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # 寫入標題行
        writer.writerow(["Frame", "Person", "Keypoint", "X", "Y", "Confidence"])

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 處理當前幀
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            # 獲取骨架數據
            keypoints = datum.poseKeypoints

            # 如果有檢測到骨架
            if keypoints is not None:
                for person_idx, person in enumerate(keypoints):
                    for keypoint_idx, keypoint in enumerate(person):
                        # 寫入每個節點的數據 (Frame, Person, Keypoint, X, Y, Confidence)
                        writer.writerow([frame_id, person_idx, keypoint_idx, keypoint[0], keypoint[1], keypoint[2]])

            frame_id += 1

    cap.release()
    print(f"Processed {video_file} -> {output_csv}")

print("All videos processed successfully.")


import os
import cv2
import re
import argparse

def images_to_video(image_folder, fps=500):
    if not os.path.isdir(image_folder):
        print(f"Error: Path {image_folder} does not exist!")
        return

    # get the parent folder name
    parent_folder_name = os.path.basename(os.path.abspath(image_folder))
    output_video_path = os.path.join(image_folder, f"{parent_folder_name}.mp4")

    # get all image files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No image files found!")
        return

    # Define a function to extract numbers from filenames for sorting
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else -1

    image_files.sort(key=extract_number)

    # get image size
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read {img_path}, skipped.")
            continue

        video.write(img)

    video.release()
    print(f"Video generated: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert images in a folder into an MP4 video.')
    parser.add_argument('--path_to_data', type=str, required=True, help="Path to the image folder")
    parser.add_argument('--fps', type=int, default=500, help='Output frame rate')
    args = parser.parse_args()

    images_to_video(args.path_to_data, fps=args.fps)


if __name__ == "__main__":
    main()

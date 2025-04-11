import zipfile
import os
import shutil
from PIL import Image
from pytorch_fid import fid_score

def score(submission_zip_path, solution_dir_path):
    submission_extract_path = 'submission_imgs'

    with zipfile.ZipFile(submission_zip_path, 'r') as zip_ref:
        zip_ref.extractall(submission_extract_path)

    for filename in os.listdir(submission_extract_path):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(submission_extract_path, filename)
            img = Image.open(img_path).convert('RGB')
            new_name = os.path.splitext(filename)[0] + ".png"
            img.save(os.path.join(submission_extract_path, new_name))
            os.remove(img_path)

    valid_exts = ['.png']
    num_imgs = len([f for f in os.listdir(submission_extract_path) if os.path.splitext(f)[1].lower() in valid_exts])
    print(f"Found {num_imgs} images in submission folder")

    if num_imgs == 0:
        raise ValueError("No valid .png images found in submission folder.")

    gt_path = os.path.join(solution_dir_path, 'ground_truth')

    fid_value = fid_score.calculate_fid_given_paths(
        [submission_extract_path, gt_path],
        batch_size=50,
        device='cpu',
        dims=2048
    )

    shutil.rmtree(submission_extract_path)

    return float(fid_value)

if __name__ == "__main__":
    submission_zip_path = 'submission.zip'
    solution_dir_path = './'
    fid_value = score(submission_zip_path, solution_dir_path)
    print(f"FID Score: {fid_value}")

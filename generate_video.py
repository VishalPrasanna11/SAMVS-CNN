import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from config import get_config
from generate_summary import generate_summary
from model import set_model
from video_helper import VideoPreprocessor

# def pick_frames(video_path, selections, target_resolution):
#     """Extract frames based on selections and resize to target resolution."""
#     cap = cv2.VideoCapture(str(video_path))
#     frames = []
#     n_frames = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if selections[n_frames]:
#             # Resize frame to target resolution
#             frame = cv2.resize(frame, target_resolution)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(frame)
#         n_frames += 1

#     cap.release()
#     return frames
def pick_frames(video_path, selections, target_resolution):
    """Extract frames based on selections and resize to target resolution."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    n_frames = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Original video FPS
    frame_interval = int(frame_rate / 8)  # Frame interval to pick frames at 8 FPS

    if frame_interval <= 0:
        raise ValueError("Target FPS must be less than or equal to the source video FPS.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if n_frames < len(selections) and selections[n_frames]:  # Ensure within bounds and check selection
            # Resize frame to target resolution
            frame = cv2.resize(frame, target_resolution)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        n_frames += 1

    cap.release()
    return frames

def produce_video(save_path, frames, fps, frame_size):
    """Generate a video from the selected frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()

def main():
    # Load config
    config = get_config()

    # Create output directory
    out_dir = Path(config.save_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature extractor
    video_proc = VideoPreprocessor(
        sample_rate=config.sample_rate,
        device=config.device
    )

    # Search all videos with .mp4 suffix
    video_paths = [Path(config.file_path)] if config.input_is_file else sorted(Path(config.dir_path).glob(f'*.{config.ext}'))

    # Load model weights
    model = set_model(
        model_name=config.model_name,
        Scale=config.Scale,
        Softmax_axis=config.Softmax_axis,
        Balance=config.Balance,
        Positional_encoding=config.Positional_encoding,
        Positional_encoding_shape=config.Positional_encoding_shape,
        Positional_encoding_way=config.Positional_encoding_way,
        Dropout_on=config.Dropout_on,
        Dropout_ratio=config.Dropout_ratio,
        Classifier_on=config.Classifier_on,
        CLS_on=config.CLS_on,
        CLS_mix=config.CLS_mix,
        key_value_emb=config.key_value_emb,
        Skip_connection=config.Skip_connection,
        Layernorm=config.Layernorm
    )
    model.load_state_dict(torch.load(config.weight_path, map_location='cpu'))
    model.to(config.device)
    model.eval()

    target_resolution = (config.target_width, config.target_height)  # Set target resolution

    # Generate summarized videos
    with torch.no_grad():
        for video_path in tqdm(video_paths, total=len(video_paths), ncols=80, desc="Processing Videos"):
            try:
                video_name = video_path.stem
                tqdm.write(f"Processing video: {video_name}")

                n_frames, features, cps, pick = video_proc.run(video_path)
                inputs = features.to(config.device)
                inputs = inputs.unsqueeze(0).expand(3, -1, -1).unsqueeze(0)
                outputs = model(inputs)

                predictions = outputs.squeeze().clone().detach().cpu().numpy().tolist()
                selections = generate_summary([cps], [predictions], [n_frames], [pick])[0]

                frames = pick_frames(video_path=video_path, selections=selections, target_resolution=target_resolution)
                produce_video(
                    save_path=f'{config.save_path}/{video_name}.mp4',
                    frames=frames,
                    fps=video_proc.fps,
                    frame_size=target_resolution
                )
            except Exception as e:
                tqdm.write(f"Error processing {video_name}: {e}")

if __name__ == '__main__':
    main()

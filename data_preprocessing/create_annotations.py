# -*- coding: utf-8 -*-

import os

base_root = '/Users/valentineo/MLProjects/Dissertation/datasets/RAVDESS'
video_root = os.path.join(base_root, 'videos')
audio_root = os.path.join(base_root, 'audio')

n_folds = 1
folds = [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]]
for fold in range(n_folds):
    fold_ids = folds[fold]
    test_ids, val_ids, train_ids = fold_ids

    # annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
    annotation_file = 'annotations.txt'

    # Clear the annotation file if it exists
    with open(annotation_file, 'w') as f:
        pass

    for i, actor in enumerate(os.listdir(video_root)):
        # Skip hidden files
        if actor.startswith('.'):
            continue

        actor_video_path = os.path.join(video_root, actor)
        actor_audio_path = os.path.join(audio_root, actor)

        # Skip if not a directory
        if not os.path.isdir(actor_video_path):
            continue

        for video in os.listdir(actor_video_path):
            if not video.endswith('.npy') or 'croppad' not in video:
                continue

            # Extract the emotion label
            label = str(int(video.split('-')[2]))

            # Construct corresponding audio filename
            audio = '03' + video.split('_face')[0][2:] + '_cropped.wav'

            # Determine the split type based on actor index
            split_type = 'training' if i in train_ids else 'validation' if i in val_ids else 'testing'

            # Write to annotation file
            with open(annotation_file, 'a') as f:
                f.write(
                    f"{os.path.join(actor_video_path, video)};{os.path.join(actor_audio_path, audio)};{label};{split_type}\n")

    print(f"Annotation file created successfully at: {os.path.abspath(annotation_file)}")

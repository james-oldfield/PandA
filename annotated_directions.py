annotated_directions = {
    'stylegan2_ffhq1024': {
        # Directions used in paper with a single decomposition:
        'big_eyes': {
            'parameters': [7, 6, 30],  # used in main paper
            'layer': 5,
            'ranks': [512, 8],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_5-rank_8.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_5-rank_512.npy',
            ],
        },
        'long_nose': {
            'parameters': [5, 82, 30],  # used in main paper
            'layer': 5,
            'ranks': [512, 8],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_5-rank_8.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_5-rank_512.npy',
            ],
        },
        'smile': {
            'parameters': [4, 46, -30],  # used in sup. material
            'layer': 5,
            'ranks': [512, 8],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_5-rank_8.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_5-rank_512.npy',
            ],
        },
        'open_mouth': {
            'parameters': [4, 39, 30],  # used in sup. material
            'layer': 5,
            'ranks': [512, 8],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_5-rank_8.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_5-rank_512.npy',
            ],
        },

        # Additional directions
        'big_eyeballs': {
            'parameters': [8, 27, 100],
            'layer': 6,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_6-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_6-rank_512.npy',
            ],
        },
        'wide_nose': {
            'parameters': [15, 13, 100],
            'layer': 6,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_6-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_6-rank_512.npy',
            ],
        },
        'glance_left': {
            'parameters': [8, 281, 50],
            'layer': 6,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_6-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_6-rank_512.npy',
            ],
        },
        'glance_right': {
            'parameters': [8, 281, -70],
            'layer': 6,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_6-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_6-rank_512.npy',
            ],
        },
        'bald_forehead': {
            'parameters': [3, 25, 100],
            'layer': 6,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_6-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_6-rank_512.npy',
            ],
        },
        'light_eyebrows': {
            'parameters': [8, 4, 30],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'dark_eyebrows': {
            'parameters': [8, 9, 30],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'no_eyebrows': {
            'parameters': [8, 4, 50],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'dark_eyes': {
            'parameters': [11, 176, 50],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'red_eyes': {
            'parameters': [11, 109, 60],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'eyes_short': {
            'parameters': [11, 262, 70],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'eyes_open': {
            'parameters': [11, 28, 50],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'eyes_close': {
            'parameters': [11, 398, 80],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },
        'no_eyes': {
            'parameters': [11, 0, -200],
            'layer': 7,
            'ranks': [512, 16],
            'checkpoints_path': [
                './checkpoints/Us-name_stylegan2_ffhq1024-layer_7-rank_16.npy',
                './checkpoints/Uc-name_stylegan2_ffhq1024-layer_7-rank_512.npy',
            ],
        },

    },

}

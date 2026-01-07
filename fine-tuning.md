=== Catégorie hazelnut ===
  -> terminé pour hazelnut

=== Catégorie leather ===
  -> terminé pour leather

=== Catégorie metal_nut ===
  -> terminé pour metal_nut

=== Catégorie pill ===
  -> terminé pour pill

=== Catégorie screw ===
  -> terminé pour screw

=== Catégorie tile ===
  -> terminé pour tile

=== Catégorie toothbrush ===
  -> terminé pour toothbrush

=== Catégorie transistor ===
  -> terminé pour transistor

=== Catégorie wood ===
  -> terminé pour wood

=== Catégorie zipper ===
  -> terminé pour zipper

=== Résumé ===
Nb classes YOLO : 73
Classes (id -> name) :
   0: bottle_broken_large
   1: bottle_broken_small
   2: bottle_contamination
   3: cable_bent_wire
   4: cable_cable_swap
   5: cable_combined
   6: cable_cut_inner_insulation
   7: cable_cut_outer_insulation
   8: cable_missing_cable
   9: cable_missing_wire
  10: cable_poke_insulation
  11: capsule_crack
  12: capsule_faulty_imprint
  13: capsule_poke
  14: capsule_scratch
  15: capsule_squeeze
  16: carpet_color
  17: carpet_cut
  18: carpet_hole
  19: carpet_metal_contamination
  20: carpet_thread
  21: grid_bent
  22: grid_broken
  23: grid_glue
  24: grid_metal_contamination
  25: grid_thread
  26: hazelnut_crack
  27: hazelnut_cut
  28: hazelnut_hole
  29: hazelnut_print
  30: leather_color
  31: leather_cut
  32: leather_fold
  33: leather_glue
  34: leather_poke
  35: metal_nut_bent
  36: metal_nut_color
  37: metal_nut_flip
  38: metal_nut_scratch
  39: pill_color
  40: pill_combined
  41: pill_contamination
  42: pill_crack
  43: pill_faulty_imprint
  44: pill_pill_type
  45: pill_scratch
  46: screw_manipulated_front
  47: screw_scratch_head
  48: screw_scratch_neck
  49: screw_thread_side
  50: screw_thread_top
  51: tile_crack
  52: tile_glue_strip
  53: tile_gray_stroke
  54: tile_oil
  55: tile_rough
  56: toothbrush_defective
  57: transistor_bent_lead
  58: transistor_cut_lead
  59: transistor_damaged_case
  60: transistor_misplaced
  61: wood_color
  62: wood_combined
  63: wood_hole
  64: wood_liquid
  65: wood_scratch
  66: zipper_broken_teeth
  67: zipper_combined
  68: zipper_fabric_border
  69: zipper_fabric_interior
  70: zipper_rough
  71: zipper_split_teeth
  72: zipper_squeezed_teeth

Dataset YOLO créé dans : C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all
Config YOLO écrite dans : C:\Users\othni\Projects\mvtec_ad\mvtec_yolo_all.yaml
(.venv) PS C:\Users\othni\Projects\mvtec_ad> pip install ultralytics
Collecting ultralytics
  Using cached ultralytics-8.3.233-py3-none-any.whl.metadata (37 kB)
Requirement already satisfied: numpy>=1.23.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (2.3.5)
Requirement already satisfied: matplotlib>=3.3.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (3.10.7)
Collecting opencv-python>=4.6.0 (from ultralytics)
  Using cached opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl.metadata (19 kB)
Requirement already satisfied: pillow>=7.1.2 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (12.0.0)
Requirement already satisfied: pyyaml>=5.3.1 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (6.0.3)
Requirement already satisfied: requests>=2.23.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (2.32.5)
Requirement already satisfied: scipy>=1.4.1 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (1.16.3)
Requirement already satisfied: torch>=1.8.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (2.9.1)
Requirement already satisfied: torchvision>=0.9.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (0.24.1)
Requirement already satisfied: psutil>=5.8.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from ultralytics) (7.1.3)
Collecting polars>=0.20.0 (from ultralytics)
  Using cached polars-1.35.2-py3-none-any.whl.metadata (10 kB)
Collecting ultralytics-thop>=2.0.18 (from ultralytics)
  Using cached ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (1.3.3)
Requirement already satisfied: cycler>=0.10 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (4.61.0)     
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (1.4.9)
Requirement already satisfied: packaging>=20.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (25.0)
Requirement already satisfied: pyparsing>=3 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (3.2.5)
Requirement already satisfied: python-dateutil>=2.7 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)
Collecting numpy>=1.23.0 (from ultralytics)
  Using cached numpy-2.2.6-cp312-cp312-win_amd64.whl.metadata (60 kB)
Collecting polars-runtime-32==1.35.2 (from polars>=0.20.0->ultralytics)
  Using cached polars_runtime_32-1.35.2-cp39-abi3-win_amd64.whl.metadata (1.5 kB)
Requirement already satisfied: six>=1.5 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from requests>=2.23.0->ultralytics) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from requests>=2.23.0->ultralytics) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from requests>=2.23.0->ultralytics) (2.5.0)      
Requirement already satisfied: certifi>=2017.4.17 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from requests>=2.23.0->ultralytics) (2025.11.12) 
Requirement already satisfied: filelock in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (3.20.0)
Requirement already satisfied: typing-extensions>=4.10.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (4.15.0)  
Requirement already satisfied: sympy>=1.13.3 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (3.6)
Requirement already satisfied: jinja2 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (2025.10.0)
Requirement already satisfied: setuptools in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from torch>=1.8.0->ultralytics) (80.9.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from sympy>=1.13.3->torch>=1.8.0->ultralytics) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\othni\projects\mvtec_ad\.venv\lib\site-packages (from jinja2->torch>=1.8.0->ultralytics) (3.0.3)     
Using cached ultralytics-8.3.233-py3-none-any.whl (1.1 MB)
Using cached opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl (39.0 MB)
Using cached numpy-2.2.6-cp312-cp312-win_amd64.whl (12.6 MB)
Using cached polars-1.35.2-py3-none-any.whl (783 kB)
Using cached polars_runtime_32-1.35.2-cp39-abi3-win_amd64.whl (41.3 MB)
Using cached ultralytics_thop-2.0.18-py3-none-any.whl (28 kB)
Installing collected packages: polars-runtime-32, numpy, polars, opencv-python, ultralytics-thop, ultralytics
  Attempting uninstall: numpy
    Found existing installation: numpy 2.3.5
    Uninstalling numpy-2.3.5:
      Successfully uninstalled numpy-2.3.5
Successfully installed numpy-2.2.6 opencv-python-4.12.0.88 polars-1.35.2 polars-runtime-32-1.35.2 ultralytics-8.3.233 ultralytics-thop-2.0.18
(.venv) PS C:\Users\othni\Projects\mvtec_ad> yolo detect train `
>>     model=yolov8n.pt `
>>     data=mvtec_yolo_all.yaml `
>>     imgsz=640 `
>>     epochs=100 `
>>     batch=16 `
>>     workers=0
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt to 'yolov8n.pt': 100% ━━━━━━━━━━━━ 6.2MB 2.3MB/s 2.7s
Ultralytics 8.3.233  Python-3.12.6 torch-2.9.1+cpu CPU (12th Gen Intel Core i5-12450H)
engine\trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=mvtec_yolo_all.yaml, degrees=0.0, deterministic=True, device=cpu, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=C:\Users\othni\Projects\mvtec_ad\runs\detect\train, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=0, workspace=None
Overriding model.yaml nc=80 with nc=73

                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    837205  ultralytics.nn.modules.head.Detect           [73, [64, 128, 256]]
Model summary: 129 layers, 3,096,741 parameters, 3,096,725 gradients, 8.6 GFLOPs

Transferred 319/355 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
train: Fast image access  (ping: 0.30.1 ms, read: 73.717.1 MB/s, size: 1294.9 KB)
train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 72 images, 23 backgrounds, 0 corrupt: 2% ──────────── 72/4606 148.7it/s 0.5s<30.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 103 images, 54 backgrounds, 0 corrupt: 2% ──────────── 103/4606 193.5it/s 0.6s<2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 135 images, 86 backgrounds, 0 corrupt: 3% ──────────── 135/4606 227.2it/s 0.7s<1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 168 images, 119 backgrounds, 0 corrupt: 4% ──────────── 168/4606 254.8it/s 0.8s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 201 images, 152 backgrounds, 0 corrupt: 4% ╸─────────── 201/4606 272.7it/s 0.9s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 232 images, 183 backgrounds, 0 corrupt: 5% ╸─────────── 232/4606 283.6it/s 1.0s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 259 images, 209 backgrounds, 0 corrupt: 6% ╸─────────── 259/4606 249.8it/s 1.1s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 276 images, 209 backgrounds, 0 corrupt: 6% ╸─────────── 276/4606 224.2it/s 1.2s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 284 images, 209 backgrounds, 0 corrupt: 6% ╸─────────── 284/4606 178.9it/s 1.4s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 294 images, 209 backgrounds, 0 corrupt: 6% ╸─────────── 294/4606 149.0it/s 1.5s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 311 images, 216 backgrounds, 0 corrupt: 7% ╸─────────── 311/4606 151.8it/s 1.6s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 327 images, 232 backgrounds, 0 corrupt: 7% ╸─────────── 327/4606 149.3it/s 1.7s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 344 images, 249 backgrounds, 0 corrupt: 7% ╸─────────── 344/4606 154.3it/s 1.8s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 360 images, 265 backgrounds, 0 corrupt: 8% ╸─────────── 360/4606 154.7it/s 1.9s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 375 images, 280 backgrounds, 0 corrupt: 8% ╸─────────── 375/4606 151.1it/s 2.0s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 385 images, 290 backgrounds, 0 corrupt: 8% ━─────────── 385/4606 124.3it/s 2.2s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 397 images, 302 backgrounds, 0 corrupt: 9% ━─────────── 397/4606 122.4it/s 2.3s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 415 images, 320 backgrounds, 0 corrupt: 9% ━─────────── 415/4606 131.5it/s 2.4s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 432 images, 337 backgrounds, 0 corrupt: 9% ━─────────── 432/4606 135.8it/s 2.5s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 446 images, 351 backgrounds, 0 corrupt: 10% ━─────────── 446/4606 136.9it/s 2.6strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 458 images, 363 backgrounds, 0 corrupt: 10% ━─────────── 458/4606 130.4it/s 2.7strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 476 images, 381 backgrounds, 0 corrupt: 10% ━─────────── 476/4606 130.1it/s 2.8strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 494 images, 399 backgrounds, 0 corrupt: 11% ━─────────── 494/4606 134.1it/s 3.0strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 510 images, 415 backgrounds, 0 corrupt: 11% ━─────────── 510/4606 140.9it/s 3.1strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 527 images, 432 backgrounds, 0 corrupt: 11% ━─────────── 527/4606 142.9it/s 3.2strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 537 images, 433 backgrounds, 0 corrupt: 12% ━─────────── 537/4606 120.6it/s 3.3strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 545 images, 433 backgrounds, 0 corrupt: 12% ━─────────── 545/4606 106.5it/s 3.4strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 553 images, 433 backgrounds, 0 corrupt: 12% ━─────────── 553/4606 92.4it/s 3.6s<train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 571 images, 433 backgrounds, 0 corrupt: 12% ━─────────── 571/4606 112.4it/s 3.7strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 591 images, 436 backgrounds, 0 corrupt: 13% ━╸────────── 591/4606 128.0it/s 3.8strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 609 images, 454 backgrounds, 0 corrupt: 13% ━╸────────── 609/4606 138.4it/s 3.9strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 626 images, 471 backgrounds, 0 corrupt: 14% ━╸────────── 626/4606 144.3it/s 4.0strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 645 images, 490 backgrounds, 0 corrupt: 14% ━╸────────── 645/4606 154.4it/s 4.1strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 663 images, 508 backgrounds, 0 corrupt: 14% ━╸────────── 663/4606 158.4it/s 4.2strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 680 images, 525 backgrounds, 0 corrupt: 15% ━╸────────── 680/4606 156.3it/s 4.4strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 697 images, 542 backgrounds, 0 corrupt: 15% ━╸────────── 697/4606 157.4it/s 4.5strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 716 images, 561 backgrounds, 0 corrupt: 16% ━╸────────── 716/4606 161.4it/s 4.6strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 732 images, 577 backgrounds, 0 corrupt: 16% ━╸────────── 732/4606 158.1it/s 4.7strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 749 images, 594 backgrounds, 0 corrupt: 16% ━╸────────── 749/4606 155.2it/s 4.8strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 767 images, 612 backgrounds, 0 corrupt: 17% ━╸────────── 767/4606 151.5it/s 4.9strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 784 images, 629 backgrounds, 0 corrupt: 17% ━━────────── 784/4606 151.5it/s 5.0strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 801 images, 646 backgrounds, 0 corrupt: 17% ━━────────── 801/4606 154.6it/s 5.1strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 817 images, 652 backgrounds, 0 corrupt: 18% ━━────────── 817/4606 142.7it/s 5.3strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 834 images, 652 backgrounds, 0 corrupt: 18% ━━────────── 834/4606 147.3it/s 5.4strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 846 images, 652 backgrounds, 0 corrupt: 18% ━━────────── 846/4606 138.7it/s 5.5strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 857 images, 652 backgrounds, 0 corrupt: 19% ━━────────── 857/4606 128.7it/s 5.6strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 867 images, 652 backgrounds, 0 corrupt: 19% ━━────────── 867/4606 118.2it/s 5.7strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 879 images, 652 backgrounds, 0 corrupt: 19% ━━────────── 879/4606 109.6it/s 5.8strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 895 images, 662 backgrounds, 0 corrupt: 19% ━━────────── 895/4606 110.1it/s 6.0strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 911 images, 678 backgrounds, 0 corrupt: 20% ━━────────── 911/4606 122.7it/s 6.1strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 920 images, 687 backgrounds, 0 corrupt: 20% ━━────────── 920/4606 105.5it/s 6.2strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 935 images, 702 backgrounds, 0 corrupt: 20% ━━────────── 935/4606 115.5it/s 6.3strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 946 images, 713 backgrounds, 0 corrupt: 21% ━━────────── 946/4606 108.0it/s 6.5strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 959 images, 726 backgrounds, 0 corrupt: 21% ━━────────── 959/4606 114.3it/s 6.6strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 970 images, 737 backgrounds, 0 corrupt: 21% ━━╸───────── 970/4606 107.6it/s 6.7strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 983 images, 750 backgrounds, 0 corrupt: 21% ━━╸───────── 983/4606 110.4it/s 6.8strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 994 images, 761 backgrounds, 0 corrupt: 22% ━━╸───────── 994/4606 100.2it/s 6.9strain: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1007 images, 774 backgrounds, 0 corrupt: 22% ━━╸───────── 1007/4606 106.8it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1018 images, 785 backgrounds, 0 corrupt: 22% ━━╸───────── 1018/4606 101.6it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1029 images, 796 backgrounds, 0 corrupt: 22% ━━╸───────── 1029/4606 101.8it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1043 images, 810 backgrounds, 0 corrupt: 23% ━━╸───────── 1043/4606 112.6it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1054 images, 821 backgrounds, 0 corrupt: 23% ━━╸───────── 1054/4606 105.7it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1063 images, 830 backgrounds, 0 corrupt: 23% ━━╸───────── 1063/4606 100.7it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1079 images, 846 backgrounds, 0 corrupt: 23% ━━╸───────── 1079/4606 102.0it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1090 images, 857 backgrounds, 0 corrupt: 24% ━━╸───────── 1090/4606 103.4it/s 7.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1101 images, 868 backgrounds, 0 corrupt: 24% ━━╸───────── 1101/4606 101.9it/s 8.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1113 images, 880 backgrounds, 0 corrupt: 24% ━━╸───────── 1113/4606 103.7it/s 8.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1125 images, 892 backgrounds, 0 corrupt: 24% ━━╸───────── 1125/4606 104.5it/s 8.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1137 images, 904 backgrounds, 0 corrupt: 25% ━━╸───────── 1137/4606 105.6it/s 8.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1150 images, 917 backgrounds, 0 corrupt: 25% ━━╸───────── 1150/4606 106.3it/s 8.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1162 images, 929 backgrounds, 0 corrupt: 25% ━━━───────── 1162/4606 109.4it/s 8.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1174 images, 932 backgrounds, 0 corrupt: 25% ━━━───────── 1174/4606 99.8it/s 8.7train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1183 images, 932 backgrounds, 0 corrupt: 26% ━━━───────── 1183/4606 93.6it/s 8.8train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1192 images, 932 backgrounds, 0 corrupt: 26% ━━━───────── 1192/4606 90.8it/s 8.9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1204 images, 932 backgrounds, 0 corrupt: 26% ━━━───────── 1204/4606 98.8it/s 9.0train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1218 images, 932 backgrounds, 0 corrupt: 26% ━━━───────── 1218/4606 110.6it/s 9.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1237 images, 937 backgrounds, 0 corrupt: 27% ━━━───────── 1237/4606 129.2it/s 9.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1268 images, 968 backgrounds, 0 corrupt: 28% ━━━───────── 1268/4606 179.7it/s 9.train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1302 images, 1002 backgrounds, 0 corrupt: 28% ━━━───────── 1302/4606 226.0it/s 9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1340 images, 1040 backgrounds, 0 corrupt: 29% ━━━───────── 1340/4606 264.6it/s 9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1373 images, 1073 backgrounds, 0 corrupt: 30% ━━━╸──────── 1373/4606 280.2it/s 9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1405 images, 1105 backgrounds, 0 corrupt: 31% ━━━╸──────── 1405/4606 291.9it/s 9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1441 images, 1141 backgrounds, 0 corrupt: 31% ━━━╸──────── 1441/4606 309.7it/s 9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1479 images, 1179 backgrounds, 0 corrupt: 32% ━━━╸──────── 1479/4606 327.8it/s 9train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1507 images, 1196 backgrounds, 0 corrupt: 33% ━━━╸──────── 1507/4606 301.0it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1525 images, 1196 backgrounds, 0 corrupt: 33% ━━━╸──────── 1525/4606 251.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1543 images, 1200 backgrounds, 0 corrupt: 33% ━━━━──────── 1543/4606 229.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1560 images, 1217 backgrounds, 0 corrupt: 34% ━━━━──────── 1560/4606 210.2it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1574 images, 1231 backgrounds, 0 corrupt: 34% ━━━━──────── 1574/4606 188.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1590 images, 1247 backgrounds, 0 corrupt: 35% ━━━━──────── 1590/4606 179.0it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1601 images, 1258 backgrounds, 0 corrupt: 35% ━━━━──────── 1601/4606 158.2it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1622 images, 1279 backgrounds, 0 corrupt: 35% ━━━━──────── 1622/4606 164.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1638 images, 1295 backgrounds, 0 corrupt: 36% ━━━━──────── 1638/4606 157.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1656 images, 1313 backgrounds, 0 corrupt: 36% ━━━━──────── 1656/4606 155.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1674 images, 1331 backgrounds, 0 corrupt: 36% ━━━━──────── 1674/4606 161.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1691 images, 1348 backgrounds, 0 corrupt: 37% ━━━━──────── 1691/4606 159.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1709 images, 1366 backgrounds, 0 corrupt: 37% ━━━━──────── 1709/4606 159.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1726 images, 1383 backgrounds, 0 corrupt: 37% ━━━━──────── 1726/4606 158.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1742 images, 1399 backgrounds, 0 corrupt: 38% ━━━━╸─────── 1742/4606 154.2it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1760 images, 1417 backgrounds, 0 corrupt: 38% ━━━━╸─────── 1760/4606 156.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1776 images, 1433 backgrounds, 0 corrupt: 39% ━━━━╸─────── 1776/4606 157.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1792 images, 1449 backgrounds, 0 corrupt: 39% ━━━━╸─────── 1792/4606 155.2it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1805 images, 1462 backgrounds, 0 corrupt: 39% ━━━━╸─────── 1805/4606 146.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1824 images, 1481 backgrounds, 0 corrupt: 40% ━━━━╸─────── 1824/4606 154.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1839 images, 1496 backgrounds, 0 corrupt: 40% ━━━━╸─────── 1839/4606 152.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1852 images, 1509 backgrounds, 0 corrupt: 40% ━━━━╸─────── 1852/4606 143.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1860 images, 1517 backgrounds, 0 corrupt: 40% ━━━━╸─────── 1860/4606 123.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1874 images, 1531 backgrounds, 0 corrupt: 41% ━━━━╸─────── 1874/4606 120.0it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1891 images, 1548 backgrounds, 0 corrupt: 41% ━━━━╸─────── 1891/4606 134.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1907 images, 1564 backgrounds, 0 corrupt: 41% ━━━━╸─────── 1907/4606 139.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1923 images, 1580 backgrounds, 0 corrupt: 42% ━━━━━─────── 1923/4606 143.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1939 images, 1587 backgrounds, 0 corrupt: 42% ━━━━━─────── 1939/4606 144.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1950 images, 1587 backgrounds, 0 corrupt: 42% ━━━━━─────── 1950/4606 130.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1963 images, 1587 backgrounds, 0 corrupt: 43% ━━━━━─────── 1963/4606 125.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1976 images, 1587 backgrounds, 0 corrupt: 43% ━━━━━─────── 1976/4606 123.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1988 images, 1587 backgrounds, 0 corrupt: 43% ━━━━━─────── 1988/4606 121.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 1999 images, 1587 backgrounds, 0 corrupt: 43% ━━━━━─────── 1999/4606 116.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2013 images, 1587 backgrounds, 0 corrupt: 44% ━━━━━─────── 2013/4606 121.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2029 images, 1601 backgrounds, 0 corrupt: 44% ━━━━━─────── 2029/4606 130.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2044 images, 1616 backgrounds, 0 corrupt: 44% ━━━━━─────── 2044/4606 134.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2056 images, 1628 backgrounds, 0 corrupt: 45% ━━━━━─────── 2056/4606 128.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2072 images, 1644 backgrounds, 0 corrupt: 45% ━━━━━─────── 2072/4606 127.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2088 images, 1660 backgrounds, 0 corrupt: 45% ━━━━━─────── 2088/4606 134.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2104 images, 1676 backgrounds, 0 corrupt: 46% ━━━━━─────── 2104/4606 125.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2123 images, 1695 backgrounds, 0 corrupt: 46% ━━━━━╸────── 2123/4606 137.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2139 images, 1711 backgrounds, 0 corrupt: 46% ━━━━━╸────── 2139/4606 131.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2155 images, 1727 backgrounds, 0 corrupt: 47% ━━━━━╸────── 2155/4606 135.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2171 images, 1743 backgrounds, 0 corrupt: 47% ━━━━━╸────── 2171/4606 141.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2185 images, 1757 backgrounds, 0 corrupt: 47% ━━━━━╸────── 2185/4606 139.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2203 images, 1775 backgrounds, 0 corrupt: 48% ━━━━━╸────── 2203/4606 150.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2216 images, 1788 backgrounds, 0 corrupt: 48% ━━━━━╸────── 2216/4606 143.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2234 images, 1806 backgrounds, 0 corrupt: 49% ━━━━━╸────── 2234/4606 145.0it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2250 images, 1822 backgrounds, 0 corrupt: 49% ━━━━━╸────── 2250/4606 147.2it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2266 images, 1832 backgrounds, 0 corrupt: 49% ━━━━━╸────── 2266/4606 147.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2279 images, 1832 backgrounds, 0 corrupt: 49% ━━━━━╸────── 2279/4606 140.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2294 images, 1832 backgrounds, 0 corrupt: 50% ━━━━━╸────── 2294/4606 137.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2310 images, 1832 backgrounds, 0 corrupt: 50% ━━━━━━────── 2310/4606 133.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2328 images, 1832 backgrounds, 0 corrupt: 51% ━━━━━━────── 2328/4606 146.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2362 images, 1865 backgrounds, 0 corrupt: 51% ━━━━━━────── 2362/4606 201.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2396 images, 1899 backgrounds, 0 corrupt: 52% ━━━━━━────── 2396/4606 240.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2431 images, 1934 backgrounds, 0 corrupt: 53% ━━━━━━────── 2431/4606 267.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2465 images, 1968 backgrounds, 0 corrupt: 54% ━━━━━━────── 2465/4606 287.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2494 images, 1997 backgrounds, 0 corrupt: 54% ━━━━━━────── 2494/4606 287.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2531 images, 2034 backgrounds, 0 corrupt: 55% ━━━━━━╸───── 2531/4606 297.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2555 images, 2052 backgrounds, 0 corrupt: 55% ━━━━━━╸───── 2555/4606 279.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2571 images, 2052 backgrounds, 0 corrupt: 56% ━━━━━━╸───── 2571/4606 243.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2585 images, 2052 backgrounds, 0 corrupt: 56% ━━━━━━╸───── 2585/4606 211.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2604 images, 2052 backgrounds, 0 corrupt: 57% ━━━━━━╸───── 2604/4606 203.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2619 images, 2052 backgrounds, 0 corrupt: 57% ━━━━━━╸───── 2619/4606 181.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2635 images, 2052 backgrounds, 0 corrupt: 57% ━━━━━━╸───── 2635/4606 174.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2650 images, 2052 backgrounds, 0 corrupt: 58% ━━━━━━╸───── 2650/4606 165.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2684 images, 2085 backgrounds, 0 corrupt: 58% ━━━━━━╸───── 2684/4606 216.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2709 images, 2110 backgrounds, 0 corrupt: 59% ━━━━━━━───── 2709/4606 226.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2736 images, 2137 backgrounds, 0 corrupt: 59% ━━━━━━━───── 2736/4606 235.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2763 images, 2164 backgrounds, 0 corrupt: 60% ━━━━━━━───── 2763/4606 236.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2801 images, 2202 backgrounds, 0 corrupt: 61% ━━━━━━━───── 2801/4606 278.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2837 images, 2238 backgrounds, 0 corrupt: 62% ━━━━━━━───── 2837/4606 302.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2878 images, 2279 backgrounds, 0 corrupt: 62% ━━━━━━━───── 2878/4606 333.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2913 images, 2314 backgrounds, 0 corrupt: 63% ━━━━━━━╸──── 2913/4606 338.0it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2938 images, 2319 backgrounds, 0 corrupt: 64% ━━━━━━━╸──── 2938/4606 304.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 2987 images, 2362 backgrounds, 0 corrupt: 65% ━━━━━━━╸──── 2987/4606 359.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3034 images, 2409 backgrounds, 0 corrupt: 66% ━━━━━━━╸──── 3034/4606 392.0it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3082 images, 2457 backgrounds, 0 corrupt: 67% ━━━━━━━━──── 3082/4606 417.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3124 images, 2499 backgrounds, 0 corrupt: 68% ━━━━━━━━──── 3124/4606 410.3it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3169 images, 2544 backgrounds, 0 corrupt: 69% ━━━━━━━━──── 3169/4606 415.5it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3215 images, 2590 backgrounds, 0 corrupt: 70% ━━━━━━━━──── 3215/4606 428.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3261 images, 2636 backgrounds, 0 corrupt: 71% ━━━━━━━━──── 3261/4606 437.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3278 images, 2639 backgrounds, 0 corrupt: 71% ━━━━━━━━╸─── 3278/4606 345.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3314 images, 2639 backgrounds, 0 corrupt: 72% ━━━━━━━━╸─── 3314/4606 347.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3342 images, 2639 backgrounds, 0 corrupt: 73% ━━━━━━━━╸─── 3342/4606 323.2it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3356 images, 2639 backgrounds, 0 corrupt: 73% ━━━━━━━━╸─── 3356/4606 268.1it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3392 images, 2646 backgrounds, 0 corrupt: 74% ━━━━━━━━╸─── 3392/4606 292.9it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3420 images, 2674 backgrounds, 0 corrupt: 74% ━━━━━━━━╸─── 3420/4606 286.4it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3447 images, 2701 backgrounds, 0 corrupt: 75% ━━━━━━━━╸─── 3447/4606 280.6it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3478 images, 2732 backgrounds, 0 corrupt: 76% ━━━━━━━━━─── 3478/4606 288.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3504 images, 2758 backgrounds, 0 corrupt: 76% ━━━━━━━━━─── 3504/4606 279.8it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3531 images, 2785 backgrounds, 0 corrupt: 77% ━━━━━━━━━─── 3531/4606 275.7it/s 1train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3558 images, 2812 backgrounds, 0 corrupt: 77% ━━━━━━━━━─── 3558/4606 273.0it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3590 images, 2844 backgrounds, 0 corrupt: 78% ━━━━━━━━━─── 3590/4606 273.4it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3618 images, 2869 backgrounds, 0 corrupt: 79% ━━━━━━━━━─── 3618/4606 264.3it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3626 images, 2869 backgrounds, 0 corrupt: 79% ━━━━━━━━━─── 3626/4606 200.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3649 images, 2869 backgrounds, 0 corrupt: 79% ━━━━━━━━━╸── 3649/4606 203.8it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3670 images, 2869 backgrounds, 0 corrupt: 80% ━━━━━━━━━╸── 3670/4606 203.2it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3694 images, 2886 backgrounds, 0 corrupt: 80% ━━━━━━━━━╸── 3694/4606 211.4it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3718 images, 2910 backgrounds, 0 corrupt: 81% ━━━━━━━━━╸── 3718/4606 219.2it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3738 images, 2929 backgrounds, 0 corrupt: 81% ━━━━━━━━━╸── 3738/4606 206.3it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3758 images, 2929 backgrounds, 0 corrupt: 82% ━━━━━━━━━╸── 3758/4606 190.2it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3785 images, 2953 backgrounds, 0 corrupt: 82% ━━━━━━━━━╸── 3785/4606 205.5it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3811 images, 2979 backgrounds, 0 corrupt: 83% ━━━━━━━━━╸── 3811/4606 208.3it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3836 images, 3004 backgrounds, 0 corrupt: 83% ━━━━━━━━━╸── 3836/4606 213.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3856 images, 3024 backgrounds, 0 corrupt: 84% ━━━━━━━━━━── 3856/4606 209.7it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3878 images, 3046 backgrounds, 0 corrupt: 84% ━━━━━━━━━━── 3878/4606 212.6it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3902 images, 3070 backgrounds, 0 corrupt: 85% ━━━━━━━━━━── 3902/4606 217.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3921 images, 3089 backgrounds, 0 corrupt: 85% ━━━━━━━━━━── 3921/4606 207.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3944 images, 3112 backgrounds, 0 corrupt: 86% ━━━━━━━━━━── 3944/4606 206.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3968 images, 3136 backgrounds, 0 corrupt: 86% ━━━━━━━━━━── 3968/4606 213.5it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3986 images, 3142 backgrounds, 0 corrupt: 87% ━━━━━━━━━━── 3986/4606 197.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 3996 images, 3142 backgrounds, 0 corrupt: 87% ━━━━━━━━━━── 3996/4606 163.8it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4023 images, 3169 backgrounds, 0 corrupt: 87% ━━━━━━━━━━── 4023/4606 188.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4045 images, 3191 backgrounds, 0 corrupt: 88% ━━━━━━━━━━╸─ 4045/4606 197.7it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4064 images, 3210 backgrounds, 0 corrupt: 88% ━━━━━━━━━━╸─ 4064/4606 192.2it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4086 images, 3232 backgrounds, 0 corrupt: 89% ━━━━━━━━━━╸─ 4086/4606 196.4it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4108 images, 3254 backgrounds, 0 corrupt: 89% ━━━━━━━━━━╸─ 4108/4606 200.4it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4127 images, 3273 backgrounds, 0 corrupt: 90% ━━━━━━━━━━╸─ 4127/4606 196.0it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4148 images, 3294 backgrounds, 0 corrupt: 90% ━━━━━━━━━━╸─ 4148/4606 195.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4167 images, 3313 backgrounds, 0 corrupt: 90% ━━━━━━━━━━╸─ 4167/4606 191.7it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4190 images, 3336 backgrounds, 0 corrupt: 91% ━━━━━━━━━━╸─ 4190/4606 202.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4209 images, 3355 backgrounds, 0 corrupt: 91% ━━━━━━━━━━╸─ 4209/4606 198.8it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4231 images, 3377 backgrounds, 0 corrupt: 92% ━━━━━━━━━━━─ 4231/4606 195.1it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4249 images, 3389 backgrounds, 0 corrupt: 92% ━━━━━━━━━━━─ 4249/4606 188.3it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4267 images, 3389 backgrounds, 0 corrupt: 93% ━━━━━━━━━━━─ 4267/4606 184.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4284 images, 3389 backgrounds, 0 corrupt: 93% ━━━━━━━━━━━─ 4284/4606 163.0it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4324 images, 3389 backgrounds, 0 corrupt: 94% ━━━━━━━━━━━─ 4324/4606 233.0it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4379 images, 3441 backgrounds, 0 corrupt: 95% ━━━━━━━━━━━─ 4379/4606 327.5it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4426 images, 3488 backgrounds, 0 corrupt: 96% ━━━━━━━━━━━╸ 4426/4606 363.7it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4474 images, 3536 backgrounds, 0 corrupt: 97% ━━━━━━━━━━━╸ 4474/4606 395.2it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4523 images, 3585 backgrounds, 0 corrupt: 98% ━━━━━━━━━━━╸ 4523/4606 423.4it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4568 images, 3629 backgrounds, 0 corrupt: 99% ━━━━━━━━━━━╸ 4568/4606 417.9it/s 2train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4597 images, 3629 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━╸ 4597/4606 379.3it/s train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4606 images, 3629 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4606/4606 187.8it/s train: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train... 4606 images, 3629 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 4606/4606 187.8it/s 24.5s
train: New cache created: C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\train.cache
val: Fast image access  (ping: 0.70.1 ms, read: 20.014.7 MB/s, size: 830.8 KB)
val: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\val... 745 images, 467 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━╸ 745/748 238.3it/s 3.6s<0.0val: Scanning C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\val... 748 images, 467 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 748/748 206.3it/s 3.6s    
val: New cache created: C:\Users\othni\Projects\mvtec_ad\yolo_mvtec_all\labels\val.cache
Plotting labels to C:\Users\othni\Projects\mvtec_ad\runs\detect\train\labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00013, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 0 dataloader workers
Logging results to C:\Users\othni\Projects\mvtec_ad\runs\detect\train
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100         0G      1.714      10.81      1.659          8        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:20
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442    0.00196     0.0308     0.0105    0.00548

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100         0G      1.589      11.24      1.529         11        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:07
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.681     0.0342     0.0525     0.0364

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100         0G      1.649       10.2      1.528          8        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.435      0.134     0.0995     0.0507

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100         0G      1.635      8.616      1.544         12        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:59
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.474      0.206      0.143     0.0771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100         0G      1.579        8.2      1.474         13        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:45
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:14
                   all        748        442      0.552      0.188      0.182      0.103

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100         0G      1.556      7.444      1.455          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:39
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442        0.5      0.267      0.234       0.14

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100         0G      1.538      6.613       1.46          4        640: 100% ━━━━━━━━━━━━ 288/288 4.4s/it 21:05
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 7.1s/it 2:50
                   all        748        442      0.378      0.342      0.263       0.16

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100         0G      1.543      6.379      1.447          5        640: 100% ━━━━━━━━━━━━ 288/288 7.8s/it 37:25
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.417      0.333      0.282      0.177

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100         0G      1.535      5.726      1.447          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.381      0.378      0.323      0.197

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100         0G       1.52       5.56      1.431          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.55      0.331      0.335      0.211

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100         0G      1.533      5.086      1.448          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.394      0.428      0.354      0.217

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100         0G      1.486      4.488       1.44          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.532      0.388      0.391      0.234

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100         0G       1.45      4.174      1.403          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:57
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.588      0.342      0.421      0.266

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100         0G       1.47      3.993      1.411          3        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.493      0.423      0.431       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100         0G      1.433      3.779      1.386          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.513      0.423      0.446      0.284

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100         0G      1.478       3.73      1.414          8        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:59
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.418      0.508      0.458      0.286

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100         0G      1.396      3.307      1.394         14        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:57
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.424      0.538      0.473      0.305

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100         0G      1.438      3.379      1.405          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.509      0.481      0.469      0.293

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100         0G      1.442      3.076      1.395          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.451      0.556      0.498      0.316

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100         0G      1.387      2.963      1.362          6        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.641       0.47      0.521      0.327

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100         0G      1.395      2.826      1.377          1        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.447      0.522      0.498      0.314

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100         0G       1.34       2.79      1.332          7        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:04
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.57      0.519      0.532      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100         0G      1.307      2.518      1.339          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.515      0.522       0.54      0.347

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100         0G      1.355      2.813      1.339          4        640: 100% ━━━━━━━━━━━━ 288/288 4.2s/it 20:13
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.4s/it 1:22
                   all        748        442      0.502      0.544      0.533      0.342

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100         0G      1.344      2.585       1.34          3        640: 100% ━━━━━━━━━━━━ 288/288 3.9s/it 18:45
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 5.6s/it 2:14
                   all        748        442       0.51      0.543      0.535      0.349

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100         0G      1.339      2.513      1.337          5        640: 100% ━━━━━━━━━━━━ 288/288 3.9s/it 18:36
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.511      0.564      0.558      0.361

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100         0G      1.363      2.416      1.319          7        640: 100% ━━━━━━━━━━━━ 288/288 4.0s/it 19:06
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.4s/it 1:23
                   all        748        442      0.499      0.586      0.566      0.371

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100         0G      1.322      2.406      1.309          9        640: 100% ━━━━━━━━━━━━ 288/288 4.0s/it 19:22
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.593      0.534      0.573      0.375

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100         0G      1.299      2.235       1.31          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.519       0.58      0.568       0.37

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100         0G      1.316       2.16      1.295          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.55      0.571      0.576      0.377

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100         0G      1.253      2.086      1.276          7        640: 100% ━━━━━━━━━━━━ 288/288 3.9s/it 18:46
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.587       0.58      0.586      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100         0G      1.241      2.116      1.271          3        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442       0.59      0.597      0.604      0.392

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100         0G      1.288      2.048      1.301          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:43
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.545      0.584      0.579      0.381

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100         0G      1.247      2.069      1.268          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.566       0.56      0.588      0.387

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100         0G      1.242      1.966      1.281          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:14
                   all        748        442      0.605      0.591       0.61      0.398

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100         0G      1.288      1.931      1.306         11        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.615      0.566       0.61      0.405

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100         0G      1.249      1.927      1.285         14        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.605      0.609      0.617      0.407

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100         0G      1.241      1.817      1.264          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.566      0.619       0.62      0.408

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100         0G      1.249      1.786       1.27          5        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:00
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.542      0.619      0.633      0.431

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100         0G      1.217      1.842      1.263          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:16
                   all        748        442      0.529      0.636      0.634      0.419

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100         0G      1.248      1.883      1.266          1        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.59      0.591      0.621       0.41

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100         0G      1.213      1.711      1.246         21        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.575      0.621      0.629      0.419

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100         0G      1.202      1.686       1.23          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:18
                   all        748        442      0.585      0.611      0.628      0.416

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100         0G      1.183      1.588      1.228         16        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:16
                   all        748        442      0.614      0.611      0.637      0.427

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100         0G      1.203      1.684      1.246          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.608       0.59      0.612      0.418

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100         0G      1.205      1.715      1.253          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.537      0.627      0.634      0.424

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100         0G       1.15      1.523      1.219         10        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.587      0.624      0.648      0.433

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100         0G       1.17      1.627       1.22          9        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:04
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.56      0.662      0.653      0.439

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100         0G      1.182      1.551      1.218          2        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.593      0.659      0.657      0.445

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100         0G      1.173      1.513      1.241          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.605      0.631      0.658      0.445

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100         0G      1.146      1.455      1.231          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:43
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.594      0.648      0.651      0.438

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100         0G       1.15       1.45      1.207          3        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.612      0.659      0.667      0.452

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100         0G      1.138      1.491      1.211          5        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:50
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.588       0.66      0.676      0.462

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100         0G      1.138      1.472      1.205          2        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.566      0.664      0.661      0.446

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100         0G      1.103      1.363      1.197          5        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:46
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442        0.6      0.695      0.675      0.458

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100         0G       1.09      1.382      1.176          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.595      0.648      0.659      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100         0G      1.129      1.433      1.209          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.571      0.651      0.659      0.449

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100         0G      1.116      1.429      1.199          5        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.63      0.651      0.676      0.457

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100         0G      1.116      1.356      1.189          3        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.624       0.62      0.671      0.462

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100         0G      1.107      1.386      1.196          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.601      0.666      0.672       0.46

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100         0G      1.086      1.316      1.169          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:43
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.648      0.645      0.685      0.459

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100         0G      1.086      1.297      1.178          5        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.615      0.656      0.677      0.455

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100         0G      1.048      1.226      1.172          2        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:36
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:14
                   all        748        442      0.645      0.657      0.686       0.47

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100         0G      1.064      1.284      1.179         15        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:43
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.658      0.635      0.681      0.466

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100         0G       1.04      1.291      1.154          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:45
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.672      0.668      0.691      0.471

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100         0G      1.063      1.267      1.158          2        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:57
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.666      0.629      0.685      0.464

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100         0G       1.04      1.286      1.165          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.672      0.622      0.677      0.457

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100         0G      1.057      1.278      1.164          4        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.622      0.666       0.68      0.468

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100         0G      1.044      1.208       1.16          5        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:16
                   all        748        442      0.681      0.643      0.685      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     70/100         0G      1.038      1.202      1.155          8        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.616       0.69      0.687      0.467

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     71/100         0G      1.025      1.187      1.139         10        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.627      0.666      0.691       0.47

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     72/100         0G      1.056      1.171      1.163         11        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.683      0.672      0.694      0.475

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     73/100         0G      1.051      1.161      1.159          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.668       0.68      0.709      0.484

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     74/100         0G      1.005      1.176      1.134          8        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:02
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.661      0.697      0.714      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     75/100         0G      1.024      1.191      1.136          8        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.663      0.672      0.697      0.486

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     76/100         0G      1.004       1.13      1.121          5        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.684      0.664      0.697      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     77/100         0G       1.05       1.22       1.15          8        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:57
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442       0.64      0.689      0.703      0.491

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     78/100         0G      1.006      1.173      1.145         11        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:57
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.718      0.658      0.707      0.488

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     79/100         0G      1.036      1.168      1.146         12        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.665      0.672      0.706      0.487

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     80/100         0G     0.9786      1.096      1.138         11        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:54
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.649      0.683      0.703      0.479

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     81/100         0G     0.9835      1.065      1.129          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.676      0.678      0.708      0.477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     82/100         0G      1.031      1.127      1.152          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:47
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:16
                   all        748        442      0.687      0.652      0.703      0.477

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     83/100         0G      0.968       1.05      1.109          3        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:51
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.671      0.676      0.701      0.481

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     84/100         0G      1.025      1.133      1.146          7        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:58
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.668      0.685       0.71      0.484

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     85/100         0G     0.9745       1.04      1.128          9        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:53
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.722      0.651      0.708      0.484

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     86/100         0G     0.9817      1.052      1.113          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.668      0.692      0.711      0.486

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     87/100         0G     0.9977      1.034      1.135          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442        0.7      0.654        0.7      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     88/100         0G     0.9739      1.043      1.124          2        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.724       0.64      0.711      0.492

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     89/100         0G      0.954      1.028      1.115          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.716      0.648      0.708      0.492

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     90/100         0G     0.9836      1.054      1.108          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:56
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:18
                   all        748        442      0.705      0.663      0.709      0.489
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     91/100         0G     0.9135     0.9855      1.095          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.661      0.668      0.696      0.479

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     92/100         0G     0.8958     0.9467      1.087          3        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:46
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.679      0.652       0.69      0.479

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     93/100         0G     0.9068     0.9319       1.11         15        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.686      0.668      0.701      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     94/100         0G     0.8537     0.9538      1.066          8        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:43
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.665      0.678      0.697      0.482

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     95/100         0G     0.8908     0.9251      1.091          0        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:47
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.656      0.685      0.704      0.487

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     96/100         0G     0.8839     0.9575      1.086          3        640: 100% ━━━━━━━━━━━━ 288/288 5.2s/it 25:03
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 5.0s/it 1:59
                   all        748        442      0.691      0.662      0.701      0.483

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     97/100         0G     0.8564     0.8939      1.069          5        640: 100% ━━━━━━━━━━━━ 288/288 3.8s/it 18:18
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:17
                   all        748        442      0.708      0.673      0.706      0.483

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100         0G     0.8868     0.9136      1.077          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:34
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:14
                   all        748        442      0.687       0.68      0.707      0.486

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100         0G     0.8415     0.8804      1.056          4        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:33
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.2s/it 1:16
                   all        748        442      0.687      0.669      0.705      0.482

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100         0G     0.8817     0.9193      1.089          6        640: 100% ━━━━━━━━━━━━ 288/288 3.7s/it 17:33
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 3.1s/it 1:15
                   all        748        442      0.684      0.668      0.703      0.484

100 epochs completed in 32.599 hours.
Optimizer stripped from C:\Users\othni\Projects\mvtec_ad\runs\detect\train\weights\last.pt, 6.4MB
Optimizer stripped from C:\Users\othni\Projects\mvtec_ad\runs\detect\train\weights\best.pt, 6.4MB

Validating C:\Users\othni\Projects\mvtec_ad\runs\detect\train\weights\best.pt...
Ultralytics 8.3.233  Python-3.12.6 torch-2.9.1+cpu CPU (12th Gen Intel Core i5-12450H)
Model summary (fused): 72 layers, 3,091,487 parameters, 0 gradients, 8.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 24/24 2.7s/it 1:05
                   all        748        442      0.722       0.64      0.712      0.495
   bottle_broken_large          4          4       0.77          1      0.995      0.776
   bottle_broken_small          5          5          1      0.669      0.803      0.563
  bottle_contamination          5          6      0.965      0.667      0.824      0.686
       cable_bent_wire          3          3      0.511      0.704      0.597      0.328
      cable_cable_swap          3          3          1      0.936      0.995      0.866
        cable_combined          3         13          0          0      0.137     0.0434
cable_cut_inner_insulation          3          5          1      0.739      0.995      0.703
cable_cut_outer_insulation          2          3      0.777      0.333      0.344      0.237
   cable_missing_cable          3          3       0.56          1      0.863      0.812
    cable_missing_wire          2          2       0.94          1      0.995      0.895
 cable_poke_insulation          2          4      0.246       0.25      0.398      0.201
         capsule_crack          5          5      0.851        0.2      0.276      0.116
capsule_faulty_imprint          5          6      0.868      0.667      0.676      0.499
          capsule_poke          5          6      0.789      0.333       0.49      0.284
       capsule_scratch          5          5      0.879        0.8       0.92      0.592
       capsule_squeeze          4          4      0.406       0.25      0.432      0.185
          carpet_color          4          4       0.64        0.5      0.639      0.433
            carpet_cut          4          4      0.919          1      0.995      0.348
           carpet_hole          4          4      0.479       0.75      0.404      0.247
carpet_metal_contamination          4          5      0.879        0.6      0.608      0.528
         carpet_thread          4          5      0.759        0.6        0.6       0.46
             grid_bent          3          8      0.495      0.247       0.34      0.126
           grid_broken          3          9      0.939      0.667      0.727      0.313
             grid_glue          3          3      0.592      0.667      0.741      0.414
grid_metal_contamination          3          9      0.427      0.222        0.4      0.285
           grid_thread          3         18      0.357      0.167      0.364      0.217
        hazelnut_crack          4          7      0.556      0.286      0.552      0.315
          hazelnut_cut          4          4      0.677          1      0.995      0.722
         hazelnut_hole          4          4      0.881       0.75      0.945      0.745
        hazelnut_print          4         11      0.635      0.477      0.561      0.403
         leather_color          4          4          1       0.96      0.995      0.647
           leather_cut          4          4      0.655       0.75      0.912      0.581
          leather_fold          4          4       0.43       0.75      0.485       0.19
          leather_glue          4          4      0.913        0.5      0.794      0.631
          leather_poke          4          4      0.394        0.5      0.364      0.114
        metal_nut_bent          5          9      0.776      0.444      0.383      0.279
       metal_nut_color          5          5      0.879        0.6      0.928      0.623
        metal_nut_flip          5          5      0.936          1      0.995      0.937
     metal_nut_scratch          5         11      0.524      0.364      0.484       0.41
            pill_color          5         19          1      0.537      0.939      0.661
         pill_combined          4         13      0.202      0.154       0.14      0.103
    pill_contamination          5          5      0.243        0.2      0.505      0.425
            pill_crack          6          6      0.238      0.333      0.466      0.324
   pill_faulty_imprint          4          4          1      0.878      0.995      0.492
        pill_pill_type          2          2          1          1      0.995      0.995
          pill_scratch          5          5      0.759        0.8      0.858      0.677
screw_manipulated_front          5          6      0.421      0.167      0.319     0.0987
    screw_scratch_head          5          5          1      0.952      0.995      0.569
    screw_scratch_neck          5          6      0.958      0.667      0.711       0.59
     screw_thread_side          5          9      0.959      0.222      0.407      0.236
      screw_thread_top          5          5      0.827        0.4      0.554      0.303
            tile_crack          4          4      0.856          1      0.995      0.648
       tile_glue_strip          4          4      0.896          1      0.995      0.923
      tile_gray_stroke          4          4      0.916          1      0.995      0.895
              tile_oil          4          4      0.932          1      0.995      0.896
            tile_rough          3          4      0.654       0.75      0.757      0.453
  toothbrush_defective          6         11      0.657      0.525      0.553      0.339
  transistor_bent_lead          2          2          1      0.805      0.995      0.476
   transistor_cut_lead          2          2      0.851          1      0.995      0.921
transistor_damaged_case          2          2      0.799        0.5      0.828      0.499
  transistor_misplaced          2          2      0.929          1      0.995      0.895
            wood_color          2          4      0.356       0.75      0.845      0.529
         wood_combined          3         19      0.684      0.158      0.262       0.14
             wood_hole          2          7      0.843          1      0.924      0.688
           wood_liquid          2          6      0.915          1      0.995      0.808
          wood_scratch          5         18       0.68      0.474      0.596      0.427
   zipper_broken_teeth          4          5       0.71        0.6      0.705      0.378
       zipper_combined          4          9      0.549      0.144      0.509       0.29
  zipper_fabric_border          4          7      0.774      0.981      0.964      0.612
zipper_fabric_interior          4          7      0.658      0.857      0.735      0.431
          zipper_rough          4          6       0.95          1      0.995      0.701
    zipper_split_teeth          4          4       0.79          1      0.895      0.534
 zipper_squeezed_teeth          4          4      0.405      0.512      0.637      0.389
Speed: 0.7ms preprocess, 55.6ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to C:\Users\othni\Projects\mvtec_ad\runs\detect\train
 Learn more at https://docs.ultralytics.com/modes/train
VS Code: view Ultralytics VS Code Extension  at https://docs.ultralytics.com/integrations/vscode
(.venv) PS C:\Users\othni\Projects\mvtec_ad> 

yolo detect predict `
    model="C:\Users\othni\Projects\mvtec_ad\runs\detect\train\weights\best.pt" `
    source="C:\Users\othni\Projects\mvtec_ad\mvtec_anomaly_images_for_test" `
    imgsz=640 `
    conf=0.25

yolo detect train `
    model=yolov8n.pt `
    data=mvtec_yolo_all.yaml `
    imgsz=640 `
    epochs=100 `
    batch=16 `
    workers=0



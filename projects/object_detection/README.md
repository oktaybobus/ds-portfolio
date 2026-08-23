# Face and Object Detection

Detect faces with Haar cascades and objects with YOLO, on sample photographs.

| | |
|---|---|
| Task | Detection |
| Data | 10 sample images + two Haar cascade files (2.7 MB, committed) |
| Detector | `haarcascade_frontalface_default`, scale factor 1.05, min neighbours 5 |
| Reference result | **6 of 7 faces, 0 false positives** on `g8.jpg` |
| Source | `Day8 AOB Computer Vision.ipynb` |

```bash
uv sync --extra detect
uv run python projects/object_detection/detect.py
uv run python projects/object_detection/detect.py --sweep
uv run python projects/object_detection/detect.py --yolo --image cars.jpg   # needs --extra yolo
uv run dsj serve object_detection
```

## No windows

The source notebook displayed everything this way:

```python
resim = cv2.imread("input.jpg")
cv2.imshow("Merhaba CV", resim)
cv2.waitKey()
cv2.destroyAllWindows()
```

`cv2.imshow` opens a desktop window and `cv2.waitKey()` blocks until a key is
pressed. On the laptop it was written on, that is fine. On a server, in CI, or
in a notebook someone else opens, the cell hangs forever with no output and no
error. Every function here returns an array or a Matplotlib figure, the same
rule the rest of this repository follows.

The image loader also swaps BGR to RGB once, at the door. OpenCV reads BGR and
everything else expects RGB; that mismatch is why an image can look correct in
one window and blue in another.

## The default that finds nothing

`scaleFactor` controls how fast the search window grows between pyramid levels.
The notebook left it at whatever the tutorial used. Swept against `g8.jpg`,
which has seven leaders facing the camera:

| scale_factor | min_neighbours | Faces found |
|---|---|---|
| **1.05** | **3-8** | **6** |
| 1.10 | 3-5 | 6 |
| 1.10 | 8 | 5 |
| 1.20 | 5 | 4 |
| 1.20 | 8 | 2 |
| 1.30 | 3 | 3 |
| 1.30 | 5-8 | **0** |

1.30 is a common value in tutorials and finds **nothing at all** here. One
parameter, left alone, is the difference between six faces and none.

The six detections at 1.05/5 were checked by drawing the boxes and looking at
them: all six are real faces, no false positives. The seventh leader is
partially occluded by the person in front and is genuinely missed.

## The negative case

`classroom.jpg` shows students photographed from behind. A **frontal** cascade
should find nothing there, and nothing is the right answer, not a failure.

Drop `min_neighbours` to 2 and it returns two boxes: one covering half the
room, one on the whiteboard. Both were verified as false positives the same
way - by looking. That is what a low neighbour count buys you, and it is why
the ground truth in `FACE_COUNTS` is a number someone counted rather than one
the detector produced.

## Non-maximum suppression

A cascade fires several times around one face at neighbouring scales, so a raw
count is an overcount. `non_max_suppression` keeps the highest-scoring box per
cluster of overlapping detections, using intersection over union.

At the settings above it happens to change nothing on these images - raw and
deduplicated counts are equal - which is worth reporting rather than leaving as
an implied benefit.

Türkçe açıklamalar: [README.tr.md](README.tr.md)

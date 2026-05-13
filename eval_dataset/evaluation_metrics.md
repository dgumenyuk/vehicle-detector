# Evaluation metrics across distance bands

Assumption: distance is estimated from the GT box width, car length, image width, and camera horizontal FOV.

| Metric | 0-200 m | 200-400 m |
| --- | --- | --- |
| Detection rate TP / (TP + FN) | 0.022 | 0.001 |
| Precision TP / (TP + FP) | 0.080 | 1.000 |
| False alarms / min FP x 60 / N_frames | 41.559 | 0.000 |
| Time to first detection seconds | 16.000 | 235.000 |
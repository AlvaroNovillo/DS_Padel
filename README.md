# Padel 2D-Tracking from Monocular Video

## Overview

This repository contains the implementation of a novel framework for automatic two-dimensional tracking of Padel games using monocular recordings. The algorithm integrates computer vision techniques to detect and track players, the court, and the ball. By employing homography, we are able to extrapolate the detected player positions to a two-dimensional court, thus enabling automatic two-dimensional tracking of Padel games. This approach is designed to be user-friendly, cost-effective, and adaptable to different camera angles and lighting conditions, making it accessible to both amateur and professional players and coaches.

<a href="url"><img src="[http://url.to/image.png](https://github.com/AlvaroNovillo/DS_Padel/assets/81865790/a8435924-7e32-45fb-b4b7-e6eef7d03843)" align="center" height="1080" width="1920" ></a>




## Authors

- ÁLvaro Novillo (0009-0003-9888-6638)
- Víctor Aceña (0000-0003-1838-2150)
- Carmen Lancho (0000-0002-4674-1598)
- Marina Cuesta (0000-0002-1880-9225)
- Isaac Martín De Diego (0000-0001-5197-2932)

Rey Juan Carlos University, Data Science Laboratory, C/ Tulipán, s/n, 28933, Móstoles, Spain
{alvaro.novillo, victor.acena, carmen.lancho, marina.cuesta, isaac.martin}@urjc.es

## Methodology
![image](https://github.com/AlvaroNovillo/DS_Padel/assets/81865790/28ad2d6a-ac95-41f7-aaa5-bf7fa25664ee)


The framework consists of three main modules:

1. **Court Detection Module**:
   - Detects the court using image preprocessing and contour detection.

2. **Player Detection Module**:
   - Uses YOLOv8 for player detection and tracking.
   - Applies the homography matrix to transform player coordinates to a two-dimensional view.

3. **Ball Detection Module**:
   - Implements TrackNet for ball detection and tracking.

## Experiments and Results

The framework was tested on the amateur public padel dataset, with videos captured from various camera angles. The results indicate that the algorithm performs well when the camera is positioned close to standard professional angles but fails at court-level views.


## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/padel-2d-tracking.git

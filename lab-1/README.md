# Lab 1 — JetRacer Basics and Road Following

**Session 1 of 3 | Introduction, calibration, camera, OpenCV, and neural-network road following**

## 1. Goal

Learn how to control a JetRacer and use its CSI camera. Detect road boundaries with OpenCV. Train and run a neural-network road follower using NVIDIA's beginner example.

At the end of this session, compare two methods:

- **Car A:** Road following using OpenCV.
- **Car B:** Road following using a trained neural network (DNN) with TensorRT.

- **Bonus:** Find solution of detecting the distance between of them on a road


**Important:** This session covers road following. The lead-car detector and Adaptive Cruise Control (ACC) logic come in Labs 2 and 3.

## 2. Learning outcomes

After this lab, you should be able to:

1. Set and test steering and throttle commands.
2. Calibrate steering offset and steering gain, and save the settings in JSON.
3. Start, stop, and release the CSI camera.
4. Read a frame and draw lines or shapes with OpenCV.
5. Detect left and right road boundaries and calculate their center.
6. Explain a simple steering correction based on a road-center error.
7. Collect road-following examples and train a model using NVIDIA's workflow.
8. Run an optimized model on the JetRacer and compare it with OpenCV.

## 3. Before you start

Check that you have a JetRacer, its camera, a gamepad, a charged battery, and access to JupyterLab from your computer.

1. Put the car on a stand. Keep its drive wheels clear of the table.
2. Connect the car to Wi-Fi. Open its JupyterLab page in a web browser.
3. Copy the `Lab1` folder to the JetRacer or upload its files in JupyterLab.
4. Keep `jetracer_calibration.json` in the **same folder** as the supplied lab notebooks.
5. Close other notebooks that use the car or camera. Shut down their kernels before starting a new camera exercise.

The supplied notebooks use the tested JetRacer Python environment. Do not install extra packages unless the instructor asks you to.

## 4. Task sequence

| Part | Notebook / resource | Task | Result |
| --- | --- | --- | --- |
| **1A** | `Lab1_Part1_JetRacer_Calibration.ipynb` | Test motion and calibrate steering and throttle | Save working settings in JSON |
| **1B** | `Lab1_Part1_Calibration_Check.ipynb` | Load JSON and test the joystick | Confirm settings can be reused |
| **2** | `Lab1_Part2_CSI_Camera_and_Live_Control.ipynb` | Learn CSI capture and live camera/car widgets | Run the wide camera and stop it correctly |
| **3** | `Lab1_Part3_OpenCV_Drawing.ipynb` | Draw lines, circles, rectangles, and text | Show overlays on a live frame |
| **4** | `Lab1_Part4_Road_Detection_and_Center_Following.ipynb` | Detect both road edges and adjust steering | Demonstrate basic OpenCV road following |
| **5** | NVIDIA `interactive_regression.ipynb` and `road_following.ipynb` | Collect training data, train, optimize, and run a DNN | Demonstrate neural-network road following |
| **6** | Track demonstration | Compare OpenCV and DNN | Report what works and what fails |

The first four parts have guided notebooks in this folder. Parts 5 and 6 use NVIDIA's JetRacer examples and the track.

### Part 1 — Motion and calibration

**Purpose:** Learn how steering and throttle affect the car.

Use the calibration notebook to:

1. Set throttle to `0.0` before you enable the drive motor.
2. Test steering left and right.
3. Adjust `steering_offset` until the front wheels point straight when `car.steering = 0.0`.
4. Check that the steering direction is correct. Adjust `steering_gain` if necessary.
5. Try low throttle commands under supervision. Adjust `throttle_gain` only as needed.
6. Save the settings to `jetracer_calibration.json`.
7. Open the calibration-check notebook. Load the file and test the joystick.

**Remember:** `steering_offset` changes the center position. `steering_gain` affects steering direction and range. `throttle_gain` scales the drive command. Throttle is not a speed in m/s.

**Deliverable:** A saved JSON file with validated steering settings and low-speed throttle values.

### Part 2 — CSI camera

**Purpose:** Understand camera capture and control.

First, review the original beginner configuration:

```python
camera = CSICamera(width=224, height=224)
```

Then use our **wide camera configuration** for the rest of the lab:

```python
camera = CSICamera(
    width=640,
    height=360,
    capture_width=1280,
    capture_height=720,
    capture_fps=30
)
```

The wide view produces a **640 × 360** output image. It uses a **1280 × 720** sensor capture mode. It does not change the physical camera lens. The training notebook may use a smaller input tensor. Keep the training and live-inference preprocessing consistent.

Use the notebook widgets to start and stop the camera. Learn the difference between:

- `camera.running = True`: Start continuous capture.
- `camera.value`: Access the latest frame.
- `camera.running = False`: Stop continuous capture.
- `camera.cap.release()`: Release the underlying camera connection when needed.

Do not call `camera.read()` while continuous capture is running. If the camera is stuck, first stop its programs. An instructor may then use `sudo systemctl restart nvargus-daemon` from a terminal.

**Deliverable:** A live wide-camera preview with working Start and Stop controls.

### Part 3 — Draw with OpenCV

**Purpose:** Learn image coordinates and simple overlays.

1. Capture a camera frame.
2. Draw a line with `cv2.line()`.
3. Draw a circle, a rectangle, and text.
4. Draw two guide lines over the track.
5. Use the live widget to show or hide a guide line.

**Deliverable:** A camera image with visible OpenCV overlays.

### Part 4 — Find the road with OpenCV

**Purpose:** Detect the road without a neural network.

1. Convert a frame to grayscale.
2. Apply a brightness threshold to separate road and background.
3. Compare the result with Canny edge detection.
4. Find the left and right road boundaries at three horizontal scan lines.
5. Calculate the road midpoint and compare it with the image center.
6. Draw the detected boundaries, road center, and steering information.
7. Test a small proportional steering correction. Use the throttle widget for a slow, supervised drive.

For a dark road on a light background, thresholding can be a simple starting point. Its results depend on the track surface, shadows, and lighting. Canny can also detect unrelated edges. Test the detection in **preview-only mode** before you enable drive.

The basic steering idea is:

```text
Road center - Image center = Position error
Position error × Kp = Steering correction
```

The control code must limit steering commands. Stop the drive motor when a valid road center is not available.

**Deliverable:** A working visualization of the two road boundaries and a supervised low-speed road-following test.

### Part 5 — Train NVIDIA's road-following neural network

**Purpose:** Learn a different way to follow the road.

NVIDIA provides a beginner workflow for collecting road-following training examples. You will label a **target point on the intended driving path**. The model learns to predict that point from a camera image; control code uses the prediction to steer the car.

Open the original JetRacer notebooks from `~/jetracer/notebooks` (or the equivalent folder on the lab image):

1. Open **`interactive_regression.ipynb`**.
2. Set `TASK = 'road_following'` and `CATEGORIES = ['apex']` as in NVIDIA's example.
3. Place the stationary car at different parts of the track. Include curves, offsets, and different headings.
4. Select the target point in the camera image. Collect examples across the track, not only on straight sections. Start with about 50 labeled examples, then add data where the model fails.
5. Train the neural network. Review its predicted points and collect more data if needed.
6. Save the trained model as **`road_following_model.pth`**, following the NVIDIA example.
7. Open **`road_following.ipynb`**. Follow its TensorRT optimization and live-demo steps for the installed JetRacer image.
8. Test the model with the wheels off the floor first. Then test on the track at low speed.

**Do not confuse the two camera settings.** Our OpenCV exercises use 640 × 360. The native NVIDIA training workflow can use 224 × 224 images. Use the preprocessing expected by the trained model during live inference.

NVIDIA reference: [JetRacer road-following example](https://github.com/NVIDIA-AI-IOT/jetracer/blob/master/docs/examples.md).

**Deliverable:** A trained road-following model and an optimized live demonstration on the JetRacer.

### Part 6 — Compare OpenCV and DNN road following

**Purpose:** Observe how two methods behave on the same track.

- **Car A — OpenCV:** Uses thresholding, road boundaries, and center-error steering (Part 4).
- **Car B — DNN:** Uses the trained NVIDIA road-following model and TensorRT (Part 5).

**Test plan:**

1. Use the same track layout and similar lighting.
2. Begin with slow, separate runs. Do not run the cars side by side until both are stable and the instructor approves.
3. Try a straight section and a curve. If possible, repeat a complete lap.
4. Observe steering stability, loss of the road, recovery, and changes in speed.
5. Repeat the test after a small lighting change or a change in the starting position.
6. Record the results and discuss when each method works better.

Use the following table in your lab notes:

| Observation | Car A: OpenCV | Car B: DNN + TensorRT |
| --- | --- | --- |
| Completed the test route? | | |
| Left the track? How many times? | | |
| Smooth steering? | | |
| Worked on curves? | | |
| Worked after a lighting change? | | |
| One clear advantage | | |
| One clear limitation | | |

This is a qualitative comparison unless both cars use equivalent speeds, route conditions, and test repetitions. A trained DNN is not automatically better than OpenCV.

## 5. What to submit or demonstrate


A short video of each working approach is useful if the instructor requests one.

## 6. Troubleshooting

| Problem | Check |
| --- | --- |
| Car does not turn correctly | Recheck `steering_offset`, `steering_gain`, and the saved JSON |
| Joystick direction is wrong | Test steering on a stand and check the joystick mapping |
| Motor does not move | Check power, enable control, throttle deadband, and calibrated low-speed values |
| Camera reports that it is running | Stop the previous camera notebook and its kernel before opening a new camera |
| Camera cannot open | Release old camera sessions; ask the instructor before restarting `nvargus-daemon` |
| OpenCV detects the wrong road area | Check lighting, the threshold, and the lower region of interest |
| Only one road boundary is visible | Stop driving; reposition the car or improve the camera view |
| DNN model does not follow the road | Check image preprocessing, saved model, labels, and training coverage |

## 7. Safety and next step

Keep the initial tests at low speed. Use a clear track. A person must supervise the car and be ready to switch off its power. A software STOP control does not replace an emergency stop. Stop the car when detection becomes unreliable.

**Next session — Lab 2:** Collect and label images of a lead JetRacer. Train an object detector and estimate proximity. This will provide the perception input for the ACC logic in Lab 3.

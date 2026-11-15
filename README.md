# JetRacer Lab: Autonomous Driving and Adaptive Cruise Control

**Three practical lab sessions | NVIDIA JetRacer | Python, OpenCV, PyTorch, TensorRT**

## 1. Introduction

JetRacer is a small robotic car built around an NVIDIA Jetson computer. It has a camera, a steering servo, and a drive motor. Students can control the car from a Jupyter notebook and use its GPU to run computer vision and neural-network models.

Check off repo: https://github.com/NVIDIA-AI-IOT/jetracer
Check Jetracer WIKI: https://www.waveshare.com/wiki/JetRacer_Pro_AI_Kit

In these labs, we use JetRacer to study the main parts of an autonomous driving system:

- **Perception:** Read camera images, find the road, and detect another car.
- **Decision:** Decide where to steer and when to slow down or stop.
- **Control:** Send steering and throttle commands to the car.


> **Project challenge:** Build a small Adaptive Cruise Control (ACC) demonstrator. One JetRacer drives ahead. A second JetRacer follows it, stays on the track, adjusts its speed, and stops when the lead car stops or becomes too close. The follower should resume only when the path is clear.

of course this only used for educationak purpose. (in reality the system is more comples)

## 2. Hardware and software

Our lab uses the following equipment.

| Item | Purpose |
| --- | --- |
| Two JetRacer cars with NVIDIA Jetson Nano | Lead car and following car |
| CSI camera (IMX219) on each car | Capture the road and the lead car |
| Steering servo and drive motor | Control direction and speed |
| Gamepad / joystick | Manual control and testing |
| Indoor test track | Repeatable driving experiments |
| Wi-Fi and a computer with a browser | Access the Jupyter notebooks |
| Python, JetCam, OpenCV, PyTorch | Camera handling, vision, and model training |
| NVIDIA TensorRT | Run optimized neural-network inference on Jetson |

A JetRacer sends commands to its steering servo and drive motor. A **throttle command is not a speed measurement**. The relationship between throttle and speed depends on the car, battery, and surface.

## 3. Overall objective

At the end of the three sessions, demonstrate a **two-car following system** on the indoor track.

The system should aim to:

1. Keep the follower near the center of the road.
2. Detect the lead car using its front camera.
3. Reduce speed as the follower approaches the lead car.
4. Stop before reaching the lead car when the gap is too small.
5. Remain stopped while the lead car is stationary.
6. Resume following when there is sufficient clearance.
7. Stop safely when camera data or detection confidence is unreliable.

A bounding box does not directly give an accurate distance in meters. The lab will use a calibrated image-based proximity cue or distance zones for speed control. The instructor must validate these zones on the physical track.

## 4. Lab sessions

| Session | Theme | Main tasks | Expected result |
| --- | --- | --- | --- |
| **Lab 1** | JetRacer basics and road following | Calibrate steering, use the camera, draw with OpenCV, detect road boundaries, train an NVIDIA road-following network, optimize with TensorRT | A car follows the track; OpenCV and DNN methods can be compared |
| **Lab 2** | Detect the lead car | Collect and label images, train an object detector, run it on Jetson, estimate proximity from the detection | The follower recognizes the lead car and reports proximity zones |
| **Lab 3** | Adaptive Cruise Control | Combine road following and lead-car detection, adjust throttle, implement stop/resume logic, test two cars | A supervised ACC demonstration with measured results |

### Lab 1 — Introduction and road following

Start with the hardware. Calibrate steering and throttle. Learn to start and stop the CSI camera. Use OpenCV to detect the left and right road boundaries and estimate the road center. Then collect examples for NVIDIA's neural-network road follower, train it, and run its TensorRT demo.

**Final Lab 1 activity:** Compare OpenCV road following on one JetRacer with neural-network road following on another. Test one car at a time first.

[Open the Lab 1 guide](Lab1/README.md)

### Lab 2 — Object detection

Record camera images of another JetRacer under different conditions: near, far, left, right, straight sections, and curves. Label the lead car with bounding boxes. Train an object-detection network, such as SSD-MobileNetV2. Evaluate its detections and deploy the compatible model on the Jetson. If time permits, compare it with another detector or model configuration.

Use the detected bounding box to define **proximity zones**. Measure or calibrate these zones on the track before using them for control. A detector can fail when a car is partly outside the camera image; the control system must handle that case explicitly.

### Lab 3 — Adaptive Cruise Control

Integrate road following with detection and longitudinal control. The follower should stay on the road while reacting to the lead car. Start with clear rules for normal, slow, and stop behavior. If time permits, add proportional or PID control for smoother throttle changes.

Test at least three situations: the lead car moves at a constant speed, slows down, and stops. Also check what the follower does when a detection is lost. Record the response and any unsafe behavior.

## 5. How the system fits together

```text
                 CSI camera
                     |
                  Image
                     |
           +---------+---------+
           |                   |
     Road following      Car detection
     (OpenCV or DNN)     (DNN detector)
           |                   |
     Steering target     Proximity / risk
           |                   |
           +---------+---------+
                     |
              Control logic
                     |
           Steering + throttle
                     |
                JetRacer
```

Road following controls the car's **direction**. ACC primarily controls the car's **speed and following gap**. The final demonstrator needs both functions.

## 6. Final demonstration

Use two JetRacers on the same track:

- **Lead car:** Drives in front and changes speed or stops during the test.
- **Following car:** Uses its camera to follow the road and respond to the lead car.

Show a live camera view with road and car detections. Explain the selected speed or stop command. Compare runs using simple observations such as successful laps, track departures, stops, missed detections, and collisions or near misses.

The objective is to **avoid contact**, but this result is not guaranteed. An instructor must supervise all vehicle motion.

## 7. Safety and lab rules

1. Test new throttle and steering code with the drive wheels above the table.
2. Keep the first driving tests at a low speed and on a clear track.
3. Keep a person ready to use the physical power switch. A notebook STOP button is not an emergency-stop system.
4. Never allow two independent notebooks to control the same car or CSI camera at the same time.
5. Stop and disarm the car before changing models, cameras, or control code.
6. Do not drive if the camera freezes, road detection is unreliable, or the steering direction is wrong.
7. Introduce two-car following only after each car can drive reliably on its own.

## 8. Lab files

```text
JetRacer_ACC_Labs/
├── README.md                     # Project overview (this file)
└── Lab1/
    ├── README.md                 # Lab 1 tasks and instructions
    ├── jetracer_calibration.json
    ├── Lab1_Part1_JetRacer_Calibration.ipynb
    ├── Lab1_Part1_Calibration_Check.ipynb
    ├── Lab1_Part2_CSI_Camera_and_Live_Control.ipynb
    ├── Lab1_Part3_OpenCV_Drawing.ipynb
    └── Lab1_Part4_Road_Detection_and_Center_Following.ipynb
```

The NVIDIA road-following training notebooks are supplied by the **JetRacer installation**, not by this package. Lab 2 and Lab 3 guides will be provided in their respective sessions.

## 9. References

- [HSHL JetRacer project overview](https://wiki.hshl.de/wiki/index.php/JetRacer) — an example of a university JetRacer introduction and project scope.
- [NVIDIA JetRacer repository](https://github.com/NVIDIA-AI-IOT/jetracer) — original hardware/software project.
- [NVIDIA JetRacer examples](https://github.com/NVIDIA-AI-IOT/jetracer/blob/master/docs/examples.md) — basic motion and road-following workflow.
- [OpenCV documentation](https://docs.opencv.org/) — image processing and contour detection.
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — optimized inference.

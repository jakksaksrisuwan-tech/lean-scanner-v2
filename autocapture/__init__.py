"""Auto-capture document scanner.

A reference implementation of the pipeline described in the research
brief: live camera frame -> 4-corner quad detection -> quality scoring
-> Kalman smoothing + N-frame lock gate -> perspective dewarp -> shading
+ contrast cleanup -> save.

The package is structured so that every component is testable in
isolation. The verified end-to-end path uses only numpy + Pillow.
Optional dependencies (OpenCV, ultralytics) provide the production
fast paths.
"""

__version__ = "0.1.0"

# Go1 locomotion assets

This directory contains third-party assets used only by the optional legged
MuJoCo explanation environment.

- `go1_policy.onnx` and the feet-only MJCF were taken from Google DeepMind's
  MuJoCo Playground at commit
  `87a4bf98f1806adefd240d72dd53a2c3ceeb2f0d`. They are covered by
  `LICENSE.playground-apache2`.
- The five Unitree Go1 meshes were taken from MuJoCo Menagerie at commit
  `1b86ece576591213e2b666ebf59508454200ca97`. They are covered by
  `LICENSE.unitree-bsd3`.
- `go1.xml` changes only the local mesh paths and render material so the
  package is self-contained. Joint, actuator, inertial, and collision data
  remain the Playground model.

The ONNX network is an experimental command-tracking locomotion policy. It is
not a navigation, collision-avoidance, or safety policy. The model intentionally
uses feet-only terrain collision geometry, exactly as its source filename
states. The PSS reference route, forecast tube, and safety annotations are
separate outputs of this repository.

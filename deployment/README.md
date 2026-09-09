# RealMan RM65 deployment adapter

`realman_rm65.py` converts one MemoryVLA action
`[dx, dy, dz, droll, dpitch, dyaw, gripper]` to the official RM_API2 Python
SDK. Positions are metres and rotations are radians. The adapter converts only
the rotation delta to degrees for RM_API2's `rm_algo_pose_move`; its resulting
target pose is sent by `rm_movel` in metres/radians.

It is intentionally safe by default:

- importing it makes no network or robot connection;
- `connect()` supports status reads but does not move the arm;
- `execute_action()` requires `allow_motion=True` *and* calibrated Cartesian
  workspace bounds;
- each action is constrained to 2 cm and 10 degrees per step by default.

Before a real motion test, provide the controller IP, configure the active
tool/gripper and its limits, calibrate camera-to-work-frame extrinsics, and set
workspace bounds from a supervised teach-mode measurement. Start with the arm
unloaded, reduced speed, and the manufacturer's emergency-stop procedure ready.

Example (status-only):

```python
from deployment import RM65Client, RM65SafetyConfig

client = RM65Client(RM65SafetyConfig())
client.connect("192.168.1.18")  # does not move the arm
print(client.read_state())
client.close()
```

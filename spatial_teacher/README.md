# Spatial teacher

`vggt/` is the VGGT implementation vendored from the Spatial-Forcing
`openvla-SF/vggt` directory.  It is used only to produce no-gradient training
targets; it is never attached to `MemoryVLA`, FSDP, or a policy checkpoint.

Download the VGGT-1B `model.pt` checkpoint from the official Facebook VGGT
release and set `--spatial_teacher_path` to that file.  The default first-stage
configuration uses the final VGGT feature layer, removes its five special
tokens, and bilinearly pools the remaining `37 x 37` features to MemoryVLA's
`16 x 16` / 256-token grid.

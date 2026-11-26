# Tabletop scene assets

## Workshop background and bench

`workshop_sky.png` is a 2-by-3 cubemap baked from [Poly Haven University Workshop](https://polyhaven.com/a/university_workshop), by Savva Zakharov. `workbench_metal.png` is the diffuse image from [Poly Haven Blue Metal Plate](https://polyhaven.com/a/blue_metal_plate), by Rob Tuytel. Both are [CC0](https://polyhaven.com/license), including redistribution and commercial use. The sky is a photographic background; MuJoCo's scene lights illuminate the robot and bench separately.

To reproduce, download the [4K HDR panorama](https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/university_workshop_4k.hdr) (SHA-256 `f3c5838d0e3f656c08af3e00cf32982e6cbecf7d398913eedd6d410906ae4b4d`) and [1K diffuse JPG](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/blue_metal_plate/blue_metal_plate_diff_1k.jpg) (SHA-256 `a0162bffce47d4a35613a12af22571b28c18412dc5805cbb69eac343554ef750`), then run `python engine/rc_engine/assets/tabletop/bake_workshop.py path/to/university_workshop_4k.hdr path/to/blue_metal_plate_diff_1k.jpg engine/rc_engine/assets/tabletop`. The script needs NumPy and Pillow, decodes RGBE, tone maps to sRGB, samples six 1024-pixel faces in RLUDFB order, and converts the metal image to PNG.

The following assets were used in earlier scenes and remain in the repository.
The nut and both trays are NVIDIA SimReady meshes (CC-BY-4.0). The room photographs are Poly Haven (CC0). The boat hull
is drawn for this scene and uses the wooden tray's albedo.

Scaled so they fit the patch in front of the gripper. The nut is an M20
(`factory_nut_m20_loose`) at 2.8×. The blue tray is `kitchenware_tray_a01` at
0.55×. The wooden tray is `utensil_servingTray_wooden_a01` at 0.48×. Textures
were reduced to 1024 px. Geometry was read out of the SimReady USD and written
as OBJ. Colliders are simple boxes and a cylinder; the meshes are what the
cameras see.

| File | Source | Licence |
|---|---|---|
| `nut.obj`, `nut.png` | [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets) `factory_nut_m20_loose` | CC-BY-4.0. © NVIDIA. |
| `tray_blue.obj`, `tray_blue.png` | same dataset, `kitchenware_tray_a01` | CC-BY-4.0. © NVIDIA. |
| `tray_wood.obj`, `tray_wood.png` | same dataset, `utensil_servingTray_wooden_a01` | CC-BY-4.0. © NVIDIA. |
| `boat.obj` | original hull. Albedo is the wooden tray texture above | hull original; texture CC-BY-4.0 © NVIDIA |
| `table.png` | Poly Haven `brown_planks_03` | CC0 |
| `floor.png` | Poly Haven `concrete_floor_02` | CC0 |
| `wall.png` | Poly Haven `clay_plaster` | CC0 |

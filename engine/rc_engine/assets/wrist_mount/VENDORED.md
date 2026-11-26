# Wrist mount provenance

`Bracket_G336_M3.stl` is the team's own adapter bracket, exported from the
Onshape document "DraftBracket". It sits between the UR12e tool flange and the
Robotiq 2F-140 coupling and carries an Orbbec Gemini 336 on an arm. The camera
plate was re-drilled for the Gemini 336 back mount: 2 × M3 at 45 mm centres.
The file is binary STL, millimetres, watertight. SHA-256
`30377de429fa12896e657a7cf7547ac63e7c665a393cb7bd590d36068b6474b4`.

## Frame

Bracket millimetres map to UR `tool0` metres with no rotation:

    p_tool0 = (p_bracket - [0, 0, 6.4]) / 1000

- Bracket z = 6.4 mm is the recess floor. It sits on the UR flange face, which is
  the `tool0` origin. The 6.4 mm recess wraps the flange spigot.
- Bracket z = 30.8 mm is the face the Robotiq coupling bolts to. The gripper is
  therefore 24.4 mm out along `tool0` +z (`embodiment.GRIPPER_SPACER_M`).
- The camera arm points along bracket +y, which is `tool0` +y. The gripper
  fingers open along `tool0` ±x, so the arm is perpendicular to them.
- The camera plate face centroid is (0.27, 122.09, 42.92) mm, with normal
  (0, −0.38268, 0.92388). That is the approach axis tilted 22.5° toward the tool axis.

The engine draws this mesh as a visual geom on `tool0`. Contacts use primitives
fitted to it in `embodiment.py`: a Ø90 mm puck, the Ø63 mm coupling boss, a Ø22 mm
tube, the camera plate and the Gemini 336 housing (89.5 × 25 × 30.7 mm).

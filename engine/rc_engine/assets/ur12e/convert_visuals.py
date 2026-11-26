#!/usr/bin/env python3
"""Convert pinned UR10e Collada scenes to per-material OBJ meshes.

Usage: python convert_visuals.py PATH_TO_UPSTREAM/meshes/ur10e/visual
Uses only Python's standard library and rejects unsupported scene structures.
The pinned files use flat nodes, triangle-only polylists, two indexed inputs,
and uniform-scale rotation matrices. Line primitives are intentionally ignored.
"""
from collections import defaultdict
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
OUT = Path(__file__).parent / "assets"
NAMES = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")


def convert(path):
    root = ET.parse(path).getroot()
    assert root.find("c:asset/c:unit", NS).get("meter") == "1"
    assert root.find("c:asset/c:up_axis", NS).text == "Z_UP"
    geometries = {
        geometry.get("id"): geometry
        for geometry in root.findall("c:library_geometries/c:geometry", NS)
    }
    groups = defaultdict(list)
    nodes = root.findall("c:library_visual_scenes/c:visual_scene/c:node", NS)
    assert nodes and not root.findall(".//c:node/c:node", NS)
    for node in nodes:
        assert [child.tag.split("}")[-1] for child in node] == ["matrix", "instance_geometry"]
        matrix = list(map(float, node.find("c:matrix", NS).text.split()))
        assert len(matrix) == 16 and matrix[12:] == [0, 0, 0, 1]
        # Uniform positive scale lets the same rotation transform normals.
        columns = [[matrix[row * 4 + col] for row in range(3)] for col in range(3)]
        lengths = [math.sqrt(sum(value * value for value in col)) for col in columns]
        assert max(lengths) - min(lengths) < 1e-9 and min(lengths) > 0
        assert all(
            abs(sum(columns[i][k] * columns[j][k] for k in range(3))) < 1e-9
            for i in range(3) for j in range(i)
        )
        instance = node.find("c:instance_geometry", NS)
        mesh = geometries[instance.get("url")[1:]].find("c:mesh", NS)
        bindings = {
            item.get("symbol"): item.get("target")[1:]
            for item in instance.findall(".//c:instance_material", NS)
        }
        sources = {
            source.get("id"): list(map(float, source.find("c:float_array", NS).text.split()))
            for source in mesh.findall("c:source", NS)
        }
        vertices = {
            item.get("id"): item.find('c:input[@semantic="POSITION"]', NS).get("source")[1:]
            for item in mesh.findall("c:vertices", NS)
        }
        for poly in mesh.findall("c:polylist", NS):
            assert set(poly.find("c:vcount", NS).text.split()) == {"3"}
            inputs = {
                item.get("semantic"): (int(item.get("offset")), item.get("source")[1:])
                for item in poly.findall("c:input", NS)
            }
            assert set(inputs) == {"VERTEX", "NORMAL"}
            stride = max(item[0] for item in inputs.values()) + 1
            vertex_offset, vertex_id = inputs["VERTEX"]
            normal_offset, normal_id = inputs["NORMAL"]
            positions = sources[vertices[vertex_id]]
            source_normals = sources[normal_id]
            indices = list(map(int, poly.find("c:p", NS).text.split()))
            assert len(indices) == int(poly.get("count")) * 3 * stride
            material = bindings[poly.get("material")]

            def transform(vector, translation):
                return tuple(
                    sum(matrix[row * 4 + col] * vector[col] for col in range(3))
                    + (matrix[row * 4 + 3] if translation else 0)
                    for row in range(3)
                )

            points = [transform(positions[i:i + 3], True) for i in range(0, len(positions), 3)]
            normals = []
            for i in range(0, len(source_normals), 3):
                normal = transform(source_normals[i:i + 3], False)
                length = math.sqrt(sum(value * value for value in normal))
                normals.append(tuple(value / length for value in normal))
            faces = [
                (indices[i + vertex_offset], indices[i + normal_offset])
                for i in range(0, len(indices), stride)
            ]
            groups[material].append((points, normals, faces))

    for material, chunks in sorted(groups.items()):
        assert material.endswith("-material")
        output = OUT / f"{path.stem}_visual_{material[:-9].lower()}.obj"
        with output.open("w", encoding="ascii", newline="\n") as output_file:
            output_file.write("# Derived from Universal Robots UR10e Collada; see ../VENDORED.md\n")
            vertex_base = normal_base = 0
            for points, normals, faces in chunks:
                for point in points:
                    output_file.write("v %.9g %.9g %.9g\n" % point)
                for normal in normals:
                    output_file.write("vn %.9g %.9g %.9g\n" % normal)
                for i in range(0, len(faces), 3):
                    indices = " ".join(
                        f"{vertex + vertex_base + 1}//{normal + normal_base + 1}"
                        for vertex, normal in faces[i:i + 3]
                    )
                    output_file.write("f " + indices + "\n")
                vertex_base += len(points)
                normal_base += len(normals)
        print(output.name, sum(len(chunk[2]) // 3 for chunk in chunks), "triangles")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for name in NAMES:
        convert(Path(sys.argv[1]) / (name + ".dae"))

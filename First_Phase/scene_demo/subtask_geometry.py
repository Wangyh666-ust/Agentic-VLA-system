import hashlib
import json
import math

import numpy as np


_MESH_CONVEX_CACHE_LIMIT = 128
_MESH_CONVEX_CACHE = {}


class GeometryError(ValueError):
    """Raised when collision geometry can not be read exactly."""


_PLANE = 0
_SPHERE = 2
_CAPSULE = 3
_ELLIPSOID = 4
_CYLINDER = 5
_BOX = 6
_MESH = 7
_HFIELD = 1


def _finite(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (str, bytes)):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
    else:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
    if not math.isfinite(out):
        return None
    return out


def _finite_vec(values, count):
    if values is None:
        return None
    try:
        seq = list(values)
    except TypeError:
        return None
    if len(seq) != count:
        return None
    out = []
    for item in seq:
        num = _finite(item)
        if num is None:
            return None
        out.append(num)
    return out


def _finite_matrix(values, rows, cols):
    array = np.asarray(values, dtype=float)
    if array.shape != (rows, cols):
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array


def _read_attr(model, names):
    for name in names:
        if hasattr(model, name):
            return getattr(model, name)
    return None


def _geom_name(model, index):
    try:
        if hasattr(model, "geom_id2name"):
            name = model.geom_id2name(index)
        else:
            name = None
    except Exception:
        name = None
    if name is None:
        raise GeometryError("geometry %d has no name" % index)
    if not isinstance(name, str) or not name:
        raise GeometryError("geometry %d has an invalid name" % index)
    return name


def geom_bounds(model, data, index):
    """Read the exact collision bounds of one geometry as a plain dict."""

    geoms = _read_attr(model, ("geom_type",))
    if geoms is None:
        raise GeometryError("model has no geom_type array")
    try:
        count = len(geoms)
    except TypeError:
        raise GeometryError("geom_type is not readable")
    if index is None or isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise GeometryError("geometry index is not an integer")
    index = int(index)
    if index < 0 or index >= count:
        raise GeometryError("geometry index %d out of range" % index)

    def model_column(name):
        attr = _read_attr(model, (name,))
        if attr is None:
            raise GeometryError("model has no %s array" % name)
        try:
            if len(attr) != count:
                raise GeometryError("%s length mismatch" % name)
        except TypeError:
            raise GeometryError("%s is not readable" % name)
        return attr

    def data_column(name):
        attr = _read_attr(data, (name,))
        if attr is None:
            raise GeometryError("data has no %s array" % name)
        try:
            if len(attr) != count:
                raise GeometryError("%s length mismatch" % name)
        except TypeError:
            raise GeometryError("%s is not readable" % name)
        return attr

    def element(column, idx):
        return column[idx]

    contype_arr = model_column("geom_contype")
    conaffinity_arr = model_column("geom_conaffinity")
    contype = _finite(element(contype_arr, index))
    conaffinity = _finite(element(conaffinity_arr, index))
    if contype is None or conaffinity is None:
        raise GeometryError("geometry %d has unreadable contact flags" % index)
    if contype == 0 and conaffinity == 0:
        raise GeometryError("geometry %d is not collidable (zero contype/conaffinity)" % index)

    geom_type = element(geoms, index)
    if isinstance(geom_type, (bool, np.bool_)):
        raise GeometryError("geometry %d has an unreadable type" % index)
    try:
        geom_type = int(geom_type)
    except (TypeError, ValueError):
        raise GeometryError("geometry %d has an unreadable type" % index)

    name = _geom_name(model, index)

    xmat_attr = data_column("geom_xmat")
    xpos_attr = data_column("geom_xpos")
    try:
        rotation = np.asarray(xmat_attr[index], dtype=float).reshape(3, 3)
        position = np.asarray(xpos_attr[index], dtype=float).reshape(3)
    except Exception:
        raise GeometryError("geometry %d world pose is unreadable" % index)
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(position)):
        raise GeometryError("geometry %d world pose is nonfinite" % index)
    if _proper_orthogonal(rotation) is None:
        raise GeometryError("geometry %d world rotation is not proper orthogonal" % index)

    size_attr = model_column("geom_size")
    try:
        raw_size = list(np.asarray(size_attr[index], dtype=float).reshape(-1))
    except Exception:
        raise GeometryError("geometry %d size is unreadable" % index)

    def need_size(count_needed):
        if len(raw_size) < count_needed:
            raise GeometryError("geometry %d size is too short" % index)
        digits = []
        for item in raw_size[:count_needed]:
            num = _finite(item)
            if num is None or num < 0.0:
                raise GeometryError("geometry %d has invalid size" % index)
            digits.append(num)
        return digits

    localcenter = np.zeros(3, dtype=float)
    mesh_bounds = None
    convex = None

    if geom_type == _SPHERE:
        radius = need_size(1)[0]
        half = np.array([radius, radius, radius], dtype=float)
        size = [radius]
    elif geom_type == _CAPSULE:
        radius, half_length = need_size(2)
        half = np.array([radius, radius, half_length + radius], dtype=float)
        size = [radius, half_length]
    elif geom_type == _ELLIPSOID:
        radii = need_size(3)
        half = np.array(radii, dtype=float)
        size = list(radii)
    elif geom_type == _CYLINDER:
        radius, half_height = need_size(2)
        half = np.array([radius, radius, half_height], dtype=float)
        size = [radius, half_height]
    elif geom_type == _BOX:
        half_sizes = need_size(3)
        half = np.array(half_sizes, dtype=float)
        size = list(half_sizes)
    elif geom_type == _MESH:
        if len(raw_size) < 3:
            raise GeometryError("geometry %d size is too short" % index)
        size_digits = []
        for item in raw_size[:3]:
            num = _finite(item)
            if num is None or num < 0.0:
                raise GeometryError("geometry %d has invalid size" % index)
            size_digits.append(num)
        size = list(size_digits)
        dataid_attr = model_column("geom_dataid")
        dataid_raw = element(dataid_attr, index)
        if isinstance(dataid_raw, (bool, np.bool_)):
            raise GeometryError("mesh geometry %d has no data id" % index)
        try:
            dataid = int(dataid_raw)
        except (TypeError, ValueError):
            raise GeometryError("mesh geometry %d has no data id" % index)
        if dataid < 0:
            raise GeometryError("mesh geometry %d has invalid data id" % index)
        vertadr_attr = _read_attr(model, ("mesh_vertadr",))
        vertnum_attr = _read_attr(model, ("mesh_vertnum",))
        vert_attr = _read_attr(model, ("mesh_vert",))
        if vertadr_attr is None or vertnum_attr is None or vert_attr is None:
            raise GeometryError("mesh vertex data unavailable")
        try:
            vertadr = int(vertadr_attr[dataid])
            vertnum = int(vertnum_attr[dataid])
        except (TypeError, ValueError, IndexError):
            raise GeometryError("mesh geometry %d has invalid vertex range" % index)
        if vertadr < 0 or vertnum <= 0:
            raise GeometryError("mesh geometry %d has empty vertex range" % index)
        try:
            verts = np.asarray(vert_attr[vertadr:vertadr + vertnum], dtype=float)
        except Exception:
            raise GeometryError("mesh geometry %d vertices unreadable" % index)
        if verts.size == 0:
            raise GeometryError("mesh geometry %d has no vertices" % index)
        verts = verts.reshape(-1, 3)
        if not np.all(np.isfinite(verts)):
            raise GeometryError("mesh geometry %d has nonfinite vertices" % index)
        local_min = verts.min(axis=0)
        local_max = verts.max(axis=0)
        localcenter = (local_min + local_max) / 2.0
        half = (local_max - local_min) / 2.0
        mesh_bounds = {
            "min": [float(value) for value in local_min],
            "max": [float(value) for value in local_max],
        }
        convex = _mesh_convex_local(verts, localcenter)
    elif geom_type == _PLANE:
        normal_z = float(rotation[2, 2])
        floor_like = abs(normal_z) >= 0.99 and float(position[2]) <= 0.85
        if not floor_like:
            raise GeometryError("unsupported non-floor plane geometry %d" % index)
        return {
            "id": index,
            "name": name,
            "type": "plane",
            "center": None,
            "rotation": [[float(rotation[r, c]) for c in range(3)] for r in range(3)],
            "half": None,
            "geom_position": [float(value) for value in position],
            "geom_rotation": [[float(rotation[r, c]) for c in range(3)] for r in range(3)],
            "size": [float(value) for value in raw_size],
            "mesh_bounds": None,
            "aabb_min": None,
            "aabb_max": None,
            "contype": contype,
            "conaffinity": conaffinity,
            "excluded_floor_plane": True,
        }
    elif geom_type == _HFIELD:
        raise GeometryError("unsupported heightfield geometry %d" % index)
    else:
        raise GeometryError("unknown geometry type %r for geometry %d" % (geom_type, index))

    if not np.all(np.isfinite(half)):
        raise GeometryError("geometry %d has nonfinite half extents" % index)
    if np.any(half < 0.0):
        raise GeometryError("geometry %d has negative half extents" % index)

    center = position + rotation @ localcenter
    if not np.all(np.isfinite(center)):
        raise GeometryError("geometry %d center is nonfinite" % index)

    if geom_type == _MESH:
        convex = {
            "vertices": [[float(value) for value in row] for row in convex["vertices"]],
            "face_axes": [[float(value) for value in row] for row in convex["face_axes"]],
            "edge_axes": [[float(value) for value in row] for row in convex["edge_axes"]],
        }

    abs_rotation = np.abs(rotation)
    extent = abs_rotation @ half
    aabb_min = center - extent
    aabb_max = center + extent

    return {
        "id": index,
        "name": name,
        "type": {
            _SPHERE: "sphere",
            _CAPSULE: "capsule",
            _ELLIPSOID: "ellipsoid",
            _CYLINDER: "cylinder",
            _BOX: "box",
            _MESH: "mesh",
        }[geom_type],
        "center": [float(value) for value in center],
        "rotation": [[float(rotation[r, c]) for c in range(3)] for r in range(3)],
        "half": [float(value) for value in half],
        "geom_position": [float(value) for value in position],
        "geom_rotation": [[float(rotation[r, c]) for c in range(3)] for r in range(3)],
        "size": [float(value) for value in size],
        "mesh_bounds": mesh_bounds,
        "convex": convex,
        "aabb_min": [float(value) for value in aabb_min],
        "aabb_max": [float(value) for value in aabb_max],
        "contype": contype,
        "conaffinity": conaffinity,
        "excluded_floor_plane": False,
    }


def _round_floats(value, ndigits=6):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise GeometryError("nonfinite value in fingerprint")
        out = round(value, ndigits)
        if out == 0.0:
            out = 0.0
        return out
    if isinstance(value, (list, tuple)):
        return [_round_floats(item, ndigits) for item in value]
    if isinstance(value, dict):
        return {key: _round_floats(item, ndigits) for key, item in value.items()}
    return value


def geometry_fingerprint(target, geoms):
    """Stable fingerprint of the target collision OBBs, immune to naming/pose."""

    if geoms is None:
        raise GeometryError("no geometry for fingerprint")
    try:
        geom_list = list(geoms)
    except TypeError:
        raise GeometryError("geometry collection is unreadable for fingerprint")
    if not geom_list:
        raise GeometryError("no geometry for fingerprint")

    if not isinstance(target, dict):
        raise GeometryError("target is unreadable for fingerprint")
    target_position = _finite_vec(target.get("position"), 3)
    if target_position is None:
        raise GeometryError("target position is unreadable for fingerprint")
    try:
        target_orientation = np.asarray(target.get("orientation"), dtype=float).reshape(3, 3)
    except Exception:
        raise GeometryError("target orientation is unreadable for fingerprint")
    target_orientation = _proper_orthogonal(target_orientation)
    if target_orientation is None:
        raise GeometryError("target orientation is not proper orthogonal for fingerprint")

    def check_size(type_name, size_value):
        try:
            size_seq = list(size_value)
        except TypeError:
            raise GeometryError("geometry size is unreadable for fingerprint")
        needed = {
            "sphere": 1,
            "capsule": 2,
            "ellipsoid": 3,
            "cylinder": 2,
            "box": 3,
            "mesh": 3,
            "plane": 0,
        }.get(type_name)
        if needed is None:
            raise GeometryError("unknown geometry type %r for fingerprint" % (type_name,))
        if len(size_seq) < needed:
            raise GeometryError("geometry size is too short for fingerprint")
        out = []
        for item in size_seq:
            num = _finite(item)
            if num is None or num < 0.0:
                raise GeometryError("geometry size is invalid for fingerprint")
            out.append(num)
        return out

    records = []
    for geom in geom_list:
        if geom is None or not isinstance(geom, dict):
            raise GeometryError("geometry entry is missing or invalid for fingerprint")

        geom_type = geom.get("type")
        if not isinstance(geom_type, str) or geom_type not in ("sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "plane"):
            raise GeometryError("geometry type is invalid for fingerprint")

        size = check_size(geom_type, geom.get("size"))

        geom_position = _finite_vec(geom.get("geom_position"), 3)
        if geom_position is None:
            raise GeometryError("geometry position is unreadable for fingerprint")
        try:
            geom_rotation = np.asarray(geom.get("geom_rotation"), dtype=float).reshape(3, 3)
        except Exception:
            raise GeometryError("geometry rotation is unreadable for fingerprint")
        geom_rotation = _proper_orthogonal(geom_rotation)
        if geom_rotation is None:
            raise GeometryError("geometry rotation is not proper orthogonal for fingerprint")

        mesh_bounds = geom.get("mesh_bounds")
        if geom_type == "mesh":
            if not isinstance(mesh_bounds, dict):
                raise GeometryError("mesh geometry bounds are missing for fingerprint")
            bounds_min = _finite_vec(mesh_bounds.get("min"), 3)
            bounds_max = _finite_vec(mesh_bounds.get("max"), 3)
            if bounds_min is None or bounds_max is None:
                raise GeometryError("mesh geometry bounds are invalid for fingerprint")
            for min_value, max_value in zip(bounds_min, bounds_max):
                if min_value > max_value:
                    raise GeometryError("mesh geometry bounds are unordered for fingerprint")
            rounded_bounds = {
                "min": _round_floats(bounds_min),
                "max": _round_floats(bounds_max),
            }
        else:
            if mesh_bounds is not None:
                raise GeometryError("non-mesh geometry has mesh bounds for fingerprint")
            rounded_bounds = None

        relative_position = target_orientation.T @ (np.asarray(geom_position, dtype=float) - np.asarray(target_position, dtype=float))
        relative_rotation = target_orientation.T @ geom_rotation

        record = {
            "type": geom_type,
            "size": _round_floats(size),
            "position": _round_floats([float(value) for value in relative_position]),
            "rotation": _round_floats([[float(relative_rotation[r, c]) for c in range(3)] for r in range(3)]),
            "mesh_bounds": rounded_bounds,
        }
        records.append(record)

    if not records:
        raise GeometryError("no collidable geometry for fingerprint")

    records.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _proper_orthogonal(matrix):
    if matrix is None:
        return None
    try:
        arr = np.asarray(matrix, dtype=float)
    except Exception:
        return None
    if arr.shape != (3, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if not np.allclose(arr.T @ arr, np.eye(3), atol=1e-5):
        return None
    if not np.isclose(np.linalg.det(arr), 1.0, atol=1e-5):
        return None
    return arr


def _hand_prefixes(robot):
    if robot is None:
        raise GeometryError("robot object is missing")
    model = getattr(robot, "robot_model", None)
    gripper = getattr(robot, "gripper", None)
    if model is None or gripper is None:
        raise GeometryError("robot model or gripper is missing")
    robot_prefix = getattr(model, "naming_prefix", None)
    gripper_prefix = getattr(gripper, "naming_prefix", None)
    if not isinstance(robot_prefix, str) or not robot_prefix:
        raise GeometryError("robot naming_prefix is invalid")
    if not isinstance(gripper_prefix, str) or not gripper_prefix:
        raise GeometryError("gripper naming_prefix is invalid")
    return robot_prefix, gripper_prefix


def _geom_names(model, count):
    names = []
    for index in range(count):
        names.append(_geom_name(model, index))
    return names


def read_geometry(env, capability, home_orientation, calibration_profiles=None, extra_goals=None):
    """Read-only, fail-closed collision observation for the current scene."""

    if not isinstance(capability, dict):
        raise GeometryError("capability must be a dict")
    if "object_id" not in capability:
        raise GeometryError("capability must specify an arbitrary object_id")
    object_id = capability["object_id"]

    home_matrix = _proper_orthogonal(home_orientation)
    if home_matrix is None:
        raise GeometryError("home orientation is not a proper orthogonal matrix")
    home_orientation = [[float(value) for value in row] for row in home_matrix]

    import service
    import preparation_diagnostics as pd
    import placement_experiments as pe
    import local_grasp

    inner_env = service._inner_env(env)
    if inner_env is None:
        raise GeometryError("inner environment is unavailable")

    robots = list(getattr(inner_env, "robots", []) or [])
    if len(robots) != 1:
        raise GeometryError("scene must contain exactly one robot")
    robot = robots[0]

    facts = pd.controller_facts(env)
    mismatch = pd.controller_mismatch(facts)
    if mismatch:
        raise GeometryError("accepted OSC controller required: %s" % mismatch)

    pose = pd.read_eef_pose(env)
    if not isinstance(pose, dict):
        raise GeometryError("end-effector pose is unreadable")

    goals = list(capability.get("goals") or [])
    if extra_goals:
        goals = goals + list(extra_goals)
    snapshot = pe.capture_snapshot(env, goals)
    if not isinstance(snapshot, dict):
        raise GeometryError("snapshot is unreadable")

    objects_dict = getattr(inner_env, "objects_dict", None)
    if not isinstance(objects_dict, dict):
        raise GeometryError("objects_dict is unavailable")
    if object_id not in objects_dict:
        raise GeometryError("object_id %r not found in objects_dict" % (object_id,))
    person_object = objects_dict[object_id]
    contact_geoms = list(getattr(person_object, "contact_geoms", []) or [])
    if not contact_geoms:
        raise GeometryError("object %r has no contact geoms" % (object_id,))

    snapshot_objects = snapshot.get("objects")
    if not isinstance(snapshot_objects, dict):
        raise GeometryError("snapshot objects map is unavailable")
    target_snapshot = snapshot_objects.get(object_id)
    if not isinstance(target_snapshot, dict):
        raise GeometryError("snapshot has no entry for object %r" % (object_id,))

    target_position = _finite_vec(target_snapshot.get("position"), 3)
    if target_position is None:
        raise GeometryError("target snapshot position is unreadable")
    target_quat = _finite_vec(target_snapshot.get("quaternion"), 4)
    if target_quat is None:
        raise GeometryError("target snapshot quaternion is unreadable")
    if not np.isclose(math.sqrt(sum(value * value for value in target_quat)), 1.0, atol=1e-4):
        raise GeometryError("target snapshot quaternion is not normalized")
    target_matrix = local_grasp._quat_wxyz_to_matrix(target_quat)
    if target_matrix is None:
        raise GeometryError("target quaternion conversion failed")
    target_matrix = np.asarray(target_matrix, dtype=float).reshape(3, 3)
    target_matrix = _proper_orthogonal(target_matrix)
    if target_matrix is None:
        raise GeometryError("target orientation is not proper orthogonal")

    sim = getattr(inner_env, "sim", None)
    if sim is None:
        raise GeometryError("sim is unavailable")
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        raise GeometryError("model or data is unavailable")

    try:
        geom_count = int(model.ngeom)
    except Exception:
        raise GeometryError("model geom count is unreadable")
    names = _geom_names(model, geom_count)

    def _known_zero_flags(index):
        if index < 0 or index >= geom_count:
            raise GeometryError("geom index %r is out of range" % (index,))
        contype_column = getattr(model, "geom_contype", None)
        conaffinity_column = getattr(model, "geom_conaffinity", None)
        if contype_column is None or conaffinity_column is None:
            raise GeometryError("geom contype/conaffinity columns are unavailable")
        try:
            contype_length = len(contype_column)
            conaffinity_length = len(conaffinity_column)
        except Exception:
            raise GeometryError("geom contype/conaffinity columns are unreadable")
        if contype_length != geom_count or conaffinity_length != geom_count:
            raise GeometryError("geom contype/conaffinity columns have wrong length")
        contype_value = contype_column[index]
        conaffinity_value = conaffinity_column[index]
        known_zero = True
        for value in (contype_value, conaffinity_value):
            if isinstance(value, (bool, np.bool_)):
                raise GeometryError("geom contype/conaffinity value is boolean")
            try:
                as_float = float(value)
            except Exception:
                raise GeometryError("geom contype/conaffinity value is unreadable")
            if not math.isfinite(as_float):
                raise GeometryError("geom contype/conaffinity value is nonfinite")
            if as_float != math.floor(as_float):
                raise GeometryError("geom contype/conaffinity value is not an integer")
            if as_float != 0.0:
                known_zero = False
        return known_zero

    target_geoms = []
    for geom_index in contact_geoms:
        try:
            index = int(model.geom_name2id(geom_index)) if isinstance(geom_index, str) else int(geom_index)
        except Exception:
            raise GeometryError("target contact geom %r is unresolvable" % (geom_index,))
        if _known_zero_flags(index):
            continue
        bounds = geom_bounds(model, data, index)
        if bounds.get("excluded_floor_plane"):
            raise GeometryError("target contact geom %r resolves to a floor plane" % (geom_index,))
        target_geoms.append(bounds)

    if not target_geoms:
        raise GeometryError("target has no nonzero collision geometry")

    target_position_array = np.asarray(target_position, dtype=float)

    corners_local = []
    for bounds in target_geoms:
        center = np.asarray(bounds["center"], dtype=float)
        rotation = np.asarray(bounds["rotation"], dtype=float)
        half = np.asarray(bounds["half"], dtype=float)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    offset = np.array([sx * half[0], sy * half[1], sz * half[2]], dtype=float)
                    corners_local.append(center + rotation @ offset)

    if not corners_local:
        raise GeometryError("target has no readable collision geometry")

    local_points = []
    for world_corner in corners_local:
        local_points.append(target_matrix.T @ (world_corner - target_position_array))
    local_points = np.asarray(local_points, dtype=float)
    if not np.all(np.isfinite(local_points)):
        raise GeometryError("projected target corners are nonfinite")
    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)
    dimensions = (local_max - local_min).tolist()

    world_aabb_min = None
    world_aabb_max = None
    all_world = []
    for bounds in target_geoms:
        all_world.append(np.asarray(bounds["aabb_min"], dtype=float))
        all_world.append(np.asarray(bounds["aabb_max"], dtype=float))
    all_world = np.asarray(all_world, dtype=float)
    world_aabb_min = [float(value) for value in all_world.min(axis=0)]
    world_aabb_max = [float(value) for value in all_world.max(axis=0)]

    target = {
        "position": [float(value) for value in target_position],
        "orientation": [[float(target_matrix[r, c]) for c in range(3)] for r in range(3)],
        "dimensions": [float(value) for value in dimensions],
        "calibration_key": None,
        "fingerprint": None,
        "world_aabb_min": world_aabb_min,
        "world_aabb_max": world_aabb_max,
    }

    fingerprint = geometry_fingerprint(target, target_geoms)
    target["fingerprint"] = fingerprint
    if isinstance(calibration_profiles, dict):
        profile = calibration_profiles.get(fingerprint)
        if isinstance(profile, dict) and profile.get("key") is not None:
            target["calibration_key"] = profile.get("key")

    robot_prefix, gripper_prefix = _hand_prefixes(robot)

    hand_geoms = []
    obstacle_geoms = []
    excluded_geoms = []
    id_by_geom = {}

    for index in range(geom_count):
        if _known_zero_flags(index):
            continue
        bounds = geom_bounds(model, data, index)
        name = bounds["name"]
        id_by_geom[index] = bounds
        if bounds.get("excluded_floor_plane"):
            excluded_geoms.append({
                "id": bounds["id"],
                "name": name,
                "type": bounds["type"],
                "reason": "floor_plane",
            })
            continue
        is_robot = name.startswith(robot_prefix)
        is_gripper = name.startswith(gripper_prefix)
        if is_robot or is_gripper:
            if is_gripper:
                hand_geoms.append(bounds)
            continue
        obstacle_geoms.append(bounds)

    if not hand_geoms:
        raise GeometryError("no hand collision geometry found")

    hand = []
    for bounds in hand_geoms:
        hand.append({
            "id": bounds["id"],
            "name": bounds["name"],
            "type": bounds["type"],
            "center": bounds["center"],
            "half": bounds["half"],
            "rotation": bounds["rotation"],
            "convex": bounds.get("convex"),
        })

    obstacles = []
    for bounds in obstacle_geoms:
        obstacles.append({
            "id": bounds["id"],
            "name": bounds["name"],
            "type": bounds["type"],
            "center": bounds["center"],
            "half": bounds["half"],
            "rotation": bounds["rotation"],
            "convex": bounds.get("convex"),
            "contype": bounds["contype"],
            "conaffinity": bounds["conaffinity"],
            "aabb_min": bounds["aabb_min"],
            "aabb_max": bounds["aabb_max"],
        })

    try:
        ncon = int(data.ncon)
    except Exception:
        raise GeometryError("contact count is unreadable")
    if ncon < 0:
        raise GeometryError("contact count is negative")

    contacts = getattr(data, "contact", None)
    if contacts is None:
        raise GeometryError("data.contact is unavailable")
    if len(contacts) < ncon:
        raise GeometryError("data.contact is shorter than ncon")

    robot_contacts = []
    for contact_index in range(ncon):
        contact = contacts[contact_index]
        try:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
        except Exception:
            raise GeometryError("contact %d has unreadable geom ids" % contact_index)
        if geom1 < 0 or geom2 < 0 or geom1 >= geom_count or geom2 >= geom_count:
            raise GeometryError("contact %d has out-of-range geom ids" % contact_index)
        name1 = names[geom1]
        name2 = names[geom2]
        is_robot1 = name1.startswith(robot_prefix) or name1.startswith(gripper_prefix)
        is_robot2 = name2.startswith(robot_prefix) or name2.startswith(gripper_prefix)
        if is_robot1 == is_robot2:
            continue
        try:
            dist = float(contact.dist)
            position = np.asarray(contact.pos, dtype=float).reshape(3)
        except Exception:
            raise GeometryError("contact %d evidence is unreadable" % contact_index)
        if not math.isfinite(dist) or not np.all(np.isfinite(position)):
            raise GeometryError("contact %d evidence is nonfinite" % contact_index)
        if is_robot1:
            robot_name, other_name, other_index = name1, name2, geom2
        else:
            robot_name, other_name, other_index = name2, name1, geom1
        record = id_by_geom.get(other_index)
        robot_contacts.append({
            "robot_geom": robot_name,
            "other_geom": other_name,
            "distance": dist,
            "position": [float(value) for value in position],
            "other_type": record["type"] if record else None,
        })

    gripper_gap_m = None
    gripper_qpos = snapshot.get("gripper_qpos")
    if gripper_qpos is not None:
        values = _finite_vec(gripper_qpos, 2)
        if values is not None:
            gripper_gap_m = abs(values[0] - values[1])

    return {
        "geometry_source": "mujoco_collision_obbs",
        "pose": pose,
        "home_orientation": home_orientation,
        "snapshot": snapshot,
        "target": target,
        "hand": hand,
        "obstacles": obstacles,
        "robot_contacts": robot_contacts,
        "excluded_geoms": excluded_geoms,
        "gripper_gap_m": gripper_gap_m,
    }


def _mesh_convex_local(verts, localcenter):
    arr = np.ascontiguousarray(verts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 4:
        raise GeometryError("mesh geometry has too few vertices for a hull")
    if not np.all(np.isfinite(arr)):
        raise GeometryError("mesh geometry has nonfinite vertices")
    key = (hashlib.sha256(arr.tobytes()).hexdigest(), arr.shape, arr.dtype.str)
    cached = _MESH_CONVEX_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(arr)
    except Exception as exc:
        raise GeometryError("mesh convex hull failed: %s" % exc)
    hv = getattr(hull, "vertices", None)
    if hv is None or len(hv) < 4:
        raise GeometryError("mesh convex hull has no vertices")
    rel = arr[np.asarray(hv, dtype=int)] - localcenter.reshape(1, 3)
    if not np.all(np.isfinite(rel)):
        raise GeometryError("mesh convex vertices are nonfinite")
    equations = getattr(hull, "equations", None)
    if equations is None:
        raise GeometryError("mesh convex hull has no face equations")
    eq = np.asarray(equations, dtype=np.float64)
    if eq.ndim != 2 or eq.shape[1] < 3:
        raise GeometryError("mesh convex hull face equations malformed")
    face_axes = eq[:, :3]
    simplices = getattr(hull, "simplices", None)
    if simplices is None:
        raise GeometryError("mesh convex hull has no simplices")
    simp = np.asarray(simplices, dtype=int)
    if simp.ndim != 2 or simp.shape[1] != 3:
        raise GeometryError("mesh convex hull simplices malformed")
    pts = arr[np.unique(simp.reshape(-1))]
    if pts.shape[0] < 4:
        raise GeometryError("mesh convex hull has too few unique vertices")
    tri = arr[simp]
    if not np.all(np.isfinite(tri)):
        raise GeometryError("mesh convex hull triangle vertices are nonfinite")
    edges = np.concatenate(
        [tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]],
        axis=0,
    )

    def _dedup(rows):
        out = []
        seen = set()
        for row in np.asarray(rows, dtype=np.float64).reshape(-1, 3):
            norm = float(np.linalg.norm(row))
            if norm <= 1e-9:
                continue
            unit = row / norm
            if not np.all(np.isfinite(unit)):
                raise GeometryError("mesh convex axis is nonfinite")
            nonzero = np.flatnonzero(np.abs(unit) > 1e-9)
            if len(nonzero) and unit[nonzero[0]] < 0.0:
                unit = -unit
            k = tuple(np.round(unit, 9))
            if k not in seen:
                seen.add(k)
                out.append(unit)
        if not out:
            raise GeometryError("mesh convex has no usable axes")
        return np.asarray(out, dtype=np.float64)

    face_axes = _dedup(face_axes)
    edge_axes = _dedup(edges)
    if not np.all(np.isfinite(rel)) or not np.all(np.isfinite(face_axes)) or not np.all(np.isfinite(edge_axes)):
        raise GeometryError("mesh convex data is nonfinite")
    record = {
        "vertices": np.array(rel, dtype=np.float64, copy=True),
        "face_axes": np.array(face_axes, dtype=np.float64, copy=True),
        "edge_axes": np.array(edge_axes, dtype=np.float64, copy=True),
    }
    if len(_MESH_CONVEX_CACHE) >= _MESH_CONVEX_CACHE_LIMIT:
        _MESH_CONVEX_CACHE.clear()
    _MESH_CONVEX_CACHE[key] = record
    return {
        "vertices": record["vertices"].copy(),
        "face_axes": record["face_axes"].copy(),
        "edge_axes": record["edge_axes"].copy(),
    }

# blenderutils.py
""" library for Blender-related functions or classes.

Copyright (c) 2024 Sony Semiconductor Solutions Corporation
This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

from pathlib import Path
import math
import bpy

SPACE = 50  # space between two shading nodes
SHIFT_Y = 100  # shift value (y) for whole shading nodes
SHIFT_X = 400  # shift value (x) for whole shading nodes

MULTIPLY_R_VALUE = 1  # albedo (R) gain value
MULTIPLY_G_VALUE = 1  # albedo (G) gain value
MULTIPLY_B_VALUE = 1  # albedo (B) gain value
MULTIPLY_ROUGH_VALUE = 1  # roughness gain value


def uv_unwrapping_core(default_mesh_path: Path, unwrapped_mesh_path: Path, angle: float, margin: float) -> None:
    """ Performs UV unwrapping on a mesh imported from a PLY file and exports it as an OBJ file.

    Args:
        default_mesh_path (Path): file path to the default mesh to be unwrapped.
        unwrapped_mesh_path (Path): file path to the unwrapped mesh.
        angle (float): angle limit for smart UV projection in degrees.
        margin (float): margin to use between islands in the UV map.
    """

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.ply_import(filepath=str(default_mesh_path))

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=margin)
            bpy.ops.object.mode_set(mode="OBJECT")
            bpy.context.view_layer.update()
            bpy.ops.wm.obj_export(filepath=str(unwrapped_mesh_path), up_axis="Z", forward_axis="Y")
        else:
            obj.select_set(False)


def load_obj(obj_path: Path) -> bpy.types.Object:
    """ Imports an OBJ file into the current Blender scene and returns the imported object.

    Args:
        obj_path (Path): file path to the OBJ file to be imported.

    Returns:
        bpy.types.Object: object that was imported into the scene.
    """

    # delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # import obj file
    bpy.ops.wm.obj_import(filepath=str(obj_path))

    obj = bpy.context.active_object
    return obj


def apply_texture_map(obj: bpy.types.Object, albedo_uv_path: Path, roughness_uv_path: Path) -> None:
    """ Applies albedo and roughness texture maps to the given object.

    Args:
        obj (bpy.types.Object): object to which the texture maps will be applied.
        albedo_uv_path (Path): file path to the albedo texture image.
        roughness_uv_path (Path): file path to the roughness texture image.

    Returns:
        None: This function does not return any value.
    """

    # create new material node
    mat = bpy.data.materials.new(name="RGB with roughness")
    obj.data.materials.append(mat)
    mat.use_nodes = True
    mat.use_fake_user = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # add Principled BSDF node
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")

    # add Image Texture node and load rgb image
    rgb_node = nodes.new(type="ShaderNodeTexImage")
    rgb = bpy.data.images.load(str(albedo_uv_path.absolute()))
    rgb_node.image = rgb
    rgb_node.image.colorspace_settings.name = "Non-Color"

    # add separate, multiply and combine node for albedo gain
    separate_rgb_node = nodes.new(type="ShaderNodeSeparateRGB")

    multiply_R_node = nodes.new(type="ShaderNodeMath")
    multiply_R_node.operation = "MULTIPLY"
    multiply_R_node.inputs[0].default_value = MULTIPLY_R_VALUE

    multiply_G_node = nodes.new(type="ShaderNodeMath")
    multiply_G_node.operation = "MULTIPLY"
    multiply_G_node.inputs[0].default_value = MULTIPLY_G_VALUE

    multiply_B_node = nodes.new(type="ShaderNodeMath")
    multiply_B_node.operation = "MULTIPLY"
    multiply_B_node.inputs[0].default_value = MULTIPLY_B_VALUE

    combine_rgb_node = nodes.new(type="ShaderNodeCombineRGB")

    # add Image Texture node and load roughness image
    roughness_node = nodes.new(type="ShaderNodeTexImage")
    roughness = bpy.data.images.load(str(roughness_uv_path.absolute()))
    roughness_node.image = roughness
    roughness_node.image.colorspace_settings.name = "Non-Color"

    # add multiply node for roughness gain
    multiply_roughness_node = nodes.new(type="ShaderNodeMath")
    multiply_roughness_node.operation = "MULTIPLY"
    multiply_roughness_node.inputs[0].default_value = MULTIPLY_ROUGH_VALUE

    # add output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    # connect nodes
    links = mat.node_tree.links
    links.new(rgb_node.outputs["Color"], separate_rgb_node.inputs["Image"])
    links.new(separate_rgb_node.outputs["R"], multiply_R_node.inputs[1])
    links.new(separate_rgb_node.outputs["G"], multiply_G_node.inputs[1])
    links.new(separate_rgb_node.outputs["B"], multiply_B_node.inputs[1])

    links.new(multiply_R_node.outputs["Value"], combine_rgb_node.inputs["R"])
    links.new(multiply_G_node.outputs["Value"], combine_rgb_node.inputs["G"])
    links.new(multiply_B_node.outputs["Value"], combine_rgb_node.inputs["B"])
    links.new(combine_rgb_node.outputs["Image"], principled_node.inputs["Base Color"])

    links.new(roughness_node.outputs["Color"], multiply_roughness_node.inputs[1])
    links.new(multiply_roughness_node.outputs["Value"], principled_node.inputs["Roughness"])

    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    # adjust node location
    accum_space = combine_rgb_node.width + SPACE

    combine_rgb_node.location = (-accum_space + SHIFT_X, SHIFT_Y)
    accum_space += multiply_R_node.width + SPACE

    multiply_R_node.location = (-accum_space + SHIFT_X, SHIFT_Y + (multiply_R_node.height + SPACE * 1.5))
    multiply_G_node.location = (-accum_space + SHIFT_X, SHIFT_Y)
    multiply_B_node.location = (-accum_space + SHIFT_X, SHIFT_Y - (multiply_R_node.height + SPACE * 1.5))
    accum_space += separate_rgb_node.width + SPACE

    separate_rgb_node.location = (-accum_space + SHIFT_X, SHIFT_Y)
    accum_space += rgb_node.width + SPACE

    rgb_node.location = (-accum_space + SHIFT_X, SHIFT_Y)

    multiply_roughness_node.location = (
    -(multiply_roughness_node.width + SPACE) + SHIFT_X, -(rgb_node.height + SPACE * 4) + SHIFT_Y)
    roughness_node.location = (-(roughness_node.width + multiply_roughness_node.width + 2 * SPACE) + SHIFT_X,
                               -(rgb_node.height + SPACE * 5) + SHIFT_Y)

    principled_node.location = (SHIFT_X, SHIFT_Y)
    output_node.location = (principled_node.width + SPACE + SHIFT_X, SHIFT_Y)


def apply_vis_roughness_map(obj: bpy.types.Object, roughness_uv_path: Path) -> None:
    """ Applies a roughness texture map to the given object for visualization purposes.

    Args:
        obj (bpy.types.Object): object to which the roughness map will be applied for visualization.
        roughness_uv_path (Path): file path to the roughness texture image.

    Returns:
        None: This function does not return any value.
    """

    # visualize roughness map as sRGB
    mat = bpy.data.materials.new(name="roughness only")
    obj.data.materials.append(mat)
    mat.use_nodes = True
    mat.use_fake_user = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_node.inputs[
        "Roughness"].default_value = 1.0  # Since roughness up map is set as albedo, roughness value is set to 1 (this should not be changed).
    roughness_node = nodes.new(type="ShaderNodeTexImage")
    roughness = bpy.data.images.load(str(roughness_uv_path.absolute()))
    roughness_node.image = roughness
    roughness_node.image.colorspace_settings.name = "sRGB"

    output_node = nodes.new(type="ShaderNodeOutputMaterial")

    links = mat.node_tree.links
    links.new(roughness_node.outputs["Color"], principled_node.inputs["Base Color"])
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    roughness_node.location = (-(roughness_node.width + SPACE), SHIFT_Y)
    principled_node.location = (0, SHIFT_Y)
    output_node.location = (principled_node.width + SPACE, SHIFT_Y)


def modify_obj(obj: bpy.types.Object) -> None:
    """ Modifies the given object.

    Args:
        obj (bpy.types.Object): object to be modified.
    """

    # rotate object
    obj.rotation_euler[0] += math.radians(-90)


def set_circle_path_camera(focal_length: float,
                           camera_path_radius: float,
                           camera_path_z: float,
                           camera_path_duration: int) -> None:
    """ Sets up a camera to follow a circular path around the selected mesh object and look at it during the animation.

    Args:
        focal_length (float): focal length for the camera lens.
        camera_path_radius (float): radius of the circular path the camera will follow.
        camera_path_z (float): Z coordinate of the circular path's center.
        camera_path_duration (int): number of frames it takes for the camera to complete one revolution around the path.
    """

    # delete non-mesh object
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            obj.select_set(True)
        else:
            obj.select_set(False)
    bpy.ops.object.delete()

    # select mesh object
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

    # add circle path
    target_object = bpy.context.object
    bpy.ops.curve.primitive_bezier_circle_add(radius=camera_path_radius, enter_editmode=False,
                                              location=target_object.location)
    circle_path = bpy.context.object
    circle_path.location.z = camera_path_z

    # add camera
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.data.lens = focal_length

    # add a Follow Path constraint for moving the camera along a path
    follow_path_constraint = camera.constraints.new(type="FOLLOW_PATH")
    follow_path_constraint.target = circle_path
    follow_path_constraint.use_curve_follow = True

    # add a Track To constraint for the camera to face the target object
    track_to_constraint = camera.constraints.new(type="TRACK_TO")
    track_to_constraint.target = target_object
    track_to_constraint.up_axis = "UP_Y"
    track_to_constraint.track_axis = "TRACK_NEGATIVE_Z"

    # set the number of frames for the camera to complete one revolution around the path
    circle_path.data.path_duration = camera_path_duration

    # set the start and end frames for the animation
    start_frame = 0
    end_frame = camera_path_duration

    camera.constraints["Follow Path"].offset = 0.0
    camera.constraints["Follow Path"].keyframe_insert(data_path="offset", frame=start_frame)

    camera.constraints["Follow Path"].offset = camera_path_duration
    camera.constraints["Follow Path"].keyframe_insert(data_path="offset", frame=end_frame)

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame

    # set keyframe points for the offset of the Follow Path constraint
    f_curve = camera.animation_data.action.fcurves.find(
        'constraints["Follow Path"].offset')  # "constraints['Follow Path'].offset" does not work

    # set the interpolation type of the keyframe points to linear
    for keyframe_point in f_curve.keyframe_points:
        keyframe_point.interpolation = "LINEAR"


def set_world(env_path: Path) -> None:
    """ Configures the world environment in the current Blender scene using an environment texture.

    Args:
        env_path (Path): file path of the environment map.
    """

    # add World node and fine links
    world = bpy.context.scene.world
    links = world.node_tree.links
    world.use_nodes = True
    world.node_tree.nodes.clear()

    # add background, environment and output nodes
    background_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
    env_texture_node = world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    env_texture_node.image = bpy.data.images.load(str(env_path.absolute()))
    output_node = world.node_tree.nodes.new(type="ShaderNodeOutputWorld")

    # connect nodes
    links.new(env_texture_node.outputs["Color"], background_node.inputs["Color"])
    links.new(background_node.outputs["Background"], output_node.inputs["Surface"])

    # adjust node location
    env_texture_node.location = (-(env_texture_node.width + SPACE), 0)
    background_node.location = (0, 0)
    output_node.location = (background_node.width + SPACE, 0)


def set_rendering_mode(rendering_mode: str,
                       use_gpu: bool,
                       cycles_max_sampling: int,
                       camera_resolution_x: int,
                       camera_resolution_y: int) -> None:
    """ Configures the rendering settings for the current Blender scene.

    Args:
        rendering_mode (str): rendering engine to use ("EEVEE" or "Cycles").
        use_gpu (bool): If True and if rendering_mode is "Cycles", the GPU will be used for rendering.
        cycles_max_sampling (int): maximum number of samples for Cycles rendering.
        camera_resolution_x (int): horizontal resolution of the camera.
        camera_resolution_y (int): vertical resolution of the camera.
    """

    # set camera resolution
    bpy.context.scene.render.resolution_x = camera_resolution_x
    bpy.context.scene.render.resolution_y = camera_resolution_y

    # set rendering mode
    if rendering_mode == "EEVEE":
        bpy.context.scene.render.engine = "BLENDER_EEVEE"
        bpy.context.scene.eevee.shadow_cube_size = "4096"  # maximize the shadow resolution in EEVEE mode.

    elif rendering_mode == "Cycles":
        bpy.context.scene.render.engine = "CYCLES"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.scene.cycles.samples = cycles_max_sampling

        if use_gpu:
            bpy.context.scene.cycles.device = "GPU"


def save_blender_data(blender_save_path: Path) -> None:
    """ Saves the current Blender scene to the specified file path.

    Args:
        blender_save_path (Path): file path where the Blender file will be saved.
    """

    # select mesh object only
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        else:
            obj.select_set(False)

    bpy.context.preferences.filepaths.save_version = 0  # do not save version

    # save blender data
    bpy.ops.wm.save_as_mainfile(filepath=str(blender_save_path))


def save_animation(obj: bpy.types.Object, out_dir_path: Path) -> None:
    """ Renders and saves an animation of the given object with different material settings.

    Args:
        obj (bpy.types.Object): object whose animation will be rendered and saved.
        out_dir_path (Path): directory path where the animation videos will be saved.
    """

    # set animation setting
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"  # vido format: FFMPEG
    bpy.context.scene.render.ffmpeg.format = "MPEG4"  # output format: MPEG4
    bpy.context.scene.render.ffmpeg.codec = "H264"  # codec: H.264

    bpy.context.scene.camera = bpy.data.objects["Camera"]

    material_rgb = bpy.data.materials.get("RGB with roughness")
    material_roughness = bpy.data.materials.get("roughness only")

    # save animation (visualize roughness)
    obj.data.materials[0] = material_roughness
    bpy.context.scene.render.filepath = str(out_dir_path.joinpath(f"roughness_vis.mp4").absolute())
    bpy.ops.render.render(animation=True)

    # save animation (albedo + roughness)
    obj.data.materials[0] = material_rgb
    bpy.context.scene.render.filepath = str(out_dir_path.joinpath(f"albedo_roughness.mp4").absolute())
    bpy.ops.render.render(animation=True)

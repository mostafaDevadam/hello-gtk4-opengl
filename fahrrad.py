import sys
import os
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import moderngl
import numpy as np
from pyrr import Matrix44
import trimesh

# -------------------------------------------------------------------------
# strict OpenGL ES 3.0 Shaders (Standard for GLES environments)
# -------------------------------------------------------------------------
VERTEX_SHADER = """
#version 300 es
precision highp float;
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_normal;
out vec3 v_color;

uniform mat4 m_model;
uniform mat4 m_view;
uniform mat4 m_proj;

void main() {
    gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
    v_normal = mat3(m_model) * in_normal;
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 300 es
precision highp float;

in vec3 v_normal;
in vec3 v_color;
out vec4 fragColor;

void main() {
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.8));
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    
    vec3 ambient = 0.35 * v_color;
    vec3 active_diffuse = 0.65 * v_color * diffuse;
    
    fragColor = vec4(ambient + active_diffuse, 1.0);
}
"""

def load_multi_material_mesh(filepath):
    """
    Parses a multi-material scene, baking material colors directly into 
    the vertex data arrays so it can be rendered in a single draw call.
    """
    if not os.path.exists(filepath):
        print(f"Warning: '{filepath}' not found. Generating a mock colored geometric assembly.")
        mesh = trimesh.creation.icosahedron()
        colors = np.random.uniform(0.2, 0.8, (len(mesh.vertices), 3)).astype('f4')
        v_bytes = np.hstack([mesh.vertices.astype('f4'), mesh.vertex_normals.astype('f4'), colors]).tobytes()
        return v_bytes, mesh.faces.astype('i4').tobytes()

    scene = trimesh.load(filepath, force='scene')
    
    global_vertices = []
    global_normals = []
    global_colors = []
    global_faces = []
    vertex_count = 0

    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        if geometry_name is None:
            continue

        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)

        diffuse_color = np.array([0.64, 0.64, 0.64], dtype='f4')
        
        if hasattr(mesh.visual, 'material'):
            mat = mesh.visual.material
            if hasattr(mat, 'diffuse'):
                diffuse_color = np.array(mat.diffuse[:3], dtype='f4') / 255.0
            elif hasattr(mat, 'baseColorFactor'):
                diffuse_color = np.array(mat.baseColorFactor[:3], dtype='f4')

        if not hasattr(mesh, 'vertex_normals') or len(mesh.vertex_normals) == 0:
            mesh.generate_normals()

        global_vertices.append(mesh.vertices.astype('f4'))
        global_normals.append(mesh.vertex_normals.astype('f4'))
        
        mesh_colors = np.tile(diffuse_color, (len(mesh.vertices), 1))
        global_colors.append(mesh_colors.astype('f4'))
        
        global_faces.append(mesh.faces.astype('i4') + vertex_count)
        vertex_count += len(mesh.vertices)

    all_v = np.vstack(global_vertices)
    all_n = np.vstack(global_normals)
    all_c = np.vstack(global_colors)
    all_f = np.vstack(global_faces)

    bbox_center = all_v.mean(axis=0)
    all_v -= bbox_center
    max_extent = np.max(np.max(all_v, axis=0) - np.min(all_v, axis=0))
    if max_extent > 0:
        all_v = (all_v / max_extent) * 1.8

    v_bytes = np.hstack([all_v, all_n, all_c]).tobytes()
    i_bytes = all_f.tobytes()

    return v_bytes, i_bytes


class ModernGLWidget(Gtk.GLArea):
    def __init__(self, obj_file_path):
        super().__init__()
        self.obj_file_path = obj_file_path
        
        self.ctx = None
        self.program = None
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.rotation = 0.0
        self.speed = 2.0  
        self.zoom = 4.5

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_has_depth_buffer(True)
        
        self.set_required_version(3, 0)
        if hasattr(self, 'set_allowed_apis'):
            self.set_allowed_apis(Gdk.GLAPI.GLES)

    # CRITICAL FIX 1: Override native do_realize instead of using .connect()
    def do_realize(self):
        Gtk.GLArea.do_realize(self)
        self.make_current()
        if self.get_error() is not None:
            print("GL context realization failed.")
            return

        self.ctx = moderngl.get_context()
        
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )

        v_bytes, i_bytes = load_multi_material_mesh(self.obj_file_path)

        self.vbo = self.ctx.buffer(v_bytes)
        self.ibo = self.ctx.buffer(i_bytes)

        content_layout = [
            (self.vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')
        ]
        self.vao = self.ctx.vertex_array(self.program, content_layout, self.ibo)

    # CRITICAL FIX 2: Override native do_resize to guarantee matrix execution
    def do_resize(self, width, height):
        Gtk.GLArea.do_resize(self, width, height)
        self.make_current()
        if not self.ctx or not self.program:
            return

        aspect = width / max(height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        self.program['m_proj'].write(proj.astype('f4').T.tobytes())

    # CRITICAL FIX 3: Override native do_render to handle frame buffers directly
    def do_render(self, context):
        self.make_current()
        if self.ctx is None or self.program is None or self.vao is None:
            return False

        try:
            # Bind GTK's native target frame buffer object
            fbo = self.ctx.detect_framebuffer()
            fbo.use()

            width = self.get_width()
            height = self.get_height()
            self.ctx.viewport = (0, 0, width, height)

            # Force blend modes off so GTK doesn't read the alpha channels as transparent
            self.ctx.disable(moderngl.BLEND)
            self.ctx.enable(moderngl.DEPTH_TEST)
            
            # CRITICAL FIX 4: Explicit alpha clear forced to 1.0 (Opaque Black Background)
            self.ctx.clear(0.12, 0.13, 0.15, 1.0, depth=1.0)

            aspect = width / max(height, 1)
            proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
            view = Matrix44.look_at(
                eye=(0.0, 0.5, self.zoom),      
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0)
            )
            
            base_correction = Matrix44.from_x_rotation(-np.pi / 2.0)
            spin = Matrix44.from_y_rotation(self.rotation)
            model = spin * base_correction

            self.program['m_proj'].write(proj.astype('f4').T.copy().tobytes())
            self.program['m_view'].write(view.astype('f4').T.copy().tobytes())
            self.program['m_model'].write(model.astype('f4').T.copy().tobytes())

            self.vao.render(moderngl.TRIANGLES)
            return True

        except Exception as e:
            print(f"Render engine error: {e}")
            return False


def on_tick(canvas):
    canvas.rotation += (canvas.speed / 120.0)
    canvas.queue_draw()
    return True


def on_slider_changed(slider, canvas):
    canvas.speed = slider.get_value()


class GTK4App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLGTK4')

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL Fixed Mesh Viewport")
        win.set_default_size(950, 650)

        obj_path = 'models/fahrrad/fahrrad.obj'
        gl_widget = ModernGLWidget(obj_path) 

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        main_box.append(gl_widget)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)

        label = Gtk.Label(label="Rotation Speed:")
        sidebar.append(label)
        
        slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.5)
        slider.set_value(2.0)
        slider.connect("value-changed", on_slider_changed, gl_widget)
        sidebar.append(slider)

        main_box.append(sidebar)
        win.set_child(main_box)

        GLib.timeout_add(16, on_tick, gl_widget)
        win.present()


if __name__ == '__main__':
    app = GTK4App()
    app.run(sys.argv)

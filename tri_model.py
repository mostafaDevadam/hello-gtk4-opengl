import sys
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import moderngl
import numpy as np
from pyrr import Matrix44
import trimesh

def load_mesh_data(filepath):
    """Loads mesh data using trimesh into bytes for ModernGL."""
    mesh = trimesh.load(filepath, force='mesh')
    
    # trimesh automatically handles normal generation
    if not hasattr(mesh, 'vertex_normals'):
        mesh.generate_normals()

    vertices = mesh.vertices.astype('f4')
    normals = mesh.vertex_normals.astype('f4')
    
    vertex_data = np.hstack([vertices, normals])
    indices = mesh.faces.astype('i4')

    return vertex_data.tobytes(), indices.tobytes()


class ModernGLWidget(Gtk.GLArea):
    def __init__(self, obj_file_path):
        super().__init__()
        self.obj_file_path = obj_file_path
        
        # Initialize attributes
        self.ctx = None
        self.program = None
        self.vao = None
        self.rotation = 0.0

        # Request depth buffer and configure ES 3.0 target
        self.set_has_depth_buffer(True)
        self.set_required_version(3, 0) # Targets OpenGL ES 3.0 (300 es)

        # Signal Bindings
        self.connect('realize', self.on_realize)
        self.connect('render', self.on_render)
        self.connect('resize', self.on_resize)

        # Start animation frame ticking
        GLib.timeout_add(16, self.on_tick)

    def on_realize(self, area):
        area.make_current()
        if area.get_error() is not None:
            print("GLArea initialization error:", area.get_error())
            return

        # Initialize ModernGL context inside active GTK Context
        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)

        # ES 3.0 compliant Shaders with explicit high precision floats
        vertex_shader = """
        #version 300 es
        precision highp float;

        in vec3 in_position;
        in vec3 in_normal;

        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;

        out vec3 v_normal;

        void main() {
            v_normal = mat3(m_model) * in_normal;
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        out vec4 fragColor;

        void main() {
            vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
            float diff = max(dot(normalize(v_normal), light_dir), 0.2);
            vec3 color = vec3(0.2, 0.6, 1.0) * diff;
            fragColor = vec4(color, 1.0);
        }
        """

        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_shader,
                fragment_shader=fragment_shader
            )
        except Exception as e:
            print("GLSL compilation failed:\n", e)
            return

        # Load obj buffers
        try:
            v_bytes, i_bytes = load_mesh_data(self.obj_file_path)
            vbo = self.ctx.buffer(v_bytes)
            ibo = self.ctx.buffer(i_bytes)

            self.vao = self.ctx.vertex_array(
                self.program,
                [(vbo, '3f 3f', 'in_position', 'in_normal')],
                index_buffer=ibo
            )
        except Exception as e:
            print("Failed loading 3D Mesh Assets:\n", e)

    def on_resize(self, area, width, height):
        if not self.ctx or not self.program:
            return

        aspect = width / max(height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        self.program['m_proj'].write(proj.astype('f4').tobytes())

    def on_render(self, area, context):
        if not self.ctx or not self.program or not self.vao:
            return False

        # Make sure ModernGL renders to GTK's framebuffer
        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.clear(0.15, 0.15, 0.18, 1.0)

        view = Matrix44.look_at(
            eye=(0.0, 3.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        model = Matrix44.from_eulers((0.0, self.rotation, 0.0))

        self.program['m_view'].write(view.astype('f4').tobytes())
        self.program['m_model'].write(model.astype('f4').tobytes())

        self.vao.render()
        return True

    def on_tick(self):
        self.rotation += 0.02
        self.queue_draw()
        return True


class GTK4App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLGTK4')

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL ES 3.0 2D Trianlge")
        win.set_default_size(800, 600)

        # Point this to your actual OBJ model
        gl_widget = ModernGLWidget('tri_3d.obj') 
        win.set_child(gl_widget)
        win.present()

if __name__ == '__main__':
    app = GTK4App()
    app.run(sys.argv)
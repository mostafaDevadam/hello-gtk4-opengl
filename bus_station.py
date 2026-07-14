import sys
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import moderngl
import numpy as np
from pyrr import Matrix44
import trimesh

def load_combined_mesh_data(bus_path, station_path, cb):
    bus_mesh = trimesh.load(bus_path, force='mesh')
    station_mesh = trimesh.load(station_path, force='mesh')
    
    # Offset the bus slightly so it parks alongside the station
    #bus_mesh.vertices += [0.0, 0.0, 0.7]  
    # 1. Scale the bus up (adjust 1.8 to make it even larger/smaller if needed)
    bus_mesh.apply_scale(1.8)

    # 2. Rotate the bus 90 degrees (pi/2 radians) so it is parallel to the station
    # The OBJ's local 'up' axis is Z, so we rotate around Z.
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 50, 0])
    #bus_mesh.apply_transform(rotation_matrix)

    # 2. Extract coordinates
    """x = bus_mesh.vertices[:, 0]
    y = bus_mesh.vertices[:, 1]
    z = bus_mesh.vertices[:, 2]

    # 3. Swap and invert coordinates (This rotates exactly -90 degrees around Y)
    # New X becomes -Z, New Z becomes X
    #bus_mesh.vertices = np.column_stack([-z, y, x])

    R_y = np.array([
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])"""

    # 3. Rotate vertices using matrix multiplication
    #bus_mesh.vertices = bus_mesh.vertices @ R_y.T

    #bus_mesh.vertices += [0, -0.4, -0.5]
    #bus_mesh.vertices += [0.0, -0.4, -0.5]

    #bus_mesh.vertices -= bus_mesh.center_mass

    # 3. Slide the bus:
    # [X (left/right), Y (up/down), Z (forward/backward)]
    # We push it forward on Z, align it on X, and drop it down slightly on Y if needed.
    #bus_mesh.vertices += [0.0, 0.0, 7.0]
    #bus_mesh.vertices += [-1.5, -0.4, 11.8]

    # 2. Rotate left (-90 degrees) around the Z-axis (keeps it flat on its wheels)
    R_z = np.array([
        [ 0.0,  1.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ])
    #bus_mesh.vertices = bus_mesh.vertices @ R_z.T

    # 3. Compensate for the external [0.0, 0.0, 7.0] addition
    # [X remains 0.0, Y remains -0.4, Z becomes -7.5]
    #bus_mesh.vertices += [0.0, -0.4, -7.5]
    #bus_mesh.vertices += [0.0, 0.0, 7.0]

    if cb is not None:
        cb(bus_mesh) # This will now safely shift the bus on its Z axis!

   



    # Merge geometries together
    scene = trimesh.Scene([bus_mesh, station_mesh])
    combined_mesh = scene.to_mesh()

    # Global scale normalization
    combined_mesh.vertices -= combined_mesh.center_mass
    max_bound = np.max(np.abs(combined_mesh.vertices))
    if max_bound > 0:
        combined_mesh.vertices /= max_bound
        combined_mesh.vertices *= 1.4

    if not hasattr(combined_mesh, 'vertex_normals') or len(combined_mesh.vertex_normals) == 0:
        combined_mesh.generate_normals()

    color_visuals = combined_mesh.visual.to_color()
    colors = color_visuals.vertex_colors.astype('f4') / 255.0

    vertices = combined_mesh.vertices.astype('f4')
    normals = combined_mesh.vertex_normals.astype('f4')
    
    vertex_data = np.hstack([vertices, normals, colors])
    indices = combined_mesh.faces.astype('i4')

    return vertex_data.tobytes(), indices.tobytes(), bus_mesh


class ModernGLWidget(Gtk.GLArea):
    def __init__(self, bus_file, station_file):
        super().__init__()
        self.bus_file = bus_file
        self.station_file = station_file
        
        self.ctx = None
        self.program = None
        self.vao = None
        self.rotation = 0.0
        self.speed = 2.0  
        self.pos_y = 0.0  
        self.zoom = 4.5  
        self.bus_z_margin = 7.0

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_has_depth_buffer(True)
        self.set_required_version(3, 0)

        self.connect('realize', self.on_realize)
        self.connect('render', self.on_render)
        self.connect('resize', self.on_resize)

    
    def update_bus_mesh_z_margin(self, x):
                x.vertices += [0.0, 0.0, self.bus_z_margin]

    def on_realize(self, area):
        area.make_current()
        if area.get_error() is not None:
            print("GLArea initialization error:", area.get_error())
            return

        self.ctx = moderngl.get_context()
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        vertex_shader = """
        #version 300 es
        precision highp float;

        in vec3 in_position;
        in vec3 in_normal;
        in vec4 in_color;

        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;

        out vec3 v_normal;
        out vec4 v_color;

        void main() {
            v_normal = mat3(m_model) * in_normal;
            v_color = in_color;
            gl_Position = m_proj * m_view * m_model * vec4(in_position, 1.0);
        }
        """

        fragment_shader = """
        #version 300 es
        precision highp float;

        in vec3 v_normal;
        in vec4 v_color;

        out vec4 fragColor;

        void main() {
            vec3 light_dir = normalize(vec3(1.5, 3.5, 2.5));
            float diff = max(dot(normalize(v_normal), light_dir), 0.3);
            
            vec3 diffuse = v_color.rgb * diff;
            vec3 ambient = v_color.rgb * 0.22;
            
            fragColor = vec4(diffuse + ambient, v_color.a);
        }
        """

        try:
            self.program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        except Exception as e:
            print("Shader compilation failed:\n", e)
            return

        try:
            
                
            
            v_bytes, i_bytes, _ = load_combined_mesh_data(self.bus_file, self.station_file, 
                                                                     lambda x: self.update_bus_mesh_z_margin(x)
                                                                 )
            

            
            self.vbo = self.ctx.buffer(v_bytes)
            self.ibo = self.ctx.buffer(i_bytes)

            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, '3f 3f 4f', 'in_position', 'in_normal', 'in_color')],
                index_buffer=self.ibo
            )
            print("Successfully merged and loaded Bus + Station Scene!")
        except Exception as e:
            print("Error parsing data files:", e)

    def on_resize(self, area, width, height):
        if not self.ctx or not self.program:
            return
        aspect = width / max(height, 1)
        proj = Matrix44.perspective_projection(45.0, aspect, 0.1, 100.0)
        self.program['m_proj'].write(proj.astype('f4').tobytes())

    def on_render(self, area, context):
        if not self.ctx or not self.program or not self.vao:
            return False

        fbo = self.ctx.detect_framebuffer()
        fbo.use()
        self.ctx.clear(0.1, 0.11, 0.13, 1.0)

        view = Matrix44.look_at(
            eye=(0.0, 1.2, self.zoom),      
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0)
        )
        
        base_correction = Matrix44.from_x_rotation(-np.pi / 100)
        spin = Matrix44.from_y_rotation(self.rotation)
        translation = Matrix44.from_translation((0.0, self.pos_y, 0.0))
        
        model = translation * spin * base_correction

        self.program['m_view'].write(view.astype('f4').tobytes())
        self.program['m_model'].write(model.astype('f4').tobytes())

        self.vao.render()
        return True

    def update_mesh_buffers(self, v_bytes, i_bytes):
        # 'orphan=True' tells ModernGL to safely reallocate memory 
        # if the number of vertices changed
        self.vbo.write(v_bytes, orphan=True) 
        self.ibo.write(i_bytes, orphan=True)

# --- Timing UI Callbacks ---

def on_tick(canvas):
    canvas.rotation += (canvas.speed / 120.0)
    canvas.queue_draw()
    return True


class GTK4App(Gtk.Application):
    def __init__(self):
        super().__init__(application_id='com.example.ModernGLCombinedViewer')
        self.bus_file='bus.obj' 
        self.station_file='stationBus.obj'

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self)
        win.set_title("GTK4 + ModernGL Combined Scene Viewer")
        win.set_default_size(950, 650)

       
        # Passes both targets explicitly to the canvas initializer
        gl_widget = ModernGLWidget(bus_file=self.bus_file, station_file=self.station_file) 

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        main_box.append(gl_widget)

        # Sidebar controls layout
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        sidebar.set_size_request(220, -1)

        # Speed Control
        s_label = Gtk.Label(label="Rotation Speed:")
        s_label.set_halign(Gtk.Align.START)
        speed_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
        speed_slider.set_value(gl_widget.speed)
        speed_slider.connect("value-changed", lambda sl: setattr(gl_widget, 'speed', sl.get_value()))
        sidebar.append(s_label)
        sidebar.append(speed_slider)

        # Height Control
        h_label = Gtk.Label(label="Height (Y Position):")
        h_label.set_halign(Gtk.Align.START)
        height_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, -2.0, 2.0, 0.1)
        height_slider.set_value(gl_widget.pos_y)
        height_slider.connect("value-changed", lambda sl: setattr(gl_widget, 'pos_y', sl.get_value()))
        sidebar.append(h_label)
        sidebar.append(height_slider)

        # Zoom Control
        zoom_label = Gtk.Label(label="Zoom Window:")
        zoom_label.set_halign(Gtk.Align.START)
        zoom_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.5, 10.0, 0.1)
        zoom_slider.set_value(gl_widget.zoom)
        zoom_footer = Gtk.Label(label=f"zoom: {gl_widget.zoom:.1f}")
        zoom_footer.set_halign(Gtk.Align.START)

        def update_zoom(sl):
            val = sl.get_value()
            gl_widget.zoom = val
            zoom_footer.set_label(f"zoom: {val:.1f}")

        zoom_slider.connect("value-changed", update_zoom)
        sidebar.append(zoom_label)
        sidebar.append(zoom_slider)
        sidebar.append(zoom_footer)

        # bus margin z
        bus_margin_z_label = Gtk.Label(label="Bus Z Margin Window:")
        bus_margin_z_label.set_halign(Gtk.Align.START)
        bus_margin_z_slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.5, 10.0, 0.1)
        bus_margin_z_slider.set_value(gl_widget.bus_z_margin)
        
        bus_margin_z_footer = Gtk.Label(label=f"Bus Z Margin: {gl_widget.bus_z_margin:.1f}")
        bus_margin_z_footer.set_halign(Gtk.Align.START)

        def update_bus_z_margin(sl):
            val = sl.get_value()
            gl_widget.bus_z_margin = val
            bus_margin_z_footer.set_label(f"Bus Z Margin: {val:.1f}")
           
            
            # Re-run the mesh pipeline using the vertex callback
            v_bytes, i_bytes, _ = load_combined_mesh_data(
                self.bus_file, 
                self.station_file, 
                lambda x: gl_widget.update_bus_mesh_z_margin(x)
            )

            # 2. ModernGL built-in data overwrite (orphan=True clears the old memory slot safely)
            gl_widget.vbo.write(v_bytes)
            gl_widget.ibo.write(i_bytes)
            
            # Push the updated buffers back into your ModernGL widget context
            #gl_widget.update_mesh_buffers(v_bytes, i_bytes) # Or however your widget uploads bytes to GPU
            gl_widget.queue_draw()

        bus_margin_z_slider.connect("value-changed", update_bus_z_margin)
        sidebar.append(bus_margin_z_label)
        sidebar.append(bus_margin_z_slider)
        sidebar.append(bus_margin_z_footer)


        main_box.append(sidebar)
        win.set_child(main_box)
        win.present()
        
        GLib.timeout_add(16, on_tick, gl_widget)

if __name__ == '__main__':
    app = GTK4App()
    app.run(sys.argv)
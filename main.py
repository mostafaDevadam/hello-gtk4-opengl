import sys
import array
import time
import math
import gi

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

import moderngl

class ModernGLCanvas(Gtk.GLArea):
    def __init__(self):
        super().__init__()
        self.set_required_version(3, 0)
        self.set_use_es(True)
        self.set_hexpand(True)
        self.set_vexpand(True)
        
        self.ctx = None
        self.prog = None
        self.vao = None
        
        # Rotations-Variablen
        self.angle = 0.0
        self.speed = 1.0  # Standardgeschwindigkeit
        self.last_time = time.time()

    def do_realize(self):
        Gtk.GLArea.do_realize(self)
        self.make_current()
        self.ctx = moderngl.get_context()

        # VERTEX SHADER: Jetzt mit einer Rotationsmatrix basierend auf 'u_angle'
        vertex_shader = """
            #version 300 es
            in vec2 in_vert;
            in vec3 in_color;
            out vec3 v_color;
            
            uniform float u_angle; // Wert wird von Python übergeben
            
            void main() {
                // Rotationsmatrix berechnen
                float s = sin(u_angle);
                float c = cos(u_angle);
                mat2 rotation_matrix = mat2(c, s, -s, c);
                
                // Rotierte Position berechnen
                vec2 rotated_position = rotation_matrix * in_vert;
                
                gl_Position = vec4(rotated_position, 0.0, 1.0);
                v_color = in_color;
            }
        """
        
        fragment_shader = """
            #version 300 es
            precision mediump float;
            in vec3 v_color;
            out vec4 f_color;
            void main() {
                f_color = vec4(v_color, 1.0);
            }
        """
        self.prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        # Ein etwas kleineres Dreieck (Radius 0.4), damit es beim Drehen nicht aus dem Bild ragt
        vertices = array.array('f', [
             0.0,  0.4,   1.0, 0.0, 0.0,
            -0.4, -0.4,   0.0, 1.0, 0.0,
             0.4, -0.4,   0.0, 0.0, 1.0,
        ])
        vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, '2f 3f', 'in_vert', 'in_color')])

    def do_resize(self, width, height):
        Gtk.GLArea.do_resize(self, width, height)
        if self.ctx:
            self.ctx.viewport = (0, 0, width, height)

    def do_render(self, gl_context):
        if not self.ctx or not self.vao:
            return False

        # Zeitunterschied berechnen, um eine flüssige Animation zu garantieren
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time

        # Winkel inkrementieren basierend auf Slider-Geschwindigkeit
        self.angle += self.speed * delta_time
        
        # CRITICAL: Den berechneten Winkel an die Uniform-Variable im Shader senden!
        if 'u_angle' in self.prog:
            self.prog['u_angle'].value = self.angle

        # Framebuffer binden und initialisieren
        fbo = self.ctx.detect_framebuffer()
        fbo.use()

        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        self.ctx.clear(0.12, 0.12, 0.14, 1.0)
        self.vao.render()
        
        return True

def on_slider_changed(slider, canvas):
    # Aktualisiere die Rotationsgeschwindigkeit im Canvas, wenn der Slider bewegt wird
    canvas.speed = slider.get_value()

def on_tick(canvas):
    # Diese Schleife signalisiert GTK im Hintergrund, den Canvas permanent neu zu zeichnen
    canvas.queue_render()
    return True # True hält den Timer am Leben

def on_activate(app):
    window = Gtk.ApplicationWindow(application=app)
    window.set_title("GTK4 Layout: Animiertes ModernGL Dreieck")
    window.set_default_size(800, 600)

    # Haupt-Box
    main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
    main_box.set_margin_top(10)
    main_box.set_margin_bottom(10)
    main_box.set_margin_start(10)
    main_box.set_margin_end(10)

    # Canvas
    canvas = ModernGLCanvas()
    main_box.append(canvas)

    # Seitenleiste
    sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
    sidebar.set_size_request(200, -1)

    slider_label = Gtk.Label(label="Geschwindigkeit:")
    slider_label.set_halign(Gtk.Align.START)
    
    # Ein Slider von 0.0 (Stopp) bis 10.0 (Schnell), voreingestellt auf 2.0
    slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.0, 10.0, 0.2)
    slider.set_value(2.0)
    canvas.speed = 2.0
    
    # Event anbinden: Wenn der Slider sich bewegt, wird die Funktion ausgeführt
    slider.connect("value-changed", on_slider_changed, canvas)

    sidebar.append(slider_label)
    sidebar.append(slider)
    main_box.append(sidebar)

    window.set_child(main_box)
    window.present()

    # CRITICAL ANIMATION TICKER: Weist GTK an, alle 16 Millisekunden (~60 FPS) 
    # die 'on_tick' Funktion aufzurufen, welche 'queue_render()' triggert.
    GLib.timeout_add(16, on_tick, canvas)


app = Gtk.Application(application_id="com.example.gtk4modernglanimate")
app.connect("activate", on_activate)
sys.exit(app.run(sys.argv))

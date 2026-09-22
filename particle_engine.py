import cv2
import numpy as np
import config


class Particle:

  def __init__(self, x, y, magnitude, angle):
    self.x = float(x)
    self.y = float(y)
    speed = float(magnitude) * 0.5
    self.vx = np.cos(angle) * speed
    self.vy = np.sin(angle) * speed
    self.alpha = 1.0

    norm_angle = float(angle) % (2 * np.pi)
    hue = int((norm_angle / (2 * np.pi)) * 180)
    hsv_pixel = np.uint8([[[hue, 255, 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    self.color = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))

  def update(self):
    self.x += self.vx
    self.y += self.vy
    self.alpha -= config.PARTICLE_DECAY_RATE
    return self.alpha > 0.0


class ParticleEngine:

  def __init__(
      self, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT
  ):
    self.width = width
    self.height = height
    self.particles = []
    self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

  def spawn_from_flow(self, magnitude, angle):
    motion_mask = magnitude > config.MOTION_THRESHOLD
    y_coords, x_coords = np.where(motion_mask)

    if len(y_coords) == 0:
      return

    step = config.PARTICLE_SPAWN_STEP
    for y, x in zip(y_coords[::step], x_coords[::step]):
      if len(self.particles) >= config.MAX_PARTICLES:
        break
      mag_val = magnitude[y, x]
      ang_val = angle[y, x]
      self.particles.append(Particle(x, y, mag_val, ang_val))

  def update_and_render(self):
    self.canvas = (self.canvas * config.CANVAS_FADE_FACTOR).astype(np.uint8)

    surviving_particles = []
    for p in self.particles:
      if p.update():
        px, py = int(p.x), int(p.y)
        if 0 <= px < self.width and 0 <= py < self.height:
          color_faded = (
              int(p.color[0] * p.alpha),
              int(p.color[1] * p.alpha),
              int(p.color[2] * p.alpha),
          )
          cv2.circle(self.canvas, (px, py), 3, color_faded, -1)
          surviving_particles.append(p)

    self.particles = surviving_particles
    return self.canvas

  def clear(self):
    self.particles.clear()
    self.canvas.fill(0)


if __name__ == "__main__":
  engine = ParticleEngine()
  dummy_mag = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH), 5.0)
  dummy_ang = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH), np.pi / 4)

  engine.spawn_from_flow(dummy_mag, dummy_ang)
  canvas = engine.update_and_render()
  print(
      f"Particle Engine Test: Active Particles = {len(engine.particles)}, Canvas"
      f" Shape = {canvas.shape}"
  )
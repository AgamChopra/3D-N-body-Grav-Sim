import pygame
import sys
import random
import math
import numpy as np
from math import sin, cos, pi


pygame.init()
FONT = pygame.font.SysFont("Arial" , 18 , bold = True)
clock = pygame.time.Clock()
SCREEN_WIDTH = 1900
SCREEN_HEIGHT = 1080
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("3D Gravity Simulator")
bg_color = (0, 0, 0)
num_particles = 200
T = 1E2
running = True
CAM = [0, 0, 0]
f = 1000
tx, ty, tz = 0, 0, 0
psi, theta, phi = 0, 0, 0
SCALE = 5
SHOW_MAG = True
DISP_TH = 0
Q = 1
show_dir = False
show_acc_dir = False
show_blob = True
show_arc = False
G = 1E-1
TR_FACT = 20
TRAIL = TR_FACT * T * 1E-3
L = 5
EPSILON = 1E0
SHOW_DM = False
FPS = 20


def fps_counter():
    flag = int(clock.get_fps())
    c = "RED" if flag < 5 else "YELLOW" if flag < 10 else "GREEN"
    fps = str(flag)
    fps_t = FONT.render(fps , 1, pygame.Color(c))
    screen.blit(fps_t,(0,0))


def proj(obj_coord, cam_coord, fov):
    o = [obj_coord[i] - cam_coord[i] for i in range(len(obj_coord))]
    try:
        m = o[1]/o[2]
    except:
        m = 0
    screen_y = m * fov + o[1]
    try:
        m = o[0]/o[2]
    except:
        m = 0
    screen_x = m * fov + o[0]
    return screen_x, screen_y


def rot_matx(psi, theta, phi):
    r11 = cos(theta) * cos(phi)
    r12 = sin(psi) * sin(theta) * cos(phi) - cos(psi) * sin(phi)
    r13 = cos(psi) * sin(theta) * cos(phi) + sin(psi) * sin(phi)

    r21 = cos(theta) * sin(phi)
    r22 = sin(psi) * sin(theta) * sin(phi) + cos(psi) * cos(phi)
    r23 = cos(psi) * sin(theta) * sin(phi) - sin(psi) * cos(phi)

    r31 = -sin(theta)
    r32 = sin(psi) * cos(theta)
    r33 = cos(psi) * cos(theta)

    return r11, r12, r13, r21, r22, r23, r31, r32, r33


def proj2(cam_coord, obj_coord, f, tx=0, ty=0, tz=0, psi=0, theta=0, phi=0):
    axis = int(SCREEN_WIDTH/2), int(SCREEN_HEIGHT/2)

    r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_matx(psi, theta, phi)

    X = np.array((obj_coord[0], obj_coord[1], obj_coord[2], 1)).T
    K = np.array(((f, 0, cam_coord[0]), (0, f, cam_coord[1]), (0, 0, 1)))
    E = np.array(
        ((r11, r12, r13, tx), (r21, r22, r23, ty), (r31, r32, r33, tz)))

    X_ = K @ E @ X
    x = X_[0]/X_[2]
    y = X_[1]/X_[2]
    lam = X_[2]

    return x + axis[0], y + axis[1], lam


class Particle:
    def __init__(self, x, y, z, vx=0, vy=0, vz=0):
        self.x = x
        self.y = y
        self.z = z

        self.counter = 0
        self.xarc = [self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, 
                     self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x, self.x]
        self.yarc = [self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, 
                     self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y, self.y]
        self.zarc = [self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, 
                     self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z, self.z]

        self.vx = vx
        self.vy = vy
        self.vz = vz

        self.Fx = 0
        self.Fy = 0
        self.Fz = 0
        
        self.m = random.randint(2, 63)

    def update(self, dt):
        self.counter += 1
        if self.counter % TRAIL == 0:
            self.xarc = [self.x]+self.xarc[:-1]
            self.yarc = [self.y]+self.yarc[:-1]
            self.zarc = [self.z]+self.zarc[:-1]
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

    def attract(self, particle, dt):
        dx = particle.x - self.x
        dy = particle.y - self.y
        dz = particle.z - self.z
        
        r = math.sqrt(dx**2 + dy**2 + dz**2)

        F = (G * (self.m + ((2 + 63)/2) * 6) * (particle.m + ((2 + 63)/2) * 6)) / (r**3 + EPSILON**2)

        ax = F * dx / (self.m + ((2 + 63)/2) * 6)
        ay = F * dy / (self.m + ((2 + 63)/2) * 6)
        az = F * dz / (self.m + ((2 + 63)/2) * 6)
        
        self.Fx += F * dx
        self.Fy += F * dy
        self.Fz += F * dz

        self.vx += ax * dt
        self.vy += ay * dt
        self.vz += az * dt
        
        ax = F * dx / (particle.m + ((2 + 63)/2) * 6)
        ay = F * dy / (particle.m + ((2 + 63)/2) * 6)
        az = F * dz / (particle.m + ((2 + 63)/2) * 6)
        
        particle.Fx -= F * dx
        particle.Fy -= F * dy
        particle.Fz -= F * dz

        particle.vx -= ax * dt
        particle.vy -= ay * dt
        particle.vz -= az * dt
        
    def clear(self):
        self.Fx = 0
        self.Fy = 0
        self.Fz = 0             


particles = []
for i in range(int(num_particles/2)):
    x = random.uniform(-L -int(L*5), L -int(L*5))
    y = random.uniform(-L -int(L*5), L -int(L*5))
    z = random.uniform(-L -int(L*2), L -int(L*2))
    particles.append(Particle(x, y, z, random.randint(-5, 5), random.randint(-5, 5) ,random.randint(-5, 5)))
    
    x = random.uniform(-L +int(L*5), L +int(L*5))
    y = random.uniform(-L +int(L*5), L +int(L*5))
    z = random.uniform(-L +int(L*2), L +int(L*2))
    particles.append(Particle(x, y, z, random.randint(-5, 5), random.randint(-5, 5) , random.randint(-5, 5)))


while running:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                tz -= 1*Q
            if event.key == pygame.K_s:
                tz += 1*Q

            if event.key == pygame.K_a:
                tx += 1*Q
            if event.key == pygame.K_d:
                tx -= 1*Q

            if event.key == pygame.K_q:
                ty -= 1*Q
            if event.key == pygame.K_e:
                ty += 1*Q

            if event.key == pygame.K_f:
                f -= 10
            if event.key == pygame.K_g:
                f += 10

            if event.key == pygame.K_r:
                if Q > 0:
                    Q -= 1
            if event.key == pygame.K_t:
                Q += 1

            if event.key == pygame.K_k:
                psi -= pi*Q/64
            if event.key == pygame.K_i:
                psi += pi*Q/64

            if event.key == pygame.K_l:
                theta += pi*Q/64
            if event.key == pygame.K_j:
                theta -= pi*Q/64

            if event.key == pygame.K_u:
                phi -= pi*Q/64
            if event.key == pygame.K_o:
                phi += pi*Q/64

            if event.key == pygame.K_b:
                tx, ty, tz = 0, 0, 0
                psi, theta, phi = 0, 0, 0
                Q = 1

            if event.key == pygame.K_v:
                f = 2E3

            if event.key == pygame.K_y:
                f = 5E2

            if event.key == pygame.K_z:
                if show_dir:
                    show_dir = False
                else:
                    show_dir = True

            if event.key == pygame.K_x:
                if show_acc_dir:
                    show_acc_dir = False
                else:
                    show_acc_dir = True

            if event.key == pygame.K_c:
                if show_blob:
                    show_blob = False
                else:
                    show_blob = True

            if event.key == pygame.K_m:
                if show_arc:
                    show_arc = False
                else:
                    show_arc = True

            if event.key == pygame.K_n:
                TRAIL -= 10
                if TRAIL <= 0:
                    TRAIL = 1

            if event.key == pygame.K_p:
                TRAIL += 10
                if TRAIL > 1500:
                    TRAIL = 1500
            
            if event.key == pygame.K_LEFT:
                CAM[0] -= 1*Q
                
            if event.key == pygame.K_RIGHT:
                CAM[0] += 1*Q
                
            if event.key == pygame.K_UP:
                CAM[1] -= 1*Q
                
            if event.key == pygame.K_DOWN:
                CAM[1] += 1*Q
                                
    screen.fill(bg_color)
    dt = 1/T
    
    for i in range(len(particles)):
        for j in range(i+1, len(particles)):
            particles[i].attract(particles[j], dt)
        particles[i].update(dt)

    temp = sorted(particles, key=lambda x: x.z, reverse=True)

    for particle in temp:
        x, y, lam = proj2(CAM, (particle.x, particle.y, particle.z),
                              f, tx=tx, ty=ty, tz=tz, psi=psi, phi=phi, theta=theta)
    
        if lam > CAM[2]:
            scale = SCALE / ((lam - CAM[2])**2)
            if scale < 1:
                scale = 1
            if scale > 800:
                scale = 800
    
            vmag = (particle.vx**2 + particle.vy**2 + particle.vz**2)**(1/2)
            xx, yy, _ = proj2(CAM, (particle.x + (particle.vx/(vmag+EPSILON)), particle.y + (particle.vy/(vmag+EPSILON)),
                                  particle.z + (particle.vz/(vmag+EPSILON))), f, tx=tx, ty=ty, tz=tz, psi=psi, phi=phi, theta=theta)
    
            Fmag = (particle.Fx**2 + particle.Fy**2 + particle.Fz ** 2)**(1/2)
            dx, dy, _ = proj2(CAM, (particle.x + (particle.Fx/(Fmag+EPSILON)), particle.y + (particle.Fy/(Fmag+EPSILON)),
                                  particle.z + (particle.Fz/(Fmag+EPSILON))), f, tx=tx, ty=ty, tz=tz, psi=psi, phi=phi, theta=theta)
    
            if particle.z + tz > CAM[2]:
    
                if show_acc_dir:
                    pygame.draw.line(screen, (int(250*abs(particle.Fx)/(Fmag+EPSILON)), int(250*abs(particle.Fy)/(
                            Fmag+EPSILON)), int(250*abs(particle.Fz)/(Fmag+EPSILON))), (int(x), int(y)), (int(dx), int(dy)))
    
                if show_dir:
                    pygame.draw.line(screen, (int(250*abs(particle.vx)/(vmag+EPSILON)), int(250*abs(particle.vy)/(
                            vmag+EPSILON)), int(250*abs(particle.vz)/(vmag+EPSILON))), (int(x), int(y)), (int(xx), int(yy)))
    
                if show_arc:
                    x_, y_ = x, y
    
                    for i in range(len(particle.xarc)):
                        xi, yi, _ = proj2(
                                CAM, (particle.xarc[i], particle.yarc[i], particle.zarc[i]), f, tx=tx, ty=ty, tz=tz, psi=psi, phi=phi, theta=theta)
                        pygame.draw.line(screen, (150, 75, 75),
                                             (int(x_), int(y_)), (int(xi), int(yi)))
                        x_, y_ = xi, yi
    
                if show_blob:
                    pygame.draw.circle(
                        screen, (255 - 4*particle.m, 2*particle.m, 4*particle.m), (int(x), int(y)), int(scale))
                        
                particle.clear()
                
    fps_counter()
    pygame.display.update()
    clock.tick(240)

pygame.quit()
sys.exit()
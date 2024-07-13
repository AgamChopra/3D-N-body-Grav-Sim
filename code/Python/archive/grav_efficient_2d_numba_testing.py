import pygame
from numpy import concatenate, nan_to_num, sum, dot, ones, array, asarray, einsum, sqrt, empty, int32, float64, int64
from numpy.random import randint
from numba import jit, cuda

pygame.init()

G = 1E-1
FPS = 60
SPF = 2.5E-1/FPS
RADIUS = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 1000, 1000
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Gravity Simulator Efficient CPU')
FONT = pygame.font.SysFont("Arial" , 18 , bold = True)
clock = pygame.time.Clock()

EPSILON = 1E-1

DEVICE = 'cpu'


@jit(target_backend=DEVICE)
def newtonian_gravitational_dynamics(ringo=empty((500,4), dtype=float64), color=empty((500,3), dtype=int32), M=empty((500,1), dtype=float64), SPF=1/144, EPSILON = 1E-1):
    #x = ringo[:, :2]
    x = empty((500,2), dtype=float64)
    x = array([[a[0],a[1]] for a in ringo])
    #y = x.reshape(x.shape[0], 1, x.shape[1])
    
    #R = sqrt(einsum('ijk, ijk->ij', x-y, x-y))
    R = empty((500,500), dtype=float64)
    R = array([[((b[0]-a[0])**2+(b[1]-a[1])**2)**0.5 for b in x]for a in x])
    M_ = empty((500,500), dtype=float64)
    M_ = dot(M,M.T)
    R_ = empty((500,500,2), dtype=float64)
    R_ = array([(x - y) / (r.reshape(R.shape[0], 1) + EPSILON) for y,r in zip(x,R)])
    
    a = empty((500,2), dtype=float64)
    a = sum(((1/(M * ones((R.shape[0], R.shape[0])))) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = empty((500,2), dtype=float64)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    # Outputs
    ringi = empty((500, 4), dtype=float64)
    ringi = concatenate((x, v), axis=1)
    artu = empty((500, 5), dtype=int32)
    artu = concatenate((x.astype('int32'), color), axis=1)
    return artu, ringi

#newtonian_gravitational_dynamics()
#assert newtonian_gravitational_dynamics.nopython_signatures


#@jit(target_backend=DEVICE)
def fps_counter():
    flag = int(clock.get_fps())
    c = "RED" if flag < 5 else "YELLOW" if flag < 10 else "GREEN"
    fps = str(flag)
    fps_t = FONT.render(fps , 1, pygame.Color(c))
    DISH.blit(fps_t,(0,0))


#@jit(target_backend=DEVICE)
def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS) for cell in arty]
    fps_counter()
    pygame.display.update()


#@jit(target_backend=DEVICE)
def main():
    run = True
    
    N = 500
    M = randint(40, 64, (N, 1)).astype('float64') * 7
    M[0] = 4E7 * 7
    
    width = randint(160, 370, (N, 1)).astype('float64')
    width[0] = WIDTH/2
    height = randint(160, 370, (N, 1)).astype('float64')
    height[0] = HEIGHT/2
    
    vx = randint(240, 241, (N, 1)).astype('float64')
    vy = randint(3, 4, (N, 1)).astype('float64')*0.
    vx[0] = 0.
    vy[0] = 0.
    
    ringo = concatenate((width, height, vx, vy), axis=1)
    color = randint(150, 255, (N, 3)).astype('int32')
    time_step = 0
    lighting = BLACK
    #counter = range(N)

    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
        arty, ringo = newtonian_gravitational_dynamics(ringo, color, M=M, SPF=SPF)
        
        draw_window(arty, lighting)
        time_step += 1
        
    pygame.quit()


if __name__ == '__main__':
    main()
import pygame
from numpy import concatenate, nan_to_num, sum, dot, ones, asarray, einsum, sqrt
from numpy.random import randint
#from numba import jit

pygame.init()

G = 6.67430E-11
FPS = 20
SPF = 1E-1/FPS
RADIUS = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 1000, 1000
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Gravity Simulator Efficient CPU')
FONT = pygame.font.SysFont("Arial" , 18 , bold = True)
clock = pygame.time.Clock()

EPSILON = 1E-1


#@jit(nopython=False)
def newtonian_gravitational_dynamics(ringo, color, counter, M, SPF=1/144, WIDTH=600, HEIGHT=600):
    x = ringo[:, :2]    
    y = x.reshape(x.shape[0], 1, x.shape[1])
    
    R = sqrt(einsum('ijk, ijk->ij', x-y, x-y))
    M_ = dot(M,M.T)
    R_ = asarray([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in counter])
    
    a = sum(nan_to_num((1/(M * ones((R.shape[0], R.shape[0])))) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 2:] + (a * SPF)
    x = x + (v * SPF)
    
    # Boundry Condition
    #v = v * concatenate((where(((x[:,0] > WIDTH) * v[:,0]) > 0, -1, 1).reshape(x.shape[0], 1),
    #                    where(((x[:,1] > HEIGHT) * v[:,1]) > 0, -1, 1).reshape(x.shape[0], 1)),1) *\
    #        concatenate((where(((x[:,0] < 0) * v[:,0]) < 0, -1, 1).reshape(x.shape[0], 1),
    #                    where(((x[:,1] < 0) * v[:,1]) < 0, -1, 1).reshape(x.shape[0], 1)),1)
    
    # Outputs
    ringo = concatenate((x, v), axis=1)
    arty = concatenate((x.astype('int32'), color), axis=1)
    return arty, ringo


def fps_counter():
    flag = int(clock.get_fps())
    c = "RED" if flag < 5 else "YELLOW" if flag < 10 else "GREEN"
    fps = str(flag)
    fps_t = FONT.render(fps , 1, pygame.Color(c))
    DISH.blit(fps_t,(0,0))


def draw_window(arty, lighting):
    DISH.fill(lighting)
    [pygame.draw.circle(DISH, [cell[2], cell[3], cell[4]], (cell[0], cell[1]), RADIUS) for cell in arty]
    fps_counter()
    pygame.display.update()


def main():
    run = True
    
    N = 1000
    M = randint(3E7, 4E7, (N, 1)).astype('float64')
    M[0] = M[0]*1E10
    
    width = randint(160, 370, (N, 1)).astype('float64')
    width[0] = WIDTH/2
    height = randint(160, 370, (N, 1)).astype('float64')
    height[0] = HEIGHT/2
    
    vx = randint(240, 241, (N, 1)).astype('float64')
    vy = randint(3, 4, (N, 1)).astype('float64')*0.
    vx[0] = 0.
    vy[0] = 0.
    
    ringo = concatenate((width, height, vx, vy), axis=1)
    color = randint(150, 255, (N, 3))
    time_step = 0
    lighting = BLACK
    counter = range(N)

    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
        arty, ringo = newtonian_gravitational_dynamics(ringo, color, counter, M=M, WIDTH=WIDTH, HEIGHT=HEIGHT, SPF=SPF)
        draw_window(arty, lighting)
        time_step += 1
        
    pygame.quit()


if __name__ == '__main__':
    main()
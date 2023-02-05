import pygame
from torch import cat, nan_to_num, sum, ones, einsum, sqrt, randint, float64, int64, stack

pygame.init()

T = 2.5E-1 # step constant.
G = 1E-1 # 6.67430E-11
FPS = 14 # frame per second
SPF = T/FPS # step per frame
RADIUS = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 900, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Gravity Simulator Efficient CUDA')
FONT = pygame.font.SysFont("Arial" , 18 , bold = True)
clock = pygame.time.Clock()

EPSILON = 1E-1


def newtonian_gravitational_dynamics(ringo, color, counter, M, SPF=1/144, WIDTH=600, HEIGHT=600):
    x = ringo[:, :3]    
    y = x.reshape(x.shape[0], 1, x.shape[1])
    
    R = sqrt(einsum('ijk, ijk->ij', x-y, x-y))  
    M_ = M @ M.T
    R_ = stack([(x - x[i]) / (R[i].reshape(R.shape[0], 1) + EPSILON) for i in counter],dim=0)
    
    a = sum(nan_to_num((1/(M * ones((R.shape[0], R.shape[0])).cuda())) * G * (M_/((R ** 2) + EPSILON))).reshape(R.shape[0], R.shape[0], 1) * R_, axis=1)
    v = ringo[:, 3:] + (a * SPF)
    x = x + (v * SPF)
    
    #!!!Logic for 3D world coordinates to 2D camera projection
    x_ = x
    
    # Outputs
    ringo = cat((x, v), axis=1)
    arty = cat((x_.to(dtype = int64), color), axis=1)
    return arty, ringo


def fps_counter():
    flag = int(clock.get_fps())
    c = "RED" if flag < 5 else "YELLOW" if flag < 10 else "GREEN"
    fps = str(flag)
    fps_t = FONT.render(fps , 1, pygame.Color(c))
    DISH.blit(fps_t,(0,0))


def draw_window(arty, lighting):
    DISH.fill(lighting)
    #!!! TO DO: ADD CONDITION FOR 3D RENDRING ON SCREEN
    [pygame.draw.circle(DISH, [cell[3], cell[4], cell[5]], (cell[0], cell[1]), RADIUS)  if cell[0]>0 and cell[0]<WIDTH and cell[1]>0 and cell[1]<HEIGHT and cell[2]>-500  else '' for cell in arty.detach().cpu().numpy()]
    fps_counter()
    pygame.display.update()


def main():
    run = True
    
    N = 1000    
    M = randint(int(40), int(63), (N, 1)).to(dtype = float64).cuda() * 7
    
    width = randint(int(WIDTH/2)-100, int(WIDTH/2)+100, (N, 1)).to(dtype = float64).cuda()
    height = randint(int(HEIGHT/2)-100, int(HEIGHT/2)+100, (N, 1)).to(dtype = float64).cuda()
    depth = randint(-100, 100, (N, 1)).to(dtype = float64).cuda()
    
    vx = randint(-2, 2, (N, 1)).to(dtype = float64).cuda()
    vy = randint(-2, 2, (N, 1)).to(dtype = float64).cuda()
    vz = randint(-2, 2, (N, 1)).to(dtype = float64).cuda()  
    
    ringo = cat((width, height, depth, vx, vy, vz), axis=1).cuda()
    color = randint(int(150), int(255), (N, 3)).cuda()   
    
    time_step = 0
    lighting = BLACK
    counter = range(N)
    
    while run:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                
        arty, ringo = newtonian_gravitational_dynamics(ringo, color, counter, M = M, WIDTH = WIDTH, HEIGHT = HEIGHT, SPF = SPF)
        draw_window(arty, lighting)
        time_step += 1
        
    pygame.quit()


if __name__ == '__main__':
    main()
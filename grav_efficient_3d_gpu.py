import pygame
from torch import cat, nan_to_num, sum, ones, einsum, sqrt, randint, float64, int64, stack

T = 1E1 # step constant.
G = 1E-1 # 6.67430E-11
FPS = 240 # frame per second
SPF = T * 1/FPS # step per frame
RADIUS = 3

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 900, 900
DISH = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('2D Gravity Simulator Efficient CUDA')

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


def draw_window(arty, lighting):
    DISH.fill(lighting)
    #!!! TO DO: ADD CONDITION FOR 3D RENDRING ON SCREEN
    [pygame.draw.circle(DISH, [cell[3], cell[4], cell[5]], (cell[0], cell[1]), RADIUS)  if cell[0]>0 and cell[0]<WIDTH and cell[1]>0 and cell[1]<HEIGHT and cell[2]>0 and cell[2]<HEIGHT  else '' for cell in arty.detach().cpu().numpy()]
    pygame.display.update()


def main():
    clock = pygame.time.Clock()
    run = True
    
    N = 500    
    M = randint(int(2), int(63), (N, 1)).to(dtype = float64).cuda() + (((2 + 63)/2) * 6)
    
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